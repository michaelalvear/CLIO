#!/usr/bin/env bash
# Build the image with Cloud Build and deploy it to Cloud Run: private by default,
# public (no sign-in) with PUBLIC=true.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require NETCDF_FILE CHROMADB_COLLECTION GEMINI_MODEL MIN_INSTANCES MAX_INSTANCES \
        DAILY_QUERY_LIMIT QUERY_RATE_PER_MINUTE REGION_RATE_PER_MINUTE \
        QUERY_DAILY_PER_CLIENT REGION_DAILY_PER_CLIENT REGION_DAILY_LIMIT \
        MAX_MESSAGE_CHARS MAX_REQUEST_BYTES

case "$GEMINI_MODEL" in
    *latest*)
        echo "GEMINI_MODEL must be a pinned model ID, not '$GEMINI_MODEL'." >&2
        exit 1
        ;;
esac

cd "$REPO_ROOT"
# IMAGE may be set to deploy an image that is already built (eval/deploy_arms.sh).
BUILD_IMAGE=false
if [ -z "${IMAGE:-}" ]; then
    IMAGE="$IMAGE_BASE:$(git rev-parse --short HEAD)-$(date -u +%Y%m%d%H%M%S)"
    BUILD_IMAGE=true
fi
DATA="/data/$DATA_VERSION"

ENV_VARS=(
    "NETCDF_DATA_PATH=$DATA/netcdf/$NETCDF_FILE"
    "CHROMADB_PATH=/tmp/chromadb"
    "CHROMADB_COLLECTION=$CHROMADB_COLLECTION"
    "DOCS_PATH=$DATA/documents"
    "GEMINI_MODEL=$GEMINI_MODEL"
    "DAILY_QUERY_LIMIT=$DAILY_QUERY_LIMIT"
    "QUERY_RATE_PER_MINUTE=$QUERY_RATE_PER_MINUTE"
    "REGION_RATE_PER_MINUTE=$REGION_RATE_PER_MINUTE"
    "QUERY_DAILY_PER_CLIENT=$QUERY_DAILY_PER_CLIENT"
    "REGION_DAILY_PER_CLIENT=$REGION_DAILY_PER_CLIENT"
    "REGION_DAILY_LIMIT=$REGION_DAILY_LIMIT"
    "REGION_MAX_CONCURRENT=${REGION_MAX_CONCURRENT:-2}"
    "EXPORT_RATE_PER_MINUTE=${EXPORT_RATE_PER_MINUTE:-3}"
    "EXPORT_DAILY_PER_CLIENT=${EXPORT_DAILY_PER_CLIENT:-20}"
    "EXPORT_DAILY_LIMIT=${EXPORT_DAILY_LIMIT:-100}"
    "MAX_EXPORT_MB=${MAX_EXPORT_MB:-30}"
    "MAX_EXPORT_SOURCE_MB=${MAX_EXPORT_SOURCE_MB:-300}"
    "LIMITS_STORE=${LIMITS_STORE:-memory}"
    "MAX_MESSAGE_CHARS=$MAX_MESSAGE_CHARS"
    "MAX_REQUEST_BYTES=$MAX_REQUEST_BYTES"
    "ALLOWED_ORIGINS=${ALLOWED_ORIGINS:-}"
    "NOMINATIM_USER_AGENT=${NOMINATIM_USER_AGENT:-}"
    "NOMINATIM_MIN_INTERVAL=${NOMINATIM_MIN_INTERVAL:-1.1}"
    "AUDIT_LOG=${AUDIT_LOG:-false}"
)
# RAG_ENABLED=false starts with an empty knowledge base, e.g. before the Chroma DB is built.
if [ "${RAG_ENABLED:-true}" = true ]; then
    ENV_VARS+=("CHROMADB_SOURCE_PATH=$DATA/chromadb")
fi
# "^|^" tells gcloud to split on | instead of commas: ALLOWED_ORIGINS may hold commas
# and NOMINATIM_USER_AGENT usually holds an email address.
ENV_ARG="^|^$(IFS="|"; echo "${ENV_VARS[*]}")"

if [ "$BUILD_IMAGE" = true ]; then
    echo "==> Building $IMAGE"
    gcloud builds submit --project "$PROJECT_ID" --region "$REGION" --tag "$IMAGE" .
fi

if [ "${PUBLIC:-false}" = true ]; then
    ACCESS=--allow-unauthenticated
else
    ACCESS=--no-allow-unauthenticated
fi

echo "==> Deploying $SERVICE (data $DATA_VERSION, $NETCDF_FILE, model $GEMINI_MODEL, $ACCESS)"
gcloud run deploy "$SERVICE" \
    --project "$PROJECT_ID" \
    --region "$REGION" \
    --image "$IMAGE" \
    --service-account "$SERVICE_ACCOUNT" \
    "$ACCESS" \
    --execution-environment gen2 \
    --cpu "${CPU:-2}" --memory "${MEMORY:-4Gi}" \
    --concurrency 8 \
    --timeout 300 \
    --cpu-boost \
    --min-instances "$MIN_INSTANCES" \
    --max-instances "$MAX_INSTANCES" \
    --add-volume "name=data,type=cloud-storage,bucket=$BUCKET,readonly=true" \
    --add-volume-mount "volume=data,mount-path=/data" \
    --set-secrets "GEMINI_API_KEY=$SECRET_NAME:latest,GOOGLE_API_KEY=$SECRET_NAME:latest" \
    --set-env-vars "$ENV_ARG"

echo
if [ "${PUBLIC:-false}" = true ]; then
    echo "Deployed publicly at:"
    gcloud run services describe "$SERVICE" --project "$PROJECT_ID" --region "$REGION" \
        --format 'value(status.url)'
else
    echo "Deployed. To open it, run:"
    echo "  gcloud run services proxy $SERVICE --project $PROJECT_ID --region $REGION --port 8080"
    echo "then browse to http://localhost:8080"
fi
