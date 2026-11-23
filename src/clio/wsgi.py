"""
Production entry point for CLIO (gunicorn clio.wsgi:app).

Wraps the Flask app from clio.app with usage guardrails for public hosting.
The routes and their behavior are unchanged; local development still uses
`python -m clio.app`, which does not load this module.

Guardrails (all configurable via env vars):
  MAX_REQUEST_BYTES          largest accepted request body           (default 262144)
  MAX_MESSAGE_CHARS          longest /query message                  (default 4000)
  QUERY_RATE_PER_MINUTE      /query calls per client per minute      (default 6)
  REGION_RATE_PER_MINUTE     /region/select calls per client/minute  (default 6)
  QUERY_DAILY_PER_CLIENT     /query calls per client per UTC day     (default 20)
  REGION_DAILY_PER_CLIENT    /region/select calls per client per day (default 60)
  DAILY_QUERY_LIMIT          /query calls per instance per UTC day   (default 200)
  REGION_DAILY_LIMIT         /region/select calls per instance/day   (default 300)
  EXPORT_RATE_PER_MINUTE     /region/export calls per client/minute  (default 3)
  EXPORT_DAILY_PER_CLIENT    /region/export calls per client per day (default 20)
  EXPORT_DAILY_LIMIT         /region/export calls per instance/day   (default 100)
  MAX_EXPORT_MB              largest /region/export file, compressed (default 30, read
                             by clio.app; keep under Cloud Run's 32 MiB response cap)
  MAX_EXPORT_SOURCE_MB       largest selection /region/export will   (default 300, read
                             write, uncompressed                      by clio.app)
  REGION_MAX_CONCURRENT      /region/select and /region/export calls processed at
                             once, shared                            (default 2)
  LIMITS_STORE               "memory" (default) or "firestore". Daily counts kept in
                             memory reset whenever Cloud Run restarts or scales the
                             instance to zero; "firestore" keeps them across restarts
                             and instances (clients are stored as salted hashes).
  ALLOWED_ORIGINS            comma-separated origins allowed by CORS (default: none,
                             the bundled frontend is same-origin and needs no CORS)

Gemini quota and overload errors (429 RESOURCE_EXHAUSTED, 503 UNAVAILABLE)
are returned as 429/503 with a plain message instead of a generic 500.

OpenStreetMap Nominatim (reverse geocoding in /region/select):
  Lookups are serialized across threads, spaced at least NOMINATIM_MIN_INTERVAL
  seconds apart (default 1.1, the usage policy allows 1 request/second) and
  cached, so concurrent users cannot exceed the policy from one instance.
  NOMINATIM_USER_AGENT       identifying User-Agent with a contact (optional)

Audit log (off by default):
  AUDIT_LOG=true             write one structured log entry per /query with the
                             question, the selection statistics, the answer and the
                             retrieved sources/pages, for benchmarking and for
                             human-on-the-loop review. Records what users type.
"""
import hashlib
import json
import os
import re
import threading
import time
from datetime import datetime, timedelta, timezone

from flask import g, request, jsonify

import clio.data_processor as data_processor
from clio.app import app

MAX_REQUEST_BYTES      = int(os.getenv("MAX_REQUEST_BYTES", "262144"))
MAX_MESSAGE_CHARS      = int(os.getenv("MAX_MESSAGE_CHARS", "4000"))
QUERY_RATE_PER_MINUTE  = int(os.getenv("QUERY_RATE_PER_MINUTE", "6"))
REGION_RATE_PER_MINUTE = int(os.getenv("REGION_RATE_PER_MINUTE", "6"))
QUERY_DAILY_PER_CLIENT  = int(os.getenv("QUERY_DAILY_PER_CLIENT", "20"))
REGION_DAILY_PER_CLIENT = int(os.getenv("REGION_DAILY_PER_CLIENT", "60"))
DAILY_QUERY_LIMIT      = int(os.getenv("DAILY_QUERY_LIMIT", "200"))
REGION_DAILY_LIMIT     = int(os.getenv("REGION_DAILY_LIMIT", "300"))
EXPORT_RATE_PER_MINUTE  = int(os.getenv("EXPORT_RATE_PER_MINUTE", "3"))
EXPORT_DAILY_PER_CLIENT = int(os.getenv("EXPORT_DAILY_PER_CLIENT", "20"))
EXPORT_DAILY_LIMIT      = int(os.getenv("EXPORT_DAILY_LIMIT", "100"))
REGION_MAX_CONCURRENT  = int(os.getenv("REGION_MAX_CONCURRENT", "2"))
LIMITS_STORE           = os.getenv("LIMITS_STORE", "memory").lower()
ALLOWED_ORIGINS        = {o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()}
NOMINATIM_MIN_INTERVAL = float(os.getenv("NOMINATIM_MIN_INTERVAL", "1.1"))
NOMINATIM_USER_AGENT   = os.getenv("NOMINATIM_USER_AGENT", "")
AUDIT_LOG              = os.getenv("AUDIT_LOG", "").lower() in ("1", "true", "yes")
GEMINI_MODEL           = os.getenv("GEMINI_MODEL", "gemini-pro-latest")
DATASET                = os.path.basename(os.getenv("NETCDF_DATA_PATH", ""))

RATE_LIMITS = {
    "query":         QUERY_RATE_PER_MINUTE,
    "region_select": REGION_RATE_PER_MINUTE,
    "region_export": EXPORT_RATE_PER_MINUTE,
}
DAILY_PER_CLIENT = {
    "query":         QUERY_DAILY_PER_CLIENT,
    "region_select": REGION_DAILY_PER_CLIENT,
    "region_export": EXPORT_DAILY_PER_CLIENT,
}
DAILY_LIMITS = {
    "query":         DAILY_QUERY_LIMIT,
    "region_select": REGION_DAILY_LIMIT,
    "region_export": EXPORT_DAILY_LIMIT,
}
DAILY_LIMIT_MESSAGES = {
    "query":         "CLIO has reached its daily question limit. Please try again tomorrow.",
    "region_select": "CLIO has reached its daily map selection limit. Please try again tomorrow.",
    "region_export": "CLIO has reached its daily download limit. Please try again tomorrow.",
}
CLIENT_LIMIT_MESSAGES = {
    "query":         "You have reached today's question limit. Please try again tomorrow.",
    "region_select": "You have reached today's map selection limit. Please try again tomorrow.",
    "region_export": "You have reached today's download limit. Please try again tomorrow.",
}

app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_BYTES

_lock          = threading.Lock()
_minute_window = 0
_minute_counts = {}     # (endpoint, client) -> calls in the current minute
_day           = ""
_day_totals    = {}     # endpoint -> calls today on this instance
_day_counts    = {}     # (endpoint, client) -> calls today
_region_slots  = threading.BoundedSemaphore(REGION_MAX_CONCURRENT)

_store = None
if LIMITS_STORE == "firestore":
    from google.cloud import firestore
    _store = firestore.Client()
    _NAMESPACE = os.getenv("K_SERVICE", "clio")   # Cloud Run sets K_SERVICE


def _client_id() -> str:
    # Cloud Run's front end appends the caller's real IP as the last
    # X-Forwarded-For entry; earlier entries can be set by the client.
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[-1].strip()
    return request.remote_addr or "unknown"


def _too_many(message: str, retry_after: int = 60):
    response = jsonify({"error": message})
    response.status_code = 429
    response.headers["Retry-After"] = str(retry_after)
    return response


def _take_daily_in_memory(endpoint: str, client: str):
    """Count one call against today's limits; return an error message if over."""
    global _day, _day_totals, _day_counts
    with _lock:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != _day:
            _day, _day_totals, _day_counts = today, {}, {}
        key = (endpoint, client)
        if _day_counts.get(key, 0) >= DAILY_PER_CLIENT[endpoint]:
            return CLIENT_LIMIT_MESSAGES[endpoint]
        if _day_totals.get(endpoint, 0) >= DAILY_LIMITS[endpoint]:
            return DAILY_LIMIT_MESSAGES[endpoint]
        _day_counts[key] = _day_counts.get(key, 0) + 1
        _day_totals[endpoint] = _day_totals.get(endpoint, 0) + 1
    return None


def _take_daily_in_firestore(endpoint: str, client: str):
    """Same as _take_daily_in_memory, but shared by all instances and restarts."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    client_hash = hashlib.sha256(f"{today}|{client}".encode()).hexdigest()[:32]
    day_ref = _store.collection("clio_daily_totals").document(f"{_NAMESPACE}_{today}")
    client_ref = _store.collection("clio_daily_clients").document(f"{_NAMESPACE}_{today}_{client_hash}")
    expire_at = datetime.now(timezone.utc) + timedelta(days=2)   # for a Firestore TTL policy

    @firestore.transactional
    def take(transaction):
        totals = day_ref.get(transaction=transaction).to_dict() or {}
        mine = client_ref.get(transaction=transaction).to_dict() or {}
        if mine.get(endpoint, 0) >= DAILY_PER_CLIENT[endpoint]:
            return CLIENT_LIMIT_MESSAGES[endpoint]
        if totals.get(endpoint, 0) >= DAILY_LIMITS[endpoint]:
            return DAILY_LIMIT_MESSAGES[endpoint]
        transaction.set(day_ref, {endpoint: totals.get(endpoint, 0) + 1, "expireAt": expire_at}, merge=True)
        transaction.set(client_ref, {endpoint: mine.get(endpoint, 0) + 1, "expireAt": expire_at}, merge=True)
        return None

    return take(_store.transaction())


def _take_daily(endpoint: str, client: str):
    if _store is not None:
        try:
            return _take_daily_in_firestore(endpoint, client)
        except Exception as error:   # keep serving with per-instance limits
            print(json.dumps({"severity": "ERROR", "message": "clio.limits.firestore",
                              "error": str(error)}), flush=True)
    return _take_daily_in_memory(endpoint, client)


@app.before_request
def _enforce_limits():
    global _minute_window, _minute_counts

    endpoint = request.endpoint
    if endpoint not in RATE_LIMITS:
        return None

    if endpoint == "query":
        body = request.get_json(silent=True) or {}
        message = body.get("message")
        if isinstance(message, str) and len(message) > MAX_MESSAGE_CHARS:
            return jsonify({"error": f"Message is too long (limit {MAX_MESSAGE_CHARS} characters)."}), 413

    client = _client_id()
    with _lock:
        window = int(time.time() // 60)
        if window != _minute_window:
            _minute_window, _minute_counts = window, {}
        key = (endpoint, client)
        if _minute_counts.get(key, 0) >= RATE_LIMITS[endpoint]:
            return _too_many("Too many requests. Please wait a minute and try again.")
        _minute_counts[key] = _minute_counts.get(key, 0) + 1

    # Map selections and downloads load data into memory; a few at a time keeps the instance up.
    if endpoint in ("region_select", "region_export"):
        if not _region_slots.acquire(blocking=False):
            return _too_many("CLIO is busy with other map selections. Please try again in a minute.")
        g.region_slot = True

    over = _take_daily(endpoint, client)
    if over:
        return _too_many(over, retry_after=3600)
    return None


@app.teardown_request
def _release_region_slot(_error):
    if g.pop("region_slot", False):
        _region_slots.release()


@app.after_request
def _hide_internal_errors(response):
    # The routes return str(exception) on 500s; keep details in the logs only,
    # but tell users plainly when Gemini's quota is used up or it is overloaded.
    if response.status_code >= 500 and response.is_json:
        error = str((response.get_json(silent=True) or {}).get("error", ""))
        if "RESOURCE_EXHAUSTED" in error:
            message, response.status_code =                 "CLIO has reached its AI usage limit for now. Please try again later.", 429
            response.headers["Retry-After"] = "3600"
        elif "UNAVAILABLE" in error:
            message, response.status_code =                 "The AI model is busy right now. Please try again in a few minutes.", 503
            response.headers["Retry-After"] = "120"
        else:
            message = "Internal server error."
        response.set_data(jsonify({"error": message}).get_data())
    return response


_SOURCE_RE = re.compile(r"SOURCE:\s*([^\n]+)")
_PAGE_RE   = re.compile(r"PAGE:\s*([^\n]+)")


def _retrieved_sources(tool_log: list) -> list:
    """Source title + page of every chunk the RAG tool returned (same parsing as the frontend)."""
    sources = []
    for entry in tool_log or []:
        if entry.get("type") != "result":
            continue
        for chunk in str(entry.get("content", "")).split("--- DOCUMENT CHUNK ---"):
            source, page = _SOURCE_RE.search(chunk), _PAGE_RE.search(chunk)
            if source:
                sources.append({"source": source.group(1).strip(),
                                "page": page.group(1).strip() if page else None})
    return sources


# Registered after _hide_internal_errors, so it runs first and logs the real error.
@app.after_request
def _audit_query(response):
    if not AUDIT_LOG or request.endpoint != "query":
        return response
    # An oversized body was never read; reading it here would raise again.
    body = {} if response.status_code == 413 else (request.get_json(silent=True) or {})
    data = response.get_json(silent=True) or {}
    print(json.dumps({
        "severity":     "INFO" if response.status_code < 500 else "ERROR",
        "message":      "clio.query",
        "status":       response.status_code,
        "model":        GEMINI_MODEL,
        "dataset":      DATASET,
        "question":     body.get("message"),
        "historyTurns": len(body.get("history") or []),
        "selection":    body.get("selectionContext"),
        "answer":       data.get("textResponse"),
        "error":        data.get("error"),
        "toolCalls":    [e.get("args") for e in data.get("toolLog") or [] if e.get("type") == "call"],
        "retrieved":    _retrieved_sources(data.get("toolLog")),
    }, default=str), flush=True)
    return response


@app.errorhandler(413)
def _request_too_large(_error):
    return jsonify({"error": f"Request is too large (limit {MAX_REQUEST_BYTES} bytes)."}), 413


# ── OpenStreetMap Nominatim usage policy ────────────────────────────────────
# data_processor pauses between the five lookups of one selection, but
# concurrent selections on different threads would still overlap. Serialize
# all lookups, space them out and cache them, without changing their results.

_geocode_lock  = threading.Lock()
_geocode_last  = 0.0
_geocode_cache = {}
_raw_reverse   = data_processor._geolocator.reverse


def _polite_reverse(query, *args, **kwargs):
    global _geocode_last
    lat, lon = query
    key = (round(float(lat), 5), round(float(lon), 5), kwargs.get("language"))
    with _geocode_lock:
        if key in _geocode_cache:
            return _geocode_cache[key]
        wait = _geocode_last + NOMINATIM_MIN_INTERVAL - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        try:
            result = _raw_reverse(query, *args, **kwargs)
        finally:
            _geocode_last = time.monotonic()
        if len(_geocode_cache) >= 5000:
            _geocode_cache.clear()
        _geocode_cache[key] = result
        return result


data_processor._geolocator.reverse = _polite_reverse
if NOMINATIM_USER_AGENT:
    data_processor._geolocator.headers["User-Agent"] = NOMINATIM_USER_AGENT


class _RestrictCors:
    """Drop CORS headers for origins not in ALLOWED_ORIGINS.

    clio.app enables CORS for every origin; this WSGI middleware narrows it
    without touching that module.
    """

    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        origin = environ.get("HTTP_ORIGIN", "")

        def _start_response(status, headers, exc_info=None):
            if origin not in ALLOWED_ORIGINS:
                headers = [(k, v) for k, v in headers
                           if not k.lower().startswith("access-control-")]
            return start_response(status, headers, exc_info)

        return self.wsgi_app(environ, _start_response)


app.wsgi_app = _RestrictCors(app.wsgi_app)

# Record which model and dataset this revision serves, so answers in the logs
# can be matched to the configuration that was benchmarked.
print(json.dumps({
    "severity": "INFO",
    "message":  "clio.startup",
    "model":    GEMINI_MODEL,
    "dataset":  DATASET,
    "auditLog": AUDIT_LOG,
}), flush=True)
