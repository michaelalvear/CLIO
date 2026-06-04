"""
Flask server for URSA.
/region/select  — deterministic Xarray slicing driven by the user's map selection
/query          — LLM interpreter that receives the current selection as context
/dataset/info   — dataset metadata for populating the frontend UI
"""
import os
import signal
import traceback
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from pyproj import Transformer

from ursa.agent.orchestration import DS, run_agent
from ursa.data_processor import process_region

load_dotenv()

app = Flask(__name__)
CORS(app)

CURRENT_DIR  = Path(__file__).resolve().parent
REPO_ROOT    = CURRENT_DIR.parent.parent
FRONTEND_DIR = REPO_ROOT / "frontend"


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/region/select", methods=["POST"])
def region_select():
    """
    Deterministically slice the dataset to the user's map selection and return
    visualization data + summary statistics.

    Expected body:
      {
        "bbox":       {"sw": [lat, lon], "ne": [lat, lon]},
        "time_range": ["YYYY-MM-DD", "YYYY-MM-DD"],
        "variable":   "salinity"
      }
    """
    body = request.get_json()

    required = ("bbox", "time_range", "variable")
    if not body or any(k not in body for k in required):
        return jsonify({"error": f"Body must include: {', '.join(required)}"}), 400

    try:
        result = process_region(
            dataset=DS,
            bbox_latlon=body["bbox"],
            time_range=body["time_range"],
            variable=body["variable"],
        )
        return jsonify(result), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/query", methods=["POST"])
def query():
    """
    Run the LLM interpreter. The frontend forwards the current selection
    context so the agent can reference it without doing any data work itself.

    Expected body:
      {
        "message":          "...",
        "history":          [...],          (optional)
        "selectionContext": {...}           (optional, from last /region/select)
      }
    """
    body = request.get_json()

    if not body or "message" not in body:
        return jsonify({"error": "Body must include a 'message' field"}), 400

    try:
        result = run_agent(
            user_message=body["message"],
            history=body.get("history", []),
            selection_context=body.get("selectionContext"),
        )
        return jsonify({
            "textResponse": result["text"],
            "toolLog":      result["toolLog"],
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/dataset/info", methods=["GET"])
def dataset_info():
    _utm_to_latlon = Transformer.from_crs("EPSG:26917", "EPSG:4326", always_xy=True)

    x_min = float(DS["x"].min())
    x_max = float(DS["x"].max())
    y_min = float(DS["y"].min())
    y_max = float(DS["y"].max())
    lon_sw, lat_sw = _utm_to_latlon.transform(x_min, y_min)
    lon_ne, lat_ne = _utm_to_latlon.transform(x_max, y_max)

    return jsonify({
        "variables":  list(DS.data_vars),
        "time_range": {
            "start": str(DS["time"].values[0])[:10],
            "end":   str(DS["time"].values[-1])[:10],
        },
        "spatial_bounds": {
            "x_min": x_min, "x_max": x_max,
            "y_min": y_min, "y_max": y_max,
        },
        "lat_lon_bounds": {
            "sw": [round(lat_sw, 5), round(lon_sw, 5)],
            "ne": [round(lat_ne, 5), round(lon_ne, 5)],
        },
    }), 200


def _on_sigterm(signum, frame):
    print("\n[SIGTERM received] Stack trace:", flush=True)
    traceback.print_stack(frame)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _on_sigterm)
    app.run(debug=False, port=5001)
