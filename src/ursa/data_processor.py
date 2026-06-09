"""
Deterministic Xarray processing for user-defined map selections.
All data slicing lives here, entirely outside the agent.
Works with any CF-compliant rectilinear NetCDF dataset.
"""
import numpy as np
import matplotlib.colors as mcolors
import base64
from io import BytesIO

import xarray as xr
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderServiceError

from ursa import config  # sets matplotlib backend to Agg before pyplot import
import matplotlib.pyplot as plt
from ursa.cf_utils import detect_crs, get_cf_axes, get_transformers

_geolocator = Nominatim(user_agent="ursa_data_processor", timeout=5)


def _safe_slice(coord_array, lo, hi):
    """Return a slice that respects ascending or descending coordinate order."""
    if len(coord_array) > 1 and coord_array[0] > coord_array[-1]:
        return slice(hi, lo)
    return slice(lo, hi)


def _reverse_geocode(bbox_latlon: dict) -> str | None:
    """Return a human-readable place name for the bbox centroid, or None on failure."""
    sw_lat, sw_lon = bbox_latlon["sw"]
    ne_lat, ne_lon = bbox_latlon["ne"]
    lat = (sw_lat + ne_lat) / 2
    lon = (sw_lon + ne_lon) / 2
    try:
        result = _geolocator.reverse((lat, lon), language="en")
        return result.address if result else None
    except (GeocoderServiceError, Exception):
        return None


def process_region(
    dataset: xr.Dataset,
    bbox_latlon: dict,
    time_range: list,
    variable: str,
) -> dict:
    """
    Slice the dataset to a lat/lon bounding box and time range, then return
    a heatmap image, time-series data, and summary statistics.

    bbox_latlon: {"sw": [lat, lon], "ne": [lat, lon]}
    time_range:  ["YYYY-MM-DD", "YYYY-MM-DD"]
    variable:    data variable name
    """
    crs                   = detect_crs(dataset)
    axes                  = get_cf_axes(dataset)
    x_dim, y_dim, t_dim   = axes["x_dim"], axes["y_dim"], axes["t_dim"]
    to_latlon, from_latlon = get_transformers(crs)

    sw_lat, sw_lon = bbox_latlon["sw"]
    ne_lat, ne_lon = bbox_latlon["ne"]

    if from_latlon is None:
        # Dataset is already in lat/lon
        x_min, x_max = sw_lon, ne_lon
        y_min, y_max = sw_lat, ne_lat
    else:
        x_min, y_min = from_latlon.transform(sw_lon, sw_lat)
        x_max, y_max = from_latlon.transform(ne_lon, ne_lat)

    x_vals = dataset[x_dim].values
    y_vals = dataset[y_dim].values

    selection = dataset.sel({
        x_dim: _safe_slice(x_vals, x_min, x_max),
        y_dim: _safe_slice(y_vals, y_min, y_max),
        t_dim: slice(time_range[0], time_range[1]),
    })

    if any(selection.sizes.get(dim, 0) == 0 for dim in (x_dim, y_dim, t_dim)):
        raise ValueError(
            "No data found in the selected region and time range. "
            "Try expanding your bounding box or adjusting the dates."
        )

    da = selection[variable]

    # ── Statistics ────────────────────────────────────────────────────────────
    flat  = da.values.flatten()
    valid = flat[~np.isnan(flat)]

    stats = {
        "variable":    variable,
        "long_name":   str(da.attrs.get("long_name", variable)),
        "units":       str(da.attrs.get("units", "unknown")),
        "time_range":  [
            str(selection[t_dim].values[0])[:10],
            str(selection[t_dim].values[-1])[:10],
        ],
        "point_count": int(da.size),
        "null_pct":    round(float(da.isnull().mean() * 100), 1),
        "mean": round(float(valid.mean()), 4) if len(valid) else None,
        "max":  round(float(valid.max()),  4) if len(valid) else None,
        "min":  round(float(valid.min()),  4) if len(valid) else None,
        "std":  round(float(valid.std()),  4) if len(valid) else None,
        "location_name": _reverse_geocode(bbox_latlon),
    }

    # ── Heatmap (time mean → spatial PNG) ─────────────────────────────────────
    spatial = da.mean(dim=t_dim)

    # Ensure north-up rendering (y must decrease top→bottom in the image array)
    y_coord_vals = spatial[y_dim].values
    if y_coord_vals[0] < y_coord_vals[-1]:
        spatial = spatial.isel({y_dim: slice(None, None, -1)})

    grid = spatial.transpose(y_dim, x_dim).values

    max_val = float(np.nanmax(grid)) if not np.all(np.isnan(grid)) else 1.0
    cmap    = plt.get_cmap("turbo")
    norm    = mcolors.Normalize(vmin=0, vmax=max_val)
    rgba    = cmap(norm(grid))
    rgba[np.isnan(grid), 3] = 0.0  # transparent NaN (land / out-of-domain mask)

    buf = BytesIO()
    plt.imsave(buf, rgba, format="png")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")

    sel_x = selection[x_dim].values
    sel_y = selection[y_dim].values

    if to_latlon is None:
        lat_sw, lon_sw = float(sel_y.min()), float(sel_x.min())
        lat_ne, lon_ne = float(sel_y.max()), float(sel_x.max())
    else:
        lon_sw, lat_sw = to_latlon.transform(float(sel_x.min()), float(sel_y.min()))
        lon_ne, lat_ne = to_latlon.transform(float(sel_x.max()), float(sel_y.max()))

    heatmap = {
        "image_b64": img_b64,
        "bounds":    [[round(lat_sw, 5), round(lon_sw, 5)],
                      [round(lat_ne, 5), round(lon_ne, 5)]],
        "max_val":   round(max_val, 4),
    }

    # ── Time series (spatial mean over selected region) ────────────────────────
    ts     = da.mean(dim=[x_dim, y_dim])
    labels = [str(t)[:10] for t in selection[t_dim].values]
    values = [None if np.isnan(v) else round(float(v), 4) for v in ts.values.flatten()]

    timeseries = {"labels": labels, "data": values}

    return {"heatmap": heatmap, "timeseries": timeseries, "stats": stats}
