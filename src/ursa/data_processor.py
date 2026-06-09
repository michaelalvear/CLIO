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


def _trend_per_year(ts_values: np.ndarray, t_values) -> float | None:
    """
    Fit a linear trend to a time series and return the slope in units per year.
    Handles both numpy datetime64 and cftime coordinate arrays.
    Returns None if there are fewer than 2 valid points.
    """
    y = ts_values.flatten().astype(float)
    valid = ~np.isnan(y)
    if valid.sum() < 2:
        return None
    try:
        days = (t_values - t_values[0]) / np.timedelta64(1, "D")
    except TypeError:
        # cftime objects
        ord0 = t_values[0].toordinal()
        days = np.array([t.toordinal() - ord0 for t in t_values], dtype=float)
    x = np.asarray(days, dtype=float) / 365.25
    slope, _ = np.polyfit(x[valid], y[valid], 1)
    return round(float(slope), 6)


def _reverse_geocode(bbox_latlon: dict) -> list[str]:
    """
    Reverse geocode five points (four corners + centroid) of the bbox and return
    a deduplicated list of unique place names. Nominatim requires ≤1 req/sec so
    a 1-second pause is inserted between calls.
    """
    import time
    sw_lat, sw_lon = bbox_latlon["sw"]
    ne_lat, ne_lon = bbox_latlon["ne"]
    mid_lat = (sw_lat + ne_lat) / 2
    mid_lon = (sw_lon + ne_lon) / 2

    points = [
        (sw_lat,  sw_lon),   # SW corner
        (sw_lat,  ne_lon),   # SE corner
        (ne_lat,  sw_lon),   # NW corner
        (ne_lat,  ne_lon),   # NE corner
        (mid_lat, mid_lon),  # centroid
    ]

    seen   = set()
    names  = []
    for i, (lat, lon) in enumerate(points):
        if i > 0:
            time.sleep(1.1)
        try:
            result = _geolocator.reverse((lat, lon), language="en")
            if result and result.address not in seen:
                seen.add(result.address)
                names.append(result.address)
        except (GeocoderServiceError, Exception):
            pass

    return names


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

    # ── Time-mean spatial grid (used for both stats and heatmap) ──────────────
    spatial = da.mean(dim=t_dim)

    # ── Statistics ────────────────────────────────────────────────────────────
    flat  = da.values.flatten()
    valid = flat[~np.isnan(flat)]

    # Trend: slope of the spatially-averaged time series in units per year
    ts_raw   = da.mean(dim=[x_dim, y_dim])
    t_coords = selection[t_dim].values
    trend    = _trend_per_year(ts_raw.values, t_coords)

    # Location of maximum value in the time-mean grid
    spatial_vals = spatial.values
    max_location = None
    if not np.all(np.isnan(spatial_vals)):
        flat_idx   = int(np.nanargmax(spatial_vals))
        dim_order  = spatial.dims
        idx_map    = dict(zip(dim_order, np.unravel_index(flat_idx, spatial_vals.shape)))
        max_x      = float(spatial[x_dim].values[idx_map[x_dim]])
        max_y      = float(spatial[y_dim].values[idx_map[y_dim]])
        if to_latlon is None:
            max_lat, max_lon = max_y, max_x
        else:
            max_lon, max_lat = to_latlon.transform(max_x, max_y)
        max_location = {"lat": round(max_lat, 5), "lon": round(max_lon, 5)}

    stats = {
        "variable":    variable,
        "long_name":   str(da.attrs.get("long_name", variable)),
        "units":       str(da.attrs.get("units", "unknown")),
        "time_range":  [
            str(t_coords[0])[:10],
            str(t_coords[-1])[:10],
        ],
        "point_count": int(da.size),
        "null_pct":    round(float(da.isnull().mean() * 100), 1),
        "mean":              round(float(valid.mean()), 4) if len(valid) else None,
        "max":               round(float(valid.max()),  4) if len(valid) else None,
        "min":               round(float(valid.min()),  4) if len(valid) else None,
        "std":               round(float(valid.std()),  4) if len(valid) else None,
        "trend_per_year":    trend,
        "max_value_location": max_location,
        "location_names":    _reverse_geocode(bbox_latlon),
    }

    # ── Heatmap (time mean → spatial PNG) ─────────────────────────────────────
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
    labels = [str(t)[:10] for t in t_coords]
    values = [None if np.isnan(v) else round(float(v), 4) for v in ts_raw.values.flatten()]

    timeseries = {"labels": labels, "data": values}

    return {"heatmap": heatmap, "timeseries": timeseries, "stats": stats}
