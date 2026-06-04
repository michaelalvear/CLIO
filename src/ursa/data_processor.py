"""
Deterministic Xarray processing for user-defined map selections.
All data slicing lives here, entirely outside the agent.
"""
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import base64
from io import BytesIO

import xarray as xr
from pyproj import Transformer

from ursa import config  # sets matplotlib backend to Agg before pyplot import
import matplotlib.pyplot as plt

_utm_to_latlon = Transformer.from_crs("EPSG:26917", "EPSG:4326", always_xy=True)
_latlon_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:26917", always_xy=True)


def _safe_slice(coord_array, lo, hi):
    """Return a slice that respects the coordinate's ascending/descending order."""
    if len(coord_array) > 1 and coord_array[0] > coord_array[-1]:
        return slice(hi, lo)
    return slice(lo, hi)


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
    variable:    name of the data variable to extract (e.g. "salinity")
    """
    sw_lat, sw_lon = bbox_latlon["sw"]
    ne_lat, ne_lon = bbox_latlon["ne"]

    x_min, y_min = _latlon_to_utm.transform(sw_lon, sw_lat)
    x_max, y_max = _latlon_to_utm.transform(ne_lon, ne_lat)

    t_start = pd.to_datetime(time_range[0])
    t_end   = pd.to_datetime(time_range[1])

    x_vals = dataset["x"].values
    y_vals = dataset["y"].values

    selection = dataset.sel(
        x=_safe_slice(x_vals, x_min, x_max),
        y=_safe_slice(y_vals, y_min, y_max),
        time=slice(t_start, t_end),
    )

    if selection.sizes.get("time", 0) == 0 or selection.sizes.get("x", 0) == 0:
        raise ValueError(
            "No data found in the selected region and time range. "
            "Try expanding your bounding box or adjusting the dates."
        )

    da = selection[variable]

    # ── Statistics ──────────────────────────────────────────────────────────
    flat  = da.values.flatten()
    valid = flat[~np.isnan(flat)]

    stats = {
        "variable":    variable,
        "units":       str(da.attrs.get("units", "unknown")),
        "time_range":  [
            str(selection["time"].values[0])[:10],
            str(selection["time"].values[-1])[:10],
        ],
        "point_count": int(da.size),
        "null_pct":    round(float(da.isnull().mean() * 100), 1),
        "mean": round(float(valid.mean()), 4) if len(valid) else None,
        "max":  round(float(valid.max()),  4) if len(valid) else None,
        "min":  round(float(valid.min()),  4) if len(valid) else None,
        "std":  round(float(valid.std()),  4) if len(valid) else None,
    }

    # ── Heatmap (time mean → spatial PNG) ───────────────────────────────────
    spatial = da.mean(dim="time")

    # Ensure north-up rendering (y must decrease top→bottom)
    if spatial["y"].values[0] < spatial["y"].values[-1]:
        spatial = spatial.isel(y=slice(None, None, -1))

    grid = spatial.transpose("y", "x").values

    max_val = float(np.nanmax(grid)) if not np.all(np.isnan(grid)) else 1.0
    cmap = plt.get_cmap("turbo")
    norm = mcolors.Normalize(vmin=0, vmax=max_val)
    rgba = cmap(norm(grid))
    rgba[np.isnan(grid), 3] = 0.0  # transparent for NaN (land / ocean mask)

    buf = BytesIO()
    plt.imsave(buf, rgba, format="png")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")

    sel_x = selection["x"].values
    sel_y = selection["y"].values
    lon_sw, lat_sw = _utm_to_latlon.transform(float(sel_x.min()), float(sel_y.min()))
    lon_ne, lat_ne = _utm_to_latlon.transform(float(sel_x.max()), float(sel_y.max()))
    cell_size_m = abs(float(sel_x[1] - sel_x[0])) if len(sel_x) > 1 else 250.0

    heatmap = {
        "image_b64":  img_b64,
        "bounds":     [[round(lat_sw, 5), round(lon_sw, 5)],
                       [round(lat_ne, 5), round(lon_ne, 5)]],
        "max_val":    round(max_val, 4),
        "cell_size_m": round(cell_size_m, 1),
    }

    # ── Time series (spatial mean over selected region) ──────────────────────
    ts     = da.mean(dim=["x", "y"])
    labels = [str(t)[:10] for t in selection["time"].values]
    values = [None if np.isnan(v) else round(float(v), 4) for v in ts.values.flatten()]

    timeseries = {"labels": labels, "data": values}

    return {"heatmap": heatmap, "timeseries": timeseries, "stats": stats}
