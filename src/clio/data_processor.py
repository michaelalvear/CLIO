"""
Deterministic Xarray processing for user-defined map selections.
All data slicing lives here, entirely outside the agent.
Works with any CF-compliant rectilinear NetCDF dataset.
"""
import os
import re
import shutil
import tempfile
import numpy as np
import matplotlib.colors as mcolors
import base64
from datetime import datetime, timezone
from io import BytesIO

import xarray as xr
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderServiceError
from pyproj import Geod

from clio import config  # sets matplotlib backend to Agg before pyplot import
import matplotlib.pyplot as plt
from clio.cf_utils import detect_crs, get_cf_axes, get_transformers, format_units

_geolocator = Nominatim(user_agent="clio_data_processor", timeout=5)
_geod       = Geod(ellps="WGS84")


def _bbox_area_km2(lat_sw: float, lon_sw: float, lat_ne: float, lon_ne: float) -> float:
    """Geodesic area (WGS84) of the lat/lon rectangle defined by its corners."""
    lons = [lon_sw, lon_ne, lon_ne, lon_sw]
    lats = [lat_sw, lat_sw, lat_ne, lat_ne]
    area_m2, _ = _geod.polygon_area_perimeter(lons, lats)
    return round(abs(area_m2) / 1e6, 3)


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


def select_region(
    dataset: xr.Dataset,
    bbox_latlon: dict,
    time_range: list,
    variable: str,
) -> xr.Dataset:
    """
    Slice the dataset to a lat/lon bounding box and time range. This is the
    single source of the selection used by both the dashboard (process_region)
    and the NetCDF download (export_region_netcdf), so both see the same cells.

    Raises ValueError if the variable is unknown or the selection is empty.
    """
    if variable not in dataset.data_vars:
        raise ValueError(f"Unknown variable: {variable}")

    axes                  = get_cf_axes(dataset)
    x_dim, y_dim, t_dim   = axes["x_dim"], axes["y_dim"], axes["t_dim"]
    _, from_latlon        = get_transformers(detect_crs(dataset))

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
    return selection


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
    axes                  = get_cf_axes(dataset)
    x_dim, y_dim, t_dim   = axes["x_dim"], axes["y_dim"], axes["t_dim"]
    to_latlon, _          = get_transformers(detect_crs(dataset))

    selection = select_region(dataset, bbox_latlon, time_range, variable)
    da = selection[variable]

    # ── Actual (grid-snapped) extent of the selection, in lat/lon ──────────────
    # Used for both the stats block (agent-facing) and the heatmap overlay
    # (map-facing) — computed once since .sel() may include a slightly
    # different footprint than the user's drawn rectangle.
    sel_x = selection[x_dim].values
    sel_y = selection[y_dim].values

    if to_latlon is None:
        lat_sw, lon_sw = float(sel_y.min()), float(sel_x.min())
        lat_ne, lon_ne = float(sel_y.max()), float(sel_x.max())
    else:
        lon_sw, lat_sw = to_latlon.transform(float(sel_x.min()), float(sel_y.min()))
        lon_ne, lat_ne = to_latlon.transform(float(sel_x.max()), float(sel_y.max()))

    selection_extent = {
        "sw": [round(lat_sw, 5), round(lon_sw, 5)],
        "ne": [round(lat_ne, 5), round(lon_ne, 5)],
    }
    area_km2 = _bbox_area_km2(lat_sw, lon_sw, lat_ne, lon_ne)

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
        "units":       format_units(str(da.attrs.get("units", "unknown"))),
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
        "selection_extent":  selection_extent,
        "area_km2":          area_km2,
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

    heatmap = {
        "image_b64": img_b64,
        "bounds":    [selection_extent["sw"], selection_extent["ne"]],
        "max_val":   round(max_val, 4),
    }

    # ── Time series (spatial mean over selected region) ────────────────────────
    labels = [str(t)[:10] for t in t_coords]
    values = [None if np.isnan(v) else round(float(v), 4) for v in ts_raw.values.flatten()]

    timeseries = {"labels": labels, "data": values}

    return {"heatmap": heatmap, "timeseries": timeseries, "stats": stats}


# ── NetCDF export ──────────────────────────────────────────────────────────

class ExportTooLargeError(ValueError):
    """The requested selection exceeds the export size limit."""


# Encoding keys inherited from the source file that describe its on-disk
# layout and would be invalid for (or fight with) a smaller subset.
_STALE_ENCODING = ("chunksizes", "original_shape", "contiguous", "source",
                   "preferred_chunks", "zlib", "complevel", "shuffle",
                   "compression", "fletcher32", "szip_coding", "szip_pixels_per_block",
                   "blosc_shuffle", "quantize_mode", "significant_digits")


def _grid_mapping_names(dataset: xr.Dataset, variable: str) -> list[str]:
    """Names of the CF grid_mapping variable(s) referenced by `variable`."""
    da = dataset[variable]
    ref = da.attrs.get("grid_mapping") or da.encoding.get("grid_mapping")
    if not ref:
        return []
    # CF allows "crs" or the extended form "crs: x y other_crs: lat lon"
    tokens = [t.rstrip(":") for t in str(ref).split()]
    return [t for t in tokens if t in dataset.variables and t not in dataset[variable].dims]


def export_region_netcdf(
    dataset: xr.Dataset,
    bbox_latlon: dict,
    time_range: list,
    variable: str,
    max_bytes: int | None = None,
    max_source_bytes: int | None = None,
    source_name: str = "",
) -> tuple[bytes, str]:
    """
    Write the user's selection (same slice as process_region) to a CF NetCDF-4
    file and return (file_bytes, suggested_filename).

    The file keeps the variable at its native grid and time resolution, its
    coordinates, its CF grid_mapping variable (so GIS tools know the CRS) and
    the original metadata, plus attributes recording how it was subset.

    Two limits, both raising ExportTooLargeError:
      max_source_bytes  uncompressed size of the selection, checked before
                        anything is read, to bound the work and memory used.
      max_bytes         size of the finished (compressed) file. Land cells are
                        empty and compress away, so files are often several
                        times smaller than the selection.
    """
    selection = select_region(dataset, bbox_latlon, time_range, variable)
    da = selection[variable]

    if max_source_bytes is not None and da.nbytes > max_source_bytes:
        raise ExportTooLargeError(
            f"This selection holds {da.nbytes / 1e6:.0f} MB of data, above the "
            f"{max_source_bytes / 1e6:.0f} MB that can be exported at once. "
            f"Select a smaller region or a shorter date range."
        )

    keep = [variable] + _grid_mapping_names(selection, variable)
    out = selection[keep]

    for name in out.variables:
        for key in _STALE_ENCODING:
            out[name].encoding.pop(key, None)
    out[variable].encoding.update({"zlib": True, "complevel": 4, "shuffle": True})

    t_dim  = get_cf_axes(dataset)["t_dim"]
    t_vals = out[t_dim].values
    start, end = str(t_vals[0])[:10], str(t_vals[-1])[:10]
    sw, ne = bbox_latlon["sw"], bbox_latlon["ne"]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    out.attrs = dict(dataset.attrs)
    history = f"{now}: subset by CLIO to {variable}, bbox (lat/lon) SW={sw} NE={ne}, {start} to {end}"
    out.attrs["history"] = (f"{history}\n{dataset.attrs['history']}"
                            if dataset.attrs.get("history") else history)
    out.attrs["clio_source_file"]       = source_name
    out.attrs["clio_requested_bbox_sw"] = [float(v) for v in sw]
    out.attrs["clio_requested_bbox_ne"] = [float(v) for v in ne]
    out.attrs["clio_time_range"]        = f"{start}/{end}"
    out.attrs.setdefault("Conventions", "CF-1.8")

    safe_var = re.sub(r"[^A-Za-z0-9_-]+", "_", variable)
    filename = f"clio_{safe_var}_{start}_{end}.nc"

    # netCDF4 can only write to a path; Cloud Run's /tmp is in memory anyway.
    tmp_dir = tempfile.mkdtemp(prefix="clio_export_")
    try:
        path = os.path.join(tmp_dir, filename)
        out.to_netcdf(path, format="NETCDF4", engine="netcdf4")
        size = os.path.getsize(path)
        if max_bytes is not None and size > max_bytes:
            raise ExportTooLargeError(
                f"The file for this selection is {size / 1e6:.0f} MB, above the "
                f"{max_bytes / 1e6:.0f} MB download limit. Select a smaller region "
                f"or a shorter date range."
            )
        with open(path, "rb") as f:
            return f.read(), filename
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
