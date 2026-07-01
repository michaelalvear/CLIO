"""
Utilities for reading CRS, axis names, and descriptive metadata from
CF-compliant xarray Datasets. No BISECT-specific assumptions here.
"""

import re
import numpy as np
import cf_xarray  # noqa: F401 — registers .cf accessor on xr.Dataset
import xarray as xr
from pyproj import Transformer


# ── CRS detection ──────────────────────────────────────────────────────────

def detect_crs(dataset: xr.Dataset) -> str:
    """
    Return the dataset's CRS as an EPSG string (e.g. 'EPSG:26917').

    Checks, in order:
      1. grid_mapping variable referenced by a data variable (CF standard)
      2. Well-known CRS variable names (crs, spatial_ref, grid_mapping)
      3. Dataset-level attributes
      4. X coordinate units (degrees_east → EPSG:4326)
      5. X coordinate value range (−180…180 → EPSG:4326)

    Raises ValueError if the CRS cannot be determined.
    """
    # 1 & 2 — look for a grid_mapping variable
    gm_var = None
    for var_name in dataset.data_vars:
        gm_name = dataset[var_name].attrs.get("grid_mapping")
        if gm_name and gm_name in dataset:
            gm_var = dataset[gm_name]
            break
    if gm_var is None:
        for name in ("crs", "spatial_ref", "grid_mapping"):
            if name in dataset:
                gm_var = dataset[name]
                break

    if gm_var is not None:
        epsg = gm_var.attrs.get("epsg_code")
        if epsg:
            s = str(epsg).strip()
            return s if s.upper().startswith("EPSG:") else f"EPSG:{s}"
        for attr in ("crs_wkt", "spatial_ref", "proj4_params"):
            m = re.search(r"EPSG[:\s]+(\d+)", str(gm_var.attrs.get(attr, "")), re.IGNORECASE)
            if m:
                return f"EPSG:{m.group(1)}"

    # 3 — dataset-level attributes
    for attr in ("crs", "spatial_ref"):
        m = re.search(r"EPSG[:\s]+(\d+)", str(dataset.attrs.get(attr, "")), re.IGNORECASE)
        if m:
            return f"EPSG:{m.group(1)}"

    # 4 & 5 — infer from X coordinate
    try:
        x = dataset.cf["X"]
        units = x.attrs.get("units", "").lower()
        if "degrees_east" in units or "degree_east" in units:
            return "EPSG:4326"
        x_min, x_max = float(x.min()), float(x.max())
        if x_min >= -180 and x_max <= 180:
            return "EPSG:4326"
    except Exception:
        pass

    raise ValueError(
        "Could not detect CRS from dataset metadata. Ensure the dataset follows "
        "CF conventions and includes a grid_mapping variable or coordinate units."
    )


# ── Axis detection ─────────────────────────────────────────────────────────

def get_cf_axes(dataset: xr.Dataset) -> dict:
    """
    Return the *dimension* names for the X, Y, and T axes as a dict:
      {"x_dim": str, "y_dim": str, "t_dim": str}

    Detection order for each axis:
      1. CF axis/standard_name attributes (via cf_xarray)
      2. Common coordinate names (case-insensitive)
      3. For T: any 1-D coordinate with a datetime64 dtype

    Only rectilinear (1-D coordinate) grids are supported.
    """
    _FALLBACK_NAMES = {
        "X": ("x", "lon", "longitude", "rlon", "easting"),
        "Y": ("y", "lat", "latitude",  "rlat", "northing"),
        "T": ("time", "t", "datetime", "date"),
    }

    axes = {}
    for axis, key in [("X", "x_dim"), ("Y", "y_dim"), ("T", "t_dim")]:
        coord = None

        # 1. CF conventions
        try:
            coord = dataset.cf[axis]
        except KeyError:
            pass

        # 2. Common names
        if coord is None:
            for name in _FALLBACK_NAMES[axis]:
                if name in dataset.coords or name in dataset.dims:
                    coord = dataset[name]
                    break

        # 3. Datetime dtype scan (T axis only)
        if coord is None and axis == "T":
            for name in dataset.coords:
                c = dataset[name]
                if c.ndim == 1 and np.issubdtype(c.dtype, np.datetime64):
                    coord = c
                    break

        if coord is None:
            raise ValueError(
                f"Dataset does not have a recognisable {axis} axis. "
                "Check that coordinates carry CF 'axis' attributes, "
                "standard_names (e.g. 'longitude', 'latitude', 'time'), "
                f"or a common name such as {_FALLBACK_NAMES[axis]}."
            )
        if coord.ndim != 1:
            raise ValueError(
                f"Non-rectilinear (2-D) {axis} coordinates are not supported."
            )
        axes[key] = coord.dims[0]
    return axes


# ── Coordinate transformers ────────────────────────────────────────────────

def get_transformers(dataset_crs: str):
    """
    Return (to_latlon, from_latlon) Transformer pair for the given CRS.
    Both transformers are identity-like no-ops when dataset_crs is EPSG:4326.
    """
    if dataset_crs == "EPSG:4326":
        return None, None
    to_latlon   = Transformer.from_crs(dataset_crs, "EPSG:4326", always_xy=True)
    from_latlon = Transformer.from_crs("EPSG:4326", dataset_crs, always_xy=True)
    return to_latlon, from_latlon


# ── Dataset metadata for the agent system prompt ───────────────────────────

def dataset_prompt_block(dataset: xr.Dataset, lat_lon_bounds: dict | None = None) -> str:
    """
    Format CF global attributes and variable metadata into a plain-text block
    suitable for injection into the agent system prompt.

    lat_lon_bounds: optional {"sw": [lat, lon], "ne": [lat, lon]} covering the
    full dataset extent (as computed for the frontend map). When provided, a
    "Spatial extent" line is added so the agent has geographic context for the
    dataset as a whole, not just whatever region the user has selected.
    """
    a = dataset.attrs
    lines = ["DATASET INFORMATION:"]

    title = a.get("title") or a.get("Title") or "Untitled dataset"
    lines.append(f"Title: {title}")

    for key in ("summary", "description", "abstract"):
        val = a.get(key, "").strip()
        if val:
            lines.append(f"Summary: {val}")
            break

    for key in ("institution", "source", "references"):
        val = a.get(key, "").strip()
        if val:
            lines.append(f"{key.capitalize()}: {val}")

    conventions = a.get("Conventions", a.get("conventions", ""))
    if conventions:
        lines.append(f"Conventions: {conventions}")

    # Variables
    skip = {"crs", "spatial_ref", "grid_mapping"}
    var_lines = []
    for var in dataset.data_vars:
        if var in skip:
            continue
        da = dataset[var]
        if da.ndim == 0 or "grid_mapping_name" in da.attrs:
            continue
        long_name     = da.attrs.get("long_name", var)
        units         = da.attrs.get("units", "unknown")
        standard_name = da.attrs.get("standard_name", "")
        entry = f"  - {var} ({long_name}): units={units}"
        if standard_name:
            entry += f", standard_name={standard_name}"
        var_lines.append(entry)
    if var_lines:
        lines.append("Variables:\n" + "\n".join(var_lines))

    # Time range
    try:
        t = dataset.cf["T"]
        lines.append(
            f"Time range: {str(t.values[0])[:10]} to {str(t.values[-1])[:10]}"
        )
    except Exception:
        pass

    # Spatial extent
    if lat_lon_bounds:
        sw_lat, sw_lon = lat_lon_bounds["sw"]
        ne_lat, ne_lon = lat_lon_bounds["ne"]
        lines.append(
            f"Spatial extent: latitude {sw_lat} to {ne_lat}, "
            f"longitude {sw_lon} to {ne_lon} (decimal degrees, full dataset domain)"
        )

    return "\n".join(lines)
