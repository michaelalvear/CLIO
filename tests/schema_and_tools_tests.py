"""
Tests for the refactored URSA architecture.

Covers:
  - AgentState schema (simplified)
  - build_tools() returns the expected RAG tool
  - process_region() deterministic Xarray slicing
"""

import os
import pytest
import xarray as xr
import numpy as np
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

# ── Schema tests ───────────────────────────────────────────────────────────

def test_agent_state_requires_only_messages():
    """AgentState no longer needs active_selection or tools."""
    from ursa.agent.schemas import AgentState

    state = AgentState(messages=[HumanMessage(content="hello")])
    assert len(state.messages) == 1
    assert not hasattr(state, "active_selection")
    assert not hasattr(state, "tools")


# ── Tools tests ────────────────────────────────────────────────────────────

def test_build_tools_returns_rag_tool():
    """build_tools() should return exactly one tool: bisect_context_retriever."""
    from ursa.agent.tools import build_tools

    tools = build_tools()
    assert len(tools) == 1
    assert tools[0].name == "bisect_context_retriever"


# ── data_processor tests ───────────────────────────────────────────────────

@pytest.fixture(scope="module")
def real_dataset():
    """Load the real NetCDF dataset for integration tests."""
    path = os.getenv("NETCDF_DATA_PATH")
    if not path or not os.path.exists(path):
        pytest.skip("NETCDF_DATA_PATH not set or file missing")
    return xr.open_dataset(path, chunks="auto")


def test_process_region_returns_expected_keys(real_dataset):
    """process_region() should return heatmap, timeseries, and stats dicts."""
    from ursa.data_processor import process_region

    # Use a small slice near the center of the dataset
    x_mid = float(real_dataset["x"].mean())
    y_mid = float(real_dataset["y"].mean())

    from pyproj import Transformer
    utm_to_ll = Transformer.from_crs("EPSG:26917", "EPSG:4326", always_xy=True)
    lon_c, lat_c = utm_to_ll.transform(x_mid, y_mid)
    offset = 0.05  # degrees

    t_start = str(real_dataset["time"].values[0])[:10]
    t_end   = str(real_dataset["time"].values[min(30, len(real_dataset["time"]) - 1)])[:10]

    # Pick the first non-metadata variable
    skip = {"crs", "spatial_ref", "grid_mapping"}
    variable = next(v for v in real_dataset.data_vars if v not in skip)

    result = process_region(
        dataset=real_dataset,
        bbox_latlon={
            "sw": [lat_c - offset, lon_c - offset],
            "ne": [lat_c + offset, lon_c + offset],
        },
        time_range=[t_start, t_end],
        variable=variable,
    )

    assert "heatmap"    in result
    assert "timeseries" in result
    assert "stats"      in result

    hm = result["heatmap"]
    assert "image_b64"  in hm
    assert "bounds"     in hm
    assert len(hm["bounds"]) == 2

    ts = result["timeseries"]
    assert "labels" in ts
    assert "data"   in ts
    assert len(ts["labels"]) == len(ts["data"])

    stats = result["stats"]
    assert stats["variable"] == variable
    assert stats["mean"] is not None


def test_process_region_raises_on_empty_selection(real_dataset):
    """process_region() should raise ValueError for an out-of-bounds bbox."""
    from ursa.data_processor import process_region

    skip = {"crs", "spatial_ref", "grid_mapping"}
    variable = next(v for v in real_dataset.data_vars if v not in skip)
    t_start  = str(real_dataset["time"].values[0])[:10]
    t_end    = str(real_dataset["time"].values[-1])[:10]

    with pytest.raises(ValueError):
        process_region(
            dataset=real_dataset,
            bbox_latlon={"sw": [0.0, 0.0], "ne": [1.0, 1.0]},  # Gulf of Guinea
            time_range=[t_start, t_end],
            variable=variable,
        )
