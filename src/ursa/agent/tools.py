"""These are the tools provided to the Agent"""

# Environment variable access
import os
from dotenv import load_dotenv

# Essential LangChain/LangGraph packages
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

# Chroma/RAG
from langchain_chroma import Chroma
from langchain_core.tools import create_retriever_tool
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Type hinting/validation
from typing import Annotated, Any, Dict, List, Literal, Optional, Union
from langchain_core.messages import ToolMessage
from inspect import signature
from pydantic import BaseModel, Field
from ursa.agent.schemas import AgentState

# Geospatial
import xarray as xr
import cf_xarray
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
from pyproj import Transformer

# Regex
import re

# String formatting
import json

# Operators
import operator

load_dotenv()  # Load environment variables


def generate_tools(dataset: xr.Dataset) -> List[Any]:
    """
    Dynamically generates a list of tools for the LLM, modifying argument
    schemas according to dataset metadata.
    """

    # ++++++++++++++++++++ RAG Retrieval ++++++++++++++++++++
    # Setting up vector store access
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    # Connect to the existing DB directory
    vectorstore = Chroma(
        persist_directory=os.getenv("CHROMADB_PATH"),
        embedding_function=embeddings,
        collection_name="BISECT"
    )

    # Creating the retriever, retrieves records in the form of "Document" objects
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}  # K is the amount of docs to return
    )

    # This template is passed to create_retriever_tool.
    # This tells the function how to stringify the text
    # of the retrieved Document objects along with their metadata (page_label)
    custom_doc_prompt = PromptTemplate.from_template(
        "--- DOCUMENT CHUNK ---\n"
        "SOURCE PAGE: {page_label}\n"
        "CONTENT: {page_content}\n"
    )

    # This is the actual StructuredTool
    # A function that formats retriever output into a string
    bisect_context_retriever = create_retriever_tool(
        retriever=retriever,
        name="bisect_context_retriever",
        description="Search and return relevant portions of the bisect paper.",
        document_prompt=custom_doc_prompt,
        response_format="content"
    )

    # ++++++++++++++++++++ Dataset Metadata Retrieval ++++++++++++++++++++

    @tool("dataset_metadata_retriever")
    def dataset_metadata_retriever(
            # Gets the entire src Dataset
            ds: Annotated[xr.Dataset, InjectedState]
    ) -> str:

        """This tool allows you to see the metadata of the entire dataset"""
        # Capture spatial and temporal bounds
        dims_info = {}
        for dim in ds.dims:
            d_min = ds[dim].min()
            d_max = ds[dim].max()

            # We should convert date dims from an ugly numpy type to a pretty
            # Formatted string
            # This assumes the time dimension is a datetime64 type.
            if np.issubdtype(ds[dim].dtype, np.datetime64):
                its_range = [
                    str(np.datetime_as_string(d_min.values, unit='D')),
                    str(np.datetime_as_string(d_max.values, unit='D'))]
            else:
                its_range = [round(float(d_min), 2), round(float(d_max), 2)]

            dims_info[dim] = {
                "size": int(ds.sizes[dim]),  # Cast np.int64 to native int
                "range": its_range
            }

        # Extract variable information (Skipping metadata scalars)
        variables = {}
        for var in ds.data_vars:
            da = ds[var]
            if da.ndim == 0 or "grid_mapping_name" in da.attrs:
                continue

            variables[var] = {
                "dimensions": list(da.dims),
                "units": str(da.attrs.get("units", "unknown")),
                "long_name": str(da.attrs.get("long_name", var))
            }

        # Global attributes
        # Convert to a standard dict to remove any xarray-specific formatting
        # We iterate and cast each value to ensure no raw NumPy types remain
        attrs = {}
        for k, v in ds.attrs.items():
            if isinstance(v, (np.integer, np.int64, np.int32)):
                attrs[k] = int(v)
            elif isinstance(v, (np.floating, np.float64, np.float32)):
                attrs[k] = float(v)
            elif isinstance(v, np.ndarray):
                attrs[k] = v.tolist()
            else:
                attrs[
                    k] = v  # Native strings, bools, and None are already JSON safe

        # Construct the final summary
        summary = {
            "dataset_title": attrs.get("title", "Untitled Dataset"),
            "spatial_temporal_bounds": dims_info,
            "data_variables": variables,
            "global_metadata": attrs
        }

        return f"Dataset Overview:\n{json.dumps(summary, indent=2)}"

    # ++++++++++++++++++++ See Selection Details ++++++++++++++++++++

    @tool("inspect_selection")
    def inspect_selection(
            active_selection: Annotated[Optional[xr.Dataset], InjectedState]
    ) -> str:
        """
        Statistical and geospatial summary of the active_selection.
        """

        summary = {}

        # Process variable statistics (skipping metadata variables)
        for var in active_selection.data_vars:
            da = active_selection[var]

            # Skip coordinate reference system (CRS) or dummy variables
            if da.ndim == 0 or "grid_mapping_name" in da.attrs:
                continue

            # If the slice is empty (0 pixels), don't even try math
            if da.size == 0:
                summary[var] = {"error": "Empty selection"}
                continue

            # Basic Stats
            v_mean = round(float(da.mean()), 2)
            v_max = round(float(da.max()), 2)
            v_min = round(float(da.min()), 2)
            v_std = round(float(da.std()), 2)
            v_null_percentage = round(float(da.isnull().mean() * 100), 1)

            # Find landmarks

            # These methods return a point datarray of the highest/lowest value

            # For max
            flat_max_idx = int(da.argmax())  # Find which index is the highest
            indices = np.unravel_index(flat_max_idx,
                                       da.shape)  # Find the grid this index lies in in the original DA
            max_point_indexed = {dim: indices[i] for i, dim in enumerate(
                da.dims)}  # Add labels to the grid index
            max_info = da.isel(
                max_point_indexed)  # Query the dataset for the point using the dictionary above

            # For min
            flat_min_idx = int(da.argmin())
            indices = np.unravel_index(flat_min_idx, da.shape)
            min_point_indexed = {dim: indices[i] for i, dim in
                                 enumerate(da.dims)}
            min_info = da.isel(min_point_indexed)

            # These dictionary extract the value of the coordinates
            max_coord = {
                dim: (
                    str(np.datetime_as_string(max_info[dim].values, unit='D'))
                    if np.issubdtype(max_info[dim].dtype, np.datetime64)
                    else round(float(max_info[dim]), 2)
                )
                for dim in da.dims
            }

            min_coord = {
                dim: (
                    str(np.datetime_as_string(min_info[dim].values, unit='D'))
                    if np.issubdtype(min_info[dim].dtype, np.datetime64)
                    else round(float(min_info[dim]), 2)
                )
                for dim in da.dims
            }

            summary[var] = {
                "mean": v_mean,
                "max": v_max,
                "max_coord": max_coord,
                "min": v_min,
                "min_coord": min_coord,
                "std": v_std,
                "units": da.attrs.get("units", "unknown"),
                "long_name": da.attrs.get("long_name", var),
                "null_percentage": v_null_percentage
            }

        # Process coordinate information
        dims_info = {}
        for dim in active_selection.dims:
            d_min = active_selection[dim].min()
            d_max = active_selection[dim].max()

            # Handles formatting of time-based coordinates
            if dim == 'time' or np.issubdtype(active_selection[dim].dtype,
                                              np.datetime64):
                # Formats to human-readable strings like '2026-01-01'
                its_range = [
                    str(np.datetime_as_string(d_min.values, unit='D')),
                    str(np.datetime_as_string(d_max.values, unit='D'))
                ]
            else:
                # Standard numeric coordinates (e.g. x, y) can be floats
                its_range = [round(float(d_min), 2), round(float(d_max), 2)]

            dims_info[dim] = {
                "range": its_range
            }

        return f"Selection Profile: {{'variable_stats': {summary}, 'dimensions': {dims_info}}}"

    # ++++++++++++++++++++ Changing View ++++++++++++++++++++

    # The schemas below control how function arguments are allowed to be
    # structured for these GIS operations.
    # These schemas are a must for the agent not to get lost when trying to
    # execute a complex GIS manipulations.

    # We also dynamically define the type hints for Variable and Dimension to
    # match the metadata of the source dataset.
    # This prevents the LLM from hallucinating nonexistent variable names

    # This is a list of dataset variables excluding metadata variables like
    # coordinate reference systems
    vars_list = [
        var for var in dataset.data_vars if
        var not in ["crs", "spatial_ref", "grid_mapping"]
    ]
    var_names = tuple(vars_list)
    dim_names = tuple(dataset.dims)
    # Literal only understands tuples if you feed them through the .__getitem__
    # Method
    Variable = Literal.__getitem__(var_names)
    Dimension = Literal.__getitem__(dim_names)
    MathSymbol = Literal[">", "<", ">=", "<=", "==", "!="]
    StatsMethod = Literal["mean", "max", "min", "std", "median"]
    TimeFreq = Literal[
        "1D", "1W", "1MS", "1YS"]  # Daily, Weekly, Month Start, Year Start

    class SpatialTemporalSelectSchema(BaseModel):
        kwargs: Dict[
            Dimension, Union[float, str, List[Union[float, str]]]] = Field(
            ...,
            description="Coordinate constraints. Use [min, max] for a range "
                        "or a single value. Takes spatial (x, y) and "
                        "calendar dates for time.",
            example={"x": [580000, 585000], "time": "2026-01-01"}
        )

    class FilterByValueSchema(BaseModel):
        target: Variable = Field(...,
                                 description="The variable to filter (e.g., "
                                             "'salinity').")
        symbol: MathSymbol = Field(..., description="The comparison operator.")
        value: float = Field(..., description="The threshold value.")

    class ResampleTimeSeriesSchema(BaseModel):
        freq: TimeFreq = Field(..., description="Temporal grouping frequency.")
        method: StatsMethod = Field(...,
                                    description="Aggregation method (e.g., "
                                                "'mean').")

    class ReduceDimensionSchema(BaseModel):
        dim: Dimension = Field(..., description="The dimension to collapse.")
        method: StatsMethod = Field(..., description="The reduction method.")

    # Below are the actual GIS tools, make sure to never implement these using
    # in-place methods on the original dataset, because that will lead to
    # mutating the source data causing unpredictable behavior
    @tool(name_or_callable="spatial_temporal_select",
          args_schema=SpatialTemporalSelectSchema)
    def spatial_temporal_select(
            kwargs: Dict[
                Dimension, Union[float, str, List[Union[float, str]]]],
            active_selection: Annotated[xr.Dataset, InjectedState]
    ) -> xr.Dataset:
        """
        Slices the dataset. Each dimension (e.g. lat, lon, x, y, time) can only
        be targeted once per call. Use a list [min, max] for a range or a
        single value for a point.
        """

        # The 'kwargs' dictionary automatically prevents duplicate keys.
        # If the agent tries to send 'x' twice, only the last one gets passed.

        result = active_selection

        for dim, val in kwargs.items():

            # --- Boundary Check ---

            # Check if we are looking at the time dimension.
            # If it is we need to convert the llm's string args to datetime
            # type to perform boundary check

            # 1. Is the dimension explicitly named 'time'?
            # 2. Does it have 'units' metadata like 'days since...'?
            # 3. Is it already a numpy datetime64 ('M')?
            dtype_kind = active_selection[dim].dtype.kind
            units = active_selection[dim].attrs.get("units", "").lower()

            is_datetime = (
                    dim.lower() == "time" or
                    "since" in units or
                    dtype_kind in 'Mm'
            )

            if is_datetime:
                try:
                    # Convert agent input to Pandas Timestamps
                    if isinstance(val, list):
                        val = [pd.to_datetime(v) for v in val]
                    else:
                        val = pd.to_datetime(val)
                except Exception as e:
                    raise ValueError(
                        f"Invalid date format for {dim}: {val}. Use YYYY-MM-DD.") from e

            # --- Boundary Check ---
            if is_datetime:
                actual_min = pd.to_datetime(
                    active_selection.coords[dim].min().values.item())
                actual_max = pd.to_datetime(
                    active_selection.coords[dim].max().values.item())
            else:
                actual_min = active_selection.coords[dim].min().values.item()
                actual_max = active_selection.coords[dim].max().values.item()

            req_min = min(val) if isinstance(val, list) else val
            req_max = max(val) if isinstance(val, list) else val

            if req_max < actual_min or req_min > actual_max:
                raise ValueError(
                    f"Out of Bounds: {dim}={val} is outside current range "
                    f"[{actual_min}, {actual_max}]"
                )

            # --- Execution ---
            # Range logic
            if isinstance(val, list):
                # Handle degenerate ranges (e.g., [10, 10]) as a point query
                if len(val) == 1 or val[0] == val[1]:
                    result = result.sel({dim: val[0]}, method="nearest")
                else:
                    # Handle actual ranges
                    coord_vals = active_selection[dim].values

                    # Check for descending vs ascending values
                    if len(coord_vals) > 1 and coord_vals[0] > coord_vals[-1]:
                        sel_slice = slice(max(val), min(val))
                    else:
                        sel_slice = slice(min(val), max(val))

                    result = result.sel({dim: sel_slice})

            # Point logic
            else:
                result = result.sel({dim: val}, method="nearest")

        return result

    @tool(name_or_callable="filter_by_value", args_schema=FilterByValueSchema)
    def filter_by_value(
            target: Variable,
            symbol: MathSymbol,
            value: float,
            active_selection: Annotated[xr.Dataset, InjectedState]
    ) -> xr.Dataset:
        """
        Applies a mask to data based on values (e.g., keep salinity > 30).
        """

        # This dictionary maps string symbols to functional logic.
        ops = {
            ">": operator.gt,
            "<": operator.lt,
            ">=": operator.ge,
            "<=": operator.le,
            "==": operator.eq,
            "!=": operator.ne
        }

        condition = ops[symbol](active_selection[target], value)
        result = active_selection.where(condition)
        return result

    @tool(name_or_callable="resample_time_series",
          args_schema=ResampleTimeSeriesSchema)
    def resample_time_series(
            freq: TimeFreq,
            method: StatsMethod,
            active_selection: Annotated[xr.Dataset, InjectedState]
    ) -> xr.Dataset:
        """
        Aggregates the time dimension into larger bins (e.g., Monthly Mean).
        """

        if 'time' not in active_selection.dims:
            raise ValueError(
                "Time dimension is missing. It may have been reduced or "
                "incorrectly sliced.")

        resampler = active_selection.resample(time=freq)
        result = getattr(resampler, method)()
        return result

    @tool(name_or_callable="reduce_dimension",
          args_schema=ReduceDimensionSchema)
    def reduce_dimension(
            dim: Dimension,
            method: StatsMethod,
            active_selection: Annotated[xr.Dataset, InjectedState]
    ) -> xr.Dataset:
        """
        Collapses a dimension entirely. Use 'time' to create a map,
        or 'x'/'y' for profiles.
        """
        result = getattr(active_selection, method)(dim=dim)
        return result

    @tool(name_or_callable="reset_view")
    def reset_view(ds: Annotated[xr.Dataset, InjectedState]) -> xr.Dataset:
        """
        Resets the active view of the data back to the original dataset,
        so you can make new queries.
        """
        return ds

    # ++++++++++++++++++++ Geocoding ++++++++++++++++++++

    # Define a 'Safe List' of projections to select as the format for the
    # geocoding results
    VALID_PROJECTIONS = Literal[
        "EPSG:26917",  # NAD83 / UTM zone 17N (BISECT Default)
        "EPSG:4326",  # WGS84 (Standard Lat/Lon)
        "EPSG:3857",  # Web Mercator (Google Maps)
        "EPSG:2236",  # NAD83 / Florida East (State Plane)
        "EPSG:4269"  # NAD83 (Standard Geographic)
    ]

    class GeocodingInput(BaseModel):
        """Input schema for geocoding with coordinate system intelligence"""
        location_name: str = Field(
            ...,
            description="The name of the location to look up (e.g., 'Biscayne Bay')"
        )
        crs_override: Optional[VALID_PROJECTIONS] = Field(
            None,
            description="Select the EPSG code based on the dataset metadata."
        )

    @tool("geocoding_tool", args_schema=GeocodingInput)
    def geocoding_tool(
            location_name: str,
            ds: Annotated[xr.Dataset, InjectedState],
            crs_override: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Finds coordinates for a location and projects them into the dataset's
        CRS. Return coordinates in Agent-provided projection or tries to
        discover the datasets projection from metadata.
        """
        # 1. Initialize Geocoder
        geolocator = Nominatim(user_agent="ursa_science_agent")
        location = geolocator.geocode(location_name)
        if not location:
            return {"error": f"Could not find location: {location_name}"}

        # 2. Determine the CRS
        target_crs = None
        crs_source = "unknown"

        # Priority 1: Agent Override
        if crs_override:
            target_crs = crs_override
            crs_source = "agent_override"

        # Priority 2: Extract from Dataset Attributes using Regex
        else:
            raw_metadata = str(
                ds.attrs.get("spatial_ref", "") or ds.attrs.get("crs", ""))
            # Regex looks for 'EPSG:' followed by numbers, even inside parentheses
            match = re.search(r"EPSG:\d+", raw_metadata, re.IGNORECASE)

            if match:
                target_crs = match.group(0).upper()
                crs_source = "regex_metadata_extraction"

        # Priority 3: Hard Fallback
        if not target_crs:
            target_crs = "EPSG:26917"
            crs_source = "hardcoded_fallback_utm17n"

        # 3. Build Transformer with Error Handling
        try:
            transformer = Transformer.from_crs("EPSG:4326", target_crs,
                                               always_xy=True)
        except Exception as e:
            # If the extracted CRS is still 'dirty' or invalid, use the safe default
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:26917",
                                               always_xy=True)
            target_crs = "EPSG:26917 (Emergency Fallback)"
            crs_source = f"error_fallback: {str(e)}"

        # 4. Transform and Boundary Check
        easting, northing = transformer.transform(location.longitude,
                                                  location.latitude)

        try:
            x_min, x_max = float(ds.cf['X'].min()), float(ds.cf['X'].max())
            y_min, y_max = float(ds.cf['Y'].min()), float(ds.cf['Y'].max())
            is_in_bounds = (x_min <= easting <= x_max) and (
                    y_min <= northing <= y_max)
        except:
            is_in_bounds = "Unknown (Check spatial axis metadata)"

        return {
            "found_address": location.address,
            "is_within_dataset_coverage": is_in_bounds,
            "projection_used": {
                "crs": str(target_crs),
                "source": crs_source
            },
            "dataset_coords": {
                "x": round(easting, 2),
                "y": round(northing, 2)
            },
            "geographic_coords": {
                "lat": location.latitude,
                "lon": location.longitude
            }
        }

    # ++++++++++++++++++++ Reverse Geocoding ++++++++++++++++++++
    class ReverseGeocodingInput(BaseModel):
        """Input schema for reverse geocoding tool"""
        easting: float = Field(...,
                               description="The X coordinate (Easting) in dataset units.")
        northing: float = Field(...,
                                description="The Y coordinate (Northing) in dataset units.")
        crs_override: Optional[VALID_PROJECTIONS] = Field(
            None,
            description="Optional: Specify the dataset's EPSG code if known (e.g., 'EPSG:26917')."
        )

    @tool("reverse_geocoding_tool", args_schema=ReverseGeocodingInput)
    def reverse_geocoding_tool(
            easting: float,
            northing: float,
            ds: Annotated[xr.Dataset, InjectedState],
            crs_override: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Converts dataset grid coordinates (X/Y) back into a human-readable address.
        Automatically detects the dataset's projection to ensure accurate conversion.
        """

        # 1. Determine the Source CRS
        source_crs = None
        crs_source_type = "unknown"

        if crs_override:
            source_crs = crs_override
            crs_source_type = "agent_override"
        else:
            raw_metadata = str(
                ds.attrs.get("spatial_ref", "") or ds.attrs.get("crs", ""))
            match = re.search(r"EPSG:\d+", raw_metadata, re.IGNORECASE)
            if match:
                source_crs = match.group(0).upper()
                crs_source_type = "regex_metadata_extraction"

        if not source_crs:
            source_crs = "EPSG:26917"  # BISECT Default
            crs_source_type = "hardcoded_fallback"

        # 2. Build Transformer (Dataset CRS -> WGS84)
        try:
            # We reverse the order here compared to forward geocoding
            transformer = Transformer.from_crs(source_crs, "EPSG:4326",
                                               always_xy=True)
            lon, lat = transformer.transform(easting, northing)
        except Exception as e:
            return {"error": f"Coordinate transformation failed: {str(e)}"}

        # 3. Dynamic Boundary Check via cf_xarray
        try:
            x_min, x_max = float(ds.cf['X'].min()), float(ds.cf['X'].max())
            y_min, y_max = float(ds.cf['Y'].min()), float(ds.cf['Y'].max())

            # Determine if coordinates are within the actual data grid
            is_in_bounds = (x_min <= easting <= x_max) and (
                    y_min <= northing <= y_max)
        except:
            is_in_bounds = "Unknown (Check CF spatial metadata)"

        # 4. Perform Reverse Lookup
        geolocator = Nominatim(user_agent="ursa_science_agent")
        try:
            location = geolocator.reverse((lat, lon), language="en")
            if not location:
                return {"error": f"No address found for coords: {lat}, {lon}"}

            return {
                "found_address": location.address,
                "is_within_dataset_coverage": is_in_bounds,
                "projection_info": {
                    "detected_crs": source_crs,
                    "source": crs_source_type
                },
                "geographic_coords": {
                    "lat": round(lat, 6),
                    "lon": round(lon, 6)
                },
                "input_grid_coords": {
                    "x": easting,
                    "y": northing
                }
            }
        except Exception as e:
            return {"error": f"Geocoding service error: {str(e)}"}

    # +++++++++++++++++++ Return Dynamically GeneratedTools +++++++++++++++++++
    return [
        bisect_context_retriever,
        dataset_metadata_retriever,
        spatial_temporal_select,
        filter_by_value,
        resample_time_series,
        reduce_dimension,
        reset_view,
        inspect_selection,
        geocoding_tool,
        reverse_geocoding_tool
    ]


# ++++++++++++++++++++ Custom Tool Node  ++++++++++++++++++++
def ursa_tool_node(state: AgentState) -> Dict[str, Any]:
    """
    Executes one tool and updates the graph state.

    We only process one tool call per turn, so we can update the state after
    each tool.

    Assumes the latest message in the state is an AIMessage with at least one
    tool call.
    """

    last_message = state.messages[-1]

    # Extract tool call info
    tool_calls = last_message.tool_calls
    num_of_calls = len(tool_calls)

    # Only accept one tool per turn
    if num_of_calls > 1:
        error_messages = []
        for tool_call in tool_calls:
            error_content = f"You must make one tool call at a time, " \
                            f"however, you made {num_of_calls} "
            error_msg = ToolMessage(content=error_content,
                                    tool_call_id=tool_call["id"])
            error_messages.append(error_msg)

        return {"messages": error_messages}

    # If we have one tool call extract its information
    tool_call = tool_calls[0]
    name = tool_call["name"]
    call_id = tool_call["id"]

    # Make sure the tool name corresponds to a real tool
    tools_by_name = {
        t.name: t for t in state.tools
    }

    available_tool_names = tools_by_name.keys()

    if name not in available_tool_names:
        error_msg = f"{name} is not the name of any real tool, " \
                    f"you can only call tools from among the following:" \
                    f"{available_tool_names}"

        return {
            "messages": ToolMessage(
                content=error_msg,
                tool_call_id=call_id
            )
        }

    # If the LLM provided a valid tool call process the call
    actual_tool = tools_by_name[name]

    # Some parameters come from the state, the LLM can't pass these, so we have
    # to add them to the tool's arguments list manually.

    # Get complete list of tool parameters including injected params
    sig = signature(actual_tool.func)
    expected_params = sig.parameters.keys()

    # Get arguments provided by the LLM (copying, so we don't accidentally
    # mutate the original llm passed args down the line)
    llm_args = tool_call["args"].copy()

    # A dictionary containing the arguments we might need to inject
    injectables = {
        "ds": state.dataset,
        "active_selection": state.active_selection
    }

    # Add the injected arguments to the llm's arguments
    for param in expected_params:
        # If the param is an injected param add it to the list of llm params
        if param in injectables.keys():
            llm_args[param] = injectables[param]  # Add to the tool call args

    # Invoke tool
    try:
        result = actual_tool.invoke(llm_args)

        # Handle multi-dimensional return values
        if isinstance(result, (xr.Dataset, xr.DataArray)):

            # For the sake of consistent behaviour normalize the result to a
            # DataSet in case it gets flattened to a DataArray in one of the
            # tools.
            if isinstance(result, xr.DataArray):
                result = result.to_dataset()

            # Calculate Non-NaN values
            # This stacks all variables and counts non-nulls
            valid_count = int(result.to_dataarray().count())

            # Make a summary of the tool's results for the agent
            summary = (
                f"Operation successful. New shape: {dict(result.sizes)}"
            )

            # 4. Add the 'Empty' alert for the Agent's reasoning
            if valid_count == 0:
                summary += " | WARNING: Result is empty (all NaNs or 0-length dims)."

            return {
                "messages": ToolMessage(content=summary, tool_call_id=call_id,
                                        name=name),
                "active_selection": result
            }

        # Handle regular text return values
        elif isinstance(result, (str, dict)):
            return {"messages": ToolMessage(content=str(result),
                                            tool_call_id=call_id,
                                            name=name)}

    except Exception as e:
        return {"messages": ToolMessage(content=f"Error: {str(e)}",
                                        tool_call_id=call_id,
                                        name=name)}
