"""
This is the main agent orchestration file, run this to talk to the agent in the
console
"""

# Environment variables
from dotenv import load_dotenv
import os

# URSA modules
from ursa.agent.tools import generate_tools, ursa_tool_node
from ursa.agent.schemas import AgentState

# Format console stream
from ursa.agent.message_formatter import format_msg

# Langgraph/Langchain
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Xarray
import xarray as xr
import numpy as np

# Image generation for georeferenced overlay (replaces leaflet-heat KDE
# approach)
from ursa import config
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from io import BytesIO
import base64

# Coordinate conversion
from pyproj import Transformer

# Types
from typing import Literal, List

load_dotenv()  # Load environment variables


# ++++++++++ Graph setup ++++++++++
# Nodes (Some of the node code inspired by):
# https://github.com/langchain-ai/how_to_fix_your_context/blob/main/notebooks/01-rag.ipynb)
def end_session_router(
        state: AgentState
) -> Literal["session ended", "request created"]:
    """Decide whether to end session"""
    # Was the last human message exit?
    if state.messages[-1].content == "exit":
        return "session ended"
    else:
        return "request created"


# Main llm invocation
def llm_call(state: AgentState) -> dict[str, List[AIMessage]]:
    """
    LLM decides whether to call a tool or not.
    """
    # Initialize Gemini Client + bind tools
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview",
                                 temperature=0).bind_tools(state.tools)

    llm_response = llm.invoke(state.messages)
    return {"messages": [llm_response]}


# llm to tool node routing function
def tool_router(state: AgentState) -> Literal["pending tool calls", "done"]:
    """
    Decide if we should continue the tool loop or return to the user based on
    the last message.
    """
    last_message = state.messages[-1]

    # Continue tool loop
    if last_message.tool_calls:
        return "pending tool calls"

    # No tools are called -> return to user node
    return "done"


graph = StateGraph(AgentState)

# Nodes
graph.add_node("llm call", llm_call)
graph.add_node("tool node", ursa_tool_node)

# Edges
graph.add_edge(START, "llm call")

graph.add_conditional_edges(
    "llm call",
    tool_router,
    {
        "pending tool calls": "tool node",
        "done": END
    }
)

graph.add_edge("tool node", "llm call")

app = graph.compile()

# ++++++++++ Initializing Agent ++++++++++

# Initialize state variables
essential_context = """
*ALWAYS RETURN A NATURAL LANGUAGE MESSAGE WHEN 
INVOKED, EVEN WHEN MAKING TOOL CALLS, EXPLAIN YOUR THOUGHT PROCESS* 

*REMEMBER: DON'T END RESPONSES WITH NEWLINES, TABS, OR SPACES*

You are a helpful assistant tasked with retrieving and interpreting information 
from a South Florida hydrological model known as:
Biscayne and Southern Everglades Coastal Transport Model (BISECT).
 
Details about the model are recorded in the paper:
"The Hydrologic System of the South Florida Peninsula: 
Development and Application of the Biscayne and Southern 
Everglades Coastal Transport (BISECT) Model"

Authored by:
*Eric D. Swain, Melinda A. Lohmann, and Carl R. Goodwin*.

Your goal is to make this invaluable knowledge accessible 
to non technical South Florida stakeholders (city council-people, engineers,
developers, etc.).

You have been provided tools to fetch context from the paper itself as well
as a small subset of the results of the model in the form of one of the 
spatiotemporal raster GIS data tracking hydrology variable projections under an 
SSP scenario in South Florida, use your tools to figure out which one.

*REMEMBER: ONLY CALL ONE TOOL AT A TIME*

Your can get context on the paper through the tools provided to you.

Reflect on any context you fetch, and keep retrieving until you have sufficient 
context to answer the user's research request.

*ALWAYS PROVIDE COMPLETE CITATIONS*

You can also extract a subset of the raster data using the GIS tool suite 
provided too you. 
Follow argument schemas *EXACTLY*. 

*DON'T MAKE ASSUMPTIONS ABOUT THE SHAPE OF THE DATASET OR METADATA LIKE 
UNITS AND MAP PROJECTION* 

*ALWAYS MANUALLY CHECK METADATA* 

You are a data interface. You are *FORBIDDEN* from providing numerical data (
values, means, ranges) unless you have successfully received a ToolMessage 
containing that specific data in the current conversation turn. 

Nan values indicate a near zero projection typically indicating the presence of
a landmass.

Units have been converted from the original PSU units of the model for salinity
 variables into grams per liter. 

The updated data after each operation is preserved in your state, so if you 
need to perform a multistep operation you can.

To see the a statistical/geospatial summary of the extracted data held in 
your active selection use the inspect_selection tool.

To reset your view of the data back to the original data so you can run new
operations use the reset_view tool.

If the user asks a question that requires knowledge of coordinates use the
geocoding tool.

If you need to convert coordinates to a place name use the reverse geocoding 
tool.

Use plain text only in your responses. Do not use markdown formatting such as
**bold**, *italic*, or # headers.

TOOL USE RULES — follow these strictly:
- If the user asks to "show", "plot", "map", "chart", "graph", or "visualize"
  any data, you MUST use the GIS tools to extract the data. Do not just
  describe it in text.
- For a time series at a location (e.g. "show salinity at X over time"):
  use geocoding_tool to get coordinates, then spatial_temporal_select to
  pin x and y to that point and select the time range.
- For a spatial map (e.g. "show a map of salinity"):
  use reduce_dimension with dim="time" to collapse time first.
- Always call inspect_selection after extracting data to confirm the result
  before giving your final response.
"""

DS = xr.open_dataset(os.getenv("NETCDF_DATA_PATH"), chunks="auto")

# ++++++++++ Console Debug Mode ++++++++++
if __name__ == "__main__":

    # Initialize conversation history
    history = [SystemMessage(content=essential_context)]
    # Initialize token counter
    total_tokens = 0

    while True:

        user_message = HumanMessage(content=input("~$ "))

        if user_message.content == "exit":
            break

        history.append(user_message)

        # Initialize state
        inputs = {"messages": history,
                  "dataset": DS,
                  "tools": generate_tools(DS)
                  }

        # Run graph
        results = app.invoke(inputs)

        # Get only the messages from the latest invocation
        new_messages = results["messages"][len(history):]
        history.extend(new_messages)

        for msg in new_messages:
            print(format_msg(msg))

    # Count tokens
    for msg in history:
        if isinstance(msg, AIMessage) and getattr(msg,
                                                  "usage_metadata",
                                                  None):
            total_tokens += msg.usage_metadata.get("total_tokens",
                                                   0)

    # Show the conversation's total token use at the end
    token_string = f"|Token consumption: {total_tokens}|"
    bars = '-' * len(token_string)
    print(bars)
    print(token_string)
    print(bars)


# ++++++++++ Flask Interface ++++++++++

def run_agent(user_message: str, history: list = None) -> dict:
    """
    Takes a user's natural language message, runs it through the agent,
    and returns a dict: {text, chart_type, labels, data}.
    Called by app.py's /query route.

    history: list of {"role": "user"|"assistant", "content": str} dicts
             representing prior turns in the conversation.
    """
    # Flatten history into a text block prepended to the current message.
    # Injecting bare AIMessage objects into LangGraph's initial state causes
    # the add_messages reducer to fail as history grows.
    if history:
        history_text = "\n".join(
            f"{'User' if t['role'] == 'user' else 'Assistant'}: {t['content']}"
            for t in history
        )
        full_message = f"[Previous conversation]\n{history_text}\n\n[Current message]\n{user_message}"
    else:
        full_message = user_message

    initial_state = {
        "messages": [
            SystemMessage(content=essential_context),
            HumanMessage(content=full_message)
        ],
        "dataset": DS,
        "tools": generate_tools(DS)
    }

    result = app.invoke(initial_state)

    # Extract text, handling Gemini's list-of-blocks format
    content = result["messages"][-1].content
    if isinstance(content, list):
        content = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )

    # Build tool log: extract every tool call and its result from the
    # message chain. This lets the frontend show users exactly what the
    # agent did.
    tool_log = []
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_log.append({
                    "type": "call",
                    "tool": tc["name"],
                    "args": tc["args"]
                })
        elif hasattr(msg, "type") and msg.type == "tool":
            tool_content = msg.content
            if isinstance(tool_content, list):
                tool_content = " ".join(
                    block.get("text", "") if isinstance(block, dict) else str(
                        block)
                    for block in tool_content
                )
            tool_log.append({
                "type": "result",
                "tool": getattr(msg, "name", "unknown"),
                "content": str(tool_content)[:500]
            })

    # Build a charts dict keyed by chart type name.
    # Only includes chart types that are valid for the resulting dims.
    # The frontend uses this to render choice buttons after the agent responds.
    selection = result.get("active_selection", DS)
    dims = set(selection.dims)
    charts = {}

    # Build selection info: stats about the current active_selection slice.
    sel_info = {
        "dims": list(dims),
        "num_points": int(selection["salinity"].size)
    }
    if "time" in dims:
        sel_info["time_range"] = [
            str(selection["time"].values[0])[:10],
            str(selection["time"].values[-1])[:10]
        ]
    if "x" in dims:
        sel_info["x_range"] = [
            round(float(selection["x"].min()), 0),
            round(float(selection["x"].max()), 0)
        ]
    if "y" in dims:
        sel_info["y_range"] = [
            round(float(selection["y"].min()), 0),
            round(float(selection["y"].max()), 0)
        ]
    valid = selection["salinity"].values.flatten()
    valid = valid[~np.isnan(valid)]
    if len(valid) > 0:
        sel_info["salinity_min"] = round(float(valid.min()), 4)
        sel_info["salinity_max"] = round(float(valid.max()), 4)
        sel_info["salinity_mean"] = round(float(valid.mean()), 4)

    if dims == {"time"}:
        # Time series at a single point → line chart
        labels = [str(t)[:10] for t in selection["time"].values]
        raw = selection["salinity"].values.flatten()
        data = [None if np.isnan(v) else round(float(v), 4) for v in raw]
        charts["line"] = {"labels": labels, "data": data}

    elif dims == {"x", "y"}:
        # Spatial map, time collapsed → georeferenced image overlay.
        # Each pixel = one real grid cell colored by its true salinity value.
        # This preserves accuracy at all zoom levels (unlike leaflet-heat KDE).
        sal = selection["salinity"]

        # Ensure rows go north-to-south so the image top aligns with map north.
        if sal["y"].values[0] < sal["y"].values[-1]:
            sal = sal.isel(y=slice(None, None, -1))
        grid = sal.transpose("y", "x").values  # shape (n_y, n_x)

        max_sal = float(np.nanmax(grid)) if not np.all(np.isnan(grid)) else 1.0

        # Apply turbo colormap (perceptually uniform blue→green→yellow→red).
        # vmin=0 keeps fresh water anchored at the blue end of the scale.
        cmap = plt.get_cmap("turbo")
        norm = mcolors.Normalize(vmin=0, vmax=max_sal)
        rgba = cmap(norm(grid))  # float RGBA array (n_y, n_x, 4)
        rgba[np.isnan(grid), 3] = 0.0  # transparent for NaN (land/ocean mask)

        buf = BytesIO()
        plt.imsave(buf, rgba, format="png")
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")

        # Compute lat/lon bounds for Leaflet imageOverlay positioning.
        _utm_to_latlon = Transformer.from_crs("EPSG:26917", "EPSG:4326",
                                              always_xy=True)

        x_vals = selection["x"].values
        y_vals = selection["y"].values
        lon_sw, lat_sw = _utm_to_latlon.transform(float(x_vals.min()),
                                                  float(y_vals.min()))
        lon_ne, lat_ne = _utm_to_latlon.transform(float(x_vals.max()),
                                                  float(y_vals.max()))

        # Cell size in meters — used by the frontend to compute zoom-adaptive blur.
        cell_size_m = abs(float(x_vals[1] - x_vals[0])) if len(
            x_vals) > 1 else 250.0

        charts["heatmap"] = {
            "image_b64": img_b64,
            "bounds": [
                [round(lat_sw, 5), round(lon_sw, 5)],
                [round(lat_ne, 5), round(lon_ne, 5)]
            ],
            "max_sal": round(max_sal, 4),
            "cell_size_m": round(cell_size_m, 1)
        }

    return {
        "text": content,
        "dims": list(dims),
        "charts": charts,
        "toolLog": tool_log,
        "selectionInfo": sel_info
    }
