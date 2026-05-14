"""
Testing tools + state schema integration
"""
# For paths and environment variables
import os
from dotenv import load_dotenv

# Langgraph/Langchain
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, ToolMessage

# Xarray
import xarray as xr

# Typing
from typing import List, Literal

# Modules under test
from ursa.agent.tools import generate_tools, ursa_tool_node
from ursa.agent.schemas import AgentState

# ++++++++++ Initializing Test Cases ++++++++++
bisect_context_retriever_call = {
    "name": "bisect_context_retriever",
    "args": {
        "query": "groundwater",
    },
    "id": "call_1",
    "type": "tool_call"
}

dataset_metadata_retriever_call = {
    "name": "dataset_metadata_retriever",
    "args": {},
    "id": "call_2",
    "type": "tool_call"
}

spatial_temporal_select_call = {
    "name": "spatial_temporal_select",
    "args": {
        "kwargs": {
            "x": [572125, 592125],
            "y": [2810601, 2830601],
            "time": ["2096-01-01", "2096-01-31"]
        }
    },
    "id": "call_3",
    "type": "tool_call"
}

filter_by_value_call = {
    "name": "filter_by_value",
    "args": {
        "target": "salinity",
        "symbol": ">",
        "value": 30
    },
    "id": "call_4",
    "type": "tool_call"
}

resample_time_series_call = {
    "name": "resample_time_series",
    "args": {
        "freq": "1MS",
        "method": "mean"
    },
    "id": "call_5",
    "type": "tool_call"
}

reduce_dimension_call = {
    "name": "reduce_dimension",
    "args": {
        "dim": "time",
        "method": "max"
    },
    "id": "call_6",
    "type": "tool_call"
}

inspect_selection_call_1 = {
    "name": "inspect_selection",
    "args": {},
    "id": "call_7",
    "type": "tool_call"
}

geocoding_tool_call = {
    "name": "geocoding_tool",
    "args": {
        "location_name": "Biscayne National Park"
    },
    "id": "call_8",
    "type": "tool_call"
}

reverse_geocoding_tool_call = {
    "name": "reverse_geocoding_tool",
    "args": {
        "easting": 574000.0,
        "northing": 2815000.0
    },
    "id": "call_9",
    "type": "tool_call"
}

reset_view_call = {
    "name": "reset_view",
    "args": {},
    "id": "call_10",
    "type": "tool_call"
}

inspect_selection_call_2 = {
    "name": "inspect_selection",
    "args": {},
    "id": "call_11",
    "type": "tool_call"
}

tool_calls = [
    bisect_context_retriever_call,
    dataset_metadata_retriever_call,
    spatial_temporal_select_call,
    filter_by_value_call,
    resample_time_series_call,
    reduce_dimension_call,
    inspect_selection_call_1,
    geocoding_tool_call,
    reverse_geocoding_tool_call,
    reset_view_call,
    inspect_selection_call_2
]

test_cases = [tool_calls]

for call in tool_calls:
    test_cases.append(call)

# ++++++++++ Graph setup ++++++++++
graph = StateGraph(AgentState)


def add_test(state: AgentState) -> dict[str, List[AIMessage]]:
    """
        Wraps test cases in an AI message, so they can be processed by the tool
        node.
    """

    current_test = test_cases.pop(0)
    # AIMessage expects the tool calls to be contained in a list, so we double
    # checking here
    if type(current_test) != list:
        current_test = [current_test]
    new_msg = AIMessage(content="", tool_calls=current_test)
    return {"messages": [new_msg]}


def are_there_more_tests(state: AgentState) -> Literal[
    "no more tests", "more tests"]:
    """
        Decide whether we still have tests to process.
    """

    if len(test_cases) == 0:
        return "no more tests"
    else:
        return "more tests"


graph.add_node("add test", add_test)
graph.add_node("tool node", ursa_tool_node)

graph.add_edge(START, "add test")
graph.add_edge("add test", "tool node")
graph.add_conditional_edges(
    "tool node",
    are_there_more_tests,
    {
        "more tests": "add test",
        "no more tests": END
    }
)

app = graph.compile()

# ++++++++++ Check Test Results ++++++++++
load_dotenv()
DS = xr.open_dataset(os.getenv("NETCDF_DATA_PATH"), chunks="auto")
my_tools = generate_tools(DS)

initial_state = {"messages": [], "dataset": DS, "tools": my_tools}

results = app.invoke(initial_state)

print("\nSTART")
print(f"{'@' * 80}")
test_count = 0
for msg in results["messages"]:

    if isinstance(msg, AIMessage):
        test_count += 1
        print(f"\n{'=' * 30}")
        print(f"TEST {test_count}")
        print(f"{'=' * 30}")

    if isinstance(msg, ToolMessage):
        print(f"\n{'+' * 40}")
        print(f"TOOL: {msg.name.upper() if msg.name else ''}")
        print(f"RESPONDING TO: {msg.tool_call_id}")
        print(f"RESULT:\n{msg.content}")
        print(f"{'+' * 40}")

print(f"\n{'@' * 80}")
print("END\n")
