"""
Agent orchestration. The agent receives a data snapshot via its system prompt
and interprets it; it no longer performs any data manipulation itself.
"""

import json
import os
from typing import Literal, List

from dotenv import load_dotenv
import xarray as xr

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from ursa.agent.tools import build_tools
from ursa.agent.schemas import AgentState
from ursa.agent.message_formatter import format_msg

load_dotenv()

# ── Tools & LLM ─────────────────────────────────────────────────────────────

TOOLS = build_tools()
_llm  = ChatGoogleGenerativeAI(model="gemini-2.5-pro-preview-05-06", temperature=0)
_llm_with_tools = _llm.bind_tools(TOOLS)

# ── Graph nodes ─────────────────────────────────────────────────────────────

def llm_call(state: AgentState) -> dict[str, List[AIMessage]]:
    return {"messages": [_llm_with_tools.invoke(state.messages)]}


def tool_router(state: AgentState) -> Literal["use_tool", "done"]:
    return "use_tool" if state.messages[-1].tool_calls else "done"


# ── Graph assembly ───────────────────────────────────────────────────────────

graph = StateGraph(AgentState)
graph.add_node("llm_call",  llm_call)
graph.add_node("tool_node", ToolNode(TOOLS))
graph.add_edge(START, "llm_call")
graph.add_conditional_edges(
    "llm_call",
    tool_router,
    {"use_tool": "tool_node", "done": END},
)
graph.add_edge("tool_node", "llm_call")
agent = graph.compile()

# ── Dataset (loaded once at startup, used by app.py for /region/select) ─────

DS = xr.open_dataset(os.getenv("NETCDF_DATA_PATH"), chunks="auto")

# ── System prompt ─────────────────────────────────────────────────────────────

_BASE_PROMPT = """\
You are a helpful scientific interpreter for URSA (Universal Rasterized Science \
Agent). Your role is to help non-technical South Florida stakeholders — city \
council members, engineers, developers — understand hydrological data produced \
by the Biscayne and Southern Everglades Coastal Transport (BISECT) model.

The paper documenting BISECT is:
"The Hydrologic System of the South Florida Peninsula: Development and \
Application of the Biscayne and Southern Everglades Coastal Transport (BISECT) \
Model"
Authors: Eric D. Swain, Melinda A. Lohmann, and Carl R. Goodwin.

You have one tool: bisect_context_retriever. Use it to fetch supporting context \
from the paper whenever a question calls for methodological detail, historical \
background, or scientific explanation. Always provide complete citations.

The user has already selected a geographic region on an interactive map. The \
summary statistics for that selection are embedded in this prompt. You must \
treat those numbers as ground truth — do not invent or extrapolate values \
beyond what is given.

NaN values in the data indicate land or ocean areas outside the model domain.

Salinity values are in grams per liter (g/L), converted from the model's \
native PSU units.

Use plain text only. Do not use markdown such as **bold**, *italic*, or headers.\
"""


def _build_system_prompt(selection_context: dict | None) -> str:
    if selection_context:
        ctx_block = (
            "\n\nCURRENT DATA SELECTION (user-defined map region):\n"
            + json.dumps(selection_context, indent=2)
        )
    else:
        ctx_block = (
            "\n\nNo data selection is active yet. "
            "Ask the user to draw a region on the map first."
        )
    return _BASE_PROMPT + ctx_block


# ── Flask interface ───────────────────────────────────────────────────────────

def run_agent(
    user_message: str,
    history: list = None,
    selection_context: dict = None,
) -> dict:
    """
    Run the interpreter agent and return {text, toolLog}.
    selection_context: the stats dict returned by /region/select, or None.
    history: list of {"role": "user"|"assistant", "content": str} dicts.
    """
    if history:
        history_text = "\n".join(
            f"{'User' if t['role'] == 'user' else 'Assistant'}: {t['content']}"
            for t in history
        )
        full_message = (
            f"[Previous conversation]\n{history_text}\n\n"
            f"[Current message]\n{user_message}"
        )
    else:
        full_message = user_message

    initial_state = {
        "messages": [
            SystemMessage(content=_build_system_prompt(selection_context)),
            HumanMessage(content=full_message),
        ]
    }

    result = agent.invoke(initial_state)

    content = result["messages"][-1].content
    if isinstance(content, list):
        content = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )

    tool_log = []
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_log.append({"type": "call", "tool": tc["name"], "args": tc["args"]})
        elif hasattr(msg, "type") and msg.type == "tool":
            tool_content = msg.content
            if isinstance(tool_content, list):
                tool_content = " ".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in tool_content
                )
            tool_log.append({
                "type":    "result",
                "tool":    getattr(msg, "name", "unknown"),
                "content": str(tool_content)[:500],
            })

    return {"text": content, "toolLog": tool_log}


# ── Console debug mode ────────────────────────────────────────────────────────

if __name__ == "__main__":
    history = [SystemMessage(content=_build_system_prompt(None))]
    total_tokens = 0

    while True:
        user_input = input("~$ ")
        if user_input == "exit":
            break

        history.append(HumanMessage(content=user_input))
        result = agent.invoke({"messages": history})
        new_messages = result["messages"][len(history):]
        history.extend(new_messages)

        for msg in new_messages:
            print(format_msg(msg))

    for msg in history:
        if isinstance(msg, AIMessage) and getattr(msg, "usage_metadata", None):
            total_tokens += msg.usage_metadata.get("total_tokens", 0)

    bar = "-" * 30
    print(f"{bar}\n|Token consumption: {total_tokens}|\n{bar}")
