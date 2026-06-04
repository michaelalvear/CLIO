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
from ursa.cf_utils import dataset_prompt_block

load_dotenv()

# ── Tools & LLM ─────────────────────────────────────────────────────────────

TOOLS = build_tools()
_llm  = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)
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

# Build the dataset description once at startup so the prompt always reflects
# whatever file is configured — no BISECT-specific text anywhere below.
_DATASET_BLOCK = dataset_prompt_block(DS)

# ── System prompt ─────────────────────────────────────────────────────────────

_BASE_PROMPT = """\
You are a scientific data interpreter for URSA (Universal Rasterized Science \
Agent). Your role is to help users — including non-technical stakeholders such \
as engineers, planners, and decision-makers — understand spatiotemporal \
environmental data.

{dataset_block}

You have one tool: retrieve_domain_context. Use it to fetch supporting context \
from the domain knowledge base whenever a question calls for methodological \
detail, scientific background, or explanation of model assumptions. \
Always provide complete citations including source page numbers.

The user has already selected a geographic region on an interactive map. The \
summary statistics for that selection are provided in this prompt. Treat those \
numbers as ground truth — do not invent or extrapolate values beyond what is \
given. Units are as reported in the dataset metadata above.

NaN or null values in the data indicate areas outside the model domain \
(e.g. land, ocean, or masked regions).

Use plain text only. Do not use markdown such as **bold**, *italic*, or headers.\
"""


def _build_system_prompt(selection_context: dict | None) -> str:
    base = _BASE_PROMPT.format(dataset_block=_DATASET_BLOCK)
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
    return base + ctx_block


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


