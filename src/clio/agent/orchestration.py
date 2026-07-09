"""
Agent orchestration. The agent receives a data snapshot via its system prompt
and interprets it; it no longer performs any data manipulation itself.
"""

import json
from typing import Literal, List

from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from clio.agent.tools import build_tools, knowledge_base_prompt_block, get_vectorstore
from clio.agent.schemas import AgentState

load_dotenv()

# ── Tools & LLM ─────────────────────────────────────────────────────────────

_VECTORSTORE    = get_vectorstore()
TOOLS           = build_tools(_VECTORSTORE)
_KB_BLOCK       = knowledge_base_prompt_block(_VECTORSTORE)
_llm            = ChatGoogleGenerativeAI(model="gemini-pro-latest", temperature=0)
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

# ── System prompt ─────────────────────────────────────────────────────────────

_BASE_PROMPT = """\
You are a scientific data interpreter for CLIO (Climate and Land data \
Interpretation Oracle). Your role is to help users — including non-technical \
stakeholders such as engineers, planners, and decision-makers — understand \
spatiotemporal environmental data.

{dataset_block}

You have one tool: retrieve_domain_context. Use it to fetch supporting context \
from the domain knowledge base whenever a question calls for methodological \
detail, scientific background, or explanation of model assumptions. \
Always provide complete citations including source page numbers.

{kb_block}

Recognizing a document's title above does not mean you already know its \
contents. Even if a title looks familiar from your training, you must still \
call retrieve_domain_context and ground any claim about that document in an \
actual retrieved passage — never answer from memory of a similarly titled \
paper. Use the list above only to scope what you can retrieve: if a question \
falls outside every title listed, say so plainly instead of retrieving \
anyway or answering from general training knowledge.

Users select data through the map interface by drawing a single rectangular \
bounding box (not a polygon, transect line, or any other shape), then choosing \
a time range and a variable from a dropdown. This is the only selection \
method the interface supports — do not describe or suggest other selection \
tools (polygons, lines, multi-region comparison, etc.) that are not part of \
this interface.

The user has already selected a geographic region on an interactive map. The \
summary statistics for that selection are provided in this prompt. Treat those \
numbers as ground truth — do not invent or extrapolate values beyond what is \
given. Units are as reported in the dataset metadata above.

When a question asks you to interpret the current selection (e.g. what the data \
means, whether conditions are changing, what the outlook is), open your response \
with a brief recap drawn directly from the selection statistics: the variable, \
the region, the time period, the trend, the minimum and maximum values, and \
where the highest value occurs (max_value_location). Only after that recap \
should you bring in supporting context — from retrieve_domain_context or general \
domain knowledge — to explain the physical mechanism behind the numbers. \
Retrieved context supports the explanation; it does not replace stating what \
the selection's own numbers show first.

NaN or null values in the data can indicate areas permanently outside the \
model domain (e.g. land or ocean boundaries), or, in dynamic systems, areas \
that are simply inactive at that time step (e.g. a wetland cell that is dry \
on one date and inundated on another). The specific cells with valid data can \
vary between variables and across time within the same dataset — do not \
assume a NaN cell is permanently excluded from the domain, and do not assume \
two variables share the same valid-data footprint.

Use plain text only. Do not use markdown such as **bold**, *italic*, or headers.\
"""


def _build_system_prompt(dataset_block: str, selection_context: dict | None) -> str:
    base = _BASE_PROMPT.format(dataset_block=dataset_block, kb_block=_KB_BLOCK)
    if selection_context:
        ctx_block = (
            "\n\nCURRENT DATA SELECTION (user-defined map region):\n"
            + json.dumps(selection_context, indent=2)
        )
    else:
        ctx_block = (
            "\n\nNo data selection is active yet. "
            "Ask the user to draw a rectangular bounding box on the map first, "
            "then set a time range and variable."
        )
    return base + ctx_block


# ── Flask interface ───────────────────────────────────────────────────────────

def run_agent(
    user_message: str,
    history: list = None,
    selection_context: dict = None,
    dataset_block: str = "",
) -> dict:
    """
    Run the interpreter agent and return {text, toolLog}.
    dataset_block:     CF metadata description of the active dataset (from app.py).
    selection_context: stats dict returned by /region/select, or None.
    history:           list of {"role": "user"|"assistant", "content": str} dicts.
    """
    messages = [SystemMessage(content=_build_system_prompt(dataset_block, selection_context))]
    for turn in (history or []):
        cls = HumanMessage if turn["role"] == "user" else AIMessage
        messages.append(cls(content=turn["content"]))
    messages.append(HumanMessage(content=user_message))

    result = agent.invoke({"messages": messages})

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
                "content": str(tool_content),
            })

    return {"text": content, "toolLog": tool_log}
