"""
Pydantic model for the LangGraph agent state.
The agent is now an interpreter only — no dataset state needed here.
"""
from pydantic import BaseModel, ConfigDict
from typing import Annotated, List
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(BaseModel):
    messages: Annotated[List[BaseMessage], add_messages]
    model_config = ConfigDict(arbitrary_types_allowed=True)
