"""
This file holds pydantic models related to the Langgraph Agent's State
"""
from pydantic import BaseModel, Field, ConfigDict, model_validator
from typing import Annotated, List, Optional, Any
from xarray import Dataset
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(BaseModel):
    """State schema for the Hydrology Agent"""

    # Conversation History
    messages: Annotated[List[BaseMessage], add_messages]

    # Allow Xarray types
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Ensure this is opened lazily (e.g., xr.open_dataset(path, chunks="auto"))
    dataset: Dataset = Field(
        description="The primary NetCDF lazy-loaded dataset"
    )

    # This gets overwritten every time the agent decides to change the view
    active_selection: Optional[Dataset] = Field(
        None,
        description="The current focused data slice being analyzed or "
                    "visualized"
    )

    @model_validator(mode='after')
    def sync_active_selection(self) -> 'AgentState':
        """
            If we didn't provide a slice, start with the whole dataset
            The tools modify the active selection in place, so we need to
            make sure it's always there
        """
        if self.active_selection is None:
            self.active_selection = self.dataset
        return self

    # The list of tools for the specific dataset
    tools: Optional[List[Any]] = Field(
        description="A list of tools updated to match the dataset"
    )