import operator
import uuid
from typing import Annotated, List, TypedDict, Union, Dict, Any, Optional, Set
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

class Critique(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this critique")
    source: str = Field(..., description="The name of the critic (e.g. 'ATS Scanner')")
    summary: str = Field(..., description="A high-level readable summary (markdown)")
    details: Dict[str, Any] = Field(..., description="The raw structured JSON data for the Editor to use")
    resolved: bool = Field(default=False, description="Whether the critique has been resolved")

class AgentState(TypedDict):
    """
    Represents the state of the Job Seeker agent workflow.
    """
    messages : Annotated[List[BaseMessage], operator.add]
    critique_inputs: Annotated[List[Critique], operator.add]
    resolved_critique_ids: Set[str]
    refined_cv_text: str
    revision_count: int
    next_node: Optional[str]
    job_description: Optional[str]
    resume_text: Optional[str]
    actionable_critiques: List[Critique] 