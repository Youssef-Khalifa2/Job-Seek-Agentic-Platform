# File: src/agents/JobSeeker/state.py
import operator
import uuid
from typing import Annotated, List, TypedDict, Union, Dict, Any, Optional
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

class Critique(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this critique")
    source: str = Field(..., description="The name of the critic (e.g. 'ATS Scanner')")
    summary: str = Field(..., description="A high-level readable summary (markdown)")
    details: Dict[str, Any] = Field(..., description="The raw structured JSON data for the Editor to use")
    resolved: bool = Field(default=False, description="Whether the critique has been resolved")
    # --- NEW FIELDS ---
    confidence: float = Field(default=0.0, description="0-1 confidence score")
    reasoning: str = Field(default="", description="Why this critique was made")
    tool_evidence: Dict[str, Any] = Field(default_factory=dict, description="Tool outputs supporting this critique")

class AgentState(TypedDict):
    """
    Represents the state of the Job Seeker agent workflow.
    """
    messages : Annotated[List[BaseMessage], operator.add]
    critique_inputs: Annotated[List[Critique], operator.add]
    resolved_critique_ids: List[str]  # Changed from Set[str] - Sets aren't JSON serializable
    refined_cv_text: str  # Note: You might want to unify this with resume_text eventually
    revision_count: Optional[int]  # Made Optional with default 0
    next_node: Optional[str]
    job_description: Optional[str]
    resume_text: Optional[str]
    actionable_critiques: List[Critique]  # No annotation = replace on update (correct behavior)

    # --- NEW: Quality Tracking ---
    quality_scores: Optional[List[float]]  # Made Optional with default []
    revision_history: Optional[List[Dict[str, Any]]]  # Made Optional with default []

    # --- NEW: Tool Context ---
    original_pdf_path: Optional[str]  # Needed for Vision Model
    ats_parser_result: Optional[Dict[str, Any]]
    layout_analysis: Optional[Dict[str, Any]]

    # --- NEW: Verification & Retry Logic ---
    editor_retry_count: Optional[int]  # Made Optional with default 0
    max_editor_retries: Optional[int]  # Made Optional with default 2
    unresolved_critiques: Optional[List[Dict[str, Any]]]  # Made Optional
    verification_result: Optional[Dict[str, Any]]
    retry_guidance: Optional[str]  # Summary of what verifier wants editor to focus on

    # --- NEW: User Input Request ---
    needs_user_input: Optional[bool]  # Flag to pause workflow for user input
    user_input_request: Optional[Dict[str, Any]]  # What to ask the user for