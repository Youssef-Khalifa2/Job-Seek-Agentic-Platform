# File: src/agents/JobSeeker/state.py
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
    resolved_critique_ids: Set[str]
    refined_cv_text: str  # Note: You might want to unify this with resume_text eventually
    revision_count: int
    next_node: Optional[str]
    job_description: Optional[str]
    resume_text: Optional[str]
    actionable_critiques: List[Critique]
    
    # --- NEW: Quality Tracking ---
    quality_scores: List[float]  # Track quality over iterations [75.0, 80.5, ...]
    revision_history: List[Dict[str, Any]]  # Track what changed
    
    # --- NEW: Tool Context ---
    original_pdf_path: Optional[str]  # Needed for Vision Model
    ats_parser_result: Optional[Dict[str, Any]]
    layout_analysis: Optional[Dict[str, Any]]
    
    # --- NEW: Verification & Retry Logic ---
    editor_retry_count: int  # Track retries within one loop
    max_editor_retries: int  # Default e.g. 2
    unresolved_critiques: List[Dict[str, Any]]  # Critiques the editor failed to fix
    verification_result: Optional[Dict[str, Any]]
    retry_guidance: Optional[str]  # Summary of what verifier wants editor to focus on