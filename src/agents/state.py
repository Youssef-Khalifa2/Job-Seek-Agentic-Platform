from typing import TypedDict, List, Annotated
import operator

class AgentState(TypedDict):
    """
    Represents the state of the Job Seeker agent workflow.
    """
    job_description: str
    original_cv_text: str
    critiques: List[str]
    refined_cv_text: str
    revision_count: int
    next_agent: str
