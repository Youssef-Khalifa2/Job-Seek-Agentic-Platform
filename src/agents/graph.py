from langgraph.graph import StateGraph
from .state import AgentState

def build_job_seeker_graph():
    """
    Constructs the Supervisor-based graph for CV optimization.
    """
    workflow = StateGraph(AgentState)
    
    # TODO: Add nodes
    # workflow.add_node("supervisor", supervisor_node)
    # workflow.add_node("editor", editor_node)
    
    # TODO: Add edges
    
    return workflow.compile()
