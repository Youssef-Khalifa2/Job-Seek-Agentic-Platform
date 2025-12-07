from typing import Literal
from langgraph.types import Command
from src.agents.job_seeker.state import AgentState

def supervisor_node(state: AgentState) -> Command[Literal["critics_start", "__end__"]]:
    """
    The Traffic Controller Node.
    
    It checks the revision count and decides whether to:
    1. Send the CV back to the Critics (Loop)
    2. Finish the process (End)
    """
    # 1. Get current revision count (default to 0 if missing)
    current_revisions = state.get("revision_count", 0)
    
    print(f"🚦 Supervisor Check: Revision {current_revisions}/3")

    # 2. Safety Valve: Stop after 3 iterations
    if current_revisions >= 3:
        print("🛑 Max revisions reached. Stopping.")
        return Command(goto="__end__")
    
    # 3. The Loop: Send to the "Fan-Out" node
    # We target "critics_start" because that is the name of our 
    # splitter node in graph.py
    return Command(goto="critics_start")