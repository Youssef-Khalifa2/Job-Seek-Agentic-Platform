from langgraph.graph import StateGraph, START, END
from src.agents.JobSeeker.state import AgentState

# Import Nodes
from src.agents.JobSeeker.supervisor import supervisor_node
from src.agents.JobSeeker.critics import (
    ats_critic_node, match_critic_node, truth_critic_node,
    language_critic_node, impact_critic_node
)
from src.agents.JobSeeker.consolidator import consolidator_node
from src.agents.JobSeeker.editor import editor_node
from src.agents.JobSeeker.verifier import verifier_node  # NEW: Import verifier

def passthrough(state):
    return {} # Dummy node for splitting

def build_job_seeker_graph():
    workflow = StateGraph(AgentState)

    # --- 1. Add Nodes ---
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("critics_start", passthrough)  # The Parallel Splitter

    # The 5 Critics
    workflow.add_node("ats_critic", ats_critic_node)
    workflow.add_node("match_critic", match_critic_node)
    workflow.add_node("truth_critic", truth_critic_node)
    workflow.add_node("language_critic", language_critic_node)
    workflow.add_node("impact_critic", impact_critic_node)

    workflow.add_node("consolidator", consolidator_node)
    workflow.add_node("editor", editor_node)
    workflow.add_node("verifier", verifier_node)  # NEW: Add verifier node

    # --- 2. Define The Flow ---

    # A. Start -> Supervisor (Decides if we need to edit)
    workflow.add_edge(START, "supervisor")

    # B. Supervisor Routing (Command-based)
    # Supervisor returns Command[Literal["critics_start", "__end__"]]
    # When quality is good or max iterations reached -> END
    # When quality needs improvement -> critics_start
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state.get("next_node", "critics_start"),
        {
            "critics_start": "critics_start",
            "__end__": END
        }
    )

    # C. The Fan-Out (Parallel Execution) 🌬️
    # From the splitter to ALL 5 critics
    workflow.add_edge("critics_start", "ats_critic")
    workflow.add_edge("critics_start", "match_critic")
    workflow.add_edge("critics_start", "truth_critic")
    workflow.add_edge("critics_start", "language_critic")
    workflow.add_edge("critics_start", "impact_critic")

    # D. The Fan-In (Consolidation) 🔗
    # All 5 critics point to the Consolidator
    workflow.add_edge("ats_critic", "consolidator")
    workflow.add_edge("match_critic", "consolidator")
    workflow.add_edge("truth_critic", "consolidator")
    workflow.add_edge("language_critic", "consolidator")
    workflow.add_edge("impact_critic", "consolidator")

    # E. The Edit -> Verify -> Loop 🔄
    workflow.add_edge("consolidator", "editor")
    workflow.add_edge("editor", "verifier")

    # F. Verifier Routing (Command-based)
    # Verifier returns Command[Literal["editor", "supervisor"]]
    # When verification passes (≥80%) -> supervisor (continue workflow)
    # When verification fails and retries available -> editor (retry with guidance)
    # When verification fails and max retries reached -> supervisor (safety valve)
    workflow.add_conditional_edges(
        "verifier",
        lambda state: state.get("next_node", "supervisor"),
        {
            "supervisor": "supervisor",
            "editor": "editor"
        }
    )

    return workflow.compile()


