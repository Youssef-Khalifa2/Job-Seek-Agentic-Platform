# File: src/agents/JobSeeker/supervisor.py
import json
from typing import Literal
from langgraph.types import Command
from src.agents.JobSeeker.state import AgentState
from src.llm_registry import LLMRegistry
from src.agents.JobSeeker.memory import RevisionMemory

# Use a fast model for quality checks
# Using Mistral to avoid Bedrock API format issues with direct string prompts
llm = LLMRegistry.get_mistral_small() 

def calculate_quality_score(resume_text: str, job_description: str) -> float:
    """Score resume quality 0-100 based on fit and clarity."""
    if not resume_text or not job_description:
        return 0.0
        
    quality_prompt = f"""
    Evaluate this resume's quality for the job description.

    Resume:
    {resume_text}

    Job Description:
    {job_description}

    Score 0-100 based on:
    - ATS compatibility (30%)
    - Keyword match (30%)
    - Clarity and conciseness (20%)
    - Impact and metrics (20%)

    Return JSON: {{"score": 75, "reasoning": "..."}}
    """
    
    try:
        response = llm.invoke(quality_prompt)
        # Handle potential text/dict returns depending on model output
        content = response.content if hasattr(response, 'content') else str(response)
        data = json.loads(content)
        return float(data.get("score", 0))
    except Exception as e:
        print(f"⚠️ Quality check failed: {e}")
        return 0.0

def supervisor_node(state: AgentState) -> Command[Literal["critics_start", "__end__"]]:
    """
    Intelligent routing with adaptive stopping.
    """
    current_revisions = state.get("revision_count", 0)
    quality_scores = state.get("quality_scores", [])
    
    print(f"🚦 Supervisor Check: Revision {current_revisions}")

    current_score = 0.0

    log_entry = None  # Initialize before conditional to avoid NameError

    if current_revisions > 0 and state.get("resume_text"):
        current_score = calculate_quality_score(
            state["resume_text"],
            state.get("job_description", "")
        )
        quality_scores.append(current_score)
        print(f"  📊 Quality Score: {current_score:.1f}/100")

        # --- MEMORY LOGGING START ---
        # Log the previous loop if we have enough data
        if len(quality_scores) >= 2:
            prev_score = quality_scores[-2]
            
            # Check if the last loop "passed" verification
            # (If verifier passed, it returns no 'unresolved' items usually)
            ver_result = state.get("verification_result", {})
            passed = ver_result.get("pass_verification", False)
            
            # Create Log
            log_entry = RevisionMemory.log_revision(
                iteration=current_revisions,
                critiques=state.get("actionable_critiques", []),
                quality_before=prev_score,
                quality_after=current_score,
                success=passed
            )
            
            # Save to state history
            history = state.get("revision_history", [])
            history.append(log_entry)
            
            print(f"  📝 Logged revision #{current_revisions}: {log_entry['improvement_delta']:+.1f} pts")
            
            # NOW we clear the critiques for the next round
            # We return this update in the Command below
        # --- MEMORY LOGGING END ---

        # 2. Check for improvement (Stopping Logic)
        if len(quality_scores) >= 2:
            improvement = current_score - quality_scores[-2]
            if improvement < 5.0:
                print("  🛑 Diminishing returns. Stopping.")
                return Command(
                    goto="__end__",
                    update={"next_node": "__end__"}
                )
            if current_score >= 90:
                print("  ✅ High quality achieved. Stopping.")
                return Command(
                    goto="__end__",
                    update={"next_node": "__end__"}
                )

    if current_revisions >= 5:
        print("  🛑 Max iterations. Stopping.")
        return Command(
            goto="__end__",
            update={"next_node": "__end__"}
        )

    print("  → Continuing to next iteration")

    # Build history update - only add log_entry if it exists
    history_update = state.get("revision_history", [])
    if log_entry is not None:
        history_update = history_update + [log_entry]

    return Command(
        goto="critics_start",
        update={
            "next_node": "critics_start",
            "quality_scores": quality_scores,
            "revision_history": history_update,
            "actionable_critiques": []  # Clear critiques for next round
        }
    )