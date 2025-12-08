# File: src/agents/JobSeeker/supervisor.py
import json
from typing import Literal
from langgraph.types import Command
from src.agents.JobSeeker.state import AgentState
from src.llm_registry import LLMRegistry

# Use a fast model for quality checks
llm = LLMRegistry.get_mistral_small() 

def calculate_quality_score(resume_text: str, job_description: str) -> float:
    """Score resume quality 0-100 based on fit and clarity."""
    if not resume_text or not job_description:
        return 0.0
        
    quality_prompt = f"""
    Evaluate this resume's quality for the job description.

    Resume (first 5000 chars):
    {resume_text[:5000]}

    Job Description (first 2000 chars):
    {job_description[:2000]}

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
    # Initialize quality_scores if missing
    quality_scores = state.get("quality_scores", [])
    
    print(f"🚦 Supervisor Check: Revision {current_revisions}")

    # 1. Calculate current quality (skip for initial state if empty)
    if current_revisions > 0 and state.get("resume_text"):
        current_score = calculate_quality_score(
            state["resume_text"],
            state.get("job_description", "")
        )
        quality_scores.append(current_score)
        print(f"  📊 Quality Score: {current_score:.1f}/100")

        # 2. Check for improvement or stagnation
        if len(quality_scores) >= 2:
            improvement = current_score - quality_scores[-2]
            print(f"  📈 Improvement: {improvement:+.1f} points")

            # Stop if diminishing returns (< 5 points improvement)
            if improvement < 5.0:
                print("  🛑 Diminishing returns detected. Stopping.")
                return Command(goto="__end__")

            # Stop if high quality achieved (> 85)
            if current_score >= 85:
                print("  ✅ High quality achieved. Stopping.")
                return Command(goto="__end__")

    # 3. Safety Valve: Max 5 iterations (increased from 3 to allow optimization)
    if current_revisions >= 5:
        print("  🛑 Max iterations reached. Stopping.")
        return Command(goto="__end__")

    # 4. Continue Loop
    print("  → Continuing to next iteration")
    
    # We must return the Command with the update to save the scores
    return Command(
        goto="critics_start",
        update={"quality_scores": quality_scores}
    )