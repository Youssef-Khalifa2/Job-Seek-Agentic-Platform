# File: src/agents/JobSeeker/verifier.py
import json
from typing import Literal
from langgraph.types import Command
from src.agents.JobSeeker.state import AgentState
from src.llm_registry import LLMRegistry

# Ultra-fast, cheap model for verification
llm = LLMRegistry.get_nova_micro()

def verifier_node(state: AgentState) -> Command[Literal["editor", "supervisor"]]:
    """
    Verifies that editor's changes actually addressed the critiques.

    Routing Logic:
    - If e80% critiques fixed go to "supervisor"
    - If <80% fixed AND retries < 2 go to "editor" (with summary of what's missing)
    - If <80% fixed AND retries e 2 go to "supervisor" (safety valve)
    """
    print("Verifier: Checking if critiques were addressed...")

    # Get critiques that were supposed to be fixed
    applied_critiques = state.get("actionable_critiques", [])
    retry_count = state.get("editor_retry_count", 0)
    max_retries = 2  # Hardcoded as requested

    # Safety check: If no critiques, just pass through
    if not applied_critiques:
        print("No critiques to verify - passing to supervisor")
        return Command(goto="supervisor")

    print(f"Verifying {len(applied_critiques)} critiques...")
    print(f"Current retry count: {retry_count}/{max_retries}")

    # Build verification prompt
    verification_prompt = f"""
    You are a Quality Assurance agent. Verify that the editor addressed the critiques.

    CRITIQUES THAT SHOULD HAVE BEEN FIXED:
    {json.dumps([c.model_dump() for c in applied_critiques], indent=2)}

    EDITED RESUME (current version):
    {state["resume_text"]}

    For each critique, determine:
    1. Was it addressed? (yes/no)
    2. If not, why not? (missing data, overlooked, etc.)

    Return JSON:
    {{
    "critiques_addressed": 4,
    "critiques_total": 5,
    "success_rate": 0.80,
    "unresolved": [
        {{
        "critique_summary": "Brief description of what wasn't fixed",
        "reason": "Why it wasn't addressed"
        }}
    ]
    }}
    """

    try:
        response = llm.invoke(verification_prompt)

        # Parse response (handle potential markdown wrapping)
        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1]

        data = json.loads(content.strip())

        addressed = data.get("critiques_addressed", 0)
        total = data.get("critiques_total", len(applied_critiques))
        success_rate = data.get("success_rate", addressed / total if total > 0 else 0)

        print(f" Verification: {addressed}/{total} critiques addressed ({success_rate:.0%})")

        # ROUTING DECISION
        # Threshold: 80% = pass
        if success_rate >= 0.80:
            print(" Verification passed (80%) - moving to supervisor")
            return Command(
                goto="supervisor",
                update={
                    "verification_result": data,
                    "editor_retry_count": 0  # Reset for next iteration
                }
            )
        else:
            # Failed verification
            print(f" Verification failed ({success_rate:.0%} < 80%)")

            # Check if we can retry
            if retry_count < max_retries:
                print(f" Retrying editor ({retry_count + 1}/{max_retries})")

                # Create summary of unresolved items for editor to focus on
                unresolved_summary = "## Unresolved Critiques:\n"
                for item in data.get("unresolved", []):
                    unresolved_summary += f"- {item.get('critique_summary', 'Unknown')}: {item.get('reason', 'Not addressed')}\n"

                print(f" Sending back to editor with focused guidance")

                return Command(
                    goto="editor",
                    update={
                        "editor_retry_count": retry_count + 1,
                        "verification_result": data,
                        "unresolved_critiques": data.get("unresolved", []),
                        # Add summary to help editor focus
                        "retry_guidance": unresolved_summary
                    }
                )
            else:
                # Max retries exhausted - safety valve
                print(f" Max retries ({max_retries}) reached - moving to supervisor anyway")
                return Command(
                    goto="supervisor",
                    update={
                        "verification_result": data,
                        "editor_retry_count": 0,  # Reset for next iteration
                        "unresolved_critiques": data.get("unresolved", [])
                    }
                )

    except Exception as e:
        # Error handling - default to supervisor to avoid getting stuck
        print(f" Verification error: {e}")
        print(" Defaulting to supervisor (safety)")
        return Command(
            goto="supervisor",
            update={
                "verification_result": {"error": str(e)},
                "editor_retry_count": 0
            }
        )
