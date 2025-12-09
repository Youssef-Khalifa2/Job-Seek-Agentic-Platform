from langchain_core.prompts import ChatPromptTemplate
# We can use a smarter model here since this is the critical "Decision Point"
from src.llm_registry import LLMRegistry 
from src.agents.JobSeeker.state import AgentState, Critique
from typing import Set
import json

# Use Sonnet or GPT-OSS for better reasoning on conflicts, 
# or stick to Mistral Small if speed/cost is priority.
llm = LLMRegistry.get_mistral_medium()

consolidate_prompt = ChatPromptTemplate.from_template("""
You are the **Lead Editor**. Specialists have critiqued a resume.
Your job is to merge their feedback, RESOLVE CONFLICTS, and FILTER out bad advice.

INPUT CRITIQUES:
{raw_critiques}

RULES:
1. **Merge Duplicates:** If ATS and Language critics both mention the same issue, combine them.
2. **Resolve Conflicts:** If one critic says "delete summary" and another says "expand summary", pick the one that best serves the candidate and discard the other.
3. **Quality Filter (The Gatekeeper):**
   - DELETE vague advice (e.g., "make it pop").
   - DELETE advice requiring missing info (e.g., "add GPA" if unknown).
   - DELETE nitpicks unless they are critical keywords.
4. **Scoring:** Assign a confidence score (0.0 - 1.0) to each final item.

STRICT OUTPUT FORMAT IS JSON:
{{
  "consolidated_list": [
    {{
      "source": "Merged Source",
      "summary": "Clear, actionable instruction for the editor...",
      "details": {{ "original_critiques": [...] }},
      "confidence": 0.95,
      "reasoning": "Why this change is necessary...",
      "resolved": false
    }}
  ]
}}
""")

def consolidator_node(state: AgentState):
    """
    Merges critique streams, resolves conflicts, and filters low-quality advice.
    Centralizes the 'Thinking' logic here so Editor can just 'Act'.
    """
    print("🧠 Consolidating & Filtering critiques...")

    # 1. Get resolved IDs
    resolved_ids = state.get("resolved_critique_ids", set())

    # 2. Filter out already-resolved critiques
    all_critiques = state.get("critique_inputs", [])
    unresolved_critiques = [
        c for c in all_critiques
        if c.id not in resolved_ids
    ]

    # Check if we're in retry mode
    retry_mode = state.get("editor_retry_count", 0) > 0

    if not unresolved_critiques:
        if not retry_mode:
            print("  ✨ No new critiques to process.")
            return {"actionable_critiques": []}
        else:
            # During retry, keep existing critiques
            print("  ℹ️ No new critiques, but in retry mode - keeping existing critiques")
            return {}  # Return empty dict to not update actionable_critiques

    # 3. Pass to LLM
    raw_inputs = [c.model_dump() for c in unresolved_critiques]
    msg = consolidate_prompt.invoke({"raw_critiques": json.dumps(raw_inputs)})
    response = llm.invoke(msg)

    # 4. Parse LLM output
    try:
        # (Your existing JSON cleaning logic here...)
        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1]
            
        data = json.loads(content.strip())
        
        # Create objects and filter by confidence
        new_actionables = []
        for item in data.get("consolidated_list", []):
            # The Consolidator is the "Quality Gate" now
            # We can programmatically enforce a threshold here if we want
            if item.get("confidence", 0) >= 0.7: 
                new_actionables.append(Critique(**item))
            else:
                print(f"  🗑️ Dropped low-confidence item: {item.get('summary')}")

        print(f"  ✅ Consolidator produced {len(new_actionables)} actionable items")

        # DON'T mark as resolved yet - verifier will do that after confirming editor's work
        # Keep resolved IDs as-is
        return {
            "actionable_critiques": new_actionables,
            "resolved_critique_ids": resolved_ids  # No change - verifier handles resolution
        }

    except Exception as e:
        print(f"❌ Consolidation Error: {e}")
        return {
            "actionable_critiques": [],
            "resolved_critique_ids": resolved_ids 
        }