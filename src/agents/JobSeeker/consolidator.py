from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from src.agents.JobSeeker.state import AgentState, Critique
from typing import Set
import json
import config

llm = ChatMistralAI(
    model="mistral-small-latest", # Use a smarter model for synthesis
    mistral_api_key=config.MISTRAL_API_KEY
)

consolidate_prompt = ChatPromptTemplate.from_template("""
You are the **Lead Editor**. 5 specialists have critiqued a resume.
Your job is to merge their feedback and DELETE low-priority advice.

INPUT CRITIQUES:
{raw_critiques}

RULES:
1. **Merge Duplicates:** If ATS and Language critics both mention "passive voice," make it ONE item.
2. **Drop Low Priority:**
   - DELETE nitpicks about single words (unless critical keywords).
   - DELETE vague advice like "make it pop."
   - DELETE anything that requires information the candidate obviously doesn't have.
3. **Prioritize:**
   - MISSING KEYWORDS (Critical)
   - BROKEN FORMATTING (Critical)
   - UNSUPPORTED CLAIMS (Critical)
   - WEAK METRICS (Important)

STRICT OUTPUT FORMAT IS JSON AS FOLLOWS:
{{
  "consolidated_list": [
    {{
      "source": "Merged Source (e.g. 'ATS + Match')",
      "summary": "Specific instruction for the editor...",
      "details": {{ ...merged details... }},
      "resolved": false
    }}
  ]
}}
""")

def consolidator_node(state: AgentState):
    """
    Merges 5 critique streams into one actionable list.
    Filters out already-resolved critiques from previous loops.
    """
    print("🧠 Consolidating critiques...")

    # 1. Get resolved IDs from previous loops
    resolved_ids = state.get("resolved_critique_ids", set())

    # 2. Filter out already-resolved critiques
    all_critiques = state.get("critique_inputs", [])
    unresolved_critiques = [
        c for c in all_critiques
        if c.id not in resolved_ids
    ]

    print(f"📊 Total critiques: {len(all_critiques)}")
    print(f"🔍 Unresolved critiques: {len(unresolved_critiques)}")
    print(f"✅ Filtered out: {len(all_critiques) - len(unresolved_critiques)} resolved")

    # 3. Pass only unresolved critiques to LLM
    raw_inputs = [c.model_dump() for c in unresolved_critiques]
    msg = consolidate_prompt.invoke({"raw_critiques": json.dumps(raw_inputs)})
    response = llm.invoke(msg)

    # 4. Parse LLM output
    try:
        # Strip markdown code blocks if present
        content = response.content.strip()

        # Handle case where LLM adds text before JSON block
        if "```json" in content:
            # Extract content between ```json and ```
            start = content.find("```json") + 7
            end = content.find("```", start)
            if end != -1:
                content = content[start:end].strip()
            else:
                content = content[start:].strip()
        elif content.startswith("```"):
            # Simple case: starts with ```
            content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
        elif content.endswith("```"):
            content = content[:-3].strip()

        data = json.loads(content)
        new_actionables = [Critique(**item) for item in data.get("consolidated_list", [])]

        # 5. Mark consolidated critiques as resolved
        # Strategy: Mark all unresolved critiques as resolved after consolidation
        newly_resolved_ids = {c.id for c in unresolved_critiques}
        updated_resolved_ids = resolved_ids.union(newly_resolved_ids)

        return {
            "actionable_critiques": new_actionables,
            "resolved_critique_ids": updated_resolved_ids
        }
    except json.JSONDecodeError as e:
        print(f"❌ Consolidation JSON Error: {e}")
        print(f"Response content: {response.content[:200]}")
        return {
            "actionable_critiques": [],
            "resolved_critique_ids": resolved_ids  # Keep existing
        }
    except Exception as e:
        print(f"❌ Consolidation Error: {e}")
        print(f"Response content: {response.content[:200]}")
        return {
            "actionable_critiques": [],
            "resolved_critique_ids": resolved_ids  # Keep existing
        }