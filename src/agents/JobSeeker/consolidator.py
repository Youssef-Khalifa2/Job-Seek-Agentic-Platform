from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from src.agents.job_seeker.state import AgentState, Critique
import json
import config

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro", # Use a smarter model for synthesis
    google_api_key=config.GOOGLE_API_KEY,
    model_kwargs={"response_mime_type": "application/json"}
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

OUTPUT JSON:
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
    """
    print("🧠 Consolidating critiques...")
    
    # 1. Dump the raw inputs to JSON
    raw_inputs = [c.model_dump() for c in state.get("critique_inputs", [])]
    
    # 2. Invoke LLM
    # We allow the LLM to see all inputs at once
    msg = consolidate_prompt.invoke({"raw_critiques": json.dumps(raw_inputs)})
    response = llm.invoke(msg)
    
    # 3. Parse output
    try:
        data = json.loads(response.content)
        # We REWRITE the 'actionable_critiques' list (giving the Editor a fresh to-do list)
        new_critiques = [Critique(**item) for item in data.get("consolidated_list", [])]
        
        return {"actionable_critiques": new_critiques}
    except Exception as e:
        # Fallback if consolidation fails
        print(f"❌ Consolidation Error: {e}")
        return {"actionable_critiques": []}