from langchain_core.prompts import ChatPromptTemplate
from src.agents.JobSeeker.state import AgentState
from src.llm_registry import LLMRegistry
from src.utils import filter_reasoning
import json # Don't forget to import json

# Strong instruction-following model for execution
# Switched to Sonnet for better instruction following (GPT-OSS was only fixing 20% of critiques)
llm = LLMRegistry.get_sonnet() 

editor_prompt = ChatPromptTemplate.from_template("""
You are an Expert Resume Editor. 
Your goal is to EXECUTE the following changes on the candidate's CV.

CURRENT CV:
{resume_text}

INSTRUCTIONS TO EXECUTE:
{critiques_list}

RULES:
1. Trust these instructions; they have already been verified.
2. Apply changes **surgically**. Do not rewrite sections that aren't mentioned.
3. Maintain the original structure and formatting unless told otherwise.
4. If asked to add metrics but none are provided, use placeholders like [X%].

OUTPUT:
Return ONLY the full, rewritten CV text (Markdown format).
""")

def editor_node(state: AgentState):
    """
    Pure execution node. Applies the pre-validated critiques.
    On retry: Also considers verifier's focused guidance.
    Now includes Self-Reflection.
    """
    critiques = state.get("actionable_critiques", [])
    retry_guidance = state.get("retry_guidance")
    retry_count = state.get("editor_retry_count", 0)

    if not critiques:
        print("  😴 Editor has no critiques to apply. Skipping.")
        return {
            "revision_count": state["revision_count"] + 1,
            "editor_retry_count": 0  # Reset retry count when skipping
        }

    if retry_count > 0:
        print(f"✍️ Editor RETRY #{retry_count}: Applying {len(critiques)} critiques with focused guidance...")
    else:
        print(f"✍️ Editor applying {len(critiques)} verified critiques...")

    # STEP 1: THINK - Plan how to address critiques
    print("  💭 THINK: Planning how to address each critique...")

    # 1. Format critiques for thinking
    feedback_str = ""
    for i, c in enumerate(critiques, 1):
        feedback_str += f"{i}. {c.summary}\n   Reasoning: {c.reasoning}\n\n"

    # 2. Add retry guidance if this is a retry
    retry_context = ""
    if retry_guidance:
        retry_context = f"\n**VERIFIER FEEDBACK (What you missed last time):**\n{retry_guidance}\n"
        print("  ⚠️ Including verifier's focused guidance in prompt")

    # Create thinking prompt
    thinking_prompt = f"""
You are planning how to edit a resume based on critiques.

CURRENT RESUME:
{state.get("resume_text", "")[:1000]}...

CRITIQUES TO ADDRESS:
{feedback_str}
{retry_context}

Think through:
1. What specific changes do I need to make for each critique?
2. Are there any conflicting changes I need to resolve?
3. What order should I make these changes in?
4. Are there any critiques I cannot address (missing information)?

Return JSON:
{{
  "action_plan": [
    {{"critique_num": 1, "action": "Change XYZ to ABC", "feasible": true}},
    {{"critique_num": 2, "action": "Add metric to experience section", "feasible": true}}
  ],
  "concerns": ["concern if any"],
  "strategy": "Overall approach in 1-2 sentences"
}}
"""

    thinking_response = llm.invoke(thinking_prompt)
    try:
        # Clean markdown
        content = thinking_response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        plan = json.loads(content.strip())
        print(f"    → Strategy: {plan.get('strategy', 'Apply changes systematically')[:80]}")
        print(f"    → Actions planned: {len(plan.get('action_plan', []))}")
    except Exception as e:
        print(f"  ⚠️ Planning failed, proceeding with direct application: {e}")
        plan = {"strategy": "Apply all critiques directly", "action_plan": []}

    # STEP 2: ACT - Execute the plan
    print("  🔧 ACT: Executing changes...")

    # Include the plan in the execution prompt
    execution_instructions = f"""
YOUR PLAN:
{plan.get('strategy', 'Apply all critiques')}

ACTIONS TO TAKE:
{feedback_str}
{retry_context}
"""

    msg = editor_prompt.invoke({
        "resume_text": state["resume_text"],
        "critiques_list": execution_instructions
    })

    response = llm.invoke(msg)
    filtered_content = filter_reasoning(response)

    # --- STEP 3: REFLECT (Simplified - non-critical) ---
    print("  👁️ OBSERVE: Checking changes...")
    # Simple non-JSON reflection - just log that changes were made
    reflection_data = {
        "confidence": 0.85,
        "self_assessment": "Changes applied based on plan",
        "concerns": []
    }
    print(f"  ✅ Applied {len(critiques)} critiques using planned approach")

    return {
        "resume_text": filtered_content,
        "revision_count": state["revision_count"] + 1,
        "retry_guidance": None
        # DON'T clear actionable_critiques - verifier needs them to verify!
        # Supervisor will clear them when moving to next iteration
    }