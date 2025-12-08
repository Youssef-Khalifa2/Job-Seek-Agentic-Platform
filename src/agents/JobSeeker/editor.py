from langchain_core.prompts import ChatPromptTemplate
from src.agents.JobSeeker.state import AgentState
from src.llm_registry import LLMRegistry
from src.utils import filter_reasoning

# Strong instruction-following model for execution
llm = LLMRegistry.get_gpt_oss() 

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
    """
    critiques = state.get("actionable_critiques", [])
    
    if not critiques:
        print("  😴 Editor has no critiques to apply. Skipping.")
        return {
            "revision_count": state["revision_count"] + 1
        }

    print(f"✍️ Editor applying {len(critiques)} verified critiques...")
    
    # 1. Format instructions
    feedback_str = ""
    for c in critiques:
        feedback_str += f"- {c.summary} (Reasoning: {c.reasoning})\n"
    
    # 2. Execute
    msg = editor_prompt.invoke({
        "resume_text": state["resume_text"],
        "critiques_list": feedback_str
    })
    
    response = llm.invoke(msg)
    filtered_content = filter_reasoning(response)

    return {
        "resume_text": filtered_content,          
        "revision_count": state["revision_count"] + 1,
        "actionable_critiques": [] # Clear queue after applying
    }