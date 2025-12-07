from langchain_core.prompts import ChatPromptTemplate
from langchain_aws import ChatAWSBedrock
from src.agents.job_seeker.state import AgentState
import config
from src.utils import filter_reasoning

llm = ChatBedrock(
            model_id="openai.gpt-oss-120b-1:0",
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
            region_name=config.AWS_REGION,
        )

editor_prompt = ChatPromptTemplate.from_template("""
You are an Expert Resume Editor.
Your goal is to apply specific feedback to a candidate's CV.

CURRENT CV:
{resume_text}

FEEDBACK TO IMPLEMENT:
{critiques_list}

INSTRUCTIONS:
1. Apply the requested changes **surgically**. Do not rewrite parts of the CV that were not mentioned in the feedback.
2. Maintain the original structure and formatting as much as possible.
3. If the feedback asks for a metric you don't have, use a placeholder like [X%].

OUTPUT:
Return ONLY the full, rewritten CV text. Do not add conversational filler.
""")

def editor_node(state: AgentState):
    """
    Applies the actionable critiques to the resume text.
    """
    print(f"✍️ Editor is fixing {len(state['actionable_critiques'])} issues...")
    
    # 1. Format the critiques for the LLM
    feedback_str = ""
    for c in state["actionable_critiques"]:
        feedback_str += f"- {c.summary}\n"
        # We could add c.details here if we want the Editor to see the raw JSON
    
    # 2. Invoke the Editor
    msg = editor_prompt.invoke({
        "resume_text": state["resume_text"],
        "critiques_list": feedback_str
    })
    
    response = llm.invoke(msg)

    response = filter_reasoning(response)
    
    # 3. Update the State
    return {
        "resume_text": response.content,           # The new CV!
        "revision_count": state["revision_count"] + 1,
        "actionable_critiques": []                 # Clear the to-do list for the next round
    }