import json
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from src.agents.state import AgentState
import config

# We use Flash for speed, but enforce JSON mode for structure.
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    google_api_key=config.GOOGLE_API_KEY,
    model_kwargs={"response_mime_type": "application/json"} 
)

# helper to safely parse the JSON
def parse_critic_output(response_content: str, critic_name: str) -> str:
    try:
        data = json.loads(response_content)
        # Convert JSON to a readable string for the AgentState history
        # You can customize this to be as detailed or brief as you want
        summary = f"### 🕵️ {critic_name} Report\n"
        
        if "feedback_list" in data:
            for item in data["feedback_list"]:
                summary += f"- **{item.get('issue', 'Issue')}**: {item.get('fix', 'Fix it')}\n"
        
        if "missing_keywords" in data:
            summary += f"**Missing Keywords:** {', '.join(data['missing_keywords'])}\n"
            
        if "score" in data:
            summary += f"**Score:** {data['score']}/100\n"

        return summary
    except json.JSONDecodeError:
        return f"### {critic_name} Error\nFailed to parse JSON output."

# --- 1. ATS & Formatting Critic 🤖 ---
ats_prompt = ChatPromptTemplate.from_template("""
You are the **ATS Gatekeeper**. Your goal is to ensure the resume is machine-readable.

RESUME TEXT:
{resume_text}

TASK:
Identify layout or formatting risks that would break a parser (e.g., tables, columns, icons, header issues).
Since you are reading raw text, infer these structures based on odd spacing or line breaks.

OUTPUT JSON:
{{
  "score": 0-100,
  "feedback_list": [
    {{
      "issue": "string (e.g. 'Possible Table detected')",
      "location": "string (section name)",
      "fix": "string (e.g. 'Use standard bullets')"
    }}
  ]
}}
""")

# --- 2. JD Match Critic 🎯 ---
match_prompt = ChatPromptTemplate.from_template("""
You are the **Recruiter Bot**. Your goal is to see if this candidate matches the Job Description.

JOB DESCRIPTION:
{job_description}

RESUME:
{resume_text}

TASK:
1. Extract hard skills from the JD that are MISSING in the Resume.
2. Identify bullets that are totally irrelevant to this job.

OUTPUT JSON:
{{
  "match_score": 0-100,
  "missing_keywords": ["string", "string"],
  "irrelevant_bullets": [
    {{
      "bullet": "string",
      "reason": "string"
    }}
  ]
}}
""")

# --- 3. Evidence/Truth Critic ⚖️ ---
truth_prompt = ChatPromptTemplate.from_template("""
You are the **Integrity Auditor**. Your goal is to find unsupported claims.

RESUME:
{resume_text}

TASK:
1. Find skills listed in "Skills" section that appear NOWHERE in the "Experience" bullets.
2. Flag vague claims ("Led a team", "Managed projects") that lack scope/size.

OUTPUT JSON:
{{
  "feedback_list": [
    {{
      "claim": "string",
      "issue": "string (e.g. 'Unsupported Skill' or 'Vague Claim')",
      "fix": "string (e.g. 'Add a bullet about using Python')"
    }}
  ]
}}
""")

# --- 4. Language & Clarity Critic ✍️ ---
language_prompt = ChatPromptTemplate.from_template("""
You are the **Chief Editor**. Your goal is to remove fluff and passive voice.

RESUME:
{resume_text}

TASK:
1. specific sentences with passive voice ("was done by").
2. Remove buzzwords ("hardworking", "synergy", "proactive").

OUTPUT JSON:
{{
  "feedback_list": [
    {{
      "original": "string",
      "issue": "string",
      "better_rewrite": "string"
    }}
  ]
}}
""")

# --- 5. Impact Critic (Your Custom One) ---
# (Keeping the one we designed previously)
impact_prompt = ChatPromptTemplate.from_template("""
You are the **Impact Critic**.
RESUME: {resume_text}
OUTPUT JSON:
{{
  "critique_feedback": [
    {{
      "original_bullet": "string",
      "why_weak": "string",
      "metric_ideas": ["string"],
      "star_rewrite": "string"
    }}
  ],
  "metrics_wishlist": ["string"]
}}
""")

# --- Node Functions ---

def ats_critic_node(state: AgentState):
    res = llm.invoke(ats_prompt.invoke({"resume_text": state["resume_text"]}))
    return {"critique_feedback": [parse_critic_output(res.content, "ATS Critic")]}

def match_critic_node(state: AgentState):
    res = llm.invoke(match_prompt.invoke({"job_description": state["job_description"], "resume_text": state["resume_text"]}))
    return {"critique_feedback": [parse_critic_output(res.content, "JD Match Critic")]}

def truth_critic_node(state: AgentState):
    res = llm.invoke(truth_prompt.invoke({"resume_text": state["resume_text"]}))
    return {"critique_feedback": [parse_critic_output(res.content, "Truth Critic")]}

def language_critic_node(state: AgentState):
    res = llm.invoke(language_prompt.invoke({"resume_text": state["resume_text"]}))
    return {"critique_feedback": [parse_critic_output(res.content, "Language Critic")]}

def impact_critic_node(state: AgentState):
    # This one has a slightly different JSON structure, so we parse it custom or adapt the helper
    res = llm.invoke(impact_prompt.invoke({"resume_text": state["resume_text"]}))
    
    # Custom parsing for the complex Impact JSON
    try:
        data = json.loads(res.content)
        summary = "### 📈 Impact Report\n"
        summary += f"**Wishlist:** {', '.join(data.get('metrics_wishlist', []))}\n"
        for item in data.get("critique_feedback", []):
            summary += f"- **Original:** {item['original_bullet']}\n"
            summary += f"  - *Fix:* {item['star_rewrite']}\n"
        return {"critique_feedback": [summary]}
    except:
        return {"critique_feedback": ["Error parsing Impact Report"]}