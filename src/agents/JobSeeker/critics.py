import json
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from src.agents.JobSeeker.state import AgentState, Critique
from src.tools.ats_parser import check_ats_compatibility
from src.tools.layout_analyzer import analyze_resume_layout
import config
from src.llm_registry import LLMRegistry

llm = LLMRegistry.get_pixtral_large()

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
    """
    ATS Critic with DUAL analysis: 
    1. Text-based ATS parsing (The 'Black Box')
    2. Visual layout analysis (The 'Eyes')
    """
    print("🤖 ATS Critic: Running dual analysis...")
    
    # 1. Run the Tools
    # We use the original PDF path for accurate visual/parsing analysis
    pdf_path = state.get("original_pdf_path")
    
    if not pdf_path:
        print("  ⚠️ No PDF path found. Skipping deep analysis.")
        return {"critique_inputs": []}

    print("  → Testing ATS parsing...")
    ats_result = check_ats_compatibility(pdf_path)

    print("  → Analyzing visual layout with VLM...")
    layout_result = analyze_resume_layout(pdf_path)

    # 2. Synthesize findings with LLM
    # We give the LLM the hard data so it can explain *why* something is wrong.
    synthesis_prompt = f"""
    You are an Expert ATS Auditor. Analyze these forensic test results.

    EVIDENCE #1: REAL ATS PARSER OUTPUT
    {json.dumps(ats_result, indent=2)}

    EVIDENCE #2: VISUAL LAYOUT ANALYSIS (VLM)
    {json.dumps(layout_result, indent=2)}

    TASK:
    Compare the two evidences. 
    - If VLM sees a "Skills" section but ATS Parser returned "skills": [], that is a CRITICAL parsing failure caused by layout.
    - If VLM sees columns and ATS Parser missed fields, blame the columns.
    - If both are fine, give a high score.

    OUTPUT JSON:
    {{
        "score": 0-100,
        "critical_issues": ["Specific parsing failure 1", "Specific layout risk 2"],
        "warnings": ["minor issue 1"],
        "reasoning": "Explain the connection between layout and parsing errors."
    }}
    """
    
    # Use a fast model (Haiku/Mistral Small) for synthesis
    res = llm.invoke(synthesis_prompt)

    try:
        data = json.loads(res.content)
        
        # Helper logic: If tools found errors, confidence is high (we have proof).
        # If tools are clean, confidence is slightly lower (false negatives possible).
        tool_evidence_found = (
            not ats_result.get("parsable_status", True) or 
            layout_result.get("has_columns", False)
        )
        confidence = 0.95 if tool_evidence_found else 0.8

        # Create the summary
        summary = f"### 🤖 ATS Report (Score: {data.get('score')})\n"
        for issue in data.get("critical_issues", []):
            summary += f"- 🔴 **CRITICAL:** {issue}\n"
        for warn in data.get("warnings", []):
            summary += f"- ⚠️ {warn}\n"

        critique = Critique(
            source="ATS Critic",
            summary=summary,
            details=data,
            resolved=False,
            confidence=confidence,
            reasoning=data.get("reasoning", ""),
            tool_evidence={
                "ats_parser": ats_result, 
                "layout": layout_result
            }
        )

        print(f"  ✓ Analysis complete. Score: {data.get('score')}")
        return {"critique_inputs": [critique]}

    except Exception as e:
        print(f"❌ ATS Critic synthesis failed: {e}")
        return {"critique_inputs": []}

def match_critic_node(state: AgentState):
    res = llm.invoke(match_prompt.invoke({"job_description": state["job_description"], "resume_text": state["resume_text"]}))

    try:
        # Parse JSON to get both raw data and summary
        data = json.loads(res.content)
        summary = parse_critic_output(res.content, "JD Match Critic")

        # Create Critique object
        critique = Critique(
            source="JD Match Critic",
            summary=summary,
            details=data,
            resolved=False
        )

        return {"critique_inputs": [critique]}
    except json.JSONDecodeError as e:
        print(f"❌ JD Match Critic JSON Error: {e}")
        print(f"Response content: {res.content[:200]}")
        error_critique = Critique(
            source="JD Match Critic",
            summary=f"### Error\nFailed to parse JSON: {str(e)}",
            details={"error": str(e), "raw_response": res.content[:500]},
            resolved=False
        )
        return {"critique_inputs": [error_critique]}

def truth_critic_node(state: AgentState):
    res = llm.invoke(truth_prompt.invoke({"resume_text": state["resume_text"]}))

    try:
        # Parse JSON to get both raw data and summary
        data = json.loads(res.content)
        summary = parse_critic_output(res.content, "Truth Critic")

        # Create Critique object
        critique = Critique(
            source="Truth Critic",
            summary=summary,
            details=data,
            resolved=False
        )

        return {"critique_inputs": [critique]}
    except json.JSONDecodeError as e:
        print(f"❌ Truth Critic JSON Error: {e}")
        print(f"Response content: {res.content[:200]}")
        error_critique = Critique(
            source="Truth Critic",
            summary=f"### Error\nFailed to parse JSON: {str(e)}",
            details={"error": str(e), "raw_response": res.content[:500]},
            resolved=False
        )
        return {"critique_inputs": [error_critique]}

def language_critic_node(state: AgentState):
    res = llm.invoke(language_prompt.invoke({"resume_text": state["resume_text"]}))

    try:
        # Parse JSON to get both raw data and summary
        data = json.loads(res.content)
        summary = parse_critic_output(res.content, "Language Critic")

        # Create Critique object
        critique = Critique(
            source="Language Critic",
            summary=summary,
            details=data,
            resolved=False
        )

        return {"critique_inputs": [critique]}
    except json.JSONDecodeError as e:
        print(f"❌ Language Critic JSON Error: {e}")
        print(f"Response content: {res.content[:200]}")
        error_critique = Critique(
            source="Language Critic",
            summary=f"### Error\nFailed to parse JSON: {str(e)}",
            details={"error": str(e), "raw_response": res.content[:500]},
            resolved=False
        )
        return {"critique_inputs": [error_critique]}

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

        # Create Critique object
        critique = Critique(
            source="Impact Critic",
            summary=summary,
            details=data,
            resolved=False
        )

        return {"critique_inputs": [critique]}
    except Exception as e:
        # Error handling - return error critique
        error_critique = Critique(
            source="Impact Critic",
            summary=f"### Error parsing Impact Report\n{str(e)}",
            details={"error": str(e)},
            resolved=False
        )
        return {"critique_inputs": [error_critique]}