import json
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from src.agents.JobSeeker.state import AgentState, Critique
from src.tools.ats_parser import check_ats_compatibility
from src.tools.layout_analyzer import analyze_resume_layout
from src.tools.web_search import analyze_job_description, validate_market_demand 
import config
from src.tools.grammar_check import check_grammar
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
    ATS Critic with Full ReAct Pattern:
    1. THINK: Reason about what to check
    2. ACT: Use tools based on reasoning
    3. OBSERVE: Analyze tool results
    4. DECIDE: Generate critique or gather more data
    """
    print("🤖 ATS Critic: Starting ReAct analysis...")

    pdf_path = state.get("original_pdf_path")
    if not pdf_path:
        print("  ⚠️ No PDF path found. Skipping deep analysis.")
        return {"critique_inputs": []}

    # STEP 1: THINK - Reason about what to analyze
    print("  💭 THINK: Planning analysis strategy...")
    thinking_prompt = f"""
You are an ATS Expert planning how to analyze a resume.

RESUME TEXT (first 500 chars):
{state.get("resume_text", "")}

Think through:
1. What are the most common ATS compatibility issues?
2. What specific patterns should I look for in this resume?
3. Which tools should I use and why?
4. What evidence would I need to make confident critiques?

Return JSON:
{{
  "analysis_plan": ["check 1", "check 2"],
  "tools_to_use": ["ATS Parser", "VLM Layout Analyzer"],
  "hypotheses": ["potential issue 1", "potential issue 2"],
  "reasoning": "Why this approach"
}}
"""

    thinking_response = llm.invoke(thinking_prompt)
    try:
        # Clean potential markdown wrapping
        content = thinking_response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        thinking_data = json.loads(content.strip())
        print(f"    → Plan: {thinking_data.get('reasoning', 'N/A')[:80]}...")
    except json.JSONDecodeError as e:
        print(f"  ⚠️ JSON parsing failed: {e}")
        thinking_data = {"analysis_plan": ["Run standard checks"]}
    except Exception as e:
        print(f"  ❌ Unexpected error in thinking step: {e}")
        thinking_data = {"analysis_plan": ["Run standard checks"]}

    # STEP 2: ACT - Execute tools based on plan
    print("  🔧 ACT: Running analysis tools...")
    print("    → Testing ATS parsing...")
    ats_result = check_ats_compatibility(pdf_path)

    # Only run VLM on first iteration (revision 0) since we're analyzing the ORIGINAL PDF
    # After edits, the text changes but the PDF stays the same, so VLM would give stale results
    revision_count = state.get("revision_count", 0)
    if revision_count == 0:
        print("    → Analyzing visual layout with VLM (first iteration only)...")
        layout_result = analyze_resume_layout(pdf_path)
    else:
        print("    → Skipping VLM analysis (only runs on original PDF)")
        layout_result = {
            "has_columns": False,
            "has_graphics": False,
            "has_tables": False,
            "overall_score": 100,
            "layout_issues": [],
            "recommendation": "Layout already analyzed in first iteration"
        }

    # STEP 3: OBSERVE - Analyze tool outputs
    print("  👁️ OBSERVE: Interpreting tool results...")
    observation_prompt = f"""
You are analyzing tool outputs from an ATS test.

TOOL #1 - ATS PARSER:
{json.dumps(ats_result, indent=2)}

TOOL #2 - VLM LAYOUT:
{json.dumps(layout_result, indent=2)}

ORIGINAL PLAN:
{json.dumps(thinking_data, indent=2)}

Observations:
1. What did the tools reveal?
2. Do the results match my hypotheses?
3. Are there discrepancies between tools (e.g., VLM sees content ATS parser missed)?
4. Do I have enough information, or should I investigate further?

Return JSON:
{{
  "key_findings": ["finding 1", "finding 2"],
  "discrepancies": ["discrepancy 1"],
  "confidence_in_findings": 0.9,
  "need_more_data": false,
  "observations": "Summary of what I learned"
}}
"""

    observation_response = llm.invoke(observation_prompt)
    try:
        # Clean potential markdown wrapping
        content = observation_response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        observation_data = json.loads(content.strip())
        print(f"    → Findings: {len(observation_data.get('key_findings', []))} key issues identified")

        # Check if more data needed (for now, we'll skip the loop)
        if observation_data.get("need_more_data", False):
            print("    ⚠️ Agent wants more data (feature not implemented yet)")
    except json.JSONDecodeError as e:
        print(f"  ⚠️ JSON parsing failed: {e}")
        observation_data = {"key_findings": [], "confidence_in_findings": 0.5}
    except Exception as e:
        print(f"  ❌ Unexpected error in observation step: {e}")
        observation_data = {"key_findings": [], "confidence_in_findings": 0.5}

    # STEP 4: DECIDE - Generate final critique
    print("  ⚖️ DECIDE: Generating final critique...")
    decision_prompt = f"""
Based on my complete analysis, generate the final ATS critique.

THINKING (My Plan):
{json.dumps(thinking_data, indent=2)}

OBSERVATIONS (What I Found):
{json.dumps(observation_data, indent=2)}

TOOL EVIDENCE:
- ATS Parser: {json.dumps(ats_result, indent=2)}
- VLM Layout: {json.dumps(layout_result, indent=2)}

Generate final verdict:
- If VLM sees columns and ATS parser missed fields → CRITICAL
- If both tools show clean results → High score
- Focus on actionable, evidence-backed issues

OUTPUT JSON:
{{
  "score": 0-100,
  "critical_issues": ["Specific parsing failure 1"],
  "warnings": ["minor issue 1"],
  "reasoning": "Connect findings to evidence"
}}
"""

    decision_response = llm.invoke(decision_prompt)

    try:
        # Clean potential markdown wrapping
        content = decision_response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        data = json.loads(content.strip())

        tool_evidence_found = (
            not ats_result.get("parsable_status", True) or
            layout_result.get("has_columns", False)
        )
        confidence = observation_data.get("confidence_in_findings", 0.9 if tool_evidence_found else 0.8)

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
                "layout": layout_result,
                "react_thinking": thinking_data,
                "react_observations": observation_data
            }
        )

        print(f"  ✅ ATS ReAct complete. Score: {data.get('score')}")
        return {"critique_inputs": [critique]}

    except Exception as e:
        print(f"❌ ATS Critic ReAct failed: {e}")
        return {"critique_inputs": []}

def match_critic_node(state: AgentState):
    """
    Match Critic with Full ReAct Pattern:
    1. THINK: Identify what skills/requirements to investigate
    2. ACT: Extract JD requirements, validate via web search
    3. OBSERVE: Analyze JD vs Resume gap
    4. DECIDE: Generate match critique
    """
    print("📊 Match Critic: Starting ReAct analysis...")

    # STEP 1: THINK - Plan the analysis
    print("  💭 THINK: Planning match analysis...")
    thinking_prompt = f"""
You are a Recruiter analyzing how well a resume matches a job description.

JOB DESCRIPTION:
{state.get("job_description", "")}

RESUME (excerpt):
{state.get("resume_text", "")}

Think through:
1. What are the must-have skills vs nice-to-have?
2. Which skills should I validate against market demand?
3. What synonym matches should I check (e.g., React vs Next.js)?
4. How should I prioritize gaps?

Return JSON:
{{
  "priority_skills_to_check": ["skill1", "skill2"],
  "hypotheses": ["Candidate may be missing X"],
  "validation_strategy": "Why I'll use web search",
  "reasoning": "My approach"
}}
"""

    thinking_response = llm.invoke(thinking_prompt)
    try:
        # Clean potential markdown wrapping
        content = thinking_response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        thinking_data = json.loads(content.strip())
        print(f"    → Plan: Checking {len(thinking_data.get('priority_skills_to_check', []))} priority skills")
    except json.JSONDecodeError as e:
        print(f"  ⚠️ JSON parsing failed: {e}")
        thinking_data = {"priority_skills_to_check": []}
    except Exception as e:
        print(f"  ❌ Unexpected error in thinking step: {e}")
        thinking_data = {"priority_skills_to_check": []}

    # STEP 2: ACT - Extract JD and validate skills
    print("  🔧 ACT: Extracting requirements and validating...")
    print("    → Analyzing JD structure...")
    jd_analysis = analyze_job_description(state.get("job_description", ""))
    job_title = jd_analysis.get("job_title", "Candidate")
    required_skills = jd_analysis.get("required_skills", [])

    print(f"    → Role: {job_title} | Skills found: {len(required_skills)}")

    # Validate priority skills from thinking phase, or top 3
    skills_to_validate = thinking_data.get("priority_skills_to_check", required_skills[:3])
    market_validation = {}
    for skill in skills_to_validate[:3]:
        if skill in required_skills or skill.lower() in [s.lower() for s in required_skills]:
            validation = validate_market_demand(skill, job_title)
            market_validation[skill] = validation.get("validated", False)
            print(f"    → Web check: {skill} = {'✓' if validation.get('validated') else '✗'}")

    # STEP 3: OBSERVE - Analyze the gap
    print("  👁️ OBSERVE: Analyzing JD-Resume gap...")
    observation_prompt = f"""
Compare the job requirements with the candidate's resume.

JOB ANALYSIS:
{json.dumps(jd_analysis, indent=2)}

MARKET VALIDATION (Web):
{json.dumps(market_validation, indent=2)}

RESUME TEXT:
{state.get("resume_text", "")}

ORIGINAL PLAN:
{json.dumps(thinking_data, indent=2)}

Observations:
1. Which required skills are missing?
2. Which skills have synonym matches (e.g., React in JD, Next.js in resume)?
3. Do validation results confirm these are critical gaps?
4. What's the overall match quality?

Return JSON:
{{
  "critical_gaps": ["skill1"],
  "synonym_matches": [{{"jd": "React", "resume": "Next.js"}}],
  "validated_critical_gaps": ["skill2"],
  "observations": "Summary",
  "need_more_validation": false
}}
"""

    observation_response = llm.invoke(observation_prompt)
    try:
        # Clean potential markdown wrapping
        content = observation_response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        observation_data = json.loads(content.strip())
        print(f"    → Gaps: {len(observation_data.get('critical_gaps', []))} critical, {len(observation_data.get('synonym_matches', []))} matches")
    except json.JSONDecodeError as e:
        print(f"  ⚠️ JSON parsing failed: {e}")
        observation_data = {"critical_gaps": [], "observations": ""}
    except Exception as e:
        print(f"  ❌ Unexpected error in observation step: {e}")
        observation_data = {"critical_gaps": [], "observations": ""}

    # STEP 4: DECIDE - Generate final match critique
    print("  ⚖️ DECIDE: Generating match verdict...")
    decision_prompt = f"""
Generate final match critique based on complete analysis.

THINKING (Plan):
{json.dumps(thinking_data, indent=2)}

OBSERVATIONS (Findings):
{json.dumps(observation_data, indent=2)}

EVIDENCE:
- JD Analysis: {json.dumps(jd_analysis, indent=2)}
- Market Validation: {json.dumps(market_validation, indent=2)}

Focus on:
- Web-validated critical gaps
- Synonym matches to avoid false negatives
- Market-backed reasoning

OUTPUT JSON:
{{
  "match_score": 0-100,
  "missing_keywords": ["skill1"],
  "irrelevant_bullets": [{{"bullet": "text", "reason": "why"}}],
  "reasoning": "Market-validated gaps"
}}
"""

    decision_response = llm.invoke(decision_prompt)

    try:
        # Clean potential markdown wrapping
        content = decision_response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        data = json.loads(content.strip())
        summary = parse_critic_output(decision_response.content, "JD Match Critic")

        if market_validation:
            summary += "\n**Market Check:** Verified demand for " + ", ".join(market_validation.keys())

        critique = Critique(
            source="JD Match Critic",
            summary=summary,
            details=data,
            resolved=False,
            confidence=0.9,
            tool_evidence={
                "jd_analysis": jd_analysis,
                "market_validation": market_validation,
                "react_thinking": thinking_data,
                "react_observations": observation_data
            }
        )

        print(f"  ✅ Match Critic ReAct complete. Score: {data.get('match_score')}")
        return {"critique_inputs": [critique]}

    except Exception as e:
        print(f"❌ Match Critic failed: {e}")
        # Fallback to empty critique or standard prompt if needed
        return {"critique_inputs": []}

def truth_critic_node(state: AgentState):
    res = llm.invoke(truth_prompt.invoke({"resume_text": state.get("resume_text", "")}))

    try:
        # Clean potential markdown wrapping
        content = res.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        # Parse JSON to get both raw data and summary
        data = json.loads(content.strip())
        summary = parse_critic_output(content, "Truth Critic")

        # Create Critique object
        critique = Critique(
            source="Truth Critic",
            summary=summary,
            details=data,
            resolved=False,
            confidence=0.8,  # Logic-based verification has good confidence
            reasoning="Logic-based verification of claims and evidence"
        )

        return {"critique_inputs": [critique]}
    except json.JSONDecodeError as e:
        print(f"❌ Truth Critic JSON Error: {e}")
        print(f"Response content: {res.content[:200]}")
        error_critique = Critique(
            source="Truth Critic",
            summary=f"### Error\nFailed to parse JSON: {str(e)}",
            details={"error": str(e), "raw_response": res.content},
            resolved=False,
            confidence=0.0,  # Error has zero confidence
            reasoning="JSON parsing error"
        )
        return {"critique_inputs": [error_critique]}

def language_critic_node(state: AgentState):
    """
    Language Critic with Full ReAct Pattern:
    1. THINK: Identify language/style issues to check
    2. ACT: Run grammar tool
    3. OBSERVE: Analyze grammar results + style patterns
    4. DECIDE: Generate language critique
    """
    print("✍️ Language Critic: Starting ReAct analysis...")

    # STEP 1: THINK - Plan language review
    print("  💭 THINK: Planning language review...")
    thinking_prompt = f"""
You are a Chief Editor planning how to review a resume.

RESUME (excerpt):
{state.get("resume_text", "")}

Think through:
1. What grammar/style issues are common in resumes?
2. Should I focus on passive voice, buzzwords, or typos?
3. What patterns should I look for?
4. Which tool checks will be most useful?

Return JSON:
{{
  "focus_areas": ["grammar", "passive voice", "buzzwords"],
  "patterns_to_check": ["pattern1"],
  "tool_usage_plan": "Why grammar tool is needed",
  "reasoning": "My approach"
}}
"""

    thinking_response = llm.invoke(thinking_prompt)
    try:
        # Clean potential markdown wrapping
        content = thinking_response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        thinking_data = json.loads(content.strip())
        print(f"    → Focus: {', '.join(thinking_data.get('focus_areas', []))}")
    except json.JSONDecodeError as e:
        print(f"  ⚠️ JSON parsing failed: {e}")
        thinking_data = {"focus_areas": ["grammar", "style"]}
    except Exception as e:
        print(f"  ❌ Unexpected error in thinking step: {e}")
        thinking_data = {"focus_areas": ["grammar", "style"]}

    # STEP 2: ACT - Run grammar tool
    print("  🔧 ACT: Running grammar checker...")
    grammar_result = check_grammar(state.get("resume_text", ""))
    print(f"    → Tool found {grammar_result.get('issue_count', 0)} potential issues")

    # STEP 3: OBSERVE - Analyze grammar results and style
    print("  👁️ OBSERVE: Analyzing language patterns...")
    observation_prompt = f"""
Analyze the grammar tool results and resume text for style issues.

GRAMMAR TOOL OUTPUT:
{json.dumps(grammar_result, indent=2)}

RESUME TEXT:
{state.get("resume_text", "")}

ORIGINAL PLAN:
{json.dumps(thinking_data, indent=2)}

Observations:
1. Are the grammar tool findings valid or false positives?
2. Do I see passive voice patterns (e.g., "was responsible for")?
3. Are there buzzwords or weak verbs?
4. What's the overall writing quality?

Return JSON:
{{
  "valid_grammar_issues": ["issue1"],
  "false_positives": ["fp1"],
  "passive_voice_count": 3,
  "buzzwords_found": ["synergy"],
  "observations": "Summary",
  "confidence_in_findings": 0.9
}}
"""

    observation_response = llm.invoke(observation_prompt)
    try:
        # Clean potential markdown wrapping
        content = observation_response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        observation_data = json.loads(content.strip())
        print(f"    → Valid issues: {len(observation_data.get('valid_grammar_issues', []))}, Passive voice: {observation_data.get('passive_voice_count', 0)}")
    except json.JSONDecodeError as e:
        print(f"  ⚠️ JSON parsing failed: {e}")
        observation_data = {"valid_grammar_issues": [], "confidence_in_findings": 0.5}
    except Exception as e:
        print(f"  ❌ Unexpected error in observation step: {e}")
        observation_data = {"valid_grammar_issues": [], "confidence_in_findings": 0.5}

    # STEP 4: DECIDE - Generate final language critique
    print("  ⚖️ DECIDE: Generating language verdict...")
    decision_prompt = f"""
Generate final language critique based on analysis.

THINKING (Plan):
{json.dumps(thinking_data, indent=2)}

OBSERVATIONS (Findings):
{json.dumps(observation_data, indent=2)}

EVIDENCE:
- Grammar Tool: {json.dumps(grammar_result, indent=2)}

Focus on:
- Valid grammar issues (not false positives)
- Passive voice replacements
- Buzzword elimination
- Specific rewrites

OUTPUT JSON:
{{
  "score": 0-100,
  "grammar_issues": ["Specific typo or rule"],
  "style_issues": ["Passive voice usage"],
  "better_rewrites": [{{"original": "...", "fix": "..."}}]
}}
"""

    decision_response = llm.invoke(decision_prompt)

    try:
        # Clean potential markdown wrapping
        content = decision_response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        data = json.loads(content.strip())

        # Use observation confidence if available
        confidence = observation_data.get("confidence_in_findings",
                                         0.95 if grammar_result.get('issue_count', 0) > 0 else 0.8)

        summary = f"### ✍️ Language Report (Score: {data.get('score')})\n"
        if data.get("grammar_issues"):
            summary += f"- **Grammar:** {len(data['grammar_issues'])} issues detected.\n"
        for item in data.get("better_rewrites", [])[:3]:
            summary += f"- *Change:* '{item['original']}' → '{item['fix']}'\n"

        critique = Critique(
            source="Language Critic",
            summary=summary,
            details=data,
            resolved=False,
            confidence=confidence,
            tool_evidence={
                "grammar_check": grammar_result,
                "react_thinking": thinking_data,
                "react_observations": observation_data
            }
        )

        print(f"  ✅ Grammer And Language Critic ReAct complete. Score: {data.get('score')}")
        return {"critique_inputs": [critique]}

    except Exception as e:
        print(f"❌ Language Critic ReAct failed: {e}")
        return {"critique_inputs": []}

def impact_critic_node(state: AgentState):
    # This one has a slightly different JSON structure, so we parse it custom or adapt the helper
    res = llm.invoke(impact_prompt.invoke({"resume_text": state.get("resume_text", "")}))

    # Custom parsing for the complex Impact JSON
    try:
        # Clean potential markdown wrapping
        content = res.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        data = json.loads(content.strip())
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
            resolved=False,
            confidence=0.85,  # Metric-focused analysis has high confidence
            reasoning="Metric-focused analysis of achievements and quantifiable impact"
        )

        return {"critique_inputs": [critique]}
    except Exception as e:
        # Error handling - return error critique
        error_critique = Critique(
            source="Impact Critic",
            summary=f"### Error parsing Impact Report\n{str(e)}",
            details={"error": str(e)},
            resolved=False,
            confidence=0.0,  # Error has zero confidence
            reasoning="JSON parsing error"
        )
        return {"critique_inputs": [error_critique]}