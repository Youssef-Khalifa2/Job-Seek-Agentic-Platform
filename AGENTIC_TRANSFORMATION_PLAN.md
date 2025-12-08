# 
: Career Nexus CV Optimizer

## Executive Summary

Transform the current pipeline-based CV optimization system into a truly agentic multi-agent system with:
- **Tool usage**: ATS parsers, VLMs, web search, grammar checkers
- **Verification loops**: Agents verify their work actually improved quality
- **Decision-making**: Agents choose actions based on reasoning, not templates
- **Reflection**: Agents evaluate and self-correct
- **Adaptive behavior**: System learns and adapts based on results

## Current State vs. Target State

| Aspect | Current (Pipeline) | Target (Agentic) |
|--------|-------------------|------------------|
| **Critics** | Template prompts → JSON | ReAct pattern + Tools + VLM |
| **Editor** | Blindly applies all critiques | Evaluates quality, makes decisions |
| **Verification** | None | Dedicated verifier node |
| **Stopping** | Hardcoded 3 loops | Adaptive based on quality metrics |
| **Tools** | None | ATS parser, VLM, web search, grammar check |
| **Memory** | No history | Tracks what worked/failed |
| **Reflection** | None | Self-evaluation after each action |

---

## Phase 0: Model Selection & Architecture Setup

**Goal**: Choose the right LLM for each task based on capabilities and cost.

### Model Selection Strategy

#### Available Resources (December 2024 - January 2025):

**1. AWS Bedrock - Latest Models:**

**Anthropic Claude (Newest):**
- **Claude Opus 4.5**: `anthropic.claude-opus-4-5-20251101-v1:0` | 200K context | Vision + Text | $5/$25 per 1M tokens
- **Claude Sonnet 4.5**: `anthropic.claude-sonnet-4-5-20250929-v1:0` | 200K context | Vision + Text | $3/$15 per 1M tokens
- **Claude Haiku 4.5**: `anthropic.claude-haiku-4-5-20251001-v1:0` | 200K context | Vision + Text | Lower cost, AWS latency-optimized

**Amazon Nova (2024 - Cost-Optimized):**
- **Nova Micro**: `amazon.nova-micro-v1:0` | Text only | $0.000035/$0.00014 per 1K tokens | Ultra-cheap
- **Nova Lite**: `amazon.nova-lite-v1:0` | Text + Image + Video | 300K context | 75% cheaper than competitors
- **Nova Pro**: `amazon.nova-pro-v1:0` | Text + Image + Video | 300K context | Balanced performance
- **Nova Premier** (Q1 2025): High-end reasoning

**OpenAI (Open Weight via Bedrock):**
- **gpt-oss-120b**: `openai.gpt-oss-120b-1:0` | 120B params (5.1B active) | 128K context | **Currently used in editor**
- **gpt-oss-20b**: `openai.gpt-oss-20b-1:0` | 20B params | Lower cost alternative

**Meta Llama (Latest):**
- **Llama 4 Maverick 17B**: `meta.llama4-maverick-17b-instruct-v1:0` | Multimodal, latest generation
- **Llama 3.3 70B**: `meta.llama3-3-70b-instruct-v1:0` | General purpose
- **Llama 3.2 90B**: `meta.llama3-2-90b-instruct-v1:0` | Multimodal
- **Llama 3.1 405B**: AWS latency-optimized

**Mistral on Bedrock:**
- **Pixtral Large**: `mistral.pixtral-large-2502-v1:0` | 124B | Vision + Text | 128K context
- **Mistral Large 3**: `mistral.mistral-large-3-675b-instruct` | 41B active/675B total | Vision | 80% cheaper than GPT-4o

**Specialized Models:**
- **Qwen3 Coder 480B**: Best-in-class code generation
- **Cohere Rerank 3.5**: Improved RAG accuracy
- **DeepSeek R1**: Advanced reasoning

**2. Mistral AI - La Plateforme (Free Tier Available):**

**Flagship Models:**
- **Mistral Large 3**: `mistral-large-2512` | 41B active/675B total | 256K context | Text + Vision | **Currently used: mistral-small-latest**
- **Mistral Medium 3.1**: `mistral-medium-2508` | ~32B | 128K context | Text + Vision
- **Mistral Small 3.2**: `mistral-small-2506` | ~13B | 128K context | Text + Vision

**Efficient Small Models (Ministral 3 Family - December 2024):**
- **Ministral 14B**: `ministral-14b-2512` | 256K context | Text + Vision | Edge-optimized
- **Ministral 8B**: `ministral-8b-2512` | 256K context | Text + Vision | 4GB VRAM
- **Ministral 3B**: `ministral-3b-2512` | 256K context | Ultra-lightweight

**Vision Models:**
- **Pixtral Large**: `pixtral-large-2411` | 124B | Surpasses GPT-4o on ChartQA/DocVQA
- **Pixtral 12B**: `pixtral-12b-2409` | 12B | Apache 2.0, Free tier | **Recommended for ATS layout analysis**

**Specialized:**
- **Codestral 2508**: Code generation
- **Magistral Medium/Small 1.2**: Advanced reasoning
- **Mistral OCR**: Document processing
- **Voxtral Small/Mini**: Audio processing

#### Recommended Model Assignments:

| Agent/Task | Model | Model ID | Input Cost | Rationale |
|------------|-------|----------|------------|-----------|
| **ATS Critic (Text)** | Claude Haiku 4.5 | `us.anthropic.claude-3-5-haiku-20241022-v1:0` | $0.80/1M | Fast structured output, 200K context, cost-effective |
| **ATS Critic (Vision)** | Pixtral Large | `pixtral-large-2411` | FREE | Vision model to analyze CV layout, table detection |
| **Match Critic** | Amazon Nova Lite | `us.amazon.nova-lite-v1:0` | $0.06/1M | Extremely cheap semantic matching, 300K context |
| **Truth Critic** | Claude Sonnet 4.5 | `us.anthropic.claude-3-5-sonnet-20241022-v2:0` | $3.00/1M | Extended thinking for claim verification |
| **Language Critic** | Ministral 3B | `ministral-3b-2410` | FREE | Fast grammar/style checks, JSON mode |
| **Impact Critic** | Claude Haiku 4.5 | `us.anthropic.claude-3-5-haiku-20241022-v1:0` | $0.80/1M | Structured STAR rewrites, fast iteration |
| **Consolidator** | Claude Sonnet 4.5 | `us.anthropic.claude-3-5-sonnet-20241022-v2:0` | $3.00/1M | Needs reasoning to merge/prioritize critiques |
| **Editor** | GPT-OSS-120B (current) | `meta.gpt-oss-120b-instruct-v1` | $1.00/1M | Complex editing with strong instruction following |
| **Verifier** | Amazon Nova Micro | `us.amazon.nova-micro-v1:0` | $0.035/1M | Ultra-fast verification checks, cheapest option |
| **Supervisor** | Amazon Nova Micro | `us.amazon.nova-micro-v1:0` | $0.035/1M | Simple routing decisions, minimal reasoning |

### Implementation Steps:

Update Match Critic to use enhanced web search:
```python
# In critics.py, update match_critic_node:
from src.tools.web_search import analyze_job_description, validate_requirements

def match_critic_node(state: AgentState):
    """Match Critic with deep JD analysis and web validation."""

    print("📊 Match Critic: Analyzing job requirements...")

    # Step 1: Extract ALL requirements from JD using LLM
    jd_requirements = analyze_job_description(state["job_description"])
    job_title = jd_requirements.get("job_title", "Unknown")

    print(f"  → Extracted requirements for {job_title}")
    print(f"    - Required skills: {len(jd_requirements.get('required_skills', []))}")
    print(f"    - Experience requirements: {len(jd_requirements.get('experience_requirements', []))}")

    # Step 2: Validate requirements with web search
    print("  → Validating requirements via web search...")
    validation_results = validate_requirements(jd_requirements, job_title)

    # Step 3: Run standard match analysis
    res = llm.invoke(match_prompt.invoke({
        "job_description": state["job_description"],
        "resume_text": state["resume_text"]
    }))

    data = json.loads(res.content)

    # Step 4: Enrich with web-validated requirements
    data["jd_analysis"] = jd_requirements
    data["web_validation"] = validation_results

    # Rest of critique creation...
```

---

### 2.5: Grammar Checker for Language Critic (1 hour)

File: `src/tools/grammar_check.py` (NEW)

```python
import language_tool_python
from typing import List, Dict

class GrammarChecker:
    """
    Use LanguageTool for grammar and style checking.
    """

    def __init__(self):
        self.tool = language_tool_python.LanguageTool('en-US')

    def check(self, text: str) -> Dict[str, any]:
        """
        Check grammar, return issues.
        """
        matches = self.tool.check(text)

        issues = []
        for match in matches[:20]:  # Limit to 20 issues
            issues.append({
                "message": match.message,
                "context": match.context,
                "replacements": match.replacements[:3],
                "rule": match.ruleId
            })

        return {
            "issue_count": len(matches),
            "issues": issues,
            "corrected_text": language_tool_python.utils.correct(text, matches)
        }

def check_grammar(text: str) -> Dict[str, any]:
    checker = GrammarChecker()
    return checker.check(text)
```

**Install**:
```bash
pip install language-tool-python
```

Update Language Critic to use grammar checker (similar pattern to ATS critic).

---

## Phase 3: Verification & Reflection (6-8 hours)

**Goal**: Add feedback loops so agents verify and self-correct.

### 3.1: Create Verification Node with Conditional Routing (3 hours)

**Problem**: Editor makes changes but nothing checks if they actually fixed the issues. Need smart routing: if critiques not addressed, send back to editor for retry.

**Solution**: Add dedicated verifier node with conditional routing logic.

**First, update state schema** to track editor retries:
```python
# In src/agents/JobSeeker/state.py
class AgentState(TypedDict):
    # ... existing fields ...
    editor_retry_count: int  # NEW: Track retries per iteration
    max_editor_retries: int  # NEW: Default 2
    unresolved_critiques: List[Critique]  # NEW: Store what wasn't fixed
```

File: `src/agents/JobSeeker/verifier.py` (NEW)

```python
from src.agents.JobSeeker.state import AgentState, Critique
from src.llm_registry import LLMRegistry
from langgraph.types import Command
from typing import Literal
import json

llm = LLMRegistry.get_nova_micro()  # Ultra-fast verification

def verifier_node(state: AgentState) -> Command[Literal["editor", "supervisor"]]:
    """
    Verifies that editor's changes actually addressed the critiques.
    Routes back to editor if critiques not fixed (with max retry limit).
    """
    print("🔍 Verifier: Checking if critiques were addressed...")

    # Get the critiques that were supposed to be fixed
    applied_critiques = state.get("actionable_critiques", [])
    retry_count = state.get("editor_retry_count", 0)
    max_retries = state.get("max_editor_retries", 2)

    if not applied_critiques:
        print("  ℹ️ No critiques to verify - moving to supervisor")
        return Command(goto="supervisor")

    # Compare before/after
    verification_prompt = f"""
    You are a Quality Assurance agent. Verify that edits addressed critiques.

    CRITIQUES THAT SHOULD HAVE BEEN FIXED:
    {json.dumps([c.model_dump() for c in applied_critiques], indent=2)}

    EDITED RESUME (current version):
    {state["resume_text"]}

    For each critique, determine:
    1. Was it addressed? (yes/no)
    2. Quality of the fix (0-1)
    3. Any new issues introduced?
    4. Specific unresolved critiques with reasons

    Return JSON:
    {{
      "critiques_addressed": 4,
      "critiques_total": 5,
      "overall_quality": 0.85,
      "unresolved": [
        {{
          "critique_id": "...",
          "critique_summary": "...",
          "reason": "why it wasn't fixed",
          "guidance": "specific instruction for editor"
        }}
      ],
      "new_issues": ["issue1", "issue2"],
      "pass_verification": true/false
    }}
    """

    response = llm.invoke(verification_prompt)

    try:
        data = json.loads(response.content)

        addressed = data["critiques_addressed"]
        total = data["critiques_total"]
        quality = data["overall_quality"]
        pass_verification = data.get("pass_verification", addressed == total)

        print(f"  ✓ Verification: {addressed}/{total} critiques addressed")
        print(f"  📊 Quality: {quality:.0%}")

        if data.get("new_issues"):
            print(f"  ⚠️ New issues introduced: {len(data['new_issues'])}")
            for issue in data["new_issues"]:
                print(f"    - {issue}")

        # ROUTING DECISION
        if pass_verification:
            print("  ✅ Verification passed - moving to supervisor")
            return Command(
                goto="supervisor",
                update={
                    "verification_result": data,
                    "editor_retry_count": 0  # Reset for next iteration
                }
            )
        else:
            # Check if we can retry
            if retry_count < max_retries:
                print(f"  🔄 Verification failed - retrying editor ({retry_count + 1}/{max_retries})")

                # Create focused critiques for retry
                unresolved = []
                for unresolved_item in data.get("unresolved", []):
                    unresolved.append(Critique(
                        source="Verifier",
                        summary=unresolved_item["guidance"],
                        details=unresolved_item,
                        resolved=False,
                        confidence=0.9
                    ))

                return Command(
                    goto="editor",
                    update={
                        "actionable_critiques": unresolved,  # Send focused critiques
                        "editor_retry_count": retry_count + 1,
                        "verification_result": data
                    }
                )
            else:
                print(f"  ⚠️ Max retries ({max_retries}) reached - moving to supervisor anyway")
                return Command(
                    goto="supervisor",
                    update={
                        "verification_result": data,
                        "editor_retry_count": 0,
                        "unresolved_critiques": data.get("unresolved", [])
                    }
                )

    except Exception as e:
        print(f"❌ Verification error: {e} - defaulting to supervisor")
        return Command(
            goto="supervisor",
            update={"verification_result": {"error": str(e)}}
        )
```

**Expected Impact**:
- ✅ Verifier acts as quality gate
- ✅ Automatic retry if critiques not fixed (max 2 times)
- ✅ Focused feedback to editor on retry ("You missed X, Y, Z")
- ✅ Safety: Always moves forward after max retries

### 3.2: Add Reflection to Editor (2 hours)

File: `src/agents/JobSeeker/editor.py`

Add self-reflection after edits:
```python
def editor_node(state: AgentState):
    # ... existing evaluation and editing code ...

    # NEW: Self-reflection
    print("🤔 Editor: Reflecting on changes...")

    reflection_prompt = f"""
    Review the edits you just made.

    Original CV (excerpt):
    {state["resume_text"][:500]}

    Edited CV (excerpt):
    {filtered_content[:500]}

    Critiques you tried to address:
    {feedback_str}

    Self-evaluate:
    1. Did you successfully address each critique?
    2. Are the edits grammatically correct?
    3. Did you maintain the original tone and style?
    4. Any concerns about the changes?

    Return JSON:
    {{
      "confidence": 0.85,
      "concerns": ["concern1", "concern2"],
      "self_assessment": "good" or "needs_review"
    }}
    """

    reflection_response = llm.invoke(reflection_prompt)
    reflection_data = json.loads(reflection_response.content)

    confidence = reflection_data.get("confidence", 0.5)
    print(f"  💭 Self-confidence: {confidence:.0%}")

    if reflection_data.get("concerns"):
        print(f"  ⚠️ Editor has concerns:")
        for concern in reflection_data["concerns"]:
            print(f"    - {concern}")

    return {
        "resume_text": filtered_content,
        "revision_count": state["revision_count"] + 1,
        "actionable_critiques": [],
        "editor_reflection": reflection_data  # Store reflection
    }
```

### 3.3: Update Graph with Conditional Verification Routing (1 hour)

File: `src/agents/JobSeeker/graph.py`

```python
from src.agents.JobSeeker.verifier import verifier_node

def build_job_seeker_graph():
    workflow = StateGraph(AgentState)

    # Add nodes (existing + new verifier)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("critics_start", passthrough)
    workflow.add_node("ats_critic", ats_critic_node)
    workflow.add_node("match_critic", match_critic_node)
    workflow.add_node("truth_critic", truth_critic_node)
    workflow.add_node("language_critic", language_critic_node)
    workflow.add_node("impact_critic", impact_critic_node)
    workflow.add_node("consolidator", consolidator_node)
    workflow.add_node("editor", editor_node)
    workflow.add_node("verifier", verifier_node)  # NEW

    # Edges
    workflow.add_edge(START, "supervisor")

    # Critics fan-out/fan-in
    workflow.add_edge("critics_start", "ats_critic")
    workflow.add_edge("critics_start", "match_critic")
    workflow.add_edge("critics_start", "truth_critic")
    workflow.add_edge("critics_start", "language_critic")
    workflow.add_edge("critics_start", "impact_critic")

    workflow.add_edge("ats_critic", "consolidator")
    workflow.add_edge("match_critic", "consolidator")
    workflow.add_edge("truth_critic", "consolidator")
    workflow.add_edge("language_critic", "consolidator")
    workflow.add_edge("impact_critic", "consolidator")

    workflow.add_edge("consolidator", "editor")
    workflow.add_edge("editor", "verifier")  # NEW: Editor → Verifier

    # NOTE: Verifier uses Command routing (returns Command[Literal["editor", "supervisor"]])
    # No explicit edge needed - verifier node handles routing internally:
    #   - If verification passes → supervisor
    #   - If verification fails & retries < max → editor (retry loop)
    #   - If verification fails & retries >= max → supervisor (safety valve)

    return workflow.compile()
```

**New Flow Diagram:**
```
START → Supervisor
          ↓
       Critics (parallel) → Consolidator → Editor → Verifier
                                               ↑      ↓
                                               └──────┘ (retry loop if needed)
                                                      ↓
                                                 Supervisor → END or loop again
```

---

## Phase 4: Advanced Features (8-12 hours)

**Goal**: Memory, conditional routing, and full agency.

### 4.1: Add Memory System for Tracking (3 hours)

**IMPORTANT**: This is for **tracking and reporting only** - NO automatic action based on effectiveness.

File: `src/agents/JobSeeker/memory.py` (NEW)

```python
from typing import List, Dict, Any
from src.agents.JobSeeker.state import Critique
from datetime import datetime
import json

class RevisionMemory:
    """
    Tracks what was tried, what worked, what failed.
    Used for manual review and optimization - does NOT automatically skip critics.
    """

    @staticmethod
    def log_revision(
        iteration: int,
        critiques_applied: List[Critique],
        quality_before: float,
        quality_after: float,
        success: bool
    ) -> Dict[str, Any]:
        """
        Log a revision attempt for later analysis.
        """
        return {
            "iteration": iteration,
            "critiques": [c.source for c in critiques_applied],
            "quality_delta": quality_after - quality_before,
            "success": success,
            "timestamp": str(datetime.now())
        }

    @staticmethod
    def analyze_patterns(history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Identify patterns: which critics are most effective?
        FOR MANUAL REVIEW ONLY - not used for automatic routing.
        """
        critic_effectiveness = {}

        for entry in history:
            for critic in entry["critiques"]:
                if critic not in critic_effectiveness:
                    critic_effectiveness[critic] = {
                        "attempts": 0,
                        "successes": 0,
                        "avg_improvement": 0
                    }

                critic_effectiveness[critic]["attempts"] += 1
                if entry["success"]:
                    critic_effectiveness[critic]["successes"] += 1
                critic_effectiveness[critic]["avg_improvement"] += entry["quality_delta"]

        # Calculate averages
        for critic in critic_effectiveness:
            attempts = critic_effectiveness[critic]["attempts"]
            critic_effectiveness[critic]["avg_improvement"] /= attempts
            critic_effectiveness[critic]["success_rate"] = (
                critic_effectiveness[critic]["successes"] / attempts
            )

        return critic_effectiveness

    @staticmethod
    def generate_report(history: List[Dict[str, Any]]) -> str:
        """
        Generate markdown report for manual review.
        """
        effectiveness = RevisionMemory.analyze_patterns(history)

        report = "# Critic Effectiveness Report\n\n"
        report += f"**Total Iterations:** {len(history)}\n\n"
        report += "## Critic Performance\n\n"
        report += "| Critic | Attempts | Success Rate | Avg Improvement |\n"
        report += "|--------|----------|--------------|------------------|\n"

        for critic, stats in sorted(
            effectiveness.items(),
            key=lambda x: x[1]["success_rate"],
            reverse=True
        ):
            report += f"| {critic} | {stats['attempts']} | {stats['success_rate']:.0%} | {stats['avg_improvement']:+.2f} |\n"

        return report

# Helper to save report for manual review
def save_effectiveness_report(state: AgentState, output_path: str = "critic_effectiveness_report.md"):
    """
    Save effectiveness report to file for manual review.
    Call at the end of optimization.
    """
    history = state.get("revision_history", [])
    if history:
        report = RevisionMemory.generate_report(history)
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"📊 Effectiveness report saved to {output_path}")
```

**Usage in main loop:**
```python
# At end of optimization
save_effectiveness_report(final_state, "reports/effectiveness_report.md")
# User can review and decide if they want to disable any critics
```

### 4.2: Markdown Output + PDF Export (2 hours)

**Problem**: Need to output optimized CV in markdown format and allow PDF export.

**Solution**: Format output as markdown and add PDF rendering utility.

File: `src/tools/markdown_formatter.py` (NEW)

```python
from typing import Dict, Any
import markdown
from weasyprint import HTML
from io import BytesIO

class MarkdownFormatter:
    """
    Format resume as markdown and export to PDF.
    """

    @staticmethod
    def format_resume_as_markdown(resume_text: str) -> str:
        """
        Ensure resume text is properly formatted as markdown.
        Adds proper headers, bold text, bullet points, etc.
        """
        # Resume should already be markdown from editor
        # This function ensures consistent formatting

        lines = resume_text.split('\n')
        formatted_lines = []

        for line in lines:
            stripped = line.strip()

            # Ensure headers have proper markdown syntax
            if stripped and not stripped.startswith('#') and len(stripped) < 50:
                # Might be a section header - check if all caps or title case
                if stripped.isupper() or stripped.istitle():
                    formatted_lines.append(f"## {stripped}")
                    continue

            formatted_lines.append(line)

        return '\n'.join(formatted_lines)

    @staticmethod
    def markdown_to_pdf(markdown_text: str, output_path: str) -> str:
        """
        Convert markdown to PDF using WeasyPrint.

        Args:
            markdown_text: The markdown-formatted resume
            output_path: Where to save the PDF

        Returns:
            Path to generated PDF
        """
        # Convert markdown to HTML
        html_content = markdown.markdown(
            markdown_text,
            extensions=['extra', 'nl2br', 'sane_lists']
        )

        # Add CSS styling for professional resume look
        styled_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{
                    font-family: 'Calibri', 'Arial', sans-serif;
                    font-size: 11pt;
                    line-height: 1.4;
                    margin: 0.5in;
                    color: #333;
                }}
                h1 {{
                    font-size: 20pt;
                    margin-bottom: 0.2em;
                    color: #000;
                }}
                h2 {{
                    font-size: 14pt;
                    margin-top: 0.8em;
                    margin-bottom: 0.3em;
                    border-bottom: 1px solid #333;
                    color: #000;
                }}
                h3 {{
                    font-size: 12pt;
                    margin-top: 0.5em;
                    margin-bottom: 0.2em;
                }}
                ul {{
                    margin-top: 0.3em;
                    margin-bottom: 0.5em;
                    padding-left: 1.5em;
                }}
                li {{
                    margin-bottom: 0.2em;
                }}
                strong {{
                    color: #000;
                }}
                p {{
                    margin: 0.3em 0;
                }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """

        # Generate PDF
        HTML(string=styled_html).write_pdf(output_path)
        return output_path

# Helper functions
def export_resume_as_markdown(resume_text: str, output_path: str) -> str:
    """Save resume as markdown file."""
    formatter = MarkdownFormatter()
    formatted = formatter.format_resume_as_markdown(resume_text)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(formatted)

    return output_path

def export_resume_as_pdf(resume_text: str, output_path: str) -> str:
    """Convert resume to PDF."""
    formatter = MarkdownFormatter()
    formatted = formatter.format_resume_as_markdown(resume_text)
    return formatter.markdown_to_pdf(formatted, output_path)
```

**Install dependencies:**
```bash
pip install markdown weasyprint
```

**Update Editor to output markdown:**
```python
# In editor.py
def editor_node(state: AgentState):
    # ... existing code ...

    editor_prompt_text = f"""
    You are an Expert Resume Editor. Apply these critiques carefully.

    CURRENT CV:
    {state["resume_text"]}

    CRITIQUES TO IMPLEMENT (prioritized):
    {feedback_str}

    INSTRUCTIONS:
    1. Apply ONLY the critiques listed above
    2. Output the resume in MARKDOWN format:
       - Use ## for section headers (e.g., ## Experience)
       - Use **bold** for job titles, company names
       - Use bullet points (- ) for achievements
       - Use proper line spacing
    3. Maintain ATS-friendly structure (no tables, no columns)

    Return ONLY the full, rewritten CV in markdown format.
    """
    # ... rest of editor code
```

**Usage:**
```python
# After optimization completes
final_resume = final_state["resume_text"]

# Save markdown
export_resume_as_markdown(final_resume, "output/optimized_resume.md")

# Export PDF
export_resume_as_pdf(final_resume, "output/optimized_resume.pdf")

print("✅ Resume exported as:")
print("  - Markdown: output/optimized_resume.md")
print("  - PDF: output/optimized_resume.pdf")
```

### 4.3: Full ReAct Pattern (3 hours)

Convert critics to full ReAct (Reasoning + Acting) 
Example :
```python
def ats_critic_node_react(state: AgentState):
    """
    ATS Critic using full ReAct pattern:
    1. Think (reasoning)
    2. Act (use tools)
    3. Observe (analyze results)
    4. Repeat if needed
    """

    # Step 1: Think
    thinking = llm.invoke(f"""
    Analyze this resume for ATS compatibility. Think through:
    1. What are common ATS issues?
    2. What patterns should I look for?
    3. What tools should I use?

    Resume: {state["resume_text"][:300]}
    """)

    # Step 2: Act - Use tools
    ats_result = check_ats_compatibility(state["resume_text"])
    layout_result = analyze_resume_layout(state["resume_text"])

    # Step 3: Observe - Analyze tool outputs
    observation = llm.invoke(f"""
    I used two tools:
    1. ATS Parser: {ats_result}
    2. VLM Layout: {layout_result}

    Observations:
    - What did I learn?
    - Are there issues?
    - Do I need more information?
    """)

    # Step 4: Decide - Generate critique or use more tools
    final_critique = llm.invoke(f"""
    Based on my analysis:

    Thinking: {thinking}
    Tools used: ATS Parser, VLM
    Observations: {observation}

    Generate final critique with specific fixes.
    """)

    # ... rest of critique creation
```

---

## Implementation Checklist

### Phase 0: Setup ✓
- [ ] Update state schema with new fields
- [x] Create LLM registry
- [ ] Create tools directory structure
- [x] Install dependencies: `pip install language-tool-python reportlab`

### Phase 1: Quick Wins (2-3 hours) ✓
- [ ] Add quality evaluation to editor
- [ ] Add conflict resolution to editor
- [ ] Implement adaptive stopping in supervisor
- [ ] Test with sample resume

### Phase 2: Tool Integration (8-12 hours) ✓
- [ ] Build ATS parser tool
- [ ] Build layout analyzer with Pixtral VLM
- [ ] Update ATS critic to use both tools
- [ ] Add web search tool
- [ ] Update Match critic with web validation
- [ ] Add grammar checker tool
- [ ] Update Language critic
- [ ] Test all tools independently
- [ ] Integration test

### Phase 3: Verification (6-8 hours) ✓
- [ ] Create verifier node
- [ ] Add reflection to editor
- [ ] Update graph with verifier
- [ ] Test verification loop
- [ ] Handle retry logic

### Phase 4: Advanced (8-12 hours) ✓
- [ ] Implement memory system
- [ ] Add conditional routing
- [ ] Convert to full ReAct pattern
- [ ] Performance testing
- [ ] Cost optimization

---

## Testing Strategy

### Unit Tests:
```python
# Test ATS parser
def test_ats_parser():
    result = check_ats_compatibility("John Doe\nSoftware Engineer")
    assert "parsable" in result

# Test VLM layout analyzer
def test_layout_analyzer():
    result = analyze_resume_layout("Sample resume text")
    assert "has_columns" in result
```

### Integration Tests:
```python
# Test full flow
def test_agentic_flow():
    graph = build_job_seeker_graph()
    initial_state = {
        "resume_text": SAMPLE_RESUME,
        "job_description": SAMPLE_JOB,
        "revision_count": 0
    }
    final_state = graph.invoke(initial_state)

    # Check quality improved
    assert final_state["quality_scores"][-1] > 70

    # Check tool usage
    assert "tool_evidence" in final_state["critique_inputs"][0]
```


## Success Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| **ATS Pass Rate** | Unknown | 95% | Test with real ATS |
| **False Positive Critiques** | ~40% | <10% | Tool-grounded evidence |
| **Quality Improvement** | N/A | +20 points | Before/after scoring |
| **Unnecessary Iterations** | 33% (fixed 3) | <10% | Adaptive stopping |

---

## Migration Path

1. **Week 1**: Phase 0 + Phase 1 (Foundation + Quick wins)
2. **Week 2**: Phase 2 (Tool integration for ATS + Match critics)
3. **Week 3**: Phase 2 completion + Phase 3 start (All tools + Verification)
4. **Week 4**: Phase 3 completion + Phase 4 (Reflection + Memory)
5. **Week 5**: Testing, optimization, documentation

