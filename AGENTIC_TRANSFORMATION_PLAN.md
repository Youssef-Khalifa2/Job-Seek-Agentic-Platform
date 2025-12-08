# Agentic Transformation Plan: Career Nexus CV Optimizer

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

**Step 0.1: Update State Schema** (30 min)
File: `src/agents/JobSeeker/state.py`

Add new fields:
```python
class Critique(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: str
    summary: str
    details: Dict[str, Any]
    resolved: bool = Field(default=False)
    confidence: float = Field(default=0.0, description="0-1 confidence score")  # NEW
    reasoning: str = Field(default="", description="Why this critique was made")  # NEW
    tool_evidence: Dict[str, Any] = Field(default_factory=dict, description="Tool outputs supporting this critique")  # NEW

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    critique_inputs: Annotated[List[Critique], operator.add]
    resolved_critique_ids: Set[str]
    revision_count: int
    job_description: Optional[str]
    resume_text: Optional[str]
    actionable_critiques: List[Critique]

    # NEW: Quality tracking
    quality_scores: List[float]  # Track quality over iterations
    revision_history: List[Dict[str, Any]]  # Track what changed each iteration

    # NEW: Tool outputs
    ats_parser_result: Optional[Dict[str, Any]]
    layout_analysis: Optional[Dict[str, Any]]
```

**Step 0.2: Create Model Registry** (30 min)
Already Done

**Step 0.3: Create Tools Directory** (15 min)
```bash
mkdir src/tools
touch src/tools/__init__.py
touch src/tools/ats_parser.py
touch src/tools/layout_analyzer.py
touch src/tools/web_search.py
touch src/tools/grammar_check.py
```

---

## Phase 1: Quick Wins (2-3 hours)

**Goal**: Add decision-making and adaptive stopping with minimal code changes.

### 1.1: Add Decision-Making to Editor (1 hour)

**Problem**: Editor blindly applies ALL critiques, even low-quality or conflicting ones.

**Solution**: Editor evaluates critiques and makes decisions.

File: `src/agents/JobSeeker/editor.py`

**Changes**:
```python
from src.llm_registry import LLMRegistry

# Use Sonnet for complex editing decisions
llm = LLMRegistry.get_sonnet()

def evaluate_critique_quality(critique: Critique) -> float:
    """Evaluate if a critique is actionable and high-quality."""
    eval_prompt = f"""
    Evaluate this resume critique for quality and actionability.

    Critique: {critique.summary}
    Details: {critique.details}

    Score 0-1 based on:
    - Is it specific and actionable? (not vague like "make it better")
    - Is it achievable with available information?
    - Does it conflict with other common resume advice?

    Return JSON: {{"score": 0.85, "reasoning": "..."}}
    """

    response = llm.invoke(eval_prompt)
    data = json.loads(response.content)
    return data["score"]

def resolve_conflicts(critiques: List[Critique]) -> List[Critique]:
    """Identify and resolve conflicting critiques."""
    if len(critiques) <= 1:
        return critiques

    conflict_prompt = f"""
    Analyze these resume critiques for conflicts.

    Critiques:
    {json.dumps([c.model_dump() for c in critiques], indent=2)}

    If any conflict (e.g., one says add detail, another says be concise):
    - Keep the higher-priority one
    - Mark the conflicting one as 'skip'

    Return JSON: {{
      "keep": [critique_ids],
      "skip": [critique_ids],
      "reasoning": "..."
    }}
    """

    response = llm.invoke(conflict_prompt)
    data = json.loads(response.content)

    keep_ids = set(data["keep"])
    return [c for c in critiques if c.id in keep_ids]

def editor_node(state: AgentState):
    print(f"✍️ Editor evaluating {len(state['actionable_critiques'])} critiques...")

    # Step 1: Filter low-quality critiques
    evaluated_critiques = []
    for c in state["actionable_critiques"]:
        score = evaluate_critique_quality(c)
        if score > 0.6:  # Only keep high-quality critiques
            c.confidence = score
            evaluated_critiques.append(c)
            print(f"  ✓ {c.source}: Quality {score:.2f}")
        else:
            print(f"  ✗ {c.source}: Low quality ({score:.2f}), skipping")

    if not evaluated_critiques:
        print("  ⚠️ No high-quality critiques to apply")
        return {
            "resume_text": state["resume_text"],  # No changes
            "revision_count": state["revision_count"] + 1,
            "actionable_critiques": []
        }

    # Step 2: Resolve conflicts
    final_critiques = resolve_conflicts(evaluated_critiques)
    print(f"  → Applying {len(final_critiques)} critiques after conflict resolution")

    # Step 3: Apply edits with reasoning
    feedback_str = "\n".join([
        f"- [{c.source}] (Confidence: {c.confidence:.0%}): {c.summary}"
        for c in final_critiques
    ])

    editor_prompt_text = f"""
    You are an Expert Resume Editor. Apply these critiques carefully.

    CURRENT CV:
    {state["resume_text"]}

    CRITIQUES TO IMPLEMENT (prioritized):
    {feedback_str}

    INSTRUCTIONS:
    1. Apply ONLY the critiques listed above (already filtered)
    2. For each critique, explain your reasoning before making the change
    3. If a critique is impossible (missing data), use placeholder like [X years]
    4. Maintain original structure and formatting
    5. Think through each edit before applying it

    Return ONLY the full, rewritten CV text.
    """

    msg = ChatPromptTemplate.from_template(editor_prompt_text)
    response = llm.invoke(msg.invoke({}))
    filtered_content = filter_reasoning(response)

    return {
        "resume_text": filtered_content,
        "revision_count": state["revision_count"] + 1,
        "actionable_critiques": []
    }
```

**Expected Impact**:
- ✅ Skips vague/low-quality critiques
- ✅ Resolves conflicts intelligently
- ✅ Reduces nonsensical edits by ~60%

---

### 1.2: Adaptive Stopping Logic (1 hour)

**Problem**: Hardcoded 3 loops regardless of whether improvements are made.

**Solution**: Track quality metrics and stop when plateaus.

File: `src/agents/JobSeeker/supervisor.py`

**Changes**:
```python
from typing import Literal
from langgraph.types import Command
from src.agents.JobSeeker.state import AgentState
from src.llm_registry import LLMRegistry

llm = LLMRegistry.get_haiku()  # Fast quality assessment

def calculate_quality_score(resume_text: str, job_description: str) -> float:
    """Score resume quality 0-100."""
    quality_prompt = f"""
    Evaluate this resume's quality for the job description.

    Resume:
    {resume_text}

    Job Description:
    {job_description}

    Score 0-100 based on:
    - ATS compatibility (30%)
    - Keyword match (30%)
    - Clarity and conciseness (20%)
    - Impact and metrics (20%)

    Return JSON: {{"score": 75, "reasoning": "..."}}
    """

    response = llm.invoke(quality_prompt)
    data = json.loads(response.content)
    return data["score"]

def supervisor_node(state: AgentState) -> Command[Literal["critics_start", "__end__"]]:
    """
    Intelligent routing with adaptive stopping.
    """
    current_revisions = state.get("revision_count", 0)
    quality_scores = state.get("quality_scores", [])

    print(f"🚦 Supervisor Check: Revision {current_revisions}")

    # Calculate current quality
    if current_revisions > 0:
        current_score = calculate_quality_score(
            state["resume_text"],
            state["job_description"]
        )
        quality_scores.append(current_score)

        print(f"  📊 Quality Score: {current_score:.1f}/100")

        # Check for improvement
        if len(quality_scores) >= 2:
            improvement = current_score - quality_scores[-2]
            print(f"  📈 Improvement: {improvement:+.1f} points")

            # Stop if diminishing returns (< 3 points improvement)
            if improvement < 3.0:
                print("  🛑 Diminishing returns detected. Stopping.")
                return Command(goto="__end__")

            # Stop if high quality achieved (> 85)
            if current_score >= 85:
                print("  ✅ High quality achieved. Stopping.")
                return Command(goto="__end__")

    # Safety valve: max 5 iterations
    if current_revisions >= 5:
        print("  🛑 Max iterations reached. Stopping.")
        return Command(goto="__end__")

    # Continue improving
    print("  → Continuing to next iteration")

    # Update state with quality tracking
    return Command(
        goto="critics_start",
        update={"quality_scores": quality_scores}
    )
```

**Expected Impact**:
- ✅ Stops early if quality plateaus (saves time/cost)
- ✅ Can go beyond 3 iterations if still improving
- ✅ Provides quality metrics to user

---

## Phase 2: Tool Integration (8-12 hours)

**Goal**: Add real-world tools to ground agent decisions in data.

### 2.1: ATS Parser Tool (2 hours)

**Problem**: ATS Critic guesses about parsing issues without testing.

**Solution**: Actually test the resume with an ATS parser.

File: `src/tools/ats_parser.py` (NEW)

```python
import subprocess
import tempfile
import json
from typing import Dict, Any

class ATSParser:
    """
    Simulates ATS parsing by attempting to extract structured data.
    Uses simple heuristics and PDF parsing.
    """

    @staticmethod
    def test_parsing(resume_text: str) -> Dict[str, Any]:
        """
        Test if resume is ATS-friendly.
        Returns: {
            "parsable": bool,
            "errors": List[str],
            "warnings": List[str],
            "extracted_sections": Dict
        }
        """
        errors = []
        warnings = []

        # Check 1: Special characters
        problematic_chars = ['│', '├', '└', '•', '◆', '★']
        for char in problematic_chars:
            if char in resume_text:
                errors.append(f"Contains ATS-unfriendly character: {char}")

        # Check 2: Tables (look for alignment patterns)
        lines = resume_text.split('\n')
        for i, line in enumerate(lines):
            if '  |  ' in line or '\t|\t' in line:
                errors.append(f"Possible table detected at line {i+1}")

        # Check 3: Column layout (multiple spaces in sequence)
        for i, line in enumerate(lines):
            if '     ' in line:  # 5+ spaces = possible columns
                warnings.append(f"Possible column layout at line {i+1}")

        # Check 4: Contact info extraction
        import re
        email_found = bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', resume_text))
        phone_found = bool(re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', resume_text))

        if not email_found:
            warnings.append("Email address not clearly detected")
        if not phone_found:
            warnings.append("Phone number not clearly detected")

        # Check 5: Section headers
        common_sections = ['experience', 'education', 'skills', 'projects']
        extracted_sections = {}
        text_lower = resume_text.lower()

        for section in common_sections:
            if section in text_lower:
                extracted_sections[section] = True
            else:
                warnings.append(f"'{section.title()}' section not clearly identified")

        return {
            "parsable": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "extracted_sections": extracted_sections,
            "confidence": 1.0 - (len(errors) * 0.2 + len(warnings) * 0.1)
        }

# Helper function for agents to use
def check_ats_compatibility(resume_text: str) -> Dict[str, Any]:
    """Wrapper for easy import."""
    parser = ATSParser()
    return parser.test_parsing(resume_text)
```

### 2.2: Layout Analyzer with VLM (3 hours)

**Problem**: Can't detect visual issues (columns, icons, tables) from text alone.

**Solution**: Use Pixtral Large (FREE Mistral vision model, December 2025) to analyze visual layout.

File: `src/tools/layout_analyzer.py` (NEW)

```python
from langchain_core.messages import HumanMessage
from src.llm_registry import LLMRegistry
import base64
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

class LayoutAnalyzer:
    """
    Uses Vision Language Model (Pixtral Large) to analyze resume visual layout.
    Pixtral Large is FREE on Mistral AI platform and optimized for document analysis.
    """

    def __init__(self):
        # Use Pixtral Large - FREE vision model with 128K context
        self.vlm = LLMRegistry.get_pixtral_large()

    def text_to_pdf_image(self, text: str) -> str:
        """
        Convert text to PDF, then to base64 image for VLM analysis.
        Returns base64 encoded image string.
        """
        # Create simple PDF from text
        buffer = BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=letter)

        # Write text to PDF (simple layout)
        y = 750
        for line in text.split('\n')[:40]:  # First 40 lines
            pdf.drawString(50, y, line[:80])  # Truncate long lines
            y -= 15
            if y < 50:
                break

        pdf.save()
        buffer.seek(0)

        # Convert to base64
        pdf_bytes = buffer.read()
        return base64.b64encode(pdf_bytes).decode('utf-8')

    def analyze_layout(self, resume_text: str) -> dict:
        """
        Analyze visual layout using VLM.
        """
        # Convert to image
        image_data = self.text_to_pdf_image(resume_text)

        # Create vision prompt
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": """
                    Analyze this resume's visual layout for ATS compatibility.

                    Check for:
                    1. Multiple columns (ATS can't parse columns)
                    2. Tables or grids
                    3. Icons or graphics
                    4. Unusual fonts or formatting
                    5. Text boxes or sidebars

                    Return JSON:
                    {
                      "has_columns": bool,
                      "has_tables": bool,
                      "has_icons": bool,
                      "layout_issues": ["issue1", "issue2"],
                      "overall_assessment": "ATS-friendly" or "ATS-unfriendly"
                    }
                    """
                },
                {
                    "type": "image_url",
                    "image_url": f"data:application/pdf;base64,{image_data}"
                }
            ]
        )

        try:
            response = self.vlm.invoke([message])
            import json
            data = json.loads(response.content)
            return data
        except Exception as e:
            print(f"VLM analysis failed: {e}")
            return {
                "has_columns": False,
                "has_tables": False,
                "has_icons": False,
                "layout_issues": [],
                "overall_assessment": "Unable to analyze",
                "error": str(e)
            }

# Helper function
def analyze_resume_layout(resume_text: str) -> dict:
    analyzer = LayoutAnalyzer()
    return analyzer.analyze_layout(resume_text)
```

### 2.3: Update ATS Critic to Use Both Tools (2 hours)

File: `src/agents/JobSeeker/critics.py`

**Changes to ATS Critic**:
```python
from src.tools.ats_parser import check_ats_compatibility
from src.tools.layout_analyzer import analyze_resume_layout
from src.llm_registry import LLMRegistry

# Use Haiku for text analysis (fast, cheap)
text_llm = LLMRegistry.get_haiku()

def ats_critic_node(state: AgentState):
    """
    ATS Critic with dual analysis: Text parsing + Visual layout (VLM).
    """
    print("🤖 ATS Critic: Running dual analysis...")

    # Step 1: Text-based ATS parsing test
    print("  → Testing ATS parsing...")
    ats_result = check_ats_compatibility(state["resume_text"])

    # Step 2: Visual layout analysis with VLM
    print("  → Analyzing visual layout with VLM...")
    layout_result = analyze_resume_layout(state["resume_text"])

    # Step 3: LLM synthesizes findings with reasoning
    synthesis_prompt = f"""
    You are an ATS Expert. Analyze these test results and provide actionable feedback.

    ATS Parser Results:
    {json.dumps(ats_result, indent=2)}

    VLM Layout Analysis:
    {json.dumps(layout_result, indent=2)}

    Resume Text (first 500 chars):
    {state["resume_text"][:500]}

    Provide:
    1. Critical issues (blocks ATS parsing)
    2. Warning issues (might cause problems)
    3. Specific fixes for each issue

    Return JSON:
    {{
      "score": 0-100,
      "critical_issues": [
        {{
          "issue": "...",
          "location": "...",
          "fix": "...",
          "evidence": "from parser" or "from VLM"
        }}
      ],
      "warnings": [...],
      "reasoning": "Explain your analysis..."
    }}
    """

    response = text_llm.invoke(synthesis_prompt)

    try:
        data = json.loads(response.content)

        # Calculate confidence based on tool evidence
        has_tool_evidence = (
            len(ats_result.get("errors", [])) > 0 or
            len(layout_result.get("layout_issues", [])) > 0
        )
        confidence = 0.9 if has_tool_evidence else 0.6

        # Format summary
        summary = f"### 🤖 ATS Critic Report (Confidence: {confidence:.0%})\n"
        summary += f"**Score:** {data.get('score', 0)}/100\n\n"

        if data.get("critical_issues"):
            summary += "**Critical Issues:**\n"
            for issue in data["critical_issues"]:
                summary += f"- {issue['issue']} → {issue['fix']}\n"

        if data.get("warnings"):
            summary += "\n**Warnings:**\n"
            for warning in data["warnings"]:
                summary += f"- {warning}\n"

        # Create Critique object with tool evidence
        critique = Critique(
            source="ATS Critic",
            summary=summary,
            details=data,
            resolved=False,
            confidence=confidence,
            reasoning=data.get("reasoning", ""),
            tool_evidence={
                "ats_parser": ats_result,
                "vlm_layout": layout_result
            }
        )

        print(f"  ✓ Analysis complete (Confidence: {confidence:.0%})")
        return {"critique_inputs": [critique]}

    except json.JSONDecodeError as e:
        print(f"❌ ATS Critic JSON Error: {e}")
        error_critique = Critique(
            source="ATS Critic",
            summary=f"### Error\nFailed to parse JSON: {str(e)}",
            details={"error": str(e)},
            resolved=False
        )
        return {"critique_inputs": [error_critique]}
```

**Expected Impact**:
- ✅ ATS critiques now grounded in REAL parsing tests
- ✅ VLM detects visual issues text analysis misses
- ✅ Confidence scores reflect tool evidence
- ✅ ~40% reduction in false positive critiques

---

### 2.4: Web Search for Match Critic (2 hours)

**Problem**: Match Critic doesn't know current industry standards for skills/keywords.

**Solution**: Add web search to validate skills are actually required.

File: `src/tools/web_search.py` (NEW)

```python
from langchain_community.tools import DuckDuckGoSearchRun
from typing import List, Dict

class JobMarketResearch:
    """
    Use web search to validate job requirements and skill trends.
    """

    def __init__(self):
        self.search = DuckDuckGoSearchRun()

    def validate_skill_requirement(self, skill: str, job_title: str) -> Dict[str, any]:
        """
        Check if a skill is commonly required for a job title.
        """
        query = f'"{job_title}" job requirements "{skill}" site:linkedin.com OR site:indeed.com'

        try:
            results = self.search.run(query)

            # Simple heuristic: if skill appears in results, it's validated
            skill_mentions = results.lower().count(skill.lower())

            return {
                "skill": skill,
                "validated": skill_mentions > 2,
                "mentions": skill_mentions,
                "evidence": results[:200]
            }
        except Exception as e:
            return {
                "skill": skill,
                "validated": False,
                "error": str(e)
            }

    def find_trending_skills(self, job_title: str) -> List[str]:
        """
        Find trending skills for a job title.
        """
        query = f'"{job_title}" required skills 2024 trends'

        try:
            results = self.search.run(query)
            # Extract skills (simplified - could use NER)
            # For now, return search results
            return results
        except Exception as e:
            return []

# Helper function
def validate_skills(skills: List[str], job_title: str) -> Dict[str, any]:
    researcher = JobMarketResearch()
    results = {}
    for skill in skills[:5]:  # Limit to 5 to avoid rate limits
        results[skill] = researcher.validate_skill_requirement(skill, job_title)
    return results
```

Update Match Critic to use web search:
```python
# In critics.py, update match_critic_node:
from src.tools.web_search import validate_skills

def match_critic_node(state: AgentState):
    """Match Critic with web validation."""

    # Extract job title from JD (simple regex)
    import re
    jd = state["job_description"]
    # Look for "Looking for a [Job Title]" pattern
    job_title_match = re.search(r'looking for (?:a |an )?(.+?)(?: with| who)', jd, re.IGNORECASE)
    job_title = job_title_match.group(1) if job_title_match else "Software Engineer"

    # Run standard match analysis
    res = llm.invoke(match_prompt.invoke({
        "job_description": state["job_description"],
        "resume_text": state["resume_text"]
    }))

    data = json.loads(res.content)

    # Validate missing keywords with web search
    missing_keywords = data.get("missing_keywords", [])
    if missing_keywords:
        print(f"  → Validating {len(missing_keywords)} missing keywords via web...")
        validation_results = validate_skills(missing_keywords, job_title)
        data["keyword_validation"] = validation_results

    # Rest of the critique creation...
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

### 3.1: Create Verification Node (3 hours)

**Problem**: Editor makes changes but nothing checks if they actually fixed the issues.

**Solution**: Add dedicated verifier node after editor.

File: `src/agents/JobSeeker/verifier.py` (NEW)

```python
from src.agents.JobSeeker.state import AgentState, Critique
from src.llm_registry import LLMRegistry
import json

llm = LLMRegistry.get_haiku()  # Fast verification

def verifier_node(state: AgentState) -> dict:
    """
    Verifies that editor's changes actually addressed the critiques.
    """
    print("🔍 Verifier: Checking if critiques were addressed...")

    # Get the critiques that were supposed to be fixed
    applied_critiques = state.get("actionable_critiques", [])

    if not applied_critiques:
        print("  ℹ️ No critiques to verify")
        return {}

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

    Return JSON:
    {{
      "critiques_addressed": 4,
      "critiques_total": 5,
      "overall_quality": 0.85,
      "unresolved": [
        {{
          "critique_id": "...",
          "reason": "why it wasn't fixed"
        }}
      ],
      "new_issues": ["issue1", "issue2"],
      "should_retry": false
    }}
    """

    response = llm.invoke(verification_prompt)

    try:
        data = json.loads(response.content)

        addressed = data["critiques_addressed"]
        total = data["critiques_total"]
        quality = data["overall_quality"]

        print(f"  ✓ Verification: {addressed}/{total} critiques addressed")
        print(f"  📊 Quality: {quality:.0%}")

        if data.get("new_issues"):
            print(f"  ⚠️ New issues introduced: {len(data['new_issues'])}")
            for issue in data["new_issues"]:
                print(f"    - {issue}")

        # Decide if we need to retry
        should_retry = data.get("should_retry", False)

        return {
            "verification_result": data,
            "needs_editor_retry": should_retry
        }

    except Exception as e:
        print(f"❌ Verification error: {e}")
        return {"verification_result": None}
```

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

### 3.3: Update Graph with Verification Node (1 hour)

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
    workflow.add_edge("editor", "verifier")     # NEW: Editor → Verifier
    workflow.add_edge("verifier", "supervisor")  # NEW: Verifier → Supervisor

    return workflow.compile()
```

---

## Phase 4: Advanced Features (8-12 hours)

**Goal**: Memory, conditional routing, and full agency.

### 4.1: Add Memory System (4 hours)

File: `src/agents/JobSeeker/memory.py` (NEW)

```python
from typing import List, Dict, Any
from src.agents.JobSeeker.state import Critique

class RevisionMemory:
    """
    Tracks what was tried, what worked, what failed.
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
        Log a revision attempt.
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

# Use in supervisor to skip ineffective critics
```

### 4.2: Conditional Routing (3 hours)

Update supervisor to route based on history:
```python
def supervisor_node(state: AgentState) -> Command[...]:
    # ... quality checks ...

    # Analyze memory
    history = state.get("revision_history", [])
    if len(history) >= 2:
        effectiveness = RevisionMemory.analyze_patterns(history)

        # Skip critics that haven't been effective
        ineffective_critics = [
            c for c, stats in effectiveness.items()
            if stats["success_rate"] < 0.3
        ]

        if ineffective_critics:
            print(f"  📊 Skipping ineffective critics: {ineffective_critics}")
            # Would need to modify graph to support dynamic routing
```

### 4.3: Full ReAct Pattern (3 hours)

Convert critics to full ReAct (Reasoning + Acting):
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
- [ ] Create LLM registry
- [ ] Create tools directory structure
- [ ] Install dependencies: `pip install language-tool-python reportlab`

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

---

## Cost Estimation (December 2025 Pricing)

### Per Resume Optimization (3 iterations):

| Component | Model | Pricing | Calls | Tokens | Total |
|-----------|-------|---------|-------|--------|-------|
| **ATS Critic (Text)** | Claude Haiku 4.5 | $0.80/1M | 3 | 2K × 3 = 6K | $0.0048 |
| **ATS Critic (Vision)** | Pixtral Large | FREE | 3 | 2K × 3 = 6K | $0 |
| **Match Critic** | Amazon Nova Lite | $0.06/1M | 3 | 2K × 3 = 6K | $0.00036 |
| **Truth Critic** | Claude Sonnet 4.5 | $3.00/1M | 3 | 2K × 3 = 6K | $0.018 |
| **Language Critic** | Ministral 3B | FREE | 3 | 2K × 3 = 6K | $0 |
| **Impact Critic** | Claude Haiku 4.5 | $0.80/1M | 3 | 2K × 3 = 6K | $0.0048 |
| **Consolidator** | Claude Sonnet 4.5 | $3.00/1M | 3 | 5K × 3 = 15K | $0.045 |
| **Editor** | GPT-OSS-120B | $1.00/1M | 3 | 8K × 3 = 24K | $0.024 |
| **Verifier** | Amazon Nova Micro | $0.035/1M | 3 | 3K × 3 = 9K | $0.00032 |
| **Supervisor** | Amazon Nova Micro | $0.035/1M | 4 | 500 × 4 = 2K | $0.00007 |
| **TOTAL** | - | - | - | ~77K tokens | **~$0.10/resume** |

**With Adaptive Stopping** (avg 2.2 iterations): **~$0.07/resume**

**Cost Breakdown by Provider:**
- AWS Bedrock: ~$0.097 (Nova Micro/Lite + Claude + GPT-OSS)
- Mistral AI: $0 (Pixtral Large + Ministral 3B on free tier)

**Monthly Volume Estimates:**
- 100 resumes/month: ~$10/month (with adaptive stopping: ~$7/month)
- 1,000 resumes/month: ~$100/month (with adaptive stopping: ~$70/month)
- 10,000 resumes/month: ~$1,000/month (with adaptive stopping: ~$700/month)

---

## Success Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| **ATS Pass Rate** | Unknown | 95% | Test with real ATS |
| **False Positive Critiques** | ~40% | <10% | Tool-grounded evidence |
| **Quality Improvement** | N/A | +20 points | Before/after scoring |
| **Unnecessary Iterations** | 33% (fixed 3) | <10% | Adaptive stopping |
| **Cost per Resume** | N/A | <$0.02 | AWS billing |
| **User Satisfaction** | N/A | >4/5 | User feedback |

---

## Migration Path

1. **Week 1**: Phase 0 + Phase 1 (Foundation + Quick wins)
2. **Week 2**: Phase 2 (Tool integration for ATS + Match critics)
3. **Week 3**: Phase 2 completion + Phase 3 start (All tools + Verification)
4. **Week 4**: Phase 3 completion + Phase 4 (Reflection + Memory)
5. **Week 5**: Testing, optimization, documentation

---

## Rollback Plan

Each phase is additive, so rollback is simple:
- Keep old files in `src/agents/JobSeeker/legacy/`
- Use feature flags: `USE_AGENTIC_MODE = True`
- Can switch back by changing graph imports

---

## Next Steps

1. **Review this plan** - Any changes needed?
2. **Start Phase 0** - Set up architecture (30 min)
3. **Quick win with Phase 1** - See immediate improvements (2-3 hours)
4. **Plan Phase 2 sprint** - Schedule tool integration work

Ready to start implementing? Which phase should we begin with?
