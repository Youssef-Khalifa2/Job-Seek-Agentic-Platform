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
    resume_text: Optional[str]  # Extracted text for text-based analysis
    original_pdf_path: Optional[str]  # NEW: Path to original PDF for VLM analysis
    actionable_critiques: List[Critique]

    # NEW: Quality tracking
    quality_scores: List[float]  # Track quality over iterations
    revision_history: List[Dict[str, Any]]  # Track what changed each iteration

    # NEW: Tool outputs
    ats_parser_result: Optional[Dict[str, Any]]
    layout_analysis: Optional[Dict[str, Any]]

    # NEW: Verification & Retry logic
    editor_retry_count: int  # Track editor retries per iteration (default 0)
    max_editor_retries: int  # Maximum retries allowed (default 2)
    unresolved_critiques: List[Dict[str, Any]]  # Critiques that weren't fixed
    verification_result: Optional[Dict[str, Any]]  # Verifier output
```

**Update ingestion to pass original PDF path:**
```python
# In test.py or main.py
initial_state = {
    "resume_text": extract_text_from_pdf(pdf_path),  # For text analysis
    "original_pdf_path": pdf_path,  # For VLM analysis
    "job_description": job_desc,
    "revision_count": 0,
    "editor_retry_count": 0,
    "max_editor_retries": 2,
    # ... rest of fields
}
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

            # Stop if diminishing returns (< 5 points improvement)
            if improvement < 5.0:
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

**Solution**: Use Pixtral Large (FREE Mistral vision model) to analyze ORIGINAL PDF layout.

File: `src/tools/layout_analyzer.py` (NEW)

```python
from langchain_core.messages import HumanMessage
from src.llm_registry import LLMRegistry
from pdf2image import convert_from_path
import base64
from io import BytesIO
from typing import List
import json

class LayoutAnalyzer:
    """
    Uses Vision Language Model (Pixtral Large) to analyze ORIGINAL PDF layout.
    Converts the actual uploaded PDF to images to preserve all visual formatting.
    """

    def __init__(self):
        # Use Pixtral Large - FREE vision model with 128K context
        self.vlm = LLMRegistry.get_pixtral_large()

    def pdf_to_images(self, pdf_path: str) -> List[str]:
        """
        Convert ORIGINAL PDF pages to base64-encoded PNG images.
        Preserves actual layout, columns, tables, icons, etc.

        Args:
            pdf_path: Path to the original uploaded PDF file

        Returns:
            List of base64-encoded PNG images (one per page)
        """
        # Convert PDF to images (preserves actual layout)
        images = convert_from_path(
            pdf_path,
            first_page=1,
            last_page=2,  # Analyze first 2 pages
            dpi=150  # Good balance of quality vs size
        )

        base64_images = []
        for img in images:
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            base64_images.append(base64.b64encode(buffer.read()).decode('utf-8'))

        return base64_images

    def analyze_layout(self, pdf_path: str) -> dict:
        """
        Analyze ORIGINAL PDF visual layout using VLM.

        Args:
            pdf_path: Path to the ORIGINAL uploaded PDF file (from state)
        """
        try:
            # Convert original PDF to images
            images = self.pdf_to_images(pdf_path)

            # Analyze first page with VLM
            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": """
                        Analyze this resume's ACTUAL visual layout for ATS compatibility.

                        Check for:
                        1. Multiple columns (ATS can't parse columns properly)
                        2. Tables or grids (can confuse ATS parsers)
                        3. Icons, graphics, or images
                        4. Unusual fonts or decorative formatting
                        5. Text boxes, sidebars, or headers/footers with content
                        6. Color-coded sections (ATS loses color information)

                        Return JSON:
                        {
                          "has_columns": bool,
                          "has_tables": bool,
                          "has_icons": bool,
                          "has_text_boxes": bool,
                          "layout_issues": ["specific issue 1", "specific issue 2"],
                          "overall_assessment": "ATS-friendly" or "ATS-unfriendly",
                          "confidence": 0.0-1.0
                        }
                        """
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/png;base64,{images[0]}"
                    }
                ]
            )

            response = self.vlm.invoke([message])
            data = json.loads(response.content)
            return data

        except Exception as e:
            print(f"❌ VLM analysis failed: {e}")
            return {
                "has_columns": False,
                "has_tables": False,
                "has_icons": False,
                "has_text_boxes": False,
                "layout_issues": [],
                "overall_assessment": "Unable to analyze",
                "confidence": 0.0,
                "error": str(e)
            }

# Helper function
def analyze_resume_layout(pdf_path: str) -> dict:
    """
    Analyze original PDF layout.

    Args:
        pdf_path: Path to original uploaded PDF
    """
    analyzer = LayoutAnalyzer()
    return analyzer.analyze_layout(pdf_path)
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

    # Step 2: Visual layout analysis with VLM on ORIGINAL PDF
    print("  → Analyzing visual layout with VLM...")
    pdf_path = state.get("original_pdf_path")
    layout_result = analyze_resume_layout(pdf_path)

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

### 2.4: Enhanced Web Search for Match Critic (3 hours)

**Problem**: Match Critic doesn't know current industry standards, and needs to understand experience-based requirements.

**Solution**: First parse entire JD with LLM to extract requirements, then validate with web search.

File: `src/tools/web_search.py` (NEW)

```python
from langchain_community.tools import DuckDuckGoSearchRun
from src.llm_registry import LLMRegistry
from typing import List, Dict, Any
import json

class JobMarketResearch:
    """
    Analyze job descriptions and validate requirements with web search.
    """

    def __init__(self):
        self.search = DuckDuckGoSearchRun()
        self.llm = LLMRegistry.get_nova_lite()  # Cheap for extraction

    def extract_requirements_from_jd(self, job_description: str) -> Dict[str, Any]:
        """
        Use LLM to parse entire JD and extract structured requirements.
        Handles experience-based skills like "5 years managing Snowflake with Informatica".
        """
        extraction_prompt = f"""
        Analyze this job description and extract ALL requirements in detail.

        Job Description:
        {job_description}

        Extract:
        1. **Job Title**: The specific role
        2. **Required Skills**: Technical skills, tools, languages (e.g., Python, AWS, Docker)
        3. **Experience Requirements**: Years of experience + context
           - Example: "5+ years managing Snowflake data warehouses"
           - Example: "3 years with Informatica ETL integration"
        4. **Preferred Skills**: Nice-to-have skills
        5. **Certifications**: Required or preferred certifications
        6. **Domain Expertise**: Industry knowledge (e.g., fintech, healthcare)
        7. **Soft Skills**: Communication, leadership, etc.

        Return JSON:
        {{
          "job_title": "...",
          "required_skills": ["skill1", "skill2"],
          "experience_requirements": [
            {{"years": 5, "skill": "Snowflake warehouse management", "context": "with Informatica integration"}}
          ],
          "preferred_skills": ["skill1"],
          "certifications": ["cert1"],
          "domain_expertise": ["domain1"],
          "soft_skills": ["skill1"]
        }}
        """

        response = self.llm.invoke(extraction_prompt)
        try:
            return json.loads(response.content)
        except:
            return {"error": "Failed to parse JD"}

    def validate_skill_requirement(self, skill: str, job_title: str) -> Dict[str, Any]:
        """
        Check if a skill is commonly required for a job title via web search.
        """
        query = f'"{job_title}" job requirements "{skill}" site:linkedin.com OR site:indeed.com'

        try:
            results = self.search.run(query)
            skill_mentions = results.lower().count(skill.lower())

            return {
                "skill": skill,
                "validated": skill_mentions > 2,
                "mentions": skill_mentions,
                "evidence": results[:300],
                "confidence": min(skill_mentions / 10, 1.0)  # Max confidence at 10 mentions
            }
        except Exception as e:
            return {
                "skill": skill,
                "validated": False,
                "error": str(e),
                "confidence": 0.0
            }

    def validate_experience_requirement(
        self,
        experience_req: Dict[str, Any],
        job_title: str
    ) -> Dict[str, Any]:
        """
        Validate experience-based requirements (e.g., "5 years with Snowflake").
        """
        years = experience_req.get("years", "")
        skill = experience_req.get("skill", "")
        context = experience_req.get("context", "")

        query = f'"{job_title}" "{years} years" "{skill}" {context}'

        try:
            results = self.search.run(query)
            mentions = results.lower().count(skill.lower())

            return {
                "requirement": f"{years}+ years {skill} {context}",
                "validated": mentions > 1,
                "mentions": mentions,
                "evidence": results[:300]
            }
        except Exception as e:
            return {
                "requirement": f"{years}+ years {skill}",
                "validated": False,
                "error": str(e)
            }

# Helper functions
def analyze_job_description(jd: str) -> Dict[str, Any]:
    """Extract all requirements from JD using LLM."""
    researcher = JobMarketResearch()
    return researcher.extract_requirements_from_jd(jd)

def validate_requirements(
    requirements: Dict[str, Any],
    job_title: str
) -> Dict[str, Any]:
    """Validate extracted requirements with web search."""
    researcher = JobMarketResearch()

    validated = {
        "skills": {},
        "experience": []
    }

    # Validate top 5 required skills
    for skill in requirements.get("required_skills", [])[:5]:
        validated["skills"][skill] = researcher.validate_skill_requirement(skill, job_title)

    # Validate experience requirements
    for exp_req in requirements.get("experience_requirements", [])[:3]:
        validated["experience"].append(
            researcher.validate_experience_requirement(exp_req, job_title)
        )

    return validated
```

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

