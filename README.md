# 🚀 Career Nexus - AI-Powered Job Seeking Platform

An intelligent, multi-agent platform that helps job seekers optimize their CVs and enables recruiters to find the best candidates using RAG (Retrieval Augmented Generation).

## ✨ Features

### Job Seeker Agent
- **Automated CV Optimization**: Multi-agent workflow that analyzes and improves your resume
- **ATS Compatibility Checks**: Ensures your CV passes Applicant Tracking Systems
- **Job Description Matching**: Aligns your CV with specific job requirements
- **Visual Layout Analysis**: Uses Vision Language Models to detect formatting issues
- **Grammar & Style Improvements**: Professional language enhancement
- **Impact Optimization**: Transforms weak bullet points into quantified achievements

### Recruiter RAG System
- **Semantic Search**: Find candidates based on skills, experience, and qualifications
- **Vector Database**: Efficient storage and retrieval of candidate CVs
- **Context-Aware Responses**: Evidence-based answers with source attribution

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        SUPERVISOR                                │
│              (Quality scoring, iteration control)                │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     PARALLEL CRITICS                             │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │   ATS   │ │  Match  │ │  Truth  │ │Language │ │ Impact  │   │
│  │ Critic  │ │ Critic  │ │ Critic  │ │ Critic  │ │ Critic  │   │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘   │
└───────┼──────────┼──────────┼──────────┼──────────┼─────────────┘
        │          │          │          │          │
        └──────────┴──────────┴──────────┴──────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CONSOLIDATOR                                │
│         (Merge duplicates, resolve conflicts, filter)            │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                         EDITOR                                   │
│              (Apply changes with self-reflection)                │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                        VERIFIER                                  │
│         (Verify changes, retry if needed, max 2 retries)         │
└─────────────────────────────────────────────────────────────────┘
```

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Agent Framework** | LangGraph, LangChain |
| **LLMs** | Claude (Sonnet/Haiku), Mistral (Small/Medium/Large/Pixtral), Amazon Nova |
| **Vector Database** | Qdrant |
| **Embeddings** | Google Generative AI (text-embedding-004) |
| **UI** | Streamlit |
| **PDF Processing** | PyPDF, pdf2image, WeasyPrint, ReportLab |
| **NLP Tools** | spaCy, NLTK, LanguageTool |
| **Web Search** | DuckDuckGo Search |

## 📦 Installation

### Prerequisites
- Python 3.10+
- Poppler (for PDF to image conversion)
- Docker (optional, for Qdrant)

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Job-Seek-Agentic-Platform.git
cd Job-Seek-Agentic-Platform
```

### 2. Install Dependencies
```bash
pip install -r reqs.txt
python -m spacy download en_core_web_sm
```

### 3. Set Up Qdrant
Using Docker:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory:
```env
# AWS Bedrock (for Claude & Nova models)
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1

# Mistral AI
MISTRAL_API_KEY=your_mistral_api_key

# Google AI (for embeddings)
GEMINI_API_KEY=your_gemini_api_key

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

### 5. Install System Dependencies (Linux/Mac)
```bash
# Ubuntu/Debian
sudo apt-get install poppler-utils

# macOS
brew install poppler
```

## 🚀 Usage

### Running the Streamlit App
```bash
streamlit run src/main.py
```

### Job Seeker Mode
1. Upload your CV (PDF format)
2. Paste the job description
3. Click "Optimize My CV"
4. Download the optimized PDF

### Running the Test Script
```bash
python full-jobseeker-flow-test-script.py
```

### Recruiter RAG Chatbot
```bash
streamlit run app.py
```

## 📁 Project Structure

```
Job-Seek-Agentic-Platform/
├── src/
│   ├── agents/
│   │   ├── JobSeeker/
│   │   │   ├── graph.py          # LangGraph workflow definition
│   │   │   ├── state.py          # Agent state schema
│   │   │   ├── supervisor.py     # Quality control & routing
│   │   │   ├── critics.py        # 5 specialized critic agents
│   │   │   ├── consolidator.py   # Critique merging & filtering
│   │   │   ├── editor.py         # Resume modification
│   │   │   ├── verifier.py       # Change verification
│   │   │   └── memory.py         # Revision tracking
│   │   └── Recruiter/
│   │       └── email.py          # Email drafting (placeholder)
│   ├── database/
│   │   ├── client.py             # Qdrant client wrapper
│   │   └── schema.py             # Collection schema
│   ├── ingestion/
│   │   └── pipeline.py           # CV ingestion & embedding
│   ├── tools/
│   │   ├── ats_parser.py         # ATS compatibility checker
│   │   ├── grammar_check.py      # LanguageTool wrapper
│   │   ├── layout_analyzer.py    # VLM-based layout analysis
│   │   ├── markdown_formatter.py # PDF export
│   │   └── web_search.py         # Job market research
│   ├── llm_registry.py           # Centralized LLM management
│   ├── main.py                   # Streamlit entry point
│   └── utils.py                  # Helper functions
├── app.py                        # Recruiter chatbot
├── config.py                     # Configuration loader
├── reqs.txt                      # Python dependencies
└── README.md
```

## 🤖 Agent Descriptions

### Critics (Parallel Execution)

| Critic | Purpose | Tools Used |
|--------|---------|------------|
| **ATS Critic** | Checks parsing compatibility | pyresparser, VLM layout analyzer |
| **Match Critic** | Evaluates job description alignment | Web search, JD parser |
| **Truth Critic** | Identifies unsupported claims | LLM reasoning |
| **Language Critic** | Fixes grammar and style | LanguageTool |
| **Impact Critic** | Improves achievement metrics | LLM analysis |

### Other Agents

- **Supervisor**: Orchestrates the workflow, tracks quality scores, decides when to stop
- **Consolidator**: Merges duplicate critiques, resolves conflicts, filters low-quality advice
- **Editor**: Applies approved changes using ReAct pattern (Think → Act → Observe)
- **Verifier**: Validates that changes were properly applied (80% threshold, 2 retries max)

## 🔧 Configuration

### LLM Selection Strategy

The system uses different models based on task requirements:

| Task | Model | Reasoning |
|------|-------|-----------|
| Vision analysis | Pixtral Large | Best vision capabilities |
| Complex editing | Claude Sonnet | Strong instruction following |
| Fast routing | Nova Micro | Ultra-cheap, fast |
| JSON extraction | Mistral Small | Good JSON mode support |
| Consolidation | Mistral Medium | Balance of cost/quality |

### Quality Thresholds

- **Verification Pass**: ≥80% critiques addressed
- **Max Editor Retries**: 2
- **Max Workflow Iterations**: 5
- **Minimum Improvement**: 5 points (stops if diminishing returns)
- **Target Quality**: 90+ (stops if achieved)

## 📊 Workflow Visualization

Generate a workflow diagram:
```python
from src.agents.JobSeeker.graph import build_job_seeker_graph

graph = build_job_seeker_graph()
png_data = graph.get_graph().draw_mermaid_png()
with open("workflow.png", "wb") as f:
    f.write(png_data)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangGraph](https://github.com/langchain-ai/langgraph) for the agent orchestration framework
- [Qdrant](https://qdrant.tech/) for the vector database
- [LanguageTool](https://languagetool.org/) for grammar checking
- [pyresparser](https://github.com/OmkarPathak/pyresparser) for ATS simulation

---

**Built with ❤️ for job seekers and recruiters alike**
