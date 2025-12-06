# Agentic AI Project Roadmap: "Career Nexus"

## 1. Executive Summary
**Goal:** Build a production-grade, dual-mode AI platform for Career Management.
**Core Pivot:** Moving from a linear RAG script to a **Stateful Multi-Agent System** using LangGraph.
**Key Tech Stack:**
* **Orchestration:** LangGraph (State Machines, Supervisor Pattern).
* **Vector Database:** Qdrant (Docker) or Chroma (Local) for dynamic CRUD operations.
* **Persistence:** SQLite (for Human-in-the-Loop state management).
* **LLM:** Mistral/Llama (via Ollama) or OpenAI.

---

## 2. Architecture Overview

### The "Dual-Mode" Design
The application serves two distinct user personas using a shared underlying data infrastructure.

| Feature | **Mode A: Job Seeker** (Agentic Optimization) | **Mode B: Recruiter** (HITL Search) |
| :--- | :--- | :--- |
| **Primary Goal** | Optimize CV for specific Job Descriptions. | Find candidates and perform outreach. |
| **Agent Pattern** | **Supervisor Pattern** (One brain managing workers). | **Human-in-the-Loop** (Pause for approval). |
| **Key Action** | *Rewriting* the CV document (Generative). | *Drafting & Sending* Emails (Tool Use). |

---

## 3. Phased Execution Plan

### 🛑 Phase 1: The Data Foundation (Dynamic Ingestion)
**Objective:** Replace static file-based storage (FAISS) with a production-ready Vector Database to allow real-time uploads without restarting the server.

* **Tech Change:** FAISS $\to$ **Qdrant** (Local Docker) or **Chroma**.
* **Tasks:**
    1.  **Setup:** Run Qdrant locally (`docker run -p 6333:6333 qdrant/qdrant`).
    2.  **Schema:** Define a collection `cv_collection` with payload schema (Name, Exp, Skills, SessionID).
    3.  **Refactor Ingestion:**
        * Convert `store_vectors.py` into a reusable function `ingest_cv(pdf_file)`.
        * Replace `index.add()` with `client.upsert()`.
        * **Crucial:** Add `session_id` to metadata to isolate user data.
    4.  **UI Update:** Add a "Upload New CV" widget in Streamlit that triggers `ingest_cv` and updates the DB immediately.

**Definition of Done:** You can upload a PDF in the UI and immediately search for its unique keywords in the "Recruiter" tab.

---

### 🤖 Phase 2: The Job Seeker (Supervisor & Revision)
**Objective:** Build a Multi-Agent system that not only *critiques* a CV but *fixes* it.

* **Architecture:** **Supervisor Control Flow**.
* **The Team (Nodes):**
    * **Supervisor (LLM):** Routes state to the appropriate critic or the editor.
    * **ATS Critic:** Checks keyword density against a Job Description.
    * **Style Critic:** Checks formatting, tone, and active voice.
    * **The Editor (Action):** Receives critiques and *rewrites* the specific CV chunk to address issues.
* **State Schema:**
    ```python
    class AgentState(TypedDict):
        job_description: str
        original_cv_text: str
        critiques: List[str]
        refined_cv_text: str
        revision_count: int
        next_agent: str
    ```
* **Tasks:**
    1.  Define the `Supervisor` prompt to manage the conversation.
    2.  Implement the `Editor` node that takes `critiques` and outputs `refined_cv_text`.
    3.  Connect nodes in LangGraph: `Supervisor -> [Critics] -> Supervisor -> Editor -> End`.

**Definition of Done:** The system accepts a CV + Job Link, runs a critique loop, and outputs a downloadable `Optimized_CV.md` file.

---

### 🕵️ Phase 3: The Recruiter (Persistence & HITL)
**Objective:** Implement "Time Travel" debugging and Human-in-the-Loop approval workflows.

* **Tech Feature:** **LangGraph Checkpointing**.
* **Workflow:** Search $\to$ Select Candidate $\to$ Draft Email $\to$ **STOP (Interrupt)** $\to$ User Approves/Edits $\to$ Send.
* **Tasks:**
    1.  **Persistence:** Initialize `memory = SqliteSaver.from_conn_string(":memory:")`.
    2.  **The Graph:**
        * Node 1: `DraftEmail` (Uses LLM to write outreach).
        * Node 2: `SendEmail` (Mock tool to print "Sent!").
    3.  **The Interrupt:** Compile graph with `interrupt_before=["SendEmail"]`.
    4.  **Streamlit Integration:**
        * Display the "Draft" state to the user.
        * Add buttons: "Approve & Send" (resumes graph) or "Edit" (updates state).

**Definition of Done:** You can search for a candidate, see a drafted email, close the browser, reopen it, and still see the draft waiting for approval.

---

### 🎨 Phase 4: Unified Interface
**Objective:** A clean, mode-switching UI.

* **Tasks:**
    1.  **Sidebar:** Toggle switch `st.sidebar.radio("Mode", ["Job Seeker", "Recruiter"])`.
    2.  **State Management:** Ensure `st.session_state` clears when switching modes to prevent data leaks.
    3.  **Visuals:**
        * Job Seeker Mode: Focus on Document Preview (Before/After).
        * Recruiter Mode: Focus on Search Results cards and Email drafts.

---

## 4. Success Metrics
* **Latency:** Ingestion of a new PDF takes < 5 seconds.
* **Reliability:** The Editor agent actually fixes 100% of the critiques raised by the ATS agent.
* **Safety:** Email is **never** sent without explicit human button click.