# Project Structure Guide

This document outlines the organized file structure for the **Career Nexus** platform, aimed at supporting the "Dual-Mode" agentic architecture.

## `src/` Directory
The `src` directory contains the modern, modular codebase.

### 1. `src/database/` (Phase 1)
Handles connection and operations with the Vector Database (Qdrant/Chroma).
- `client.py`: Setup and connection logic.
- `schema.py`: Definitions for collections and payloads.

### 2. `src/ingestion/` (Phase 1)
Manages the parsing of CVs and updating the vector store.
- `pipeline.py`: Functions to parse PDFs, chunk text, and upsert to the DB.

### 3. `src/agents/` (Phase 2 & 3)
Contains the definition of the LangGraph agents.
- **Shared**:
    - `state.py`: Defines the `AgentState` schema.
- **Job Seeker Mode**:
    - `supervisor.py`: The routing logic.
    - `critics.py`: ATS and Style analysis nodes.
    - `editor.py`: The rewriting node.
- **Recruiter Mode**:
    - `email.py`: Draft generation and sending logic.
- **Orchestration**:
    - `graph.py`: Assembles the nodes into the specific LangGraph workflows.

### 4. `src/ui/` (Phase 4)
The frontend application logic.
- `main.py`: The entry point for the Streamlit app (imports components).
- `components.py`: Reusable UI elements (widgets, cards).

## `data/` Directory
Place your local data here.
- `CVs/`: Store PDF resumes here.

## Legacy Files
Files in the root directory (like `Plan.md`, `LLM.py`, old `app.py`) remain untouched for reference until migration is complete.
