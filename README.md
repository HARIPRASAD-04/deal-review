# Deal Review Pipeline

> **AI Engineer Internship Take-Home Assignment**
> Multi-Agent Deal Review Pipeline — Module 1: Foundation, Shared State & Agent Communication Contracts

---

## Project Purpose

A multi-agent system that reviews a financial deal document against a company
compliance/policy rule set and produces an evidence-backed deal review report.

The system demonstrates:

- Multi-agent decomposition
- Agent orchestration via LangGraph
- Structured agent-to-agent communication contracts (Handoffs)
- Shared typed workflow state
- Evidence grounding and traceability
- Compliance analysis
- Risk analysis with prioritisation
- Human escalation and uncertainty handling
- Failure handling and retries
- End-to-end report generation

---

## Architecture Overview

```
                    ┌─────────────────────────┐
                    │    ORCHESTRATOR AGENT    │
                    │   (LangGraph — graph.py) │
                    └────────────┬────────────┘
                                 │
           ┌─────────────────────┼─────────────────────┐
           │                     │                     │
           ▼                     ▼                     ▼
  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
  │  TERM EXTRACTION│  │   COMPLIANCE    │  │  RISK & SUMMARY │
  │     AGENT       │  │  REVIEW AGENT   │  │     AGENT       │
  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
           │                    │                     │
           └────────────────────┼─────────────────────┘
                                │
                      STRUCTURED HANDOFFS
                       (models/handoff.py)
                                │
                        SHARED WORKFLOW STATE
                         (models/state.py)
                                │
                                ▼
                          FINAL REPORT
```

**Key architectural principle:** Agents communicate exclusively through
structured `Handoff` objects placed into the shared `WorkflowState`.  Agents
do not call each other directly.  The Orchestrator reads the handoff queue and
routes work dynamically.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| Orchestration | LangGraph |
| LLM Integration (future) | LangChain |
| Data Contracts | Pydantic v2 |
| Configuration | pydantic-settings + python-dotenv |
| Document Parsing (future) | PyMuPDF |
| Testing | pytest |
| Storage | Local JSON / SQLite (no external services) |

---

## Current Implementation Status — Module 1

✅ Project structure established  
✅ `pyproject.toml` with all dependencies  
✅ `.env.example` with configuration placeholders  
✅ Pydantic domain models: Evidence, Terms, Policy, Compliance, Risk, Handoff, Audit, State  
✅ LangGraph orchestration skeleton (`build_graph`, `initialize_run` node)  
✅ Shared typed `WorkflowState` with all required fields  
✅ Structured `Handoff` communication contract  
✅ Immutable `AuditEvent` log  
✅ `AgentStatus` with lifecycle helpers  
✅ Synthetic sample data (`data/samples/`)  
✅ Unit tests for all models (positive + negative validation)  
✅ Integration tests for the LangGraph graph  
✅ Architecture documentation  

⬜ Term Extraction Agent (Module 2)  
⬜ PDF parsing / Evidence Registry (Module 2)  
⬜ Compliance Review Agent (Module 3)  
⬜ Risk & Summary Agent (Module 4)  
⬜ Full Orchestrator routing (Module 5)  
⬜ Report Generator (Module 6)  
⬜ CLI Interface (Module 7)  

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/HARIPRASAD-04/deal-review.git
cd deal-review
```

### 2. Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -e ".[dev]"
```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env to add your API key when ready for Module 2+
# No API key is required to run Module 1 tests.
```

---

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=app --cov-report=term-missing

# Run unit tests only
pytest tests/unit/

# Run integration tests only
pytest tests/integration/

# Run a specific test file
pytest tests/unit/test_handoff.py -v
```

> **No API key is required to run Module 1 tests.**  
> All tests use synthetic data and do not call any external LLM API.

---

## Running the Minimal Graph

```python
from app.models.state import WorkflowState
from app.orchestration.graph import deal_review_graph

# Create initial state
state = WorkflowState(document_id="DEAL-001")

# Execute the graph
result = deal_review_graph.invoke(state)

# Inspect results
print(f"Run ID: {result['run_id']}")
print(f"Audit events: {len(result['audit_events'])}")
print(f"Agent statuses: {list(result['agent_statuses'].keys())}")
```

Or run the quick demo script:

```bash
python -c "
from app.models.state import WorkflowState
from app.orchestration.graph import deal_review_graph
state = WorkflowState(document_id='DEAL-001')
result = deal_review_graph.invoke(state)
print('Run ID:', result['run_id'])
print('Audit events:', len(result['audit_events']))
print('Agents registered:', list(result['agent_statuses'].keys()))
"
```

---

## Evidence Traceability Model

Every material fact in the output traces back to the source document:

```
Document (DEAL-001)
  └─ EV-001 (page 2, section 1.1, "Facility Amount")
       └─ TERM-001 (facility_amount = "INR 10 crore")
            └─ COMP-001 (POLICY-001 FAIL — exceeds INR 8 crore)
                 └─ RISK-001 (HIGH — facility exceeds policy limit)
                      └─ Executive Summary
```

---

## Repository Structure

```
deal-review/
├── app/
│   ├── agents/
│   │   ├── extraction/    # Module 2
│   │   ├── compliance/    # Module 3
│   │   ├── risk/          # Module 4
│   │   └── orchestrator/  # Module 5
│   ├── models/
│   │   ├── evidence.py    # EvidenceSnippet
│   │   ├── terms.py       # DealTerm
│   │   ├── policy.py      # PolicyRule
│   │   ├── compliance.py  # ComplianceResult
│   │   ├── risk.py        # RiskFinding
│   │   ├── handoff.py     # Handoff (communication contract)
│   │   ├── audit.py       # AuditEvent
│   │   └── state.py       # WorkflowState + AgentStatus
│   ├── orchestration/
│   │   └── graph.py       # LangGraph workflow
│   ├── config/
│   │   └── settings.py    # pydantic-settings
│   └── utils/
├── tests/
│   ├── unit/              # Per-model tests
│   └── integration/       # Graph execution tests
├── data/
│   └── samples/           # Synthetic deal + policy data
├── docs/
│   └── architecture.md    # Architecture decisions
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md
```

---

## Current Limitations

- No real LLM calls — Module 1 is architecture only.
- No PDF parsing — PyMuPDF integration is planned for Module 2.
- LangGraph graph has only one node (`initialize_run`).
- Agent nodes (extraction, compliance, risk) are empty stubs.
- No persistent storage — state lives in memory only.
- No CLI interface yet.

---

## Upcoming Modules

| Module | Focus |
|--------|-------|
| **2** | Term Extraction Agent + PDF parsing + Evidence Registry |
| **3** | Compliance Review Agent + Policy engine |
| **4** | Risk & Summary Agent |
| **5** | Full Orchestrator: conditional routing, retries, escalation |
| **6** | Report Generator |
| **7** | CLI interface + end-to-end integration tests |
