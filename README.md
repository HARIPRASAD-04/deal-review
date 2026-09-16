# Multi-Agent Deal Review Pipeline

> **AI Engineer Internship Take-Home Assignment**
> A production-grade, multi-agent system that reviews financial deal documents against a compliance policy and produces an evidence-backed deal review report.

---

## The Problem

When a lending institution receives a deal document — a commercial loan term sheet, credit facility agreement, or similar — multiple compliance questions need to be answered before the deal can proceed:

- Does the interest rate fall within policy limits?
- Are all required covenants present?
- Does the governing law jurisdiction match approved counterparty agreements?
- Are any terms ambiguous or missing that require human resolution?

Traditionally this is done manually: a compliance officer reads the document, locates each relevant clause, and checks it against the policy rule book. This is slow, expensive, and inconsistent.

**This project automates the entire pipeline** using a multi-agent architecture. Two PDFs go in — the deal document and the policy document — and a structured, evidence-backed compliance report comes out. Every finding traces back to the exact page and paragraph in the source document.

---

## Architecture

### Design Philosophy

The system is built around three core principles:

1. **Strict separation of concerns.** Each agent answers exactly one question. The Term Extraction Agent answers *"What does the document say?"* The Compliance Agent answers *"Does it comply with policy?"* The Risk & Summary Agent answers *"What matters and what could go wrong?"* The Orchestrator answers *"What happens next?"* No agent crosses these boundaries.

2. **No hallucination, no silent failures.** Every extracted term is grounded to a specific evidence snippet with a page number. Every evidence ID the LLM returns is validated against the registry before being accepted. Ambiguous values are surfaced as `AMBIGUOUS` — never silently resolved. Missing terms are reported as `INSUFFICIENT_EVIDENCE` — never skipped. The pipeline does not fabricate data to make itself look complete.

3. **Structured communication via handoffs.** Agents do not call each other directly. They publish typed `Handoff` objects into shared `WorkflowState`. The Orchestrator reads the handoff queue and routes work. This makes inter-agent communication observable, auditable, and testable independently.

### The Four Agents

```
                    ┌─────────────────────────┐
                    │    ORCHESTRATOR AGENT    │
                    │   (LangGraph — graph.py) │
                    │   Routing + Report       │
                    └────────────┬────────────┘
                                 │
           ┌─────────────────────┼─────────────────────┐
           │                     │                     │
           ▼                     ▼                     ▼
  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
  │  TERM EXTRACTION│  │   COMPLIANCE    │  │  RISK & SUMMARY │
  │     AGENT       │  │  REVIEW AGENT   │  │     AGENT       │
  │  "What does it  │  │  "Does it pass  │  │  "What could go │
  │     say?"       │  │   the rules?"   │  │    wrong?"      │
  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
           │                    │                     │
           └────────────────────┼─────────────────────┘
                                │
                  STRUCTURED HANDOFFS (models/handoff.py)
                  Agents communicate through WorkflowState.
                  No direct agent-to-agent calls.
                                │
                        SHARED WORKFLOW STATE
                         (models/state.py)
                                │
                                ▼
                   FINAL DEAL REVIEW REPORT (JSON)
```

### LangGraph Execution Graph

```
START
  → initialize_run       (orchestrator setup, audit RUN_STARTED)
  → ingest_document      (PDF → EvidenceRegistry, DOCUMENT_INGESTION_*)
  → extract_terms        (LLM → DealTerm[], TERM_EXTRACTION_*)
  → review_compliance    (deterministic → ComplianceResult[], COMPLIANCE_REVIEW_*)
  → route_handoffs       (HandoffRouter → targeted clarification + re-check)
  → risk_summary         (RiskSummaryAgent → RiskFinding[], EXECUTIVE_SUMMARY_*)
  → report_assembly      (OrchestratorAgent → FinalDealReviewReport)
  → END
```

### Evidence Traceability Chain

Every material fact in the output traces all the way back to the source page:

```
Document (deal.pdf, Page 2)
  └─ EV-018 (page 2, section "4.1 - Interest Rate", raw text)
       └─ TERM-001 (interest_rate = "8.5% per annum", normalized: 0.085)
            └─ COMP-001 (POLICY-001 PASS — 0.085 ≤ 0.09)
                 └─ No risk finding raised
```

```
Document (deal.pdf, Page 3)
  └─ EV-027 (page 3, missing governing jurisdiction)
       └─ (no term extracted — INSUFFICIENT_EVIDENCE)
            └─ COMP-005 (POLICY-005 INSUFFICIENT_EVIDENCE)
                 └─ RISK-001 (HIGH — Missing Required Term: governing_law)
                       └─ Executive Summary: "Requires Human Review"
```

---

## Why These Choices

### Why LangGraph?

We needed a DAG-based orchestration framework with typed shared state and conditional routing. LangGraph provides exactly this: nodes are pure functions `(state) → dict`, the graph topology is declared separately, and state is immutable between steps. This made the orchestration testable — you can unit test any node function in isolation simply by passing in a `WorkflowState`.

### Why Pydantic v2 for everything?

Every data contract in the system — evidence snippets, extracted terms, policy rules, compliance results, risk findings, handoffs, audit events, workflow state — is a Pydantic v2 model. This gives us:
- **Runtime validation** at the system boundaries (LLM output is validated before it enters the pipeline)
- **Serialization/deserialization** for state persistence and the JSON report
- **Self-documenting schemas** for inter-agent contracts
- **Strict typing** that catches category/ID format errors at construction time, not at runtime

### Why a Protocol-based LLM client?

The `LLMClient` is defined as a `runtime_checkable Protocol` — any object implementing `get_structured_completion()` satisfies it. This means:
- `GoogleLLMClient` (real Gemini integration with `temperature=0`) for production
- `FakeLLMClient` (configurable test double with `call_count` tracking) for tests
- No mocking framework needed — just inject a different implementation

All 584 tests run without an API key using `FakeLLMClient`.

### Why separate LLM schemas from domain models?

`ExtractedTermSchema` (what the LLM returns) is kept strictly separate from `DealTerm` (the application domain model). The LLM schema is a prompt/response contract that may change as we tune the model. The domain model is stable. Conflating them would mean LLM prompt changes could break application logic.

### Why deterministic compliance evaluation?

The compliance evaluator (`app/agents/compliance/evaluator.py`) is 100% deterministic — no LLM involved. Given a `PolicyRule` and a `DealTerm`, it always produces the same `ComplianceResult`. This was a deliberate choice: compliance is a rule-following exercise, not a judgment call. The LLM's job is to *extract* values; the evaluator's job is to *compare* them against thresholds. Mixing these concerns would make compliance results non-reproducible.

### Why require both PDFs at runtime?

The policy PDF is not bundled as a static file. Every run requires an explicit policy PDF. This was a safety decision: different deals may be reviewed against different policy versions. Bundling a default policy would mean a stale policy silently applies when the caller forgets to supply one. Instead, the system hard-stops if the policy PDF is missing or unparseable.

### Why an Application Service boundary (`service.py`)?

The UI and all external callers interact exclusively with `DealReviewService.run()`. They never import from `app.orchestration.graph` directly. This keeps the UI genuinely presentation-only and means the internal graph topology can change freely without touching the UI or integration contracts.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| Orchestration | LangGraph |
| LLM Provider | Google Gemini 2.5 Flash (via `langchain-google-genai`) |
| LLM Interface | Protocol-based `LLMClient` abstraction |
| Data Contracts | Pydantic v2 |
| Configuration | pydantic-settings + python-dotenv |
| Document Parsing | PyMuPDF (`fitz`) |
| UI | Streamlit |
| Testing | pytest (584 tests, 0 failures) |
| Package Management | pip / pyproject.toml |

---

## Module-by-Module Achievements

### Module 1 — Foundation, Shared State & Agent Communication Contracts
**Tests: 126 passing**

Established the entire architectural skeleton before writing a single agent:

- Full Pydantic v2 domain model suite: `EvidenceSnippet`, `DealTerm`, `PolicyRule`, `ComplianceResult`, `RiskFinding`, `Handoff`, `AuditEvent`, `WorkflowState`, `AgentStatus`
- Typed `Handoff` communication contract — the only mechanism by which agents communicate
- Immutable `AuditEvent` log for full run traceability
- LangGraph `StateGraph` skeleton with `initialize_run` node
- `AgentStatus` lifecycle helpers (`mark_running()`, `mark_completed()`, `mark_failed()`)
- Synthetic sample datasets for testing without real documents

The key decision here was to design the *communication contracts before the agents*. This forced clarity about what each agent produces and consumes, and prevented ad-hoc coupling.

---

### Module 2 — Document Ingestion & Evidence Registry
**Tests: 258 passing (+132)**

Turned a PDF into structured, traceable evidence without any LLM:

- `load_pdf()` — PyMuPDF-backed loader returning `DocumentPage` objects
- `segment_page()` — heuristic segmenter that splits pages into `EvidenceSnippet` objects at section boundaries, preserving all financial qualifiers
- `EvidenceRegistry` — dict-backed provenance layer (not a vector store — simple add/get/list, no embeddings)
- Deterministic sequential IDs (`EV-001`, `EV-002`, ...) — same unchanged PDF always produces the same IDs
- `ingest_document` LangGraph node with graceful failure (never raises; records `DOCUMENT_INGESTION_FAILED` audit event)

The registry is intentionally not a vector store. This system does not use RAG. The LLM receives the *full serialized evidence registry* in its prompt and reasons over it directly. For deal documents (typically 5–30 pages), this fits comfortably within the context window and avoids the retrieval approximation errors that RAG introduces.

---

### Module 3 — Term Extraction Agent & LLM Abstraction
**Tests: 344 passing (+86)**

Built the first real AI agent and the LLM infrastructure layer:

- `LLMClient` Protocol + `GoogleLLMClient` (real) + `FakeLLMClient` (test double)
- `TermExtractionAgent.extract()` — single LLM call per document, structured output via `ExtractionOutput` schema
- Hallucination guard: any EV-ID returned by the LLM that doesn't exist in the registry is rejected with a recorded reason (not silently dropped)
- Ambiguity handling: same term name with different values → both kept as `AMBIGUOUS`; same name with same value → merged with combined evidence IDs
- Clarification capability: `TermExtractionAgent.clarify()` for targeted re-extraction when compliance requests it

The system prompt explicitly forbids the LLM from performing compliance or risk analysis. It is scoped to extraction only. This is enforced in the prompt, not just in the architecture.

---

### Module 4 — Compliance Review Agent & Deterministic Rule Evaluator
**Tests: 450 passing (+106)**

Built the compliance engine — entirely deterministic, no LLM:

- `evaluate_rule()` — dispatches to the correct operator handler (`<=`, `>=`, `<`, `>`, `==`, `!=`, `exists`, `in`, `not_in`, `semantic`)
- `parse_numeric()` — handles bare floats, `N%`, `Nx` ratios, `INR N,NNN,NNN`, `N months` and more
- Four compliance statuses: `PASS`, `FAIL`, `NEEDS_HUMAN_REVIEW`, `INSUFFICIENT_EVIDENCE`
- `AMBIGUOUS` terms → `NEEDS_HUMAN_REVIEW` (never a guess)
- Missing terms → `INSUFFICIENT_EVIDENCE` (never silently skipped)
- Per-rule fault isolation — one rule's failure never aborts evaluation of the others
- Multi-term rules use worst-status selection (conservative safety behavior)

The `semantic` operator always produces `NEEDS_HUMAN_REVIEW`. Semantic compliance judgments (e.g., "borrower must maintain adequate reserves") require human judgment and are explicitly not automated.

---

### Module 5 — Agent Communication, Handoff Routing & Feedback Loop
**Tests: 463 passing (+13)**

Activated the agent-to-agent communication system and implemented the compliance → extraction feedback loop:

- `HandoffRouter` — routes `CLARIFICATION_REQUIRED` handoffs from Compliance back to Term Extraction
- Loop prevention: `clarification_attempts[rule_id]` counter with `max_attempts = 1` — prevents infinite loops
- When the attempt limit is exceeded: escalated directly to `human_escalations` with a clear reason
- When clarification succeeds: new terms update `extracted_terms`; `ComplianceReviewAgent.review_rule()` immediately re-evaluates the specific rule
- Full audit trail: `HANDOFF_ROUTING_STARTED`, `CLARIFICATION_STARTED/COMPLETED/FAILED`, `LOOP_LIMIT_REACHED`

This is what proves multi-agent communication actually happened: the Compliance Agent does not call the Term Extraction Agent directly. It creates a typed `Handoff` object in `WorkflowState.pending_handoffs`. The HandoffRouter reads it, dispatches the clarification, collects the result, and updates state. All of this is visible in the `audit_events` log.

---

### Module 6 — Risk & Summary Agent
**Tests: 505 passing (+42)**

Built the fourth agent — multi-dimensional risk synthesis with LLM-generated executive summary:

- Four risk sources synthesized: compliance failures, missing terms, low-confidence extractions, material extracted terms
- Risk categories: financial, legal, operational, compliance, credit, reputational
- Severity prioritization: `CRITICAL` → `HIGH` → `MEDIUM` → `LOW`
- Evidence ID validation against the registry (risk findings cannot reference evidence that doesn't exist)
- LLM-generated executive summary via `SummaryOutput` schema, with deterministic fallback if LLM fails or returns invalid schema
- Strict state ownership: Module 6 populates `state.executive_summary` only — `state.final_report` assembly is reserved for the Orchestrator in Module 7

---

### Module 7 — Orchestrator Agent, End-to-End Workflow & Final Report
**Tests: 531 passing (+26)**

Built the Orchestrator Agent as the fourth formal agent, closing the pipeline:

- `OrchestratorAgent` with six named responsibilities: `inspect_state()`, `check_dependencies()`, `should_retry()`, `route_handoffs()`, `collect_escalations()`, `assemble_report()`
- `FinalDealReviewReport` — structured, typed, JSON-serialisable final deliverable
- `ComplianceSummary` — aggregated rule pass/fail counts with overall status
- `ReviewStatus` enum: `COMPLETED`, `COMPLETED_WITH_HUMAN_REVIEW`, `FAILED`
- Report assembly is 100% deterministic (no LLM) — the Orchestrator does not make judgment calls
- Full graph now: `initialize_run → ingest_document → extract_terms → review_compliance → route_handoffs → risk_summary → report_assembly`

The Orchestrator does not approve or reject deals. `ReviewStatus.COMPLETED` means the pipeline ran to completion and produced a report — not that the deal is compliant. Compliance status is determined entirely by the Compliance Agent's deterministic evaluations.

---

### Module 8 — Application Service Boundary, Dynamic Policy Parser, Streamlit UI & Evaluation Engine
**Tests: 584 passing (+53)**

Delivered the production boundary: a clean external API, a dynamic policy ingestion layer, a human UI, and a non-agent evaluation framework:

**`DealReviewService` (app/service.py)**
- Single public API: `service.run(DealReviewRequest) → DealReviewResponse`
- Hard stop on policy parse failure — pipeline not invoked with empty or wrong rules
- UI and all callers are isolated from internal graph topology

**Dynamic Policy Parser (`app/ingestion/policy_parser.py`)**
- Converts unstructured policy PDFs into validated `PolicyRule` objects via LLM
- Explicit rule IDs (e.g. `POLICY-003`) are preserved verbatim
- Where no ID is present, deterministic `POLICY-9NN` IDs are assigned
- Ambiguous policy clauses are surfaced in `PolicyParseResult.ambiguous_statements` — never silently codified into fake rules

**Streamlit UI (`app/ui/streamlit_app.py`)**
- Upload Deal PDF + Policy PDF
- 5 rich tabs: Executive Summary, Compliance Matrix, Extracted Terms & Evidence, Risk Findings & Escalations, Audit & Raw JSON Report
- Dark glassmorphism styling, visual status badges, one-click JSON download
- Presentation-only: calls `DealReviewService`, never touches graph internals

**Evaluation Engine (`app/evaluation/runner.py`)**
- `EvaluationCase` + `EvaluationResult` + `evaluate_case()` for non-agent deterministic evaluation
- Assertion checks: `report_integrity`, `review_status_match`, `terms_present`, `evidence_grounding`, `compliance_status`, `risk_categories`, `risk_finding_titles`
- Evaluation CLI (`scripts/run_evaluation.py`) generates runtime policy PDFs (no committed test binaries)

---

## Setup Instructions

### 1. Prerequisites
- **Python 3.11+**
- **Git**

### 2. Clone the Repository

```bash
git clone https://github.com/HARIPRASAD-04/deal-review.git
cd deal-review
```

### 3. Create and Activate a Virtual Environment

```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate
```

### 4. Install Dependencies

Install the core package in editable mode along with all optional groups (UI, Google LLM, and Development tools):

```bash
pip install -e ".[ui,llm-google,dev]"
```

### 5. Configure Environment Variables

Create your local environment file from the provided example:

```bash
# On Windows:
copy .env.example .env

# On macOS/Linux:
cp .env.example .env
```

Open the newly created `.env` file and add your `GOOGLE_API_KEY` for real LLM extraction (using Google Gemini). 
*Note: Without an API key, the system safely falls back to a mock mode (`FakeLLMClient`).*

---

## Running the System

### Streamlit UI (recommended)

```bash
streamlit run app/ui/streamlit_app.py
```

Upload a Deal PDF and a Policy PDF, then click **Run Deal Review**. The UI will run the full 4-agent pipeline and display the report across 5 tabs.

### Evaluation CLI

```bash
python scripts/run_evaluation.py
```

Runs two evaluation cases (Standard Deal & Strict Policy Breach) and prints per-check pass/fail scores.

### Demo Scripts

```bash
python scripts/demo_term_extraction.py    # Module 3 demo
python scripts/demo_compliance.py         # Module 4 demo
python scripts/demo_feedback_loop.py      # Module 5 demo
python scripts/demo_risk_summary.py       # Module 6 demo
python scripts/demo_orchestrator.py       # Module 7 end-to-end demo
```

---

## Running Tests

```bash
# Run all tests (584 tests, 0 failures)
python -m pytest -v

# Run unit tests only
python -m pytest tests/unit/ -v

# Run integration tests only
python -m pytest tests/integration/ -v

# Run with coverage
python -m pytest --cov=app --cov-report=term-missing

# Run a specific test file
python -m pytest tests/unit/test_compliance_agent.py -v
```

> **No API key is required to run the test suite.**
> All tests use `FakeLLMClient` and synthetic data. Zero network calls.

---

## Repository Structure

```
deal-review/
├── app/
│   ├── agents/
│   │   ├── extraction/         # Term Extraction Agent (Module 3)
│   │   │   ├── term_extraction.py
│   │   │   └── prompts.py
│   │   ├── compliance/         # Compliance Review Agent (Module 4)
│   │   │   ├── compliance_review.py
│   │   │   └── evaluator.py
│   │   ├── risk/               # Risk & Summary Agent (Module 6)
│   │   │   └── risk_summary.py
│   │   └── orchestrator/       # Orchestrator Agent (Module 7)
│   │       ├── agent.py
│   │       └── router.py
│   ├── evaluation/             # Evaluation framework (Module 8)
│   │   └── runner.py
│   ├── ingestion/              # Document ingestion (Module 2)
│   │   ├── pdf_loader.py
│   │   ├── segmenter.py
│   │   ├── evidence_registry.py
│   │   └── policy_parser.py   # Dynamic policy ingestion (Module 8)
│   ├── llm/                   # LLM abstraction layer (Module 3)
│   │   ├── client.py          # LLMClient Protocol + GoogleLLMClient + FakeLLMClient
│   │   └── schemas.py         # LLM-facing Pydantic schemas (separate from domain models)
│   ├── models/                # Domain models (Module 1)
│   │   ├── evidence.py        # EvidenceSnippet (EV-NNN)
│   │   ├── terms.py           # DealTerm (TERM-NNN)
│   │   ├── policy.py          # PolicyRule (POLICY-NNN)
│   │   ├── compliance.py      # ComplianceResult (COMP-NNN)
│   │   ├── risk.py            # RiskFinding (RISK-NNN)
│   │   ├── report.py          # FinalDealReviewReport (Module 7)
│   │   ├── handoff.py         # Handoff (inter-agent communication contract)
│   │   ├── audit.py           # AuditEvent (immutable run log)
│   │   └── state.py           # WorkflowState + AgentStatus
│   ├── orchestration/
│   │   └── graph.py           # LangGraph StateGraph — full 7-node pipeline
│   ├── config/
│   │   └── settings.py        # pydantic-settings configuration
│   ├── ui/
│   │   └── streamlit_app.py   # Streamlit UI (Module 8)
│   └── service.py             # Application service boundary (Module 8)
├── tests/
│   ├── unit/                  # 28 unit test files (per-component isolation)
│   └── integration/           # 9 integration test files (full pipeline)
├── scripts/                   # Demo and utility scripts
├── data/samples/              # Synthetic deal PDFs and JSON samples
├── milestones/                # Per-module milestone records
├── docs/
│   └── architecture.md        # Architectural decision records
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md
```

---

## System Constraints (By Design)

These are intentional design constraints, not limitations:

- **Exactly four agents.** Term Extraction, Compliance Review, Risk & Summary, Orchestrator. No fifth agent was introduced at any module boundary.
- **One LLM call per document pass.** The term extraction agent makes a single structured call with the full serialized evidence registry. No per-snippet LLM calls, no RAG.
- **No autonomous deal approval.** `ReviewStatus` reflects pipeline completion quality, not loan approval. A human must act on the report.
- **No silent fallbacks.** Every failure mode produces an explicit error, warning, or audit event. The system never silently substitutes defaults.
- **No committed binary test fixtures.** Test PDFs are generated at runtime by `scripts/create_sample_pdf.py` and the evaluation runner using PyMuPDF. The repository stays lean.

---

## Commit History

```
bfcc551  module-8: implement application service boundary, dynamic policy parser, Streamlit UI & evaluation engine
babff53  module-7: implement Orchestrator Agent and final report assembly pipeline
f1f6563  module-6: implement Risk & Summary Agent and narrative summary pipeline
9801779  module-5: implement agent handoff routing and compliance-extraction feedback loop
5e8c366  module-4: compliance review agent & deterministic rule evaluator
088162d  module-3: term extraction agent, llm client abstraction & evidence grounding safety
8eafbc9  module-2: document ingestion, section segmentation & evidence registry
ffb864b  module-1: establish foundational architecture, shared state & agent handoff contracts
```
