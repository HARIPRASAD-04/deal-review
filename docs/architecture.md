# Architecture — Multi-Agent Deal Review Pipeline

## 1. Why LangGraph?

LangGraph is used as the orchestration framework for several specific reasons:

**Stateful, typed workflow graph.** LangGraph maintains a single shared state
object (`WorkflowState`) across all nodes.  Every agent reads from and writes
to the same state, so there is always one consistent view of the run.

**Conditional routing.** Standard pipelines execute steps in a fixed sequence.
LangGraph allows conditional edges — the Orchestrator can inspect the
`pending_handoffs` queue after each step and dynamically decide which agent
runs next, whether a re-check is needed, or whether to escalate.

**Built-in retry / checkpointing.** LangGraph's persistence layer (in later
modules) can checkpoint state after each node, making runs resumable.  This is
important for long-running deal reviews where an LLM call might fail.

**Not a RAG framework.** LangGraph is a workflow graph, not a retrieval
augmented generation framework.  This project uses it to orchestrate agents,
not to retrieve semantically similar documents.

---

## 2. Why Pydantic Models?

Every agent contract in this project is expressed as a Pydantic model rather
than a raw Python dictionary.

**Explicit contracts.** When the Compliance Agent sends a `Handoff`, both the
sender and recipient know exactly what fields exist, their types, and their
constraints.  This eliminates silent bugs from typos or missing keys.

**Validation at the boundary.** Pydantic validates every object when it is
created.  Invalid evidence IDs (wrong format), out-of-range confidence scores,
and unknown agent names are caught immediately, not silently passed through.

**Serialisation.** All models support `.model_dump()` / `.model_dump_json()`,
making it straightforward to persist state, log structured data, and replay
runs.

**Self-documenting.** Field descriptions and type annotations serve as
in-code documentation for future engineers.

---

## 3. Why Shared State?

The `WorkflowState` is the single source of truth for a pipeline run.

All agents read from and write to the same state object (through the
Orchestrator).  This means:

- The Orchestrator always knows the full status of every agent.
- Evidence extracted in stage 1 is immediately available to stage 2 and 3.
- A failed step can be retried without re-running earlier successful steps.
- The complete audit trail is always available.

The alternative — passing results directly between agents — would make the
system harder to debug, impossible to replay, and prone to silent data loss.

---

## 4. Why Structured Handoffs?

The `Handoff` model is the core inter-agent communication primitive.

Without structured handoffs, an agent that needs re-verification would either:
(a) call another agent directly (creating hidden dependencies), or
(b) encode the request as free-form text (losing structure and type safety).

With structured handoffs:

```
Source Agent
  → creates Handoff(type=VERIFICATION_REQUIRED, target=term_extraction, ...)
  → places it in shared state: pending_handoffs
Orchestrator
  → reads pending_handoffs
  → routes to target agent
  → marks handoff ACCEPTED
Target Agent
  → processes the request
  → marks handoff RESOLVED
  → Orchestrator continues
```

Every inter-agent communication is:
- **Observable** (in shared state)
- **Auditable** (logged as AuditEvent)
- **Retryable** (Orchestrator can re-enqueue)
- **Type-safe** (Pydantic validation)

---

## 5. Why Not a Simple Linear Pipeline?

A simple `Term Extraction → Compliance → Risk` chain fails in real-world deal
reviews because:

1. **Compliance may find an inconsistency** in a term and need term extraction
   to re-verify it before making a decision.
2. **Risk analysis may identify missing information** that requires additional
   extraction to be attempted.
3. **A term extraction failure** should trigger a retry, not silently produce
   an incomplete compliance matrix.
4. **Some compliance decisions are ambiguous** and must be escalated to a
   human before risk analysis can proceed.

The LangGraph conditional routing + handoff queue supports all of these
scenarios while keeping the Orchestrator in full control.

---

## 6. Why the Orchestrator Controls Routing?

The Orchestrator is the only entity that:
- Reads `pending_handoffs` and decides what to do next.
- Increments retry counts and decides when to escalate.
- Updates agent statuses.
- Assembles the final report.

This design keeps the workflow **observable** (all decisions are in one place),
**debuggable** (a failed run shows exactly which step the Orchestrator was on),
and **explainable** (interviewers can follow the logic in `graph.py`).

---

## 7. How Agents Communicate

```
Agent A (e.g. Compliance Agent)
  ↓
Creates: Handoff(type=VERIFICATION_REQUIRED, target=term_extraction, ...)
  ↓
Writes to: WorkflowState.pending_handoffs
  ↓
Orchestrator reads pending_handoffs at the end of each cycle
  ↓
Routes to: Term Extraction Agent
  ↓
Term Extraction processes the request, updates WorkflowState
  ↓
Marks handoff RESOLVED
  ↓
Orchestrator continues to next step
```

---

## 8. Why RAG is Not Part of the Core Architecture

This project is a **structured document analysis** system, not a semantic
question-answering chatbot.

RAG (Retrieval-Augmented Generation) is designed for systems that need to find
semantically relevant passages from a large corpus in response to a query.

This system instead:
- Receives a single known document.
- Parses it deterministically (PyMuPDF — Module 2).
- Assigns stable evidence IDs to every relevant passage.
- References those IDs throughout the analysis.

Every claim in the output traces back to a specific page, section, and
verbatim text snippet in the source document.  This is **stronger** than RAG
because the traceability is deterministic, not probabilistic.

RAG could be useful if the system needed to search across thousands of deal
documents.  For single-document analysis, it adds complexity without benefit.

---

## 9. Evidence Traceability Chain

Every material fact in the final report traces back to the source document:

```
DEAL-001 (source document)
  └─ EV-001  (page 2, section 1.1, "Facility Amount" clause)
       └─ TERM-001  (facility_amount = "INR 10 crore", confidence=0.97)
            └─ COMP-001  (POLICY-001 FAIL: exceeds INR 8 crore limit)
                 └─ RISK-001  (HIGH: Facility amount exceeds policy limit)
                      └─ Executive Summary (flagged for credit committee)
```

This chain is maintained by ID references throughout:
- `DealTerm.evidence_ids` → `EvidenceSnippet`
- `ComplianceResult.term_ids` + `evidence_ids`
- `RiskFinding.related_compliance_ids` + `evidence_ids`

---

## 10. What Future Modules Will Add

| Module | Addition |
|--------|----------|
| **Module 2** | Term Extraction Agent — LLM-backed extraction with PyMuPDF parsing |
| **Module 3** | Compliance Review Agent — policy rule evaluation engine |
| **Module 4** | Risk & Summary Agent — risk synthesis and executive summary |
| **Module 5** | Full Orchestrator routing — conditional edges, retries, escalation |
| **Module 6** | Report Generator — final Markdown + JSON evidence-backed report |
| **Module 7** | CLI + integration tests + sample PDF fixtures |

---

## Module 1 Component Map

```
app/
├── models/
│   ├── evidence.py      — EvidenceSnippet (immutable, EV-NNN)
│   ├── terms.py         — DealTerm (TERM-NNN, lifecycle statuses)
│   ├── policy.py        — PolicyRule (immutable, POLICY-NNN)
│   ├── compliance.py    — ComplianceResult (COMP-NNN, PASS/FAIL/...)
│   ├── risk.py          — RiskFinding (RISK-NNN, severity/likelihood)
│   ├── handoff.py       — Handoff (core communication contract)
│   ├── audit.py         — AuditEvent (immutable, append-only log)
│   └── state.py         — WorkflowState + AgentStatus
├── orchestration/
│   └── graph.py         — LangGraph StateGraph skeleton
└── config/
    └── settings.py      — Pydantic-settings configuration
```
