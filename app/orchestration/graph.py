"""LangGraph orchestration graph — Module 2 update.

Graph layout after Module 2:

    START → initialize_run → ingest_document → END

Future modules will add nodes for:

* term_extraction   (Module 3)
* compliance_review (Module 4)
* risk_analysis     (Module 5)
* handoff_router    (Module 6)
* human_escalation  (Module 6)
* report_assembly   (Module 7)

Conditional routing between those nodes will be determined by the
Orchestrator, which inspects the shared WorkflowState after each step.

WHY LANGGRAPH?
--------------
LangGraph allows us to:
* Define the workflow as an explicit directed graph with typed state.
* Add conditional edges so the Orchestrator can route dynamically.
* Benefit from built-in checkpointing / replay when needed.
* Integrate LLM-backed agent nodes cleanly alongside deterministic nodes.

The graph uses ``WorkflowState`` (a Pydantic model) as its shared state.
LangGraph requires that state updates are returned as *dicts* or *TypedDicts*
by each node, so each node returns only the fields it modified.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.ingestion.evidence_registry import EvidenceRegistry
from app.ingestion.pdf_loader import load_pdf
from app.ingestion.segmenter import segment_page
from app.models.audit import AuditEventType, make_audit_event
from app.models.handoff import (
    AGENT_ORCHESTRATOR,
    AGENT_TERM_EXTRACTION,
    AGENT_COMPLIANCE,
    AGENT_RISK_SUMMARY,
)
from app.models.state import AgentStatus, AgentStatusEnum, WorkflowRunStatus, WorkflowState

logger = logging.getLogger(__name__)

# ── Node names — kept as constants to avoid magic strings ─────────────────────

NODE_INITIALIZE = "initialize_run"
NODE_INGEST_DOCUMENT = "ingest_document"          # Module 2
NODE_TERM_EXTRACTION = "term_extraction"          # Placeholder — Module 3
NODE_COMPLIANCE_REVIEW = "compliance_review"      # Placeholder — Module 4
NODE_RISK_ANALYSIS = "risk_analysis"              # Placeholder — Module 5
NODE_HANDOFF_ROUTER = "handoff_router"            # Placeholder — Module 6
NODE_HUMAN_ESCALATION = "human_escalation"        # Placeholder — Module 6
NODE_REPORT_ASSEMBLY = "report_assembly"          # Placeholder — Module 7


# ── Helper: default agent statuses ────────────────────────────────────────────

def _initial_agent_statuses() -> dict[str, AgentStatus]:
    """Build the initial per-agent status map with all agents IDLE."""
    return {
        name: AgentStatus(agent_name=name)
        for name in [
            AGENT_ORCHESTRATOR,
            AGENT_TERM_EXTRACTION,
            AGENT_COMPLIANCE,
            AGENT_RISK_SUMMARY,
        ]
    }


# ── Graph Node Implementations ────────────────────────────────────────────────

def initialize_run(state: WorkflowState) -> dict[str, Any]:
    """Initialize the pipeline run.

    Responsibilities:
    * Set run status to INITIALIZED (already set, but explicit here).
    * Initialize per-agent statuses.
    * Record the first audit event.

    Returns only the state fields that this node modifies, as required by
    LangGraph's node contract.
    """
    logger.info("[%s] Initializing pipeline run.", state.run_id)

    start_event = make_audit_event(
        run_id=state.run_id,
        event_type=AuditEventType.RUN_STARTED,
        agent_name=AGENT_ORCHESTRATOR,
        message=f"Pipeline run started for document '{state.document_id}'.",
        metadata={"document_id": state.document_id},
    )

    agent_statuses = _initial_agent_statuses()

    # Mark orchestrator as running.
    agent_statuses[AGENT_ORCHESTRATOR] = agent_statuses[
        AGENT_ORCHESTRATOR
    ].mark_running()

    logger.info(
        "[%s] Run initialized. Agent statuses: %s",
        state.run_id,
        {k: v.status.value for k, v in agent_statuses.items()},
    )

    return {
        "run_status": WorkflowRunStatus.INITIALIZED,
        "agent_statuses": agent_statuses,
        "audit_events": [*state.audit_events, start_event],
    }


def ingest_document(state: WorkflowState) -> dict[str, Any]:
    """Ingest the deal document PDF and populate the evidence registry.

    Reads ``state.source_pdf_path`` (provided by the caller) and
    ``state.document_id``.  Calls the PDF loader, then the segmenter, builds
    the EvidenceRegistry, and returns updated state fields.

    On fatal ingestion failure (missing file, corrupt PDF), records a
    DOCUMENT_INGESTION_FAILED audit event and an error entry — but does NOT
    raise, so the LangGraph run is not aborted silently.

    Returns only the state fields that this node modifies.
    """
    logger.info("[%s] Starting document ingestion.", state.run_id)

    started_event = make_audit_event(
        run_id=state.run_id,
        event_type=AuditEventType.DOCUMENT_INGESTION_STARTED,
        agent_name=AGENT_ORCHESTRATOR,
        message=f"Document ingestion started for '{state.document_id}'.",
        metadata={"source_pdf_path": state.source_pdf_path},
    )
    audit_events = [*state.audit_events, started_event]

    # ── Guard: no path supplied ────────────────────────────────────────────────
    if not state.source_pdf_path:
        msg = "source_pdf_path is not set in WorkflowState; cannot ingest document."
        logger.warning("[%s] %s", state.run_id, msg)
        failed_event = make_audit_event(
            run_id=state.run_id,
            event_type=AuditEventType.DOCUMENT_INGESTION_FAILED,
            agent_name=AGENT_ORCHESTRATOR,
            message=msg,
        )
        return {
            "audit_events": [*audit_events, failed_event],
            "errors": {**state.errors, "ingest_document": msg},
        }

    # ── Load PDF ───────────────────────────────────────────────────────────────
    result = load_pdf(state.source_pdf_path, state.document_id)
    if not result.success:
        msg = f"PDF ingestion failed: {result.error}"
        logger.warning("[%s] %s", state.run_id, msg)
        failed_event = make_audit_event(
            run_id=state.run_id,
            event_type=AuditEventType.DOCUMENT_INGESTION_FAILED,
            agent_name=AGENT_ORCHESTRATOR,
            message=msg,
            metadata={"error": result.error},
        )
        return {
            "audit_events": [*audit_events, failed_event],
            "errors": {**state.errors, "ingest_document": msg},
        }

    # ── Segment pages into evidence snippets ───────────────────────────────────
    registry = EvidenceRegistry()
    ev_counter = 1  # Global counter across all pages for deterministic IDs.

    for page in result.pages:
        snippets = segment_page(page, evidence_id_start=ev_counter)
        for snippet in snippets:
            registry.add(snippet)
        ev_counter += len(snippets)

    # ── Emit completion audit event ────────────────────────────────────────────
    completed_event = make_audit_event(
        run_id=state.run_id,
        event_type=AuditEventType.DOCUMENT_INGESTION_COMPLETED,
        agent_name=AGENT_ORCHESTRATOR,
        message=(
            f"Document ingestion completed for '{state.document_id}'. "
            f"{result.metadata.page_count} page(s), "
            f"{len(registry)} evidence snippet(s) extracted."
        ),
        metadata={
            "page_count": result.metadata.page_count,
            "evidence_count": len(registry),
        },
    )

    logger.info(
        "[%s] Ingestion complete: %d page(s), %d evidence snippet(s).",
        state.run_id,
        result.metadata.page_count,
        len(registry),
    )

    return {
        "document_metadata": result.metadata,
        "evidence_registry": registry.to_dict(),
        "audit_events": [*audit_events, completed_event],
    }


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Build and return the compiled LangGraph StateGraph.

    After Module 2 the graph is:
        START → initialize_run → ingest_document → END

    Future modules will register additional nodes and conditional edges.

    Returns:
        A compiled LangGraph StateGraph ready for invocation.
    """
    graph = StateGraph(WorkflowState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node(NODE_INITIALIZE, initialize_run)
    graph.add_node(NODE_INGEST_DOCUMENT, ingest_document)

    # ── Future nodes (commented — to be registered in later modules) ──────────
    # graph.add_node(NODE_TERM_EXTRACTION,   term_extraction_node)
    # graph.add_node(NODE_COMPLIANCE_REVIEW, compliance_review_node)
    # graph.add_node(NODE_RISK_ANALYSIS,     risk_analysis_node)
    # graph.add_node(NODE_HANDOFF_ROUTER,    handoff_router_node)
    # graph.add_node(NODE_HUMAN_ESCALATION,  human_escalation_node)
    # graph.add_node(NODE_REPORT_ASSEMBLY,   report_assembly_node)

    # ── Edges: START → initialize_run → ingest_document → END ────────────────
    graph.add_edge(START, NODE_INITIALIZE)
    graph.add_edge(NODE_INITIALIZE, NODE_INGEST_DOCUMENT)
    graph.add_edge(NODE_INGEST_DOCUMENT, END)

    return graph.compile()


# ── Module-level compiled graph instance ─────────────────────────────────────

deal_review_graph = build_graph()
"""Pre-compiled LangGraph instance.  Import this and call ``.invoke()`` or
``.stream()`` to execute the pipeline."""
