"""LangGraph orchestration graph — Module 4 update.

Graph layout after Module 4:

    START -> initialize_run -> ingest_document -> extract_terms -> review_compliance -> END

Future modules will add nodes for:

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

LLM CLIENT INJECTION
--------------------
``build_graph(llm_client=...)`` accepts an optional LLM client so that:
* Integration tests can pass a ``FakeLLMClient`` without touching settings.
* The module-level ``deal_review_graph`` (``llm_client=None``) remains safe for
  tests that do not configure an LLM — the ``extract_terms`` node exits
  gracefully when no client is supplied.
* The demo script and production code can pass a real ``GoogleLLMClient``.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph

from app.agents.compliance.compliance_review import ComplianceReviewAgent
from app.agents.extraction.term_extraction import TermExtractionAgent
from app.ingestion.evidence_registry import EvidenceRegistry
from app.ingestion.pdf_loader import load_pdf
from app.ingestion.segmenter import segment_page
from app.llm.client import LLMClient
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
NODE_INGEST_DOCUMENT = "ingest_document"
NODE_TERM_EXTRACTION = "extract_terms"
NODE_COMPLIANCE_REVIEW = "compliance_review"
NODE_HANDOFF_ROUTER = "route_handoffs"
NODE_RISK_ANALYSIS = "risk_analysis"              # Placeholder — Module 6
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


def make_extract_terms_node(llm_client: Optional[LLMClient] = None):
    """Factory that creates an ``extract_terms`` LangGraph node.

    Captures ``llm_client`` via closure so the node function has the plain
    ``(state) -> dict`` signature required by LangGraph while remaining
    fully testable through injection.

    Args:
        llm_client: A configured ``LLMClient`` instance.  If ``None``, the
                    node records a graceful skip (no LLM configured) rather
                    than raising.

    Returns:
        A node function ``extract_terms(state: WorkflowState) -> dict``.
    """

    def extract_terms(state: WorkflowState) -> dict[str, Any]:
        """Run the Term Extraction Agent over the evidence registry.

        Responsibilities:
        1. Verify the evidence registry is populated.
        2. Call ``TermExtractionAgent.extract()``.
        3. Emit diagnostic audit events for any rejected candidates.
        4. Store accepted terms in ``extracted_terms``.
        5. Update the term_extraction agent status.
        6. Emit TERM_EXTRACTION_COMPLETED or TERM_EXTRACTION_FAILED.
        """
        logger.info("[%s] Starting term extraction.", state.run_id)

        started_event = make_audit_event(
            run_id=state.run_id,
            event_type=AuditEventType.TERM_EXTRACTION_STARTED,
            agent_name=AGENT_TERM_EXTRACTION,
            message="Term extraction started.",
        )
        audit_events = [*state.audit_events, started_event]

        # ── Guard: no LLM configured ───────────────────────────────────────────
        if llm_client is None:
            msg = (
                "No LLM client configured. Term extraction skipped. "
                "Pass llm_client to build_graph() to enable extraction."
            )
            logger.warning("[%s] %s", state.run_id, msg)
            skip_event = make_audit_event(
                run_id=state.run_id,
                event_type=AuditEventType.TERM_EXTRACTION_FAILED,
                agent_name=AGENT_TERM_EXTRACTION,
                message=msg,
            )
            updated_statuses = {
                **state.agent_statuses,
                AGENT_TERM_EXTRACTION: state.agent_statuses.get(
                    AGENT_TERM_EXTRACTION, AgentStatus(agent_name=AGENT_TERM_EXTRACTION)
                ).mark_failed(msg),
            }
            return {
                "audit_events": [*audit_events, skip_event],
                "errors": {**state.errors, "extract_terms": msg},
                "agent_statuses": updated_statuses,
            }

        # ── Guard: empty evidence registry ─────────────────────────────────────
        if not state.evidence_registry:
            msg = "Evidence registry is empty. Term extraction completed with zero terms."
            logger.info("[%s] %s", state.run_id, msg)
            completed_event = make_audit_event(
                run_id=state.run_id,
                event_type=AuditEventType.TERM_EXTRACTION_COMPLETED,
                agent_name=AGENT_TERM_EXTRACTION,
                message=msg,
                metadata={"term_count": 0, "rejection_count": 0},
            )
            updated_statuses = {
                **state.agent_statuses,
                AGENT_TERM_EXTRACTION: state.agent_statuses.get(
                    AGENT_TERM_EXTRACTION, AgentStatus(agent_name=AGENT_TERM_EXTRACTION)
                ).mark_running().mark_completed(),
            }
            return {
                "extracted_terms": {},
                "audit_events": [*audit_events, completed_event],
                "agent_statuses": updated_statuses,
            }

        # ── Run extraction ─────────────────────────────────────────────────────
        agent = TermExtractionAgent(llm_client=llm_client)
        updated_statuses = {
            **state.agent_statuses,
            AGENT_TERM_EXTRACTION: state.agent_statuses.get(
                AGENT_TERM_EXTRACTION, AgentStatus(agent_name=AGENT_TERM_EXTRACTION)
            ).mark_running(),
        }

        try:
            accepted_terms, rejection_reasons = agent.extract(state.evidence_registry)
        except Exception as exc:  # noqa: BLE001
            msg = f"Term extraction failed: {exc}"
            logger.error("[%s] %s", state.run_id, msg, exc_info=True)
            failed_event = make_audit_event(
                run_id=state.run_id,
                event_type=AuditEventType.TERM_EXTRACTION_FAILED,
                agent_name=AGENT_TERM_EXTRACTION,
                message=msg,
                metadata={"error": str(exc)},
            )
            updated_statuses[AGENT_TERM_EXTRACTION] = updated_statuses[
                AGENT_TERM_EXTRACTION
            ].mark_failed(msg)
            return {
                "audit_events": [*audit_events, failed_event],
                "errors": {**state.errors, "extract_terms": msg},
                "agent_statuses": updated_statuses,
            }

        # ── Emit diagnostic events for rejections ──────────────────────────────
        for reason in rejection_reasons:
            rejection_event = make_audit_event(
                run_id=state.run_id,
                event_type=AuditEventType.GENERIC,
                agent_name=AGENT_TERM_EXTRACTION,
                message=f"[WARNING] Candidate term rejected: {reason}",
            )
            audit_events.append(rejection_event)

        # ── Build extracted_terms dict (TERM-NNN → DealTerm) ──────────────────
        extracted_terms = {term.term_id: term for term in accepted_terms}

        completed_event = make_audit_event(
            run_id=state.run_id,
            event_type=AuditEventType.TERM_EXTRACTION_COMPLETED,
            agent_name=AGENT_TERM_EXTRACTION,
            message=(
                f"Term extraction completed. "
                f"{len(accepted_terms)} term(s) accepted, "
                f"{len(rejection_reasons)} candidate(s) rejected."
            ),
            metadata={
                "term_count": len(accepted_terms),
                "rejection_count": len(rejection_reasons),
                "term_ids": list(extracted_terms.keys()),
            },
        )

        updated_statuses[AGENT_TERM_EXTRACTION] = updated_statuses[
            AGENT_TERM_EXTRACTION
        ].mark_completed()

        logger.info(
            "[%s] Term extraction complete: %d term(s) accepted, %d rejected.",
            state.run_id,
            len(accepted_terms),
            len(rejection_reasons),
        )

        return {
            "extracted_terms": extracted_terms,
            "audit_events": [*audit_events, completed_event],
            "agent_statuses": updated_statuses,
        }

    return extract_terms


# ── Compliance Review Node ────────────────────────────────────────────────────

def make_compliance_node():
    """Factory that creates a ``review_compliance`` LangGraph node.

    The compliance agent is fully deterministic (no LLM required), so this
    factory requires no arguments beyond capturing the agent class.  It uses
    the same closure pattern as ``make_extract_terms_node()`` for consistency.

    Returns:
        A node function ``review_compliance(state: WorkflowState) -> dict``.
    """

    def review_compliance(state: WorkflowState) -> dict[str, Any]:
        """Run the Compliance Review Agent over extracted terms and policy rules.

        Responsibilities:
        1. Guard: no policy rules → complete with zero results (not a failure).
        2. Guard: no extracted terms → all rules become INSUFFICIENT_EVIDENCE.
        3. Call ``ComplianceReviewAgent.review()``.
        4. Store results in ``compliance_results`` (keyed COMP-NNN).
        5. Emit audit events (STARTED / COMPLETED / FAILED).
        6. Update agent status.
        """
        logger.info("[%s] Starting compliance review.", state.run_id)

        started_event = make_audit_event(
            run_id=state.run_id,
            event_type=AuditEventType.COMPLIANCE_REVIEW_STARTED,
            agent_name=AGENT_COMPLIANCE,
            message=(
                f"Compliance review started. "
                f"{len(state.policy_rules)} policy rule(s) to evaluate."
            ),
            metadata={"rule_count": len(state.policy_rules)},
        )
        audit_events = [*state.audit_events, started_event]

        # Update compliance agent status to RUNNING.
        updated_statuses = {
            **state.agent_statuses,
            AGENT_COMPLIANCE: state.agent_statuses.get(
                AGENT_COMPLIANCE, AgentStatus(agent_name=AGENT_COMPLIANCE)
            ).mark_running(),
        }

        # ── Guard: no policy rules ─────────────────────────────────────────
        if not state.policy_rules:
            msg = "No policy rules provided. Compliance review completed with zero results."
            logger.info("[%s] %s", state.run_id, msg)
            completed_event = make_audit_event(
                run_id=state.run_id,
                event_type=AuditEventType.COMPLIANCE_REVIEW_COMPLETED,
                agent_name=AGENT_COMPLIANCE,
                message=msg,
                metadata={
                    "result_count": 0,
                    "pass_count": 0,
                    "fail_count": 0,
                    "needs_review_count": 0,
                    "insufficient_evidence_count": 0,
                },
            )
            updated_statuses[AGENT_COMPLIANCE] = updated_statuses[
                AGENT_COMPLIANCE
            ].mark_completed()
            return {
                "compliance_results": {},
                "audit_events": [*audit_events, completed_event],
                "agent_statuses": updated_statuses,
            }

        # ── Run compliance review ──────────────────────────────────────────
        agent = ComplianceReviewAgent()
        try:
            results = agent.review(
                policy_rules=state.policy_rules,
                extracted_terms=state.extracted_terms,
                evidence_registry=state.evidence_registry,
            )
        except Exception as exc:  # noqa: BLE001
            msg = f"Compliance review failed with unexpected error: {exc}"
            logger.error("[%s] %s", state.run_id, msg, exc_info=True)
            failed_event = make_audit_event(
                run_id=state.run_id,
                event_type=AuditEventType.COMPLIANCE_REVIEW_FAILED,
                agent_name=AGENT_COMPLIANCE,
                message=msg,
                metadata={"error": str(exc)},
            )
            updated_statuses[AGENT_COMPLIANCE] = updated_statuses[
                AGENT_COMPLIANCE
            ].mark_failed(msg)
            return {
                "audit_events": [*audit_events, failed_event],
                "errors": {**state.errors, "review_compliance": msg},
                "agent_statuses": updated_statuses,
            }

        # ── Build compliance_results dict (COMP-NNN → ComplianceResult) ───
        from app.models.compliance import ComplianceStatus
        compliance_results = {r.compliance_id: r for r in results}

        pass_count = sum(1 for r in results if r.status == ComplianceStatus.PASS)
        fail_count = sum(1 for r in results if r.status == ComplianceStatus.FAIL)
        needs_review_count = sum(
            1 for r in results if r.status == ComplianceStatus.NEEDS_HUMAN_REVIEW
        )
        insufficient_count = sum(
            1 for r in results if r.status == ComplianceStatus.INSUFFICIENT_EVIDENCE
        )

        completed_event = make_audit_event(
            run_id=state.run_id,
            event_type=AuditEventType.COMPLIANCE_REVIEW_COMPLETED,
            agent_name=AGENT_COMPLIANCE,
            message=(
                f"Compliance review completed. "
                f"{len(results)} result(s): "
                f"{pass_count} PASS, {fail_count} FAIL, "
                f"{needs_review_count} NEEDS_HUMAN_REVIEW, "
                f"{insufficient_count} INSUFFICIENT_EVIDENCE."
            ),
            metadata={
                "result_count": len(results),
                "pass_count": pass_count,
                "fail_count": fail_count,
                "needs_review_count": needs_review_count,
                "insufficient_evidence_count": insufficient_count,
                "comp_ids": list(compliance_results.keys()),
            },
        )

        updated_statuses[AGENT_COMPLIANCE] = updated_statuses[
            AGENT_COMPLIANCE
        ].mark_completed()

        logger.info(
            "[%s] Compliance review complete: %d PASS, %d FAIL, "
            "%d NEEDS_HUMAN_REVIEW, %d INSUFFICIENT_EVIDENCE.",
            state.run_id, pass_count, fail_count, needs_review_count, insufficient_count,
        )

        # ── Generate clarification handoffs for INSUFFICIENT_EVIDENCE rules ──
        clarification_handoffs = agent.generate_clarification_handoffs(
            results=results,
            policy_rules=state.policy_rules,
            evidence_registry=state.evidence_registry,
        )
        updated_pending = [*state.pending_handoffs, *clarification_handoffs]

        for h in clarification_handoffs:
            audit_events.append(
                make_audit_event(
                    run_id=state.run_id,
                    event_type=AuditEventType.HANDOFF_CREATED,
                    agent_name=AGENT_COMPLIANCE,
                    message=f"Created clarification handoff '{h.handoff_id}' for rule {h.rule_ids}.",
                    metadata={"handoff_id": h.handoff_id, "rule_ids": h.rule_ids},
                )
            )

        return {
            "compliance_results": compliance_results,
            "pending_handoffs": updated_pending,
            "audit_events": [*audit_events, completed_event],
            "agent_statuses": updated_statuses,
        }

    return review_compliance


def make_route_handoffs_node(llm_client: Optional[LLMClient] = None):
    """Factory that creates a ``route_handoffs`` LangGraph node.

    Args:
        llm_client: Optional LLM client passed to HandoffRouter for targeted clarification.

    Returns:
        A node function ``route_handoffs(state: WorkflowState) -> dict``.
    """
    def route_handoffs(state: WorkflowState) -> dict[str, Any]:
        """Route pending handoffs in state via HandoffRouter."""
        from app.agents.orchestrator.router import HandoffRouter

        logger.info("[%s] Routing pending handoffs.", state.run_id)
        router = HandoffRouter()
        new_state = router.route(state, llm_client=llm_client, max_attempts=1)

        return {
            "pending_handoffs": new_state.pending_handoffs,
            "resolved_handoffs": new_state.resolved_handoffs,
            "extracted_terms": new_state.extracted_terms,
            "compliance_results": new_state.compliance_results,
            "clarification_attempts": new_state.clarification_attempts,
            "escalations": new_state.escalations,
            "audit_events": new_state.audit_events,
        }

    return route_handoffs


def build_graph(llm_client: Optional[LLMClient] = None):
    """Build and return the compiled LangGraph StateGraph.

    After Module 5 the graph is:
        START -> initialize_run -> ingest_document -> extract_terms -> review_compliance -> route_handoffs -> END

    Args:
        llm_client: Optional LLM client injected into ``extract_terms`` and
                    ``route_handoffs`` nodes.  When ``None`` (the default), nodes exit
                    gracefully without calling any LLM.

    Returns:
        A compiled LangGraph StateGraph ready for invocation.
    """
    graph = StateGraph(WorkflowState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node(NODE_INITIALIZE, initialize_run)
    graph.add_node(NODE_INGEST_DOCUMENT, ingest_document)
    graph.add_node(NODE_TERM_EXTRACTION, make_extract_terms_node(llm_client))
    graph.add_node(NODE_COMPLIANCE_REVIEW, make_compliance_node())
    graph.add_node(NODE_HANDOFF_ROUTER, make_route_handoffs_node(llm_client))

    # ── Edges ─────────────────────────────────────────────────────────────────
    graph.add_edge(START, NODE_INITIALIZE)
    graph.add_edge(NODE_INITIALIZE, NODE_INGEST_DOCUMENT)
    graph.add_edge(NODE_INGEST_DOCUMENT, NODE_TERM_EXTRACTION)
    graph.add_edge(NODE_TERM_EXTRACTION, NODE_COMPLIANCE_REVIEW)
    graph.add_edge(NODE_COMPLIANCE_REVIEW, NODE_HANDOFF_ROUTER)
    graph.add_edge(NODE_HANDOFF_ROUTER, END)

    return graph.compile()


# ── Module-level compiled graph instance ─────────────────────────────────────

deal_review_graph = build_graph()
"""Pre-compiled LangGraph instance with no LLM client (safe for existing tests).
Import this and call ``.invoke()`` or ``.stream()`` to execute the pipeline.
For term extraction with a real LLM, call ``build_graph(llm_client=...)``."""
