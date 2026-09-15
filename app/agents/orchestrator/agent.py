"""OrchestratorAgent — Module 7.

The OrchestratorAgent is the central workflow authority of the Multi-Agent Deal
Review Pipeline.  It is the fourth and final named agent in the system.

Architecture position
---------------------
::

    Pipeline graph (LangGraph)
      initialize_run          [Orchestrator-owned node]
      ingest_document         [Orchestrator-owned node]
      extract_terms           [TermExtractionAgent node]
      review_compliance       [ComplianceReviewAgent node]
      route_handoffs          [Orchestrator-owned node → calls self.route_handoffs()]
      risk_summary            [RiskSummaryAgent node]
      assemble_report         [Orchestrator-owned node → calls self.assemble_report()]

Responsibilities
----------------
1. **Workflow state inspection** — ``inspect_state(state)`` reads agent statuses,
   errors, and escalations to surface what has run, failed, or is pending.
2. **Dependency checking** — ``check_dependencies(state)`` verifies pipeline
   prerequisites are met before each stage (e.g. evidence registry populated
   before term extraction, policy rules present before compliance review).
3. **Retry policy** — ``should_retry(agent_name, state)`` decides whether a
   failed agent is eligible for another attempt, based on per-agent retry counts
   and a configurable maximum.
4. **Handoff routing** — ``route_handoffs(state, ...)`` is the Orchestrator's
   workflow routing responsibility.  It delegates the per-handoff clarification
   loop to the internal ``HandoffRouter`` subcomponent, which implements the
   Compliance → Extraction feedback cycle from Module 5.
5. **Escalation management** — ``collect_escalations(state)`` aggregates all
   human-escalation items from handoffs, risk findings, and state escalations
   into a unified, deduplicated list.
6. **Final report assembly** — ``assemble_report(state)`` constructs a
   ``FinalDealReviewReport`` from all four agents' outputs.  This is the primary
   Module 7 deliverable.

Internal subcomponents
----------------------
* ``HandoffRouter`` (``app.agents.orchestrator.router``) — handles the
  clarification feedback loop between Compliance Review and Term Extraction.
  The Orchestrator owns it as a dependency; graph nodes call the Orchestrator,
  which in turn calls the Router.  The Router is not a separate orchestration
  authority.

What OrchestratorAgent does NOT do
------------------------------------
* It does not extract terms (``TermExtractionAgent``).
* It does not evaluate compliance rules (``ComplianceReviewAgent``).
* It does not derive risk findings (``RiskSummaryAgent``).
* It does not use an LLM — all Orchestrator logic is deterministic.
* It does not approve or reject deals.  ``ReviewStatus`` describes pipeline
  completion quality, never a lending or commercial decision.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.agents.orchestrator.router import HandoffRouter
from app.llm.client import LLMClient
from app.models.compliance import ComplianceStatus
from app.models.report import ComplianceSummary, FinalDealReviewReport, ReviewStatus
from app.models.risk import RiskSeverity
from app.models.state import AgentStatusEnum, WorkflowState

logger = logging.getLogger(__name__)

# Severity ordering used for sorting risk findings in the report.
_SEVERITY_RANK: dict[RiskSeverity, int] = {
    RiskSeverity.CRITICAL: 4,
    RiskSeverity.HIGH: 3,
    RiskSeverity.MEDIUM: 2,
    RiskSeverity.LOW: 1,
}


class OrchestratorAgent:
    """Central workflow authority and final report assembler.

    The OrchestratorAgent coordinates all pipeline agents and owns the
    ``route_handoffs`` and ``assemble_report`` LangGraph nodes.
    ``HandoffRouter`` is an internal subcomponent — not a peer agent.

    Args:
        max_retries: Maximum retry attempts per agent (default 2).
    """

    AGENT_NAME: str = "orchestrator"

    def __init__(self, max_retries: int = 2) -> None:
        self._max_retries = max_retries
        # HandoffRouter is an internal subcomponent of the Orchestrator.
        # The graph delegates routing to the Orchestrator, which in turn
        # delegates the per-handoff logic to the Router.
        self.router = HandoffRouter()

    # ── 1. Workflow State Inspection ──────────────────────────────────────────

    def inspect_state(self, state: WorkflowState) -> dict:
        """Return a structured summary of the current workflow state.

        Provides a lightweight snapshot for debugging, observability, and
        routing decisions without exposing the full ``WorkflowState``.

        Args:
            state: Current pipeline state.

        Returns:
            dict with keys:
            - ``run_id``, ``run_status``
            - ``agents``: ``{agent_name: status_value}``
            - ``errors``: ``{node_name: error_message}``
            - ``pending_handoffs``, ``resolved_handoffs``, ``escalations`` (counts)
            - ``risk_findings_count``, ``missing_information_count``
            - ``compliance_results_count``, ``extracted_terms_count``
        """
        return {
            "run_id": state.run_id,
            "run_status": state.run_status.value,
            "agents": {
                name: status.status.value
                for name, status in state.agent_statuses.items()
            },
            "errors": dict(state.errors),
            "pending_handoffs": len(state.pending_handoffs),
            "resolved_handoffs": len(state.resolved_handoffs),
            "escalations": len(state.escalations),
            "risk_findings_count": len(state.risk_findings),
            "missing_information_count": len(state.missing_information),
            "compliance_results_count": len(state.compliance_results),
            "extracted_terms_count": len(state.extracted_terms),
        }

    # ── 2. Dependency Checking ────────────────────────────────────────────────

    def check_dependencies(self, state: WorkflowState) -> list[str]:
        """Return a list of unmet pipeline dependency descriptions.

        Each entry describes a missing prerequisite in human-readable form.
        An empty list means all currently checkable dependencies are satisfied.

        Checks performed:
        - Evidence registry must be non-empty before term extraction can run.
        - At least one policy rule must be configured for compliance to be meaningful.

        Args:
            state: Current pipeline state.

        Returns:
            List of unmet dependency descriptions.  Empty if all met.
        """
        issues: list[str] = []

        if not state.evidence_registry:
            issues.append(
                "Evidence registry is empty. "
                "Document ingestion must complete successfully before term extraction."
            )

        if not state.policy_rules:
            issues.append(
                "No policy rules configured. "
                "Compliance review requires at least one policy rule to evaluate."
            )

        return issues

    # ── 3. Retry Policy ───────────────────────────────────────────────────────

    def should_retry(
        self,
        agent_name: str,
        state: WorkflowState,
        max_retries: Optional[int] = None,
    ) -> bool:
        """Return True if the named agent is eligible for another retry attempt.

        The agent must currently have a FAILED status and must not have
        exhausted its retry budget.

        Args:
            agent_name:  Canonical agent name (e.g. ``"term_extraction"``).
            state:       Current pipeline state.
            max_retries: Per-call override; falls back to ``self._max_retries``.

        Returns:
            True if retry is eligible; False if limit reached, agent not found,
            or agent is not in a FAILED state.
        """
        limit = max_retries if max_retries is not None else self._max_retries
        agent_status = state.agent_statuses.get(agent_name)

        if agent_status is None:
            return False
        if agent_status.status != AgentStatusEnum.FAILED:
            return False

        attempts = state.retry_counts.get(agent_name, 0)
        return attempts < limit

    # ── 4. Handoff Routing (delegates to internal HandoffRouter) ─────────────

    def route_handoffs(
        self,
        state: WorkflowState,
        llm_client: Optional[LLMClient] = None,
        max_attempts: int = 1,
    ) -> WorkflowState:
        """Route pending handoffs via the internal HandoffRouter subcomponent.

        Routing is a workflow responsibility owned by the Orchestrator.  The
        ``HandoffRouter`` subcomponent implements the per-handoff clarification
        logic (Compliance → Extraction feedback loop from Module 5).  The
        Orchestrator exposes routing as its own operation, so graph nodes call
        the Orchestrator rather than the Router directly.

        Args:
            state:        Current pipeline state.
            llm_client:   Optional LLM client for targeted term clarification.
            max_attempts: Maximum clarification attempts per rule before escalating
                          to human review.

        Returns:
            Updated ``WorkflowState`` with handoffs routed and escalations updated.
        """
        logger.debug(
            "[%s] Routing %d pending handoff(s) via HandoffRouter.",
            self.AGENT_NAME,
            len(state.pending_handoffs),
        )
        return self.router.route(state, llm_client=llm_client, max_attempts=max_attempts)

    # ── 5. Escalation Management ──────────────────────────────────────────────

    def collect_escalations(self, state: WorkflowState) -> list[dict]:
        """Aggregate all human-escalation items from state into a unified list.

        Sources combined:
        1. ``state.escalations`` — from HandoffRouter loop-limit escalations.
        2. Risk findings with ``requires_human_review=True`` — added as
           escalation context entries if not already represented.

        The list is deduplicated by ``handoff_id`` / ``escalation_id``.

        Args:
            state: Current pipeline state.

        Returns:
            Unified, deduplicated list of escalation context dicts.
        """
        escalations: list[dict] = list(state.escalations)
        seen_ids: set[str] = {
            str(e.get("handoff_id", e.get("escalation_id", "")))
            for e in escalations
        }

        for finding in state.risk_findings:
            if finding.requires_human_review:
                esc_id = f"risk-{finding.risk_id}"
                if esc_id not in seen_ids:
                    seen_ids.add(esc_id)
                    escalations.append(
                        {
                            "escalation_id": esc_id,
                            "risk_id": finding.risk_id,
                            "reason": finding.description,
                            "recommended_action": finding.recommended_action,
                            "severity": finding.severity.value,
                            "status": "REQUIRES_HUMAN_REVIEW",
                        }
                    )

        return escalations

    # ── 6. Final Report Assembly ──────────────────────────────────────────────

    def assemble_report(self, state: WorkflowState) -> FinalDealReviewReport:
        """Assemble the FinalDealReviewReport from all pipeline outputs.

        This is the OrchestratorAgent's primary Module 7 deliverable.  It reads
        the structured outputs of all four agents from ``WorkflowState`` and
        constructs a typed, auditable report — without making any lending or
        approval decisions.

        ``ReviewStatus`` derivation (pipeline quality, NOT a loan decision):

        +---------------------------------+------------------------------------+
        | Condition                       | ReviewStatus                       |
        +=================================+====================================+
        | ``state.errors`` non-empty      | ``FAILED``                         |
        +---------------------------------+------------------------------------+
        | Any finding with                | ``COMPLETED_WITH_HUMAN_REVIEW``    |
        | ``requires_human_review=True``, |                                    |
        | OR escalations non-empty        |                                    |
        +---------------------------------+------------------------------------+
        | Otherwise                       | ``COMPLETED``                      |
        +---------------------------------+------------------------------------+

        Args:
            state: The final ``WorkflowState`` after all agents have run.

        Returns:
            A fully populated ``FinalDealReviewReport``.
        """
        # ── Derive ReviewStatus ────────────────────────────────────────────────
        if state.errors:
            review_status = ReviewStatus.FAILED
            rationale = (
                f"Pipeline encountered {len(state.errors)} unrecoverable error(s). "
                "Report may be partial. Error(s): "
                + "; ".join(state.errors.values())
            )
        else:
            human_review_findings = [
                f for f in state.risk_findings if f.requires_human_review
            ]
            if human_review_findings or state.escalations:
                review_status = ReviewStatus.COMPLETED_WITH_HUMAN_REVIEW
                parts: list[str] = []
                if human_review_findings:
                    parts.append(
                        f"{len(human_review_findings)} finding(s) require human review"
                    )
                if state.escalations:
                    parts.append(
                        f"{len(state.escalations)} item(s) escalated to human"
                    )
                rationale = ". ".join(parts) + "."
            else:
                review_status = ReviewStatus.COMPLETED
                n = len(state.risk_findings)
                rationale = (
                    f"Pipeline completed successfully. "
                    f"{n} risk finding(s) identified; no human review required."
                )

        # ── Build ComplianceSummary ────────────────────────────────────────────
        compliance_summary = self._build_compliance_summary(state)

        # ── Order risk findings CRITICAL → LOW ────────────────────────────────
        ordered_findings = sorted(
            state.risk_findings,
            key=lambda f: _SEVERITY_RANK.get(f.severity, 0),
            reverse=True,
        )
        # Serialise findings to plain dicts for JSON-safe storage
        risk_findings_dicts = [f.model_dump() for f in ordered_findings]

        # ── Collect escalations ────────────────────────────────────────────────
        all_escalations = self.collect_escalations(state)

        # ── Serialise agent statuses ───────────────────────────────────────────
        agent_statuses_serialised: dict = {
            name: {
                "status": s.status.value,
                "attempt_count": s.attempt_count,
                "last_error": s.last_error,
            }
            for name, s in state.agent_statuses.items()
        }

        # ── Build metadata ─────────────────────────────────────────────────────
        metadata: dict = {
            "evidence_count": len(state.evidence_registry),
            "extracted_terms_count": len(state.extracted_terms),
            "compliance_results_count": len(state.compliance_results),
            "resolved_handoffs_count": len(state.resolved_handoffs),
        }
        if state.document_metadata:
            metadata["page_count"] = state.document_metadata.page_count
            metadata["file_name"] = state.document_metadata.filename

        return FinalDealReviewReport(
            run_id=state.run_id,
            document_id=state.document_id,
            review_status=review_status,
            review_status_rationale=rationale,
            executive_summary=state.executive_summary or "",
            risk_findings=risk_findings_dicts,
            compliance_summary=compliance_summary,
            missing_information=list(state.missing_information),
            follow_ups=list(state.follow_ups),
            escalations=all_escalations,
            agent_statuses=agent_statuses_serialised,
            audit_event_count=len(state.audit_events),
            metadata=metadata,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_compliance_summary(self, state: WorkflowState) -> ComplianceSummary:
        """Derive ``ComplianceSummary`` from ``state.compliance_results``."""
        results = list(state.compliance_results.values())
        total = len(results)

        if total == 0:
            return ComplianceSummary(total_rules=0, overall_status="NO_RULES")

        pass_count = sum(1 for r in results if r.status == ComplianceStatus.PASS)
        fail_count = sum(1 for r in results if r.status == ComplianceStatus.FAIL)
        needs_review = sum(
            1 for r in results if r.status == ComplianceStatus.NEEDS_HUMAN_REVIEW
        )
        insufficient = sum(
            1 for r in results if r.status == ComplianceStatus.INSUFFICIENT_EVIDENCE
        )

        if fail_count > 0:
            overall = "HAS_FAILURES"
        elif needs_review > 0:
            overall = "NEEDS_REVIEW"
        elif insufficient > 0:
            overall = "INSUFFICIENT"
        else:
            overall = "ALL_PASS"

        return ComplianceSummary(
            total_rules=total,
            pass_count=pass_count,
            fail_count=fail_count,
            needs_review_count=needs_review,
            insufficient_evidence_count=insufficient,
            overall_status=overall,
        )
