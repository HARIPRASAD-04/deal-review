"""Application service — Module 8.

``DealReviewService`` is the single callable boundary between external callers
(the UI, CLI scripts, integration tests) and the backend pipeline.

Architecture position
---------------------
::

    [UI / CLI / tests]
          |
    DealReviewService.run(DealReviewRequest)
          |
    ┌─────────────────────────────────────────┐
    │  1. parse_policy_pdf()  →  PolicyRules  │   (ingestion infrastructure)
    │  2. build_graph(llm_client)             │   (LangGraph orchestration)
    │  3. graph.invoke(WorkflowState)         │   (4-agent pipeline)
    │  4. OrchestratorAgent.assemble_report() │   (inside graph node)
    └─────────────────────────────────────────┘
          |
    DealReviewResponse (report, warnings, errors)

Design decisions
----------------
* The UI must never import from ``app.orchestration.graph`` directly.
  It calls ``DealReviewService.run()``.  This keeps the UI genuinely
  presentation-only.
* Policy parse failure is a hard stop — the pipeline is NOT invoked with
  empty or wrong policy rules.  ``DealReviewResponse.success`` will be
  ``False`` and ``error`` will describe the failure.
* ``DealReviewService`` does not contain any agent logic.  It wires
  infrastructure together; decisions remain in the agents.
* LLM client injection: the caller supplies the ``LLMClient`` (or ``None``
  for demo/test mode).  The service never reads ``settings`` itself.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4

from app.ingestion.policy_parser import PolicyParseResult, parse_policy_pdf
from app.llm.client import LLMClient
from app.models.report import FinalDealReviewReport
from app.models.state import WorkflowState
from app.orchestration.graph import build_graph

logger = logging.getLogger(__name__)


@dataclass
class DealReviewRequest:
    """Input contract for a deal review run.

    Attributes:
        deal_pdf_path:   Absolute path to the deal PDF file.
        policy_pdf_path: Absolute path to the policy PDF file.
                         Both PDFs are required — there is no bundled fallback.
        document_id:     Caller-supplied document identifier.  Auto-generated
                         as ``DEAL-<uuid8>`` when not provided.
        llm_client:      A configured ``LLMClient`` for term extraction, summary
                         generation, and policy parsing.  ``None`` uses
                         ``FakeLLMClient`` for demo/test runs.
    """

    deal_pdf_path: str
    policy_pdf_path: str
    document_id: str = field(default_factory=lambda: f"DEAL-{uuid4().hex[:8].upper()}")
    llm_client: Optional[LLMClient] = None


@dataclass
class DealReviewResponse:
    """Output contract for a deal review run.

    Attributes:
        success:               True when the pipeline ran to completion without
                               a fatal error.  Note: ``success=True`` does not
                               mean the deal is compliant — it means the review
                               pipeline completed and produced a report.
        final_state:           The terminal ``WorkflowState`` after all agents
                               have run.  ``None`` if the pipeline was not
                               invoked (e.g. policy parse failure).
        report:                The assembled ``FinalDealReviewReport``.  ``None``
                               if the pipeline did not complete report assembly.
        error:                 Top-level error description; ``None`` on success.
        policy_parse_warnings: Ambiguous policy statements surfaced from the
                               policy PDF parser.  Non-empty warnings do NOT
                               block a successful run — they indicate statements
                               that could not be safely codified as rules.
    """

    success: bool
    final_state: Optional[WorkflowState] = None
    report: Optional[FinalDealReviewReport] = None
    error: Optional[str] = None
    policy_parse_warnings: list[str] = field(default_factory=list)


class DealReviewService:
    """Orchestrates a complete deal review from two PDFs to a final report.

    Usage::

        service = DealReviewService()
        response = service.run(DealReviewRequest(
            deal_pdf_path="/path/to/deal.pdf",
            policy_pdf_path="/path/to/policy.pdf",
            llm_client=llm,
        ))
        if response.success:
            print(response.report.review_status)
        else:
            print(response.error)
    """

    def run(self, request: DealReviewRequest) -> DealReviewResponse:
        """Run the full four-agent deal review pipeline.

        Steps:
        1. Parse the policy PDF into ``PolicyRule`` objects.
           On failure, return immediately — pipeline is NOT invoked.
        2. Build the LangGraph pipeline graph.
        3. Invoke the graph with ``WorkflowState``.
        4. Deserialise the final report from state.
        5. Return ``DealReviewResponse``.

        Args:
            request: A ``DealReviewRequest`` with both PDF paths and optional
                     LLM client.

        Returns:
            A ``DealReviewResponse``.  Check ``response.success`` and
            ``response.error`` before using ``response.report``.
        """
        logger.info(
            "DealReviewService.run() — document_id='%s', deal='%s', policy='%s'.",
            request.document_id,
            request.deal_pdf_path,
            request.policy_pdf_path,
        )

        # ── Step 1: Parse policy PDF ───────────────────────────────────────────
        logger.info("Parsing policy PDF: %s", request.policy_pdf_path)
        parse_result: PolicyParseResult = parse_policy_pdf(
            pdf_path=request.policy_pdf_path,
            llm_client=request.llm_client,
        )

        if not parse_result.success:
            msg = f"Policy ingestion failed: {parse_result.error}"
            logger.error(msg)
            return DealReviewResponse(
                success=False,
                error=msg,
                policy_parse_warnings=parse_result.ambiguous_statements,
            )

        logger.info(
            "Policy PDF parsed successfully: %d rule(s), %d ambiguous statement(s).",
            len(parse_result.rules),
            len(parse_result.ambiguous_statements),
        )

        # ── Step 2: Build graph ────────────────────────────────────────────────
        graph = build_graph(llm_client=request.llm_client)

        # ── Step 3: Invoke pipeline ────────────────────────────────────────────
        initial_state = WorkflowState(
            document_id=request.document_id,
            source_pdf_path=request.deal_pdf_path,
            policy_rules=parse_result.rules,
        )

        try:
            raw_result = graph.invoke(initial_state)
        except Exception as exc:  # noqa: BLE001
            msg = f"Pipeline invocation failed unexpectedly: {exc}"
            logger.error(msg, exc_info=True)
            return DealReviewResponse(
                success=False,
                error=msg,
                policy_parse_warnings=parse_result.ambiguous_statements,
            )

        # ── Step 4: Reconstruct final state ───────────────────────────────────
        final_state: WorkflowState = (
            WorkflowState(**raw_result) if isinstance(raw_result, dict) else raw_result
        )

        # ── Step 5: Deserialise report ─────────────────────────────────────────
        report: Optional[FinalDealReviewReport] = None
        if final_state.final_report:
            try:
                report = FinalDealReviewReport.model_validate_json(final_state.final_report)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to deserialise FinalDealReviewReport: %s", exc)
                return DealReviewResponse(
                    success=False,
                    final_state=final_state,
                    error=f"Report deserialisation failed: {exc}",
                    policy_parse_warnings=parse_result.ambiguous_statements,
                )

        success = report is not None and not final_state.errors

        logger.info(
            "DealReviewService.run() completed — success=%s, run_status=%s, "
            "report=%s.",
            success,
            final_state.run_status.value,
            report.report_id if report else "None",
        )

        return DealReviewResponse(
            success=success,
            final_state=final_state,
            report=report,
            error=(
                "; ".join(final_state.errors.values()) if final_state.errors and not success
                else None
            ),
            policy_parse_warnings=parse_result.ambiguous_statements,
        )
