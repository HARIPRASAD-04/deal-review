"""Final Deal Review Report model — Module 7.

The FinalDealReviewReport is the structured output produced by the
OrchestratorAgent's report assembly phase.  It aggregates the outputs of all
four specialised agents into a single, typed, auditable document.

Design principles
-----------------
* **No lending decision** — this model does not approve or reject deals.
  It reflects *pipeline completion quality* via ``ReviewStatus``, not a credit
  or commercial decision.  That distinction is deliberate: our system reviews
  deals, it does not adjudicate them.
* **Factual aggregation** — every field is populated directly from structured
  pipeline state.  No inference, no generation beyond the executive summary
  already produced by the Risk & Summary Agent.
* **Serialisable to JSON** — stored in ``state.final_report`` as a JSON string
  so that LangGraph can checkpoint/replay the state without custom serialisers.

The assembled report is written to ``state.final_report`` (``Optional[str]``)
by the ``assemble_report`` LangGraph node after all four agents have run.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class ReviewStatus(str, Enum):
    """Pipeline completion quality.

    This reflects whether the review pipeline ran successfully and whether
    any items require human intervention before the deal can proceed.

    .. important::

        This is **not** a deal approval or rejection decision.  A
        ``COMPLETED`` status means the pipeline ran cleanly — it does not
        mean the deal should be approved.  A ``FAILED`` status means the
        pipeline encountered errors — it does not mean the deal is bad.
    """

    COMPLETED = "COMPLETED"
    """All agents ran successfully; no human-review flags and no escalations."""

    COMPLETED_WITH_HUMAN_REVIEW = "COMPLETED_WITH_HUMAN_REVIEW"
    """Pipeline completed but one or more items require human review before the
    deal can proceed.  Triggered by risk findings with
    ``requires_human_review=True`` or by human-escalation items in state."""

    FAILED = "FAILED"
    """Pipeline encountered unrecoverable errors.  The report is partial or
    incomplete and must not be relied upon without investigation."""


class ComplianceSummary(BaseModel):
    """Aggregated compliance evaluation results.

    Attributes:
        total_rules:                 Total number of policy rules evaluated.
        pass_count:                  Rules that passed evaluation.
        fail_count:                  Rules that failed evaluation.
        needs_review_count:          Rules that require human review.
        insufficient_evidence_count: Rules with insufficient evidence.
        overall_status:              High-level characterisation of compliance.
    """

    total_rules: int = Field(
        default=0,
        ge=0,
        description="Total number of policy rules evaluated.",
    )
    pass_count: int = Field(
        default=0,
        ge=0,
        description="Number of rules that passed.",
    )
    fail_count: int = Field(
        default=0,
        ge=0,
        description="Number of rules that failed.",
    )
    needs_review_count: int = Field(
        default=0,
        ge=0,
        description="Number of rules requiring human review.",
    )
    insufficient_evidence_count: int = Field(
        default=0,
        ge=0,
        description="Number of rules where evidence was insufficient.",
    )
    overall_status: str = Field(
        default="NO_RULES",
        description=(
            "High-level compliance characterisation. "
            "One of: ALL_PASS, HAS_FAILURES, NEEDS_REVIEW, INSUFFICIENT, NO_RULES."
        ),
    )


class FinalDealReviewReport(BaseModel):
    """The assembled output of the full deal review pipeline.

    Produced by the ``OrchestratorAgent`` after all four specialised agents
    have run.  This is a factual aggregation — it does **not** approve or
    reject the deal.

    Attributes:
        report_id:               Stable identifier in the format ``RPT-XXXXXXXX``.
        run_id:                  The pipeline run that produced this report.
        document_id:             The deal document that was reviewed.
        generated_at:            UTC timestamp of report assembly.
        review_status:           Pipeline completion quality (not a deal decision).
        review_status_rationale: Plain-English explanation of ``review_status``.
        executive_summary:       Narrative summary from the Risk & Summary Agent.
        risk_findings:           All risk findings, ordered CRITICAL → LOW.
        compliance_summary:      Aggregated compliance counts and overall status.
        missing_information:     Unresolvable information gaps.
        follow_ups:              Recommended next steps.
        escalations:             Human-escalation items with context.
        agent_statuses:          Per-agent execution state at assembly time
                                 (serialised to plain dicts for JSON compatibility).
        audit_event_count:       Number of audit events recorded in the run.
        metadata:                Supplementary pipeline metadata.
    """

    report_id: str = Field(
        default_factory=lambda: f"RPT-{uuid4().hex[:8].upper()}",
        description="Stable report identifier.",
        examples=["RPT-A1B2C3D4"],
    )
    run_id: str = Field(
        ...,
        min_length=1,
        description="Pipeline run ID that produced this report.",
    )
    document_id: str = Field(
        ...,
        min_length=1,
        description="Document identifier for the reviewed deal.",
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when this report was assembled.",
    )
    review_status: ReviewStatus = Field(
        ...,
        description=(
            "Pipeline completion quality — not a deal approval/rejection decision. "
            "See ReviewStatus docstring for details."
        ),
    )
    review_status_rationale: str = Field(
        ...,
        min_length=1,
        description="Plain-English explanation of the review_status value.",
    )
    executive_summary: str = Field(
        default="",
        description="Narrative executive summary from the Risk & Summary Agent.",
    )
    risk_findings: list[dict[str, Any]] = Field(
        default_factory=list,
        description="All risk findings serialised as dicts, ordered CRITICAL → LOW.",
    )
    compliance_summary: ComplianceSummary = Field(
        default_factory=ComplianceSummary,
        description="Aggregated compliance evaluation results.",
    )
    missing_information: list[str] = Field(
        default_factory=list,
        description="Unresolvable information gaps identified during review.",
    )
    follow_ups: list[str] = Field(
        default_factory=list,
        description="Recommended next steps from the Risk & Summary Agent.",
    )
    escalations: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Human-escalation items with context.",
    )
    agent_statuses: dict[str, Any] = Field(
        default_factory=dict,
        description="Per-agent execution state at report assembly time (serialised).",
    )
    audit_event_count: int = Field(
        default=0,
        ge=0,
        description="Number of audit events recorded during this pipeline run.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Supplementary pipeline metadata (page count, evidence count, etc.).",
    )
