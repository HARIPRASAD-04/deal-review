"""Shared workflow state model.

The WorkflowState is the single source of truth for a pipeline run.  It is
the object that LangGraph passes between nodes and that the Orchestrator reads
to make routing decisions.

Design principles:
* Prefer typed objects over raw dicts — agents must reference models from this
  package, not arbitrary JSON blobs.
* Prefer ID references over embedded objects — e.g. compliance results reference
  ``evidence_ids`` rather than embedding EvidenceSnippet objects.
* State must be serialisable to JSON so runs can be checkpointed / replayed.
* The state is a TypedDict-compatible structure for LangGraph whilst also
  supporting Pydantic validation for integrity checks.

AgentStatus tracks per-agent execution state so the Orchestrator knows which
agents have run, failed, or are waiting.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from app.models.audit import AuditEvent
from app.models.compliance import ComplianceResult
from app.models.document import DocumentMetadata
from app.models.evidence import EvidenceSnippet
from app.models.handoff import Handoff
from app.models.policy import PolicyRule
from app.models.risk import RiskFinding
from app.models.terms import DealTerm


# ── Agent Status ──────────────────────────────────────────────────────────────

class AgentStatusEnum(str, Enum):
    """Execution state of a single agent within a pipeline run."""

    IDLE = "IDLE"
    """Agent has not been invoked yet."""

    PENDING = "PENDING"
    """Orchestrator has queued the agent for execution."""

    RUNNING = "RUNNING"
    """Agent is currently executing."""

    COMPLETED = "COMPLETED"
    """Agent finished successfully."""

    FAILED = "FAILED"
    """Agent encountered an unrecoverable error."""

    RETRYING = "RETRYING"
    """Orchestrator is retrying the agent after a failure."""

    WAITING_FOR_REVIEW = "WAITING_FOR_REVIEW"
    """Agent is blocked waiting for human input."""

    BLOCKED = "BLOCKED"
    """Agent cannot proceed due to unresolved dependencies."""


class AgentStatus(BaseModel):
    """Execution state of a single agent within a pipeline run.

    Attributes:
        agent_name:    Canonical agent identifier.
        status:        Current execution state.
        attempt_count: How many times this agent has been invoked (for retry
                       tracking).
        last_error:    Description of the most recent error, if any.
        started_at:    UTC timestamp when the agent last started.
        completed_at:  UTC timestamp when the agent last completed/failed.
    """

    agent_name: str = Field(
        ...,
        min_length=1,
        description="Canonical agent identifier.",
    )
    status: AgentStatusEnum = Field(
        default=AgentStatusEnum.IDLE,
        description="Current execution state.",
    )
    attempt_count: int = Field(
        default=0,
        ge=0,
        description="Number of times this agent has been invoked.",
    )
    last_error: Optional[str] = Field(
        default=None,
        description="Description of the most recent error, if any.",
    )
    started_at: Optional[datetime] = Field(
        default=None,
        description="UTC timestamp when the agent last started.",
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="UTC timestamp when the agent last completed or failed.",
    )

    def mark_running(self) -> "AgentStatus":
        """Return a new AgentStatus representing the agent starting execution."""
        return self.model_copy(
            update={
                "status": AgentStatusEnum.RUNNING,
                "attempt_count": self.attempt_count + 1,
                "started_at": datetime.now(timezone.utc),
                "completed_at": None,
                "last_error": None,
            }
        )

    def mark_completed(self) -> "AgentStatus":
        """Return a new AgentStatus representing successful completion."""
        return self.model_copy(
            update={
                "status": AgentStatusEnum.COMPLETED,
                "completed_at": datetime.now(timezone.utc),
            }
        )

    def mark_failed(self, error: str) -> "AgentStatus":
        """Return a new AgentStatus representing a failure."""
        return self.model_copy(
            update={
                "status": AgentStatusEnum.FAILED,
                "completed_at": datetime.now(timezone.utc),
                "last_error": error,
            }
        )

    def mark_retrying(self) -> "AgentStatus":
        """Return a new AgentStatus representing a retry in progress."""
        return self.model_copy(
            update={
                "status": AgentStatusEnum.RETRYING,
                "completed_at": None,
            }
        )


# ── Workflow Run Status ────────────────────────────────────────────────────────

class WorkflowRunStatus(str, Enum):
    """Overall status of the pipeline run."""

    INITIALIZED = "INITIALIZED"
    EXTRACTING_TERMS = "EXTRACTING_TERMS"
    REVIEWING_COMPLIANCE = "REVIEWING_COMPLIANCE"
    ANALYZING_RISKS = "ANALYZING_RISKS"
    RESOLVING_HANDOFF = "RESOLVING_HANDOFF"
    AWAITING_HUMAN = "AWAITING_HUMAN"
    GENERATING_REPORT = "GENERATING_REPORT"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# ── Shared Workflow State ──────────────────────────────────────────────────────

class WorkflowState(BaseModel):
    """Shared pipeline state passed between all LangGraph nodes.

    This is the single source of truth for a deal review run.  Every agent
    reads from and writes to this state object through the Orchestrator.

    Field naming follows the structured ID conventions used throughout:
    * ``evidence_registry``   — keyed by ``EV-NNN``
    * ``extracted_terms``     — keyed by ``TERM-NNN``
    * ``compliance_results``  — keyed by ``COMP-NNN``
    * ``agent_statuses``      — keyed by agent name

    Attributes:
        run_id:              Unique identifier for this pipeline run.
        document_id:         Identifier of the deal document being reviewed.
        run_status:          Overall status of the pipeline run.
        evidence_registry:   All evidence snippets indexed by EV-NNN.
        policy_rules:        Ordered list of policy rules to evaluate.
        extracted_terms:     All extracted deal terms indexed by TERM-NNN.
        compliance_results:  All compliance decisions indexed by COMP-NNN.
        risk_findings:       Prioritised list of risk findings.
        missing_information: Plain-text list of unresolvable gaps and ambiguities
                             (populated by Risk & Summary Agent).
        follow_ups:          Recommended next steps from the Risk & Summary Agent.
        executive_summary:   Narrative executive summary (Risk & Summary Agent output).
                             Consumed by Module 7 when assembling the final report.
        pending_handoffs:    Handoffs awaiting Orchestrator routing.
        resolved_handoffs:   Handoffs that have been resolved or rejected.
        escalations:         Human-escalation items with context.
        agent_statuses:      Per-agent execution state.
        audit_events:        Ordered, immutable audit log.
        errors:              Unrecoverable errors keyed by agent name.
        retry_counts:        Per-agent retry counter.
        final_report:        The assembled final report (set by Module 7 Orchestrator).
        created_at:          UTC timestamp when this run was created.
        updated_at:          UTC timestamp of the last state update.
    """

    run_id: str = Field(
        default_factory=lambda: f"RUN-{uuid4().hex[:8].upper()}",
        description="Unique pipeline run identifier.",
    )
    document_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the deal document under review.",
    )
    source_pdf_path: Optional[str] = Field(
        default=None,
        description=(
            "Absolute or relative path to the source PDF file. "
            "Provided by the caller; consumed by the ingest_document node. "
            "None means no PDF has been supplied for this run."
        ),
    )
    run_status: WorkflowRunStatus = Field(
        default=WorkflowRunStatus.INITIALIZED,
        description="Overall status of the pipeline run.",
    )
    document_metadata: Optional[DocumentMetadata] = Field(
        default=None,
        description=(
            "Structured metadata for the ingested deal document. "
            "Populated by the ingest_document node after successful ingestion."
        ),
    )

    # ── Evidence & Terms ──────────────────────────────────────────────────────
    evidence_registry: dict[str, EvidenceSnippet] = Field(
        default_factory=dict,
        description="All evidence snippets indexed by EV-NNN.",
    )
    policy_rules: list[PolicyRule] = Field(
        default_factory=list,
        description="Ordered list of policy rules to be evaluated.",
    )
    extracted_terms: dict[str, DealTerm] = Field(
        default_factory=dict,
        description="All extracted deal terms indexed by TERM-NNN.",
    )

    # ── Compliance & Risk ─────────────────────────────────────────────────────
    compliance_results: dict[str, ComplianceResult] = Field(
        default_factory=dict,
        description="Compliance decisions indexed by COMP-NNN.",
    )
    risk_findings: list[RiskFinding] = Field(
        default_factory=list,
        description="Prioritised list of identified risk findings.",
    )

    # ── Risk & Summary Agent outputs ───────────────────────────────────────────
    missing_information: list[str] = Field(
        default_factory=list,
        description=(
            "Plain-text list of unresolvable information gaps and ambiguities "
            "identified by the Risk & Summary Agent. "
            "Consumed by Module 7 when assembling the final report."
        ),
    )
    follow_ups: list[str] = Field(
        default_factory=list,
        description=(
            "Recommended next steps produced by the Risk & Summary Agent, "
            "keyed to specific risk findings and ordered by urgency."
        ),
    )
    executive_summary: Optional[str] = Field(
        default=None,
        description=(
            "Concise narrative executive summary produced by the Risk & Summary Agent. "
            "Consumed by Module 7 when assembling the FinalDealReviewReport. "
            "Not the same as final_report, which is Module 7's assembled output."
        ),
    )

    # ── Handoffs & Escalations ────────────────────────────────────────────────
    pending_handoffs: list[Handoff] = Field(
        default_factory=list,
        description="Handoffs awaiting Orchestrator routing.",
    )
    resolved_handoffs: list[Handoff] = Field(
        default_factory=list,
        description="Handoffs that have been resolved or rejected.",
    )
    escalations: list[dict] = Field(
        default_factory=list,
        description="Human-escalation items with context dictionaries.",
    )

    # ── Agent Tracking ────────────────────────────────────────────────────────
    agent_statuses: dict[str, AgentStatus] = Field(
        default_factory=dict,
        description="Per-agent execution state keyed by agent name.",
    )
    audit_events: list[AuditEvent] = Field(
        default_factory=list,
        description="Ordered, immutable audit log of all pipeline events.",
    )
    errors: dict[str, str] = Field(
        default_factory=dict,
        description="Unrecoverable errors keyed by agent name.",
    )
    retry_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Per-agent retry counter.",
    )
    clarification_attempts: dict[str, int] = Field(
        default_factory=dict,
        description="Clarification attempt counter per rule_id (loop tracking).",
    )

    # ── Output ────────────────────────────────────────────────────────────────
    final_report: Optional[str] = Field(
        default=None,
        description=(
            "The assembled final report, serialised as a JSON string. "
            "Set by the OrchestratorAgent's ``assemble_report`` LangGraph node "
            "(Module 7) after all four agents have completed. "
            "Deserialise with ``FinalDealReviewReport.model_validate_json()`` "
            "from ``app.models.report``. "
            "None until the report_assembly node has run."
        ),
    )

    # ── Metadata ──────────────────────────────────────────────────────────────
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when this run was created.",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the most recent state update.",
    )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def add_audit_event(self, event: AuditEvent) -> "WorkflowState":
        """Return a new WorkflowState with the audit event appended."""
        return self.model_copy(
            update={
                "audit_events": [*self.audit_events, event],
                "updated_at": datetime.now(timezone.utc),
            }
        )

    def add_pending_handoff(self, handoff: Handoff) -> "WorkflowState":
        """Return a new WorkflowState with a handoff queued for routing."""
        return self.model_copy(
            update={
                "pending_handoffs": [*self.pending_handoffs, handoff],
                "updated_at": datetime.now(timezone.utc),
            }
        )

    def get_pending_handoffs_for(self, target_agent: str) -> list[Handoff]:
        """Return all pending handoffs addressed to a specific agent."""
        return [
            h for h in self.pending_handoffs
            if h.target_agent == target_agent
        ]

    def increment_retry(self, agent_name: str) -> "WorkflowState":
        """Return a new WorkflowState with the retry counter incremented."""
        updated = {**self.retry_counts, agent_name: self.retry_counts.get(agent_name, 0) + 1}
        return self.model_copy(
            update={
                "retry_counts": updated,
                "updated_at": datetime.now(timezone.utc),
            }
        )

    def set_document_metadata(self, metadata: DocumentMetadata) -> "WorkflowState":
        """Return a new WorkflowState with document_metadata populated."""
        return self.model_copy(
            update={
                "document_metadata": metadata,
                "updated_at": datetime.now(timezone.utc),
            }
        )

    model_config = {"arbitrary_types_allowed": True}
