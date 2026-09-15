"""Handoff model — structured inter-agent communication contract.

This is the most architecturally significant model in Module 1.

Instead of agents calling each other directly, every inter-agent communication
is expressed as a typed ``Handoff`` object that:

1. An agent places into the shared workflow state (``pending_handoffs``).
2. The Orchestrator inspects on each cycle.
3. The Orchestrator routes to the target agent and marks the handoff
   ``ACCEPTED``.
4. The target agent processes the request and marks it ``RESOLVED`` (or
   ``REJECTED`` if it cannot comply).

This design keeps the workflow:
* observable  — all communication is in the shared state
* auditable   — handoffs are logged as audit events
* retryable   — the Orchestrator can re-enqueue failed handoffs
* debuggable  — failed runs can be replayed by inspecting ``pending_handoffs``

Communication model (enforced):

    Source Agent
        ↓ creates Handoff
    Shared State (pending_handoffs)
        ↓ Orchestrator reads + routes
    Target Agent
        ↓ processes + returns Handoff (status=RESOLVED)
    Shared State (resolved_handoffs)
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


class HandoffType(str, Enum):
    """The purpose / intent of an inter-agent handoff."""

    TASK_REQUEST = "TASK_REQUEST"
    """Orchestrator delegating a task to an agent."""

    VERIFICATION_REQUIRED = "VERIFICATION_REQUIRED"
    """An agent requests that another agent re-verify a specific term/evidence."""

    REANALYSIS_REQUIRED = "REANALYSIS_REQUIRED"
    """An agent requests a full re-analysis of a prior result."""

    CLARIFICATION_REQUIRED = "CLARIFICATION_REQUIRED"
    """An agent needs more information before it can complete its work."""

    ESCALATION_REQUIRED = "ESCALATION_REQUIRED"
    """The issue cannot be resolved automatically — a human must intervene."""

    RESULT_READY = "RESULT_READY"
    """An agent signals that its output is ready for the Orchestrator to consume."""

    FAILURE = "FAILURE"
    """An agent reports that it has failed and cannot recover."""

    RETRY_REQUEST = "RETRY_REQUEST"
    """The Orchestrator requests that an agent retries its last task."""


class HandoffStatus(str, Enum):
    """Current lifecycle state of a handoff object."""

    PENDING = "PENDING"       # Created; awaiting Orchestrator routing.
    ACCEPTED = "ACCEPTED"     # Orchestrator has routed it to the target agent.
    RESOLVED = "RESOLVED"     # Target agent completed the requested work.
    REJECTED = "REJECTED"     # Target agent could not fulfil the request.
    EXPIRED = "EXPIRED"       # Handoff was not processed within allowed retries.


class HandoffPriority(str, Enum):
    """Processing priority of a handoff."""

    CRITICAL = "critical"    # Must be handled before any other work.
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# Canonical agent name constants — used as string literals in handoffs.
AGENT_ORCHESTRATOR = "orchestrator"
AGENT_TERM_EXTRACTION = "term_extraction"
AGENT_COMPLIANCE = "compliance"
AGENT_RISK_SUMMARY = "risk_summary"
AGENT_HUMAN = "human"

KNOWN_AGENTS: frozenset[str] = frozenset(
    {
        AGENT_ORCHESTRATOR,
        AGENT_TERM_EXTRACTION,
        AGENT_COMPLIANCE,
        AGENT_RISK_SUMMARY,
        AGENT_HUMAN,
    }
)


class Handoff(BaseModel):
    """Structured communication contract between agents.

    Attributes:
        handoff_id:            Auto-generated stable UUID for this handoff.
        handoff_type:          Purpose / intent of the handoff.
        source_agent:          Name of the agent creating this handoff.
        target_agent:          Name of the intended recipient agent.
        reason:                Human-readable explanation of why this handoff
                               was created.
        priority:              Processing priority.
        related_term_ids:      ``TERM-NNN`` IDs relevant to this handoff.
        evidence_ids:          ``EV-NNN`` IDs relevant to this handoff.
        related_compliance_ids:``COMP-NNN`` IDs relevant to this handoff.
        payload:               Optional structured additional data.  Kept
                               minimal — prefer referencing IDs over embedding
                               large objects.
        status:                Current lifecycle state.
        created_at:            UTC timestamp of creation.
        resolved_at:           UTC timestamp of resolution (set by recipient).
        resolution_notes:      Optional notes from the target agent on how
                               the handoff was resolved.
    """

    handoff_id: str = Field(
        default_factory=lambda: f"HDOFF-{uuid4().hex[:8].upper()}",
        description="Auto-generated unique handoff identifier.",
    )
    handoff_type: HandoffType = Field(
        ...,
        description="Purpose of this inter-agent handoff.",
    )
    source_agent: str = Field(
        ...,
        min_length=1,
        description="Name of the agent creating this handoff.",
    )
    target_agent: str = Field(
        ...,
        min_length=1,
        description="Name of the intended recipient agent.",
    )
    reason: str = Field(
        ...,
        min_length=1,
        description="Human-readable explanation of why this handoff was created.",
    )
    priority: HandoffPriority = Field(
        default=HandoffPriority.MEDIUM,
        description="Processing priority.",
    )
    related_term_ids: list[str] = Field(
        default_factory=list,
        description="TERM-NNN IDs relevant to this handoff.",
    )
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="EV-NNN IDs relevant to this handoff.",
    )
    related_compliance_ids: list[str] = Field(
        default_factory=list,
        description="COMP-NNN IDs relevant to this handoff.",
    )
    payload: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "Optional additional structured data.  Prefer ID references "
            "over embedding large objects here."
        ),
    )
    # ── Module 5 additions — feedback loop metadata ────────────────────────────
    rule_ids: list[str] = Field(
        default_factory=list,
        description=(
            "POLICY-NNN IDs of the policy rules that triggered this handoff. "
            "Used by the router to perform targeted rule re-evaluation after "
            "clarification, rather than re-running the entire compliance pass."
        ),
    )
    requested_term_names: list[str] = Field(
        default_factory=list,
        description=(
            "Term names the recipient agent should focus on in its clarification "
            "pass.  Allows targeted extraction rather than a full re-run."
        ),
    )
    clarification_attempt: int = Field(
        default=0,
        ge=0,
        description=(
            "Which clarification attempt this handoff represents (0-indexed). "
            "Used for loop-prevention: if this equals max_clarification_attempts, "
            "the Orchestrator escalates to human rather than re-queueing."
        ),
    )
    status: HandoffStatus = Field(
        default=HandoffStatus.PENDING,
        description="Current lifecycle state of this handoff.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when this handoff was created.",
    )
    resolved_at: Optional[datetime] = Field(
        default=None,
        description="UTC timestamp when this handoff was resolved.",
    )
    resolution_notes: Optional[str] = Field(
        default=None,
        description="Notes from the target agent on how the handoff was resolved.",
    )

    @field_validator("source_agent", "target_agent")
    @classmethod
    def validate_agent_name(cls, v: str) -> str:
        """Ensure agent names are from the known set to prevent typos."""
        if v not in KNOWN_AGENTS:
            raise ValueError(
                f"Unknown agent name: '{v}'. "
                f"Must be one of: {sorted(KNOWN_AGENTS)}"
            )
        return v

    @model_validator(mode="after")
    def source_and_target_must_differ(self) -> "Handoff":
        """An agent must not send a handoff to itself."""
        if self.source_agent == self.target_agent:
            raise ValueError(
                f"source_agent and target_agent must differ; "
                f"both are '{self.source_agent}'."
            )
        return self

    def accept(self) -> "Handoff":
        """Return a new Handoff with status set to ACCEPTED."""
        return self.model_copy(update={"status": HandoffStatus.ACCEPTED})

    def resolve(self, notes: Optional[str] = None) -> "Handoff":
        """Return a new Handoff with status set to RESOLVED."""
        return self.model_copy(
            update={
                "status": HandoffStatus.RESOLVED,
                "resolved_at": datetime.now(timezone.utc),
                "resolution_notes": notes,
            }
        )

    def reject(self, notes: Optional[str] = None) -> "Handoff":
        """Return a new Handoff with status set to REJECTED."""
        return self.model_copy(
            update={
                "status": HandoffStatus.REJECTED,
                "resolved_at": datetime.now(timezone.utc),
                "resolution_notes": notes,
            }
        )
