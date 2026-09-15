"""Audit event model.

An AuditEvent is an immutable record of something that happened during a
pipeline run.  The audit log allows engineers and reviewers to reconstruct:

    "What happened during this deal review?"
    "Which agent ran when, and what did it do?"
    "Why was a handoff created?"
    "How many retries occurred?"

Audit events are appended to the shared workflow state and are never deleted
or mutated during a run.  This gives us a full, ordered timeline that can be
used for debugging, observability, and interview explanation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class AuditEventType(str, Enum):
    """Categorises what kind of thing happened."""

    RUN_STARTED = "RUN_STARTED"
    RUN_COMPLETED = "RUN_COMPLETED"
    RUN_FAILED = "RUN_FAILED"

    AGENT_STARTED = "AGENT_STARTED"
    AGENT_COMPLETED = "AGENT_COMPLETED"
    AGENT_FAILED = "AGENT_FAILED"
    AGENT_RETRYING = "AGENT_RETRYING"

    HANDOFF_CREATED = "HANDOFF_CREATED"
    HANDOFF_ACCEPTED = "HANDOFF_ACCEPTED"
    HANDOFF_RESOLVED = "HANDOFF_RESOLVED"
    HANDOFF_REJECTED = "HANDOFF_REJECTED"

    ESCALATION_CREATED = "ESCALATION_CREATED"

    STATE_UPDATED = "STATE_UPDATED"

    TERM_EXTRACTED = "TERM_EXTRACTED"
    TERM_VERIFIED = "TERM_VERIFIED"
    TERM_CHALLENGED = "TERM_CHALLENGED"

    COMPLIANCE_EVALUATED = "COMPLIANCE_EVALUATED"

    RISK_IDENTIFIED = "RISK_IDENTIFIED"

    REPORT_GENERATED = "REPORT_GENERATED"

    DOCUMENT_INGESTION_STARTED = "DOCUMENT_INGESTION_STARTED"
    DOCUMENT_INGESTION_COMPLETED = "DOCUMENT_INGESTION_COMPLETED"
    DOCUMENT_INGESTION_FAILED = "DOCUMENT_INGESTION_FAILED"

    TERM_EXTRACTION_STARTED = "TERM_EXTRACTION_STARTED"
    TERM_EXTRACTION_COMPLETED = "TERM_EXTRACTION_COMPLETED"
    TERM_EXTRACTION_FAILED = "TERM_EXTRACTION_FAILED"

    COMPLIANCE_REVIEW_STARTED = "COMPLIANCE_REVIEW_STARTED"
    COMPLIANCE_REVIEW_COMPLETED = "COMPLIANCE_REVIEW_COMPLETED"
    COMPLIANCE_REVIEW_FAILED = "COMPLIANCE_REVIEW_FAILED"

    # ── Module 5: handoff routing & clarification lifecycle ───────────────────
    HANDOFF_ROUTING_STARTED = "HANDOFF_ROUTING_STARTED"
    HANDOFF_ROUTING_COMPLETED = "HANDOFF_ROUTING_COMPLETED"
    CLARIFICATION_STARTED = "CLARIFICATION_STARTED"
    CLARIFICATION_COMPLETED = "CLARIFICATION_COMPLETED"
    CLARIFICATION_FAILED = "CLARIFICATION_FAILED"
    COMPLIANCE_RECHECK_STARTED = "COMPLIANCE_RECHECK_STARTED"
    COMPLIANCE_RECHECK_COMPLETED = "COMPLIANCE_RECHECK_COMPLETED"
    LOOP_LIMIT_REACHED = "LOOP_LIMIT_REACHED"

    # ── Module 6: risk analysis & summary lifecycle ──────────────────────────
    RISK_ANALYSIS_STARTED = "RISK_ANALYSIS_STARTED"
    RISK_ANALYSIS_COMPLETED = "RISK_ANALYSIS_COMPLETED"
    RISK_ANALYSIS_FAILED = "RISK_ANALYSIS_FAILED"
    SUMMARY_GENERATION_STARTED = "SUMMARY_GENERATION_STARTED"
    SUMMARY_GENERATION_COMPLETED = "SUMMARY_GENERATION_COMPLETED"
    SUMMARY_GENERATION_FAILED = "SUMMARY_GENERATION_FAILED"

    # ── Module 7: report assembly lifecycle ───────────────────────────────────
    REPORT_ASSEMBLY_STARTED = "REPORT_ASSEMBLY_STARTED"
    REPORT_ASSEMBLY_COMPLETED = "REPORT_ASSEMBLY_COMPLETED"
    REPORT_ASSEMBLY_FAILED = "REPORT_ASSEMBLY_FAILED"

    GENERIC = "GENERIC"


class AuditEvent(BaseModel):
    """An immutable record of a significant pipeline event.

    Attributes:
        event_id:   Auto-generated stable UUID for this event.
        run_id:     The pipeline run this event belongs to.
        timestamp:  UTC timestamp of the event.
        event_type: Categorised type of the event.
        agent_name: Name of the agent that emitted the event (if applicable).
        message:    Human-readable description of what happened.
        metadata:   Optional structured data associated with the event (e.g.
                    the handoff_id, term_id, or retry count).
    """

    event_id: str = Field(
        default_factory=lambda: f"EVT-{uuid4().hex[:10].upper()}",
        description="Auto-generated unique event identifier.",
    )
    run_id: str = Field(
        ...,
        min_length=1,
        description="Pipeline run ID this event belongs to.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the event.",
    )
    event_type: AuditEventType = Field(
        ...,
        description="Categorised type of the audit event.",
    )
    agent_name: Optional[str] = Field(
        default=None,
        description="Name of the agent that emitted this event.",
    )
    message: str = Field(
        ...,
        min_length=1,
        description="Human-readable description of what happened.",
    )
    metadata: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional structured data associated with the event.",
    )

    model_config = {"frozen": True}   # Audit events are immutable once created.


def make_audit_event(
    run_id: str,
    event_type: AuditEventType,
    message: str,
    agent_name: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> AuditEvent:
    """Convenience factory for creating audit events.

    Args:
        run_id:      The current pipeline run ID.
        event_type:  Type of the event.
        message:     Human-readable description.
        agent_name:  Originating agent (optional).
        metadata:    Structured supplementary data (optional).

    Returns:
        A new, immutable ``AuditEvent`` instance.
    """
    return AuditEvent(
        run_id=run_id,
        event_type=event_type,
        message=message,
        agent_name=agent_name,
        metadata=metadata,
    )
