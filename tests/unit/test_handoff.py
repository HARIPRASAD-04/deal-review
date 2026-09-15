"""Unit tests for the Handoff model — the core communication contract."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models.handoff import (
    AGENT_COMPLIANCE,
    AGENT_ORCHESTRATOR,
    AGENT_TERM_EXTRACTION,
    AGENT_RISK_SUMMARY,
    Handoff,
    HandoffPriority,
    HandoffStatus,
    HandoffType,
)


class TestHandoffCreation:
    def test_valid_verification_handoff(self):
        """Compliance → Term Extraction verification request."""
        handoff = Handoff(
            handoff_type=HandoffType.VERIFICATION_REQUIRED,
            source_agent=AGENT_COMPLIANCE,
            target_agent=AGENT_TERM_EXTRACTION,
            reason="Facility amount appears inconsistent across clauses.",
            related_term_ids=["TERM-001"],
            evidence_ids=["EV-001", "EV-004"],
        )
        assert handoff.status == HandoffStatus.PENDING
        assert handoff.priority == HandoffPriority.MEDIUM  # Default.
        assert handoff.handoff_id.startswith("HDOFF-")

    def test_valid_task_request_handoff(self):
        """Orchestrator → Term Extraction task delegation."""
        handoff = Handoff(
            handoff_type=HandoffType.TASK_REQUEST,
            source_agent=AGENT_ORCHESTRATOR,
            target_agent=AGENT_TERM_EXTRACTION,
            reason="Extract all material terms from DEAL-001.",
            priority=HandoffPriority.HIGH,
        )
        assert handoff.handoff_type == HandoffType.TASK_REQUEST

    def test_escalation_handoff(self):
        handoff = Handoff(
            handoff_type=HandoffType.ESCALATION_REQUIRED,
            source_agent=AGENT_COMPLIANCE,
            target_agent="human",
            reason="DSCR below minimum — requires senior credit officer review.",
            priority=HandoffPriority.CRITICAL,
            related_compliance_ids=["COMP-003"],
        )
        assert handoff.priority == HandoffPriority.CRITICAL

    def test_handoff_id_auto_generated(self):
        h1 = Handoff(
            handoff_type=HandoffType.TASK_REQUEST,
            source_agent=AGENT_ORCHESTRATOR,
            target_agent=AGENT_COMPLIANCE,
            reason="Run compliance review.",
        )
        h2 = Handoff(
            handoff_type=HandoffType.TASK_REQUEST,
            source_agent=AGENT_ORCHESTRATOR,
            target_agent=AGENT_COMPLIANCE,
            reason="Run compliance review.",
        )
        # IDs should be unique.
        assert h1.handoff_id != h2.handoff_id

    def test_created_at_auto_set(self):
        handoff = Handoff(
            handoff_type=HandoffType.TASK_REQUEST,
            source_agent=AGENT_ORCHESTRATOR,
            target_agent=AGENT_TERM_EXTRACTION,
            reason="Start extraction.",
        )
        assert handoff.created_at is not None

    def test_payload_optional(self):
        handoff = Handoff(
            handoff_type=HandoffType.TASK_REQUEST,
            source_agent=AGENT_ORCHESTRATOR,
            target_agent=AGENT_TERM_EXTRACTION,
            reason="Start extraction.",
            payload={"document_id": "DEAL-001"},
        )
        assert handoff.payload == {"document_id": "DEAL-001"}

    def test_all_handoff_types(self):
        for ht in HandoffType:
            handoff = Handoff(
                handoff_type=ht,
                source_agent=AGENT_ORCHESTRATOR,
                target_agent=AGENT_TERM_EXTRACTION,
                reason="Test.",
            )
            assert handoff.handoff_type == ht


class TestHandoffLifecycle:
    def test_accept(self):
        handoff = Handoff(
            handoff_type=HandoffType.VERIFICATION_REQUIRED,
            source_agent=AGENT_COMPLIANCE,
            target_agent=AGENT_TERM_EXTRACTION,
            reason="Verify facility amount.",
        )
        accepted = handoff.accept()
        assert accepted.status == HandoffStatus.ACCEPTED
        assert handoff.status == HandoffStatus.PENDING  # Original unchanged.

    def test_resolve(self):
        handoff = Handoff(
            handoff_type=HandoffType.VERIFICATION_REQUIRED,
            source_agent=AGENT_COMPLIANCE,
            target_agent=AGENT_TERM_EXTRACTION,
            reason="Verify facility amount.",
        )
        resolved = handoff.resolve(notes="Facility amount confirmed as INR 10 crore.")
        assert resolved.status == HandoffStatus.RESOLVED
        assert resolved.resolved_at is not None
        assert "confirmed" in resolved.resolution_notes

    def test_reject(self):
        handoff = Handoff(
            handoff_type=HandoffType.VERIFICATION_REQUIRED,
            source_agent=AGENT_COMPLIANCE,
            target_agent=AGENT_TERM_EXTRACTION,
            reason="Verify missing clause.",
        )
        rejected = handoff.reject(notes="Clause not found in document.")
        assert rejected.status == HandoffStatus.REJECTED
        assert rejected.resolved_at is not None

    def test_serialisation_round_trip(self):
        """Handoffs must survive JSON serialisation/deserialisation."""
        handoff = Handoff(
            handoff_type=HandoffType.VERIFICATION_REQUIRED,
            source_agent=AGENT_COMPLIANCE,
            target_agent=AGENT_TERM_EXTRACTION,
            reason="Cross-check facility amount.",
            related_term_ids=["TERM-001"],
            evidence_ids=["EV-001"],
        )
        data = handoff.model_dump()
        restored = Handoff(**data)
        assert restored.handoff_id == handoff.handoff_id
        assert restored.status == handoff.status
        assert restored.handoff_type == handoff.handoff_type


class TestHandoffValidation:
    def test_self_handoff_raises(self):
        """An agent must not create a handoff to itself."""
        with pytest.raises(ValidationError):
            Handoff(
                handoff_type=HandoffType.TASK_REQUEST,
                source_agent=AGENT_TERM_EXTRACTION,
                target_agent=AGENT_TERM_EXTRACTION,
                reason="Self-loop.",
            )

    def test_unknown_source_agent_raises(self):
        with pytest.raises(ValidationError):
            Handoff(
                handoff_type=HandoffType.TASK_REQUEST,
                source_agent="unknown_agent",
                target_agent=AGENT_TERM_EXTRACTION,
                reason="Test.",
            )

    def test_unknown_target_agent_raises(self):
        with pytest.raises(ValidationError):
            Handoff(
                handoff_type=HandoffType.TASK_REQUEST,
                source_agent=AGENT_ORCHESTRATOR,
                target_agent="made_up_agent",
                reason="Test.",
            )

    def test_empty_reason_raises(self):
        with pytest.raises(ValidationError):
            Handoff(
                handoff_type=HandoffType.TASK_REQUEST,
                source_agent=AGENT_ORCHESTRATOR,
                target_agent=AGENT_TERM_EXTRACTION,
                reason="",
            )

    def test_missing_handoff_type_raises(self):
        with pytest.raises(ValidationError):
            Handoff(
                source_agent=AGENT_ORCHESTRATOR,
                target_agent=AGENT_TERM_EXTRACTION,
                reason="Missing type.",
            )

    def test_invalid_handoff_type_raises(self):
        with pytest.raises(ValidationError):
            Handoff(
                handoff_type="DIRECT_CALL",  # type: ignore[arg-type]
                source_agent=AGENT_ORCHESTRATOR,
                target_agent=AGENT_TERM_EXTRACTION,
                reason="Test.",
            )
