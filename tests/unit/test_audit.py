"""Unit tests for AuditEvent model and make_audit_event factory."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models.audit import AuditEvent, AuditEventType, make_audit_event


class TestAuditEventCreation:
    def test_valid_audit_event(self):
        event = AuditEvent(
            run_id="RUN-ABCD1234",
            event_type=AuditEventType.RUN_STARTED,
            message="Pipeline run started for DEAL-001.",
            agent_name="orchestrator",
        )
        assert event.run_id == "RUN-ABCD1234"
        assert event.event_type == AuditEventType.RUN_STARTED
        assert event.event_id.startswith("EVT-")

    def test_event_id_auto_generated_and_unique(self):
        e1 = AuditEvent(
            run_id="RUN-001",
            event_type=AuditEventType.GENERIC,
            message="Event 1.",
        )
        e2 = AuditEvent(
            run_id="RUN-001",
            event_type=AuditEventType.GENERIC,
            message="Event 2.",
        )
        assert e1.event_id != e2.event_id

    def test_timestamp_auto_set(self):
        event = AuditEvent(
            run_id="RUN-001",
            event_type=AuditEventType.AGENT_STARTED,
            message="Term extraction agent started.",
        )
        assert event.timestamp is not None

    def test_metadata_optional(self):
        event = AuditEvent(
            run_id="RUN-001",
            event_type=AuditEventType.HANDOFF_CREATED,
            message="Handoff created.",
            metadata={"handoff_id": "HDOFF-ABCD1234", "target": "term_extraction"},
        )
        assert event.metadata["handoff_id"] == "HDOFF-ABCD1234"

    def test_agent_name_optional(self):
        event = AuditEvent(
            run_id="RUN-001",
            event_type=AuditEventType.RUN_COMPLETED,
            message="Run completed.",
        )
        assert event.agent_name is None

    def test_audit_event_is_immutable(self):
        event = AuditEvent(
            run_id="RUN-001",
            event_type=AuditEventType.GENERIC,
            message="Test.",
        )
        with pytest.raises(Exception):
            event.message = "Modified."  # type: ignore[misc]

    def test_all_event_types(self):
        for etype in AuditEventType:
            event = AuditEvent(
                run_id="RUN-001",
                event_type=etype,
                message=f"Test for {etype.value}.",
            )
            assert event.event_type == etype


class TestAuditEventValidation:
    def test_empty_run_id_raises(self):
        with pytest.raises(ValidationError):
            AuditEvent(
                run_id="",
                event_type=AuditEventType.GENERIC,
                message="Test.",
            )

    def test_empty_message_raises(self):
        with pytest.raises(ValidationError):
            AuditEvent(
                run_id="RUN-001",
                event_type=AuditEventType.GENERIC,
                message="",
            )

    def test_invalid_event_type_raises(self):
        with pytest.raises(ValidationError):
            AuditEvent(
                run_id="RUN-001",
                event_type="NOT_A_REAL_TYPE",  # type: ignore[arg-type]
                message="Test.",
            )


class TestMakeAuditEvent:
    def test_factory_creates_valid_event(self):
        event = make_audit_event(
            run_id="RUN-ABCD1234",
            event_type=AuditEventType.AGENT_COMPLETED,
            message="Term extraction completed.",
            agent_name="term_extraction",
            metadata={"terms_extracted": 5},
        )
        assert isinstance(event, AuditEvent)
        assert event.run_id == "RUN-ABCD1234"
        assert event.agent_name == "term_extraction"
        assert event.metadata["terms_extracted"] == 5

    def test_factory_without_optional_args(self):
        event = make_audit_event(
            run_id="RUN-001",
            event_type=AuditEventType.RUN_STARTED,
            message="Run started.",
        )
        assert event.agent_name is None
        assert event.metadata is None
