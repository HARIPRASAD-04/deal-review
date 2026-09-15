"""Unit tests for WorkflowState and AgentStatus models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models.audit import AuditEventType, make_audit_event
from app.models.handoff import (
    AGENT_COMPLIANCE,
    AGENT_ORCHESTRATOR,
    AGENT_TERM_EXTRACTION,
    Handoff,
    HandoffStatus,
    HandoffType,
)
from app.models.state import AgentStatus, AgentStatusEnum, WorkflowRunStatus, WorkflowState


# ── AgentStatus ────────────────────────────────────────────────────────────────

class TestAgentStatus:
    def test_default_status_is_idle(self):
        agent = AgentStatus(agent_name="term_extraction")
        assert agent.status == AgentStatusEnum.IDLE
        assert agent.attempt_count == 0

    def test_mark_running(self):
        agent = AgentStatus(agent_name="term_extraction")
        running = agent.mark_running()
        assert running.status == AgentStatusEnum.RUNNING
        assert running.attempt_count == 1
        assert running.started_at is not None
        assert agent.status == AgentStatusEnum.IDLE  # Original unchanged.

    def test_mark_running_increments_attempt_count(self):
        agent = AgentStatus(agent_name="term_extraction")
        r1 = agent.mark_running()
        assert r1.attempt_count == 1
        r2 = r1.mark_failed("timeout").mark_retrying().mark_running()
        assert r2.attempt_count == 2

    def test_mark_completed(self):
        agent = AgentStatus(agent_name="term_extraction").mark_running()
        completed = agent.mark_completed()
        assert completed.status == AgentStatusEnum.COMPLETED
        assert completed.completed_at is not None

    def test_mark_failed(self):
        agent = AgentStatus(agent_name="term_extraction").mark_running()
        failed = agent.mark_failed("LLM timeout after 30s.")
        assert failed.status == AgentStatusEnum.FAILED
        assert failed.last_error == "LLM timeout after 30s."
        assert failed.completed_at is not None

    def test_mark_retrying(self):
        agent = AgentStatus(agent_name="term_extraction").mark_running().mark_failed("err")
        retrying = agent.mark_retrying()
        assert retrying.status == AgentStatusEnum.RETRYING

    def test_all_statuses(self):
        for status in AgentStatusEnum:
            agent = AgentStatus(agent_name="compliance", status=status)
            assert agent.status == status

    def test_invalid_agent_name_raises(self):
        with pytest.raises(ValidationError):
            AgentStatus(agent_name="")

    def test_negative_attempt_count_raises(self):
        with pytest.raises(ValidationError):
            AgentStatus(agent_name="compliance", attempt_count=-1)


# ── WorkflowState ──────────────────────────────────────────────────────────────

class TestWorkflowState:
    def _make_state(self) -> WorkflowState:
        return WorkflowState(document_id="DEAL-001")

    def test_default_state_creation(self):
        state = self._make_state()
        assert state.document_id == "DEAL-001"
        assert state.run_id.startswith("RUN-")
        assert state.run_status == WorkflowRunStatus.INITIALIZED
        assert state.evidence_registry == {}
        assert state.extracted_terms == {}
        assert state.compliance_results == {}
        assert state.risk_findings == []
        assert state.pending_handoffs == []
        assert state.resolved_handoffs == []
        assert state.audit_events == []

    def test_run_id_unique_per_run(self):
        s1 = WorkflowState(document_id="DEAL-001")
        s2 = WorkflowState(document_id="DEAL-001")
        assert s1.run_id != s2.run_id

    def test_add_audit_event(self):
        state = self._make_state()
        event = make_audit_event(
            run_id=state.run_id,
            event_type=AuditEventType.RUN_STARTED,
            message="Run started.",
        )
        updated = state.add_audit_event(event)
        assert len(updated.audit_events) == 1
        assert updated.audit_events[0].event_type == AuditEventType.RUN_STARTED
        assert len(state.audit_events) == 0  # Original unchanged.

    def test_add_multiple_audit_events(self):
        state = self._make_state()
        for i in range(3):
            event = make_audit_event(
                run_id=state.run_id,
                event_type=AuditEventType.GENERIC,
                message=f"Event {i}.",
            )
            state = state.add_audit_event(event)
        assert len(state.audit_events) == 3

    def test_add_pending_handoff(self):
        state = self._make_state()
        handoff = Handoff(
            handoff_type=HandoffType.VERIFICATION_REQUIRED,
            source_agent=AGENT_COMPLIANCE,
            target_agent=AGENT_TERM_EXTRACTION,
            reason="Verify facility amount.",
        )
        updated = state.add_pending_handoff(handoff)
        assert len(updated.pending_handoffs) == 1
        assert updated.pending_handoffs[0].handoff_type == HandoffType.VERIFICATION_REQUIRED
        assert len(state.pending_handoffs) == 0  # Original unchanged.

    def test_get_pending_handoffs_for_agent(self):
        state = self._make_state()
        h1 = Handoff(
            handoff_type=HandoffType.TASK_REQUEST,
            source_agent=AGENT_ORCHESTRATOR,
            target_agent=AGENT_TERM_EXTRACTION,
            reason="Extract terms.",
        )
        h2 = Handoff(
            handoff_type=HandoffType.TASK_REQUEST,
            source_agent=AGENT_ORCHESTRATOR,
            target_agent=AGENT_COMPLIANCE,
            reason="Run compliance.",
        )
        state = state.add_pending_handoff(h1).add_pending_handoff(h2)

        extraction_handoffs = state.get_pending_handoffs_for(AGENT_TERM_EXTRACTION)
        assert len(extraction_handoffs) == 1
        assert extraction_handoffs[0].target_agent == AGENT_TERM_EXTRACTION

    def test_get_pending_handoffs_for_nonexistent_agent(self):
        state = self._make_state()
        result = state.get_pending_handoffs_for(AGENT_TERM_EXTRACTION)
        assert result == []

    def test_increment_retry(self):
        state = self._make_state()
        state = state.increment_retry(AGENT_TERM_EXTRACTION)
        assert state.retry_counts[AGENT_TERM_EXTRACTION] == 1
        state = state.increment_retry(AGENT_TERM_EXTRACTION)
        assert state.retry_counts[AGENT_TERM_EXTRACTION] == 2

    def test_handoff_inserted_and_retrieved(self):
        """Full round-trip: create handoff, insert into state, retrieve."""
        state = self._make_state()
        handoff = Handoff(
            handoff_type=HandoffType.VERIFICATION_REQUIRED,
            source_agent=AGENT_COMPLIANCE,
            target_agent=AGENT_TERM_EXTRACTION,
            reason="Inconsistent facility amount detected.",
            related_term_ids=["TERM-001"],
            evidence_ids=["EV-001"],
        )
        state = state.add_pending_handoff(handoff)

        # Retrieve by target agent.
        pending = state.get_pending_handoffs_for(AGENT_TERM_EXTRACTION)
        assert len(pending) == 1
        retrieved = pending[0]
        assert retrieved.handoff_id == handoff.handoff_id
        assert retrieved.status == HandoffStatus.PENDING
        assert "TERM-001" in retrieved.related_term_ids
        assert "EV-001" in retrieved.evidence_ids

    def test_missing_document_id_raises(self):
        with pytest.raises(ValidationError):
            WorkflowState(document_id="")
