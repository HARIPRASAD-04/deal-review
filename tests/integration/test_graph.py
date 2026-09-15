"""Integration tests for the LangGraph orchestration graph."""

from __future__ import annotations

import pytest

from app.models.audit import AuditEventType
from app.models.handoff import (
    AGENT_COMPLIANCE,
    AGENT_ORCHESTRATOR,
    AGENT_TERM_EXTRACTION,
    Handoff,
    HandoffStatus,
    HandoffType,
)
from app.models.state import AgentStatusEnum, WorkflowRunStatus, WorkflowState
from app.orchestration.graph import (
    NODE_INITIALIZE,
    build_graph,
    deal_review_graph,
    initialize_run,
)


# ── Graph Construction ─────────────────────────────────────────────────────────

class TestGraphConstruction:
    def test_build_graph_returns_compiled_graph(self):
        """build_graph() must return a compiled LangGraph object."""
        graph = build_graph()
        assert graph is not None

    def test_module_level_graph_exists(self):
        """The pre-compiled module-level instance must exist."""
        assert deal_review_graph is not None

    def test_graph_has_initialize_node(self):
        """The graph must contain the initialize_run node."""
        graph = build_graph()
        # LangGraph compiled graphs expose their graph structure.
        assert NODE_INITIALIZE in graph.nodes


# ── Graph Execution ────────────────────────────────────────────────────────────

class TestGraphExecution:
    def _make_initial_state(self) -> WorkflowState:
        return WorkflowState(document_id="DEAL-001")

    def test_graph_executes_without_error(self):
        """The minimal graph must complete without raising an exception."""
        initial_state = self._make_initial_state()
        result = deal_review_graph.invoke(initial_state)
        assert result is not None

    def test_graph_result_has_audit_events(self):
        """After initialization, at least one audit event must exist."""
        initial_state = self._make_initial_state()
        result = deal_review_graph.invoke(initial_state)
        final_state = WorkflowState(**result) if isinstance(result, dict) else result
        assert len(final_state.audit_events) >= 1

    def test_graph_records_run_started_event(self):
        """The initialize_run node must emit a RUN_STARTED audit event."""
        initial_state = self._make_initial_state()
        result = deal_review_graph.invoke(initial_state)
        final_state = WorkflowState(**result) if isinstance(result, dict) else result
        event_types = [e.event_type for e in final_state.audit_events]
        assert AuditEventType.RUN_STARTED in event_types

    def test_graph_sets_agent_statuses(self):
        """After initialization, all four core agents must have statuses."""
        initial_state = self._make_initial_state()
        result = deal_review_graph.invoke(initial_state)
        final_state = WorkflowState(**result) if isinstance(result, dict) else result
        assert AGENT_ORCHESTRATOR in final_state.agent_statuses
        assert AGENT_TERM_EXTRACTION in final_state.agent_statuses
        assert AGENT_COMPLIANCE in final_state.agent_statuses

    def test_orchestrator_status_is_completed_after_full_run(self):
        initial_state = self._make_initial_state()
        result = deal_review_graph.invoke(initial_state)
        final_state = WorkflowState(**result) if isinstance(result, dict) else result
        orchestrator_status = final_state.agent_statuses.get(AGENT_ORCHESTRATOR)
        assert orchestrator_status is not None
        assert orchestrator_status.status == AgentStatusEnum.COMPLETED

    def test_document_id_preserved(self):
        """The document_id must survive the graph execution unchanged."""
        initial_state = WorkflowState(document_id="DEAL-TEST-999")
        result = deal_review_graph.invoke(initial_state)
        final_state = WorkflowState(**result) if isinstance(result, dict) else result
        assert final_state.document_id == "DEAL-TEST-999"

    def test_run_id_preserved(self):
        """The run_id assigned at state creation must not change during execution."""
        initial_state = self._make_initial_state()
        original_run_id = initial_state.run_id
        result = deal_review_graph.invoke(initial_state)
        final_state = WorkflowState(**result) if isinstance(result, dict) else result
        assert final_state.run_id == original_run_id


# ── Node Unit Test ─────────────────────────────────────────────────────────────

class TestInitializeRunNode:
    def test_initialize_run_directly(self):
        """Test the initialize_run node function in isolation."""
        state = WorkflowState(document_id="DEAL-001")
        output = initialize_run(state)
        assert "agent_statuses" in output
        assert "audit_events" in output
        assert "run_status" in output

    def test_initialize_run_produces_audit_event(self):
        state = WorkflowState(document_id="DEAL-001")
        output = initialize_run(state)
        events = output["audit_events"]
        assert len(events) == 1
        assert events[0].event_type == AuditEventType.RUN_STARTED

    def test_initialize_run_all_agents_present(self):
        state = WorkflowState(document_id="DEAL-001")
        output = initialize_run(state)
        statuses = output["agent_statuses"]
        assert AGENT_ORCHESTRATOR in statuses
        assert AGENT_TERM_EXTRACTION in statuses
        assert AGENT_COMPLIANCE in statuses


# ── Handoff in State integration ───────────────────────────────────────────────

class TestHandoffInState:
    """Verify that a handoff can be inserted and retrieved from WorkflowState."""

    def test_handoff_survives_state_operations(self):
        state = WorkflowState(document_id="DEAL-001")

        handoff = Handoff(
            handoff_type=HandoffType.VERIFICATION_REQUIRED,
            source_agent=AGENT_COMPLIANCE,
            target_agent=AGENT_TERM_EXTRACTION,
            reason="Facility amount appears inconsistent across clauses.",
            related_term_ids=["TERM-001"],
            evidence_ids=["EV-001", "EV-004"],
        )

        # Insert into state.
        state = state.add_pending_handoff(handoff)
        assert len(state.pending_handoffs) == 1

        # Retrieve by target agent.
        pending = state.get_pending_handoffs_for(AGENT_TERM_EXTRACTION)
        assert len(pending) == 1
        assert pending[0].handoff_id == handoff.handoff_id
        assert pending[0].status == HandoffStatus.PENDING

    def test_resolved_handoff_separate_from_pending(self):
        """Resolved handoffs must not appear in pending list."""
        state = WorkflowState(document_id="DEAL-001")

        handoff = Handoff(
            handoff_type=HandoffType.TASK_REQUEST,
            source_agent=AGENT_ORCHESTRATOR,
            target_agent=AGENT_TERM_EXTRACTION,
            reason="Start extraction.",
        )
        state = state.add_pending_handoff(handoff)

        # Simulate resolution — move from pending to resolved.
        resolved = handoff.resolve(notes="Extraction complete.")
        state = state.model_copy(
            update={
                "pending_handoffs": [],
                "resolved_handoffs": [resolved],
            }
        )

        assert len(state.pending_handoffs) == 0
        assert len(state.resolved_handoffs) == 1
        assert state.resolved_handoffs[0].status == HandoffStatus.RESOLVED
