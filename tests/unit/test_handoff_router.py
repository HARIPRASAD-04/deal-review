"""Unit tests for HandoffRouter — Module 5."""

from __future__ import annotations

import pytest

from app.agents.orchestrator.router import HandoffRouter
from app.llm.client import FakeLLMClient
from app.llm.schemas import ExtractionOutput, ExtractedTermSchema
from app.models.audit import AuditEventType
from app.models.compliance import ComplianceResult, ComplianceStatus
from app.models.evidence import EvidenceSnippet
from app.models.handoff import Handoff, HandoffPriority, HandoffStatus, HandoffType
from app.models.policy import PolicyRule, RuleCategory, RuleSeverity
from app.models.state import WorkflowState
from app.models.terms import TermCategory


@pytest.fixture
def router() -> HandoffRouter:
    return HandoffRouter()


@pytest.fixture
def sample_state() -> WorkflowState:
    rule = PolicyRule(
        rule_id="POLICY-001",
        name="interest_rate",
        description="Interest rate limit",
        category=RuleCategory.FINANCIAL,
        operator="<=",
        threshold="10%",
        severity=RuleSeverity.HIGH,
    )
    snippet = EvidenceSnippet(
        evidence_id="EV-001",
        document_id="DEAL-001",
        page_number=1,
        text="The Interest Rate is 7.5%.",
        char_start=0,
        char_end=25,
    )
    handoff = Handoff(
        source_agent="compliance",
        target_agent="term_extraction",
        handoff_type=HandoffType.CLARIFICATION_REQUIRED,
        reason="Missing interest_rate term.",
        rule_ids=["POLICY-001"],
        requested_term_names=["interest_rate"],
        priority=HandoffPriority.HIGH,
    )
    comp_result = ComplianceResult(
        compliance_id="COMP-001",
        rule_id="POLICY-001",
        status=ComplianceStatus.INSUFFICIENT_EVIDENCE,
        rationale="Missing term.",
        confidence=0.0,
        agent_name="compliance",
    )
    return WorkflowState(
        document_id="DEAL-001",
        policy_rules=[rule],
        evidence_registry={"EV-001": snippet},
        compliance_results={"COMP-001": comp_result},
        pending_handoffs=[handoff],
    )


class TestHandoffRouter:
    def test_route_empty_pending_handoffs(self, router: HandoffRouter):
        state = WorkflowState(document_id="DEAL-001")
        output = router.route(state)
        assert output == state

    def test_route_clarification_success(self, router: HandoffRouter, sample_state: WorkflowState):
        fake_output = ExtractionOutput(
            terms=[
                ExtractedTermSchema(
                    name="interest_rate",
                    value="7.5%",
                    category=TermCategory.RATE,
                    confidence=0.95,
                    evidence_ids=["EV-001"],
                )
            ]
        )
        llm = FakeLLMClient(response=fake_output)

        new_state = router.route(sample_state, llm_client=llm, max_attempts=1)

        # Check handoff resolved
        assert len(new_state.pending_handoffs) == 0
        assert len(new_state.resolved_handoffs) == 1
        assert new_state.resolved_handoffs[0].status == HandoffStatus.RESOLVED

        # Check term extracted
        assert len(new_state.extracted_terms) == 1
        term = list(new_state.extracted_terms.values())[0]
        assert term.name == "interest_rate"
        assert term.value == "7.5%"

        # Check compliance re-evaluated -> PASS
        comp_res = new_state.compliance_results["COMP-001"]
        assert comp_res.status == ComplianceStatus.PASS

        # Check attempt count incremented
        assert new_state.clarification_attempts["POLICY-001"] == 1

        # Check audit events recorded
        event_types = [e.event_type for e in new_state.audit_events]
        assert AuditEventType.HANDOFF_ROUTING_STARTED in event_types
        assert AuditEventType.CLARIFICATION_STARTED in event_types
        assert AuditEventType.CLARIFICATION_COMPLETED in event_types
        assert AuditEventType.COMPLIANCE_RECHECK_STARTED in event_types
        assert AuditEventType.COMPLIANCE_RECHECK_COMPLETED in event_types
        assert AuditEventType.HANDOFF_ROUTING_COMPLETED in event_types

    def test_loop_prevention_escalates_to_human(self, router: HandoffRouter, sample_state: WorkflowState):
        # Set clarification_attempts to max (1) to simulate prior attempt
        state_at_limit = sample_state.model_copy(
            update={"clarification_attempts": {"POLICY-001": 1}}
        )

        llm = FakeLLMClient(response=ExtractionOutput(terms=[]))
        new_state = router.route(state_at_limit, llm_client=llm, max_attempts=1)

        # Check handoff resolved & escalated
        assert len(new_state.pending_handoffs) == 0
        assert len(new_state.resolved_handoffs) == 1
        assert len(new_state.escalations) == 1
        escalation = new_state.escalations[0]
        assert escalation["rule_id"] == "POLICY-001"
        assert escalation["status"] == "ESCALATION_REQUIRED"

        # Check audit event
        event_types = [e.event_type for e in new_state.audit_events]
        assert AuditEventType.LOOP_LIMIT_REACHED in event_types
