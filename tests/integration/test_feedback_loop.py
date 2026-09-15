"""Integration tests for the Compliance → Extraction feedback loop — Module 5."""

from __future__ import annotations

import pytest

from app.llm.client import FakeLLMClient
from app.llm.schemas import ExtractionOutput, ExtractedTermSchema
from app.models.audit import AuditEventType
from app.models.compliance import ComplianceStatus
from app.models.evidence import EvidenceSnippet
from app.models.policy import PolicyRule, RuleCategory, RuleSeverity
from app.models.state import WorkflowState
from app.models.terms import TermCategory
from app.orchestration.graph import build_graph


class TestFeedbackLoopIntegration:
    def test_full_feedback_loop_resolves_insufficient_evidence(self):
        """Verify end-to-end that an INSUFFICIENT_EVIDENCE rule triggers clarification and re-checks to PASS."""
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
            text="The Interest Rate shall be 8.5% per annum.",
            char_start=0,
            char_end=42,
        )
        initial_state = WorkflowState(
            document_id="DEAL-001",
            policy_rules=[rule],
            evidence_registry={"EV-001": snippet},
        )

        # Pass 1 (initial extraction) returns no terms; Pass 2 (clarification) returns interest_rate
        initial_response = ExtractionOutput(terms=[])
        clarification_response = ExtractionOutput(
            terms=[
                ExtractedTermSchema(
                    name="interest_rate",
                    value="8.5%",
                    category=TermCategory.RATE,
                    confidence=0.9,
                    evidence_ids=["EV-001"],
                )
            ]
        )
        llm_client = FakeLLMClient(responses=[initial_response, clarification_response])
        graph = build_graph(llm_client=llm_client)

        result_dict = graph.invoke(initial_state)
        final_state = WorkflowState(**result_dict) if isinstance(result_dict, dict) else result_dict

        # Verify term was extracted during clarification loop
        assert "TERM-001" in final_state.extracted_terms
        assert final_state.extracted_terms["TERM-001"].value == "8.5%"

        # Verify compliance recheck updated to PASS
        comp_res = final_state.compliance_results["COMP-001"]
        assert comp_res.status == ComplianceStatus.PASS

        # Verify handoff resolved
        assert len(final_state.resolved_handoffs) == 1

        # Verify audit event sequence
        event_types = [e.event_type for e in final_state.audit_events]
        assert AuditEventType.CLARIFICATION_STARTED in event_types
        assert AuditEventType.CLARIFICATION_COMPLETED in event_types
        assert AuditEventType.COMPLIANCE_RECHECK_COMPLETED in event_types

    def test_feedback_loop_failed_clarification_no_infinite_loop(self):
        """Verify that when clarification fails to find terms, state completes without looping infinitely."""
        rule = PolicyRule(
            rule_id="POLICY-001",
            name="missing_term",
            description="Non-existent term rule",
            category=RuleCategory.FINANCIAL,
            operator="==",
            threshold="Value",
            severity=RuleSeverity.MEDIUM,
        )
        snippet = EvidenceSnippet(
            evidence_id="EV-001",
            document_id="DEAL-001",
            page_number=1,
            text="The document discusses missing_term in section 5.",
            char_start=0,
            char_end=48,
        )
        initial_state = WorkflowState(
            document_id="DEAL-001",
            policy_rules=[rule],
            evidence_registry={"EV-001": snippet},
        )

        llm_client = FakeLLMClient(response=ExtractionOutput(terms=[]))
        graph = build_graph(llm_client=llm_client)

        result_dict = graph.invoke(initial_state)
        final_state = WorkflowState(**result_dict) if isinstance(result_dict, dict) else result_dict

        assert final_state.compliance_results["COMP-001"].status == ComplianceStatus.INSUFFICIENT_EVIDENCE
        assert len(final_state.resolved_handoffs) == 1
        event_types = [e.event_type for e in final_state.audit_events]
        assert AuditEventType.CLARIFICATION_FAILED in event_types

    def test_feedback_loop_no_evidence_escalates_to_human_without_llm_call(self):
        """Verify that when evidence registry has no trace of a missing term, it escalates to human review without LLM call."""
        rule = PolicyRule(
            rule_id="POLICY-001",
            name="environmental_covenant",
            description="Environmental compliance certificate required",
            category=RuleCategory.FINANCIAL,
            operator="==",
            threshold="Required",
            severity=RuleSeverity.MEDIUM,
        )
        snippet = EvidenceSnippet(
            evidence_id="EV-001",
            document_id="DEAL-001",
            page_number=1,
            text="The Loan carries a fixed Interest Rate of 8.25% per annum.",
            char_start=0,
            char_end=58,
        )
        initial_state = WorkflowState(
            document_id="DEAL-001",
            policy_rules=[rule],
            evidence_registry={"EV-001": snippet},
        )

        llm_client = FakeLLMClient(response=ExtractionOutput(terms=[]))
        graph = build_graph(llm_client=llm_client)

        result_dict = graph.invoke(initial_state)
        final_state = WorkflowState(**result_dict) if isinstance(result_dict, dict) else result_dict

        # Verify escalated to human review in state.escalations
        assert len(final_state.escalations) == 1
        escalation = final_state.escalations[0]
        assert escalation["status"] == "ESCALATION_REQUIRED"
        assert "environmental_covenant" in str(escalation["requested_terms"])
        # Verify LLM was NOT called for clarification (escalated directly to human).
        # call_count == 2: one call for initial extract_terms, one for risk summary executive summary.
        # The clarification path itself made 0 LLM calls (confirmed by escalation path being taken).
        assert llm_client.call_count == 2

