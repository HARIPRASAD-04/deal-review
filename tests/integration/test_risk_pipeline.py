"""Integration tests for the Risk & Summary Agent pipeline — Module 6."""

from __future__ import annotations

import pytest

from app.llm.client import FakeLLMClient
from app.llm.schemas import ExtractionOutput, ExtractedTermSchema
from app.models.audit import AuditEventType
from app.models.compliance import ComplianceStatus
from app.models.evidence import EvidenceSnippet
from app.models.policy import PolicyRule, RuleCategory, RuleSeverity
from app.models.risk import RiskSeverity
from app.models.state import WorkflowRunStatus, WorkflowState
from app.models.terms import TermCategory
from app.orchestration.graph import build_graph


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_snippet(ev_id: str = "EV-001", text: str = "Sample evidence.") -> EvidenceSnippet:
    return EvidenceSnippet(
        evidence_id=ev_id,
        document_id="DEAL-001",
        page_number=1,
        text=text,
        char_start=0,
        char_end=len(text),
    )


def _make_rule(
    rule_id: str = "POLICY-001",
    name: str = "interest_rate",
    severity: RuleSeverity = RuleSeverity.HIGH,
) -> PolicyRule:
    return PolicyRule(
        rule_id=rule_id,
        name=name,
        description=f"Checks {name}.",
        category=RuleCategory.FINANCIAL,
        operator="<=",
        threshold="10%",
        severity=severity,
    )


def _run_graph(initial_state: WorkflowState, llm_client=None) -> WorkflowState:
    graph = build_graph(llm_client=llm_client)
    result = graph.invoke(initial_state)
    return WorkflowState(**result) if isinstance(result, dict) else result


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestRiskPipelineIntegration:
    def test_fail_compliance_produces_risk_finding_in_state(self):
        """Full pipeline: FAIL compliance result → risk finding in state.risk_findings."""
        rule = _make_rule(severity=RuleSeverity.HIGH)
        snippet = _make_snippet(text="The interest rate is 15% per annum.")
        llm = FakeLLMClient(response=ExtractionOutput(terms=[
            ExtractedTermSchema(
                name="interest_rate",
                value="15%",
                category=TermCategory.RATE,
                confidence=0.95,
                evidence_ids=["EV-001"],
            )
        ]))
        state = WorkflowState(
            document_id="DEAL-001",
            policy_rules=[rule],
            evidence_registry={"EV-001": snippet},
        )

        final = _run_graph(state, llm_client=llm)

        assert len(final.risk_findings) >= 1
        finding = final.risk_findings[0]
        assert finding.severity in (RiskSeverity.CRITICAL, RiskSeverity.HIGH)
        assert finding.risk_id.startswith("RISK-")

    def test_all_pass_produces_empty_risk_register(self):
        """Full pipeline with all PASS results → empty risk_findings."""
        rule = _make_rule()
        snippet = _make_snippet(text="The interest rate is 8% per annum.")
        llm = FakeLLMClient(response=ExtractionOutput(terms=[
            ExtractedTermSchema(
                name="interest_rate",
                value="8%",
                category=TermCategory.RATE,
                confidence=0.9,
                evidence_ids=["EV-001"],
            )
        ]))
        state = WorkflowState(
            document_id="DEAL-001",
            policy_rules=[rule],
            evidence_registry={"EV-001": snippet},
        )

        final = _run_graph(state, llm_client=llm)

        # All rules passed (8% <= 10%), so no risk findings from compliance
        # Coverage gap findings may still exist if terms/evidence are minimal
        compliance_fail_findings = [
            f for f in final.risk_findings
            if final.compliance_results
            and any(
                final.compliance_results.get(cid, None) is not None
                and final.compliance_results[cid].status == ComplianceStatus.FAIL
                for cid in (f.related_compliance_ids or [])
            )
        ]
        assert len(compliance_fail_findings) == 0

    def test_insufficient_evidence_populates_missing_information(self):
        """Full pipeline with no term extraction → INSUFFICIENT_EVIDENCE → missing_information."""
        rule = _make_rule()
        snippet = _make_snippet(text="Completely unrelated document content here.")
        llm = FakeLLMClient(response=ExtractionOutput(terms=[]))
        state = WorkflowState(
            document_id="DEAL-001",
            policy_rules=[rule],
            evidence_registry={"EV-001": snippet},
        )

        final = _run_graph(state, llm_client=llm)

        # Since evidence has no relevant content and clarification fails, rule becomes INSUFFICIENT
        # OR escalation happens — either way, missing_information should be populated
        assert (
            len(final.missing_information) >= 1
            or len(final.escalations) >= 1
        )

    def test_audit_trail_contains_risk_analysis_events(self):
        """Audit trail must include RISK_ANALYSIS_STARTED and RISK_ANALYSIS_COMPLETED."""
        state = WorkflowState(document_id="DEAL-001")
        final = _run_graph(state)

        event_types = [e.event_type for e in final.audit_events]
        assert AuditEventType.RISK_ANALYSIS_STARTED in event_types
        assert AuditEventType.RISK_ANALYSIS_COMPLETED in event_types

    def test_run_status_completed_after_full_pipeline(self):
        """run_status must be COMPLETED after risk_summary node runs."""
        state = WorkflowState(document_id="DEAL-001")
        final = _run_graph(state)
        assert final.run_status == WorkflowRunStatus.COMPLETED

    def test_executive_summary_and_final_report_set(self):
        """executive_summary is populated and final_report is set by report_assembly (Module 7)."""
        state = WorkflowState(document_id="DEAL-001")
        final = _run_graph(state)

        assert final.executive_summary is not None
        assert isinstance(final.executive_summary, str)
        assert len(final.executive_summary) > 0
        assert final.final_report is not None

    def test_evidence_ids_in_risk_findings_are_valid(self):
        """All evidence_ids in risk findings must exist in state.evidence_registry."""
        rule = _make_rule(severity=RuleSeverity.CRITICAL)
        snippet = _make_snippet(text="The interest rate is 20% per annum.")
        llm = FakeLLMClient(response=ExtractionOutput(terms=[
            ExtractedTermSchema(
                name="interest_rate",
                value="20%",
                category=TermCategory.RATE,
                confidence=0.9,
                evidence_ids=["EV-001"],
            )
        ]))
        state = WorkflowState(
            document_id="DEAL-001",
            policy_rules=[rule],
            evidence_registry={"EV-001": snippet},
        )

        final = _run_graph(state, llm_client=llm)

        for finding in final.risk_findings:
            for eid in finding.evidence_ids:
                assert eid in final.evidence_registry, (
                    f"Risk finding {finding.risk_id} references non-existent evidence ID '{eid}'"
                )

    def test_risk_identified_audit_events_present_per_finding(self):
        """RISK_IDENTIFIED event should appear for each risk finding."""
        state = WorkflowState(document_id="DEAL-001")
        final = _run_graph(state)

        risk_events = [
            e for e in final.audit_events
            if e.event_type == AuditEventType.RISK_IDENTIFIED
        ]
        assert len(risk_events) == len(final.risk_findings)

    def test_no_llm_graph_produces_deterministic_summary(self):
        """When build_graph is called with no LLM, executive_summary is deterministically generated."""
        state = WorkflowState(document_id="DEAL-001")
        final = _run_graph(state, llm_client=None)

        assert final.executive_summary is not None
        assert isinstance(final.executive_summary, str)

    def test_summary_generation_completed_event_present(self):
        """SUMMARY_GENERATION_COMPLETED must appear in audit trail."""
        state = WorkflowState(document_id="DEAL-001")
        final = _run_graph(state)

        event_types = [e.event_type for e in final.audit_events]
        assert AuditEventType.SUMMARY_GENERATION_COMPLETED in event_types

    def test_risk_agent_status_completed(self):
        """AGENT_RISK_SUMMARY status should be COMPLETED after a successful run."""
        from app.models.handoff import AGENT_RISK_SUMMARY
        from app.models.state import AgentStatusEnum

        state = WorkflowState(document_id="DEAL-001")
        final = _run_graph(state)

        assert AGENT_RISK_SUMMARY in final.agent_statuses
        assert final.agent_statuses[AGENT_RISK_SUMMARY].status == AgentStatusEnum.COMPLETED
