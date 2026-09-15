"""Unit tests for report_assembly node — Module 7."""

from app.models.evidence import EvidenceSnippet
from app.models.policy import PolicyRule, RuleCategory, RuleSeverity
from app.models.report import FinalDealReviewReport, ReviewStatus
from app.models.state import AgentStatusEnum, WorkflowRunStatus, WorkflowState
from app.orchestration.graph import make_report_assembly_node


def _make_snippet(ev_id: str = "EV-001") -> EvidenceSnippet:
    return EvidenceSnippet(
        evidence_id=ev_id,
        document_id="DEAL-001",
        page_number=1,
        text="Sample evidence text.",
        char_start=0,
        char_end=20,
    )


def _make_rule(rule_id: str = "POLICY-001") -> PolicyRule:
    return PolicyRule(
        rule_id=rule_id,
        name="dscr_min",
        description="Checks minimum DSCR.",
        category=RuleCategory.FINANCIAL,
        operator=">=",
        threshold="1.25x",
        severity=RuleSeverity.HIGH,
    )


def _make_state(**kwargs) -> WorkflowState:
    defaults = {
        "run_id": "RUN-REPORT-NODE-TEST",
        "document_id": "DEAL-001",
        "evidence_registry": {"EV-001": _make_snippet("EV-001")},
        "policy_rules": [_make_rule("POLICY-001")],
    }
    defaults.update(kwargs)
    return WorkflowState(**defaults)


class TestReportAssemblyNode:
    """Test the report_assembly node factory and execution."""

    def test_report_assembly_node_populates_final_report(self):
        node_fn = make_report_assembly_node()
        state = _make_state()

        updates = node_fn(state)

        assert "final_report" in updates
        assert updates["final_report"] is not None
        assert updates["run_status"] == WorkflowRunStatus.COMPLETED

        # Deserialise and assert report content
        report = FinalDealReviewReport.model_validate_json(updates["final_report"])
        assert report.run_id == "RUN-REPORT-NODE-TEST"
        assert report.review_status == ReviewStatus.COMPLETED

        # Check audit event and agent status
        audit_types = [e.event_type for e in updates["audit_events"]]
        assert "REPORT_ASSEMBLY_STARTED" in audit_types
        assert "REPORT_ASSEMBLY_COMPLETED" in audit_types
        assert updates["agent_statuses"]["orchestrator"].status == AgentStatusEnum.COMPLETED

    def test_report_assembly_node_handles_pipeline_errors(self):
        node_fn = make_report_assembly_node()
        state = _make_state(errors={"ingest_document": "Ingestion failed"})

        updates = node_fn(state)

        assert "final_report" in updates
        assert updates["run_status"] == WorkflowRunStatus.COMPLETED

        report = FinalDealReviewReport.model_validate_json(updates["final_report"])
        assert report.review_status == ReviewStatus.FAILED
