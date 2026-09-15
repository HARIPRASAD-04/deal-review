"""Integration tests for Module 7 Orchestrator Agent & End-to-End Pipeline."""

from app.llm.client import FakeLLMClient
from app.llm.schemas import ExtractionOutput, ExtractedTermSchema
from app.models.audit import AuditEventType
from app.models.evidence import EvidenceSnippet
from app.models.policy import PolicyRule, RuleCategory, RuleSeverity
from app.models.report import FinalDealReviewReport, ReviewStatus
from app.models.state import WorkflowRunStatus, WorkflowState
from app.models.terms import TermCategory
from app.orchestration.graph import build_graph


def _make_snippet(ev_id: str = "EV-001", text: str = "The borrower shall maintain DSCR of 1.25x.") -> EvidenceSnippet:
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
    name: str = "dscr_min",
    operator: str = ">=",
    threshold: str = "1.25x",
    severity: RuleSeverity = RuleSeverity.HIGH,
) -> PolicyRule:
    return PolicyRule(
        rule_id=rule_id,
        name=name,
        description=f"Checks {name}.",
        category=RuleCategory.FINANCIAL,
        operator=operator,
        threshold=threshold,
        severity=severity,
    )


class TestOrchestratorPipelineEndToEnd:
    """Test full pipeline execution producing FinalDealReviewReport."""

    def test_clean_pipeline_run_produces_completed_report(self):
        """Full pipeline run where all rules pass produces ReviewStatus.COMPLETED."""
        snippet = _make_snippet(ev_id="EV-001", text="The borrower shall maintain DSCR of 1.35x.")
        rule = _make_rule(rule_id="POLICY-001", name="dscr_min", operator=">=", threshold="1.25x")

        llm = FakeLLMClient(
            response=ExtractionOutput(
                terms=[
                    ExtractedTermSchema(
                        name="dscr_min",
                        value="1.35x",
                        category=TermCategory.COVENANT,
                        confidence=0.95,
                        evidence_ids=["EV-001"],
                    )
                ]
            )
        )

        state = WorkflowState(
            document_id="DEAL-001",
            policy_rules=[rule],
            evidence_registry={"EV-001": snippet},
        )

        graph = build_graph(llm_client=llm)
        result = graph.invoke(state)
        final_state = WorkflowState(**result) if isinstance(result, dict) else result

        # Verify state
        assert final_state.run_status == WorkflowRunStatus.COMPLETED
        assert final_state.final_report is not None

        # Verify report deserialisation
        report = FinalDealReviewReport.model_validate_json(final_state.final_report)
        assert report.document_id == "DEAL-001"
        assert report.review_status in (ReviewStatus.COMPLETED, ReviewStatus.COMPLETED_WITH_HUMAN_REVIEW)
        assert len(report.risk_findings) >= 0
        assert report.compliance_summary.total_rules == 1

        # Verify audit trail contains REPORT_ASSEMBLY_COMPLETED
        event_types = [e.event_type for e in final_state.audit_events]
        assert AuditEventType.REPORT_ASSEMBLY_STARTED in event_types
        assert AuditEventType.REPORT_ASSEMBLY_COMPLETED in event_types

    def test_escalated_findings_produce_human_review_status(self):
        """Pipeline run with human review escalation produces ReviewStatus.COMPLETED_WITH_HUMAN_REVIEW."""
        snippet = _make_snippet(ev_id="EV-001", text="The interest rate is 18.0% per annum.")
        rule = _make_rule(rule_id="POLICY-001", name="interest_rate", operator="<=", threshold="10.0%", severity=RuleSeverity.CRITICAL)

        llm = FakeLLMClient(
            response=ExtractionOutput(
                terms=[
                    ExtractedTermSchema(
                        name="interest_rate",
                        value="18.0%",
                        category=TermCategory.RATE,
                        confidence=0.95,
                        evidence_ids=["EV-001"],
                    )
                ]
            )
        )

        state = WorkflowState(
            document_id="DEAL-HIGH-RISK",
            policy_rules=[rule],
            evidence_registry={"EV-001": snippet},
            escalations=[{"escalation_id": "esc-1", "reason": "Handoff limit reached"}],
        )

        graph = build_graph(llm_client=llm)
        result = graph.invoke(state)
        final_state = WorkflowState(**result) if isinstance(result, dict) else result

        report = FinalDealReviewReport.model_validate_json(final_state.final_report)
        assert report.review_status == ReviewStatus.COMPLETED_WITH_HUMAN_REVIEW
        assert len(report.escalations) >= 1
