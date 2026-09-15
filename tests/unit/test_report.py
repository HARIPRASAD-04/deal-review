"""Unit tests for FinalDealReviewReport and related schemas — Module 7."""

from app.models.compliance import ComplianceResult, ComplianceStatus
from app.models.report import ComplianceSummary, FinalDealReviewReport, ReviewStatus
from app.models.risk import RiskCategory, RiskFinding, RiskSeverity
from app.models.state import WorkflowRunStatus, WorkflowState


class TestReviewStatusEnum:
    """Test ReviewStatus enum properties and string representation."""

    def test_enum_values(self):
        assert ReviewStatus.COMPLETED.value == "COMPLETED"
        assert ReviewStatus.COMPLETED_WITH_HUMAN_REVIEW.value == "COMPLETED_WITH_HUMAN_REVIEW"
        assert ReviewStatus.FAILED.value == "FAILED"

    def test_enum_string_subclass(self):
        assert isinstance(ReviewStatus.COMPLETED, str)


class TestComplianceSummary:
    """Test ComplianceSummary model validation and defaults."""

    def test_compliance_summary_instantiation(self):
        summary = ComplianceSummary(
            total_rules=5,
            pass_count=3,
            fail_count=1,
            needs_review_count=1,
            insufficient_evidence_count=0,
            overall_status="HAS_FAILURES",
        )
        assert summary.total_rules == 5
        assert summary.pass_count == 3
        assert summary.fail_count == 1
        assert summary.overall_status == "HAS_FAILURES"

    def test_compliance_summary_defaults(self):
        summary = ComplianceSummary()
        assert summary.total_rules == 0
        assert summary.pass_count == 0
        assert summary.fail_count == 0
        assert summary.overall_status == "NO_RULES"


class TestFinalDealReviewReportModel:
    """Test FinalDealReviewReport Pydantic model construction, serialisation, and validation."""

    def test_valid_report_creation(self):
        report = FinalDealReviewReport(
            run_id="RUN-12345678",
            document_id="DEAL-001",
            review_status=ReviewStatus.COMPLETED,
            review_status_rationale="All 3 rules passed cleanly.",
            executive_summary="This deal is low risk.",
            risk_findings=[],
            compliance_summary=ComplianceSummary(total_rules=3, pass_count=3, overall_status="ALL_PASS"),
            missing_information=[],
            follow_ups=["Obtain final signature."],
            escalations=[],
            agent_statuses={"orchestrator": {"status": "COMPLETED", "attempt_count": 1}},
            audit_event_count=10,
            metadata={"evidence_count": 15},
        )
        assert report.report_id.startswith("RPT-")
        assert report.run_id == "RUN-12345678"
        assert report.review_status == ReviewStatus.COMPLETED

    def test_report_json_roundtrip(self):
        report = FinalDealReviewReport(
            run_id="RUN-ABCDEF12",
            document_id="DEAL-002",
            review_status=ReviewStatus.COMPLETED_WITH_HUMAN_REVIEW,
            review_status_rationale="1 finding requires human review.",
            executive_summary="Executive summary text.",
            risk_findings=[
                {
                    "risk_id": "RISK-001",
                    "title": "High leverage",
                    "severity": "HIGH",
                    "category": "CREDIT",
                    "description": "Debt ratio exceeded",
                    "requires_human_review": True,
                }
            ],
            compliance_summary=ComplianceSummary(total_rules=1, fail_count=1, overall_status="HAS_FAILURES"),
            missing_information=["Tax returns missing"],
            follow_ups=[],
            escalations=[{"escalation_id": "esc-1", "reason": "High leverage"}],
        )

        json_str = report.model_dump_json()
        deserialised = FinalDealReviewReport.model_validate_json(json_str)

        assert deserialised.report_id == report.report_id
        assert deserialised.review_status == ReviewStatus.COMPLETED_WITH_HUMAN_REVIEW
        assert len(deserialised.risk_findings) == 1
        assert deserialised.risk_findings[0]["risk_id"] == "RISK-001"
        assert deserialised.missing_information == ["Tax returns missing"]
