"""Unit tests for the RiskFinding model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models.risk import RiskCategory, RiskFinding, RiskLikelihood, RiskSeverity


class TestRiskFindingCreation:
    def test_valid_risk_finding(self):
        risk = RiskFinding(
            risk_id="RISK-001",
            category=RiskCategory.FINANCIAL,
            title="Facility Amount Exceeds Policy Limit",
            description="The facility amount of INR 10 crore exceeds the maximum INR 8 crore.",
            severity=RiskSeverity.HIGH,
            evidence_ids=["EV-001"],
            related_term_ids=["TERM-001"],
            related_compliance_ids=["COMP-001"],
            recommended_action="Escalate to credit committee for exception approval.",
        )
        assert risk.risk_id == "RISK-001"
        assert risk.requires_human_review is False  # Default.
        assert risk.likelihood == RiskLikelihood.UNKNOWN  # Default.

    def test_human_review_flag(self):
        risk = RiskFinding(
            risk_id="RISK-002",
            category=RiskCategory.CREDIT,
            title="DSCR Below Minimum Threshold",
            description="DSCR of 1.18 is below the required 1.25.",
            severity=RiskSeverity.CRITICAL,
            recommended_action="Reject or request additional security.",
            requires_human_review=True,
        )
        assert risk.requires_human_review is True

    def test_likelihood_values(self):
        for likelihood in RiskLikelihood:
            risk = RiskFinding(
                risk_id="RISK-001",
                category=RiskCategory.FINANCIAL,
                title="Test",
                description="Test risk.",
                severity=RiskSeverity.LOW,
                recommended_action="Monitor.",
                likelihood=likelihood,
            )
            assert risk.likelihood == likelihood

    def test_all_categories(self):
        for cat in RiskCategory:
            risk = RiskFinding(
                risk_id="RISK-001",
                category=cat,
                title="Test",
                description="Test.",
                severity=RiskSeverity.MEDIUM,
                recommended_action="Review.",
            )
            assert risk.category == cat

    def test_all_severities(self):
        for severity in RiskSeverity:
            risk = RiskFinding(
                risk_id="RISK-001",
                category=RiskCategory.OPERATIONAL,
                title="Test",
                description="Test.",
                severity=severity,
                recommended_action="Review.",
            )
            assert risk.severity == severity

    def test_no_related_ids_required(self):
        risk = RiskFinding(
            risk_id="RISK-001",
            category=RiskCategory.LEGAL,
            title="Missing Guarantee Clause",
            description="No personal guarantee found.",
            severity=RiskSeverity.MEDIUM,
            recommended_action="Request guarantee from promoters.",
        )
        assert risk.evidence_ids == []
        assert risk.related_term_ids == []
        assert risk.related_compliance_ids == []


class TestRiskFindingValidation:
    def test_invalid_risk_id_raises(self):
        with pytest.raises(ValidationError):
            RiskFinding(
                risk_id="R-001",  # Wrong prefix.
                category=RiskCategory.FINANCIAL,
                title="Test",
                description="Test.",
                severity=RiskSeverity.HIGH,
                recommended_action="Action.",
            )

    def test_empty_title_raises(self):
        with pytest.raises(ValidationError):
            RiskFinding(
                risk_id="RISK-001",
                category=RiskCategory.FINANCIAL,
                title="",
                description="Test.",
                severity=RiskSeverity.HIGH,
                recommended_action="Action.",
            )

    def test_empty_description_raises(self):
        with pytest.raises(ValidationError):
            RiskFinding(
                risk_id="RISK-001",
                category=RiskCategory.FINANCIAL,
                title="Test",
                description="",
                severity=RiskSeverity.HIGH,
                recommended_action="Action.",
            )

    def test_empty_recommended_action_raises(self):
        with pytest.raises(ValidationError):
            RiskFinding(
                risk_id="RISK-001",
                category=RiskCategory.FINANCIAL,
                title="Test",
                description="Test.",
                severity=RiskSeverity.HIGH,
                recommended_action="",
            )

    def test_invalid_severity_raises(self):
        with pytest.raises(ValidationError):
            RiskFinding(
                risk_id="RISK-001",
                category=RiskCategory.FINANCIAL,
                title="Test",
                description="Test.",
                severity="extreme",  # type: ignore[arg-type]
                recommended_action="Action.",
            )
