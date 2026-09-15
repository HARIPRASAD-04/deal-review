"""Unit tests for the ComplianceResult model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models.compliance import ComplianceResult, ComplianceStatus


class TestComplianceResultCreation:
    def test_valid_pass_result(self):
        result = ComplianceResult(
            compliance_id="COMP-001",
            rule_id="POLICY-004",
            status=ComplianceStatus.PASS,
            rationale="Collateral of INR 15 crore covers 1.5x the facility amount.",
            term_ids=["TERM-004"],
            evidence_ids=["EV-004"],
            confidence=0.95,
            agent_name="compliance",
        )
        assert result.status == ComplianceStatus.PASS
        assert result.ambiguity_detected is False  # Default.

    def test_valid_fail_result(self):
        result = ComplianceResult(
            compliance_id="COMP-002",
            rule_id="POLICY-001",
            status=ComplianceStatus.FAIL,
            rationale="Facility of INR 10 crore exceeds maximum of INR 8 crore.",
            term_ids=["TERM-001"],
            evidence_ids=["EV-001"],
            confidence=0.99,
            agent_name="compliance",
        )
        assert result.status == ComplianceStatus.FAIL

    def test_needs_human_review_status(self):
        result = ComplianceResult(
            compliance_id="COMP-003",
            rule_id="POLICY-003",
            status=ComplianceStatus.NEEDS_HUMAN_REVIEW,
            rationale="DSCR of 1.18 is below 1.25 threshold but borrower claims additional income.",
            term_ids=["TERM-003"],
            evidence_ids=["EV-003"],
            confidence=0.6,
            agent_name="compliance",
            ambiguity_detected=True,
        )
        assert result.ambiguity_detected is True

    def test_insufficient_evidence_status(self):
        result = ComplianceResult(
            compliance_id="COMP-004",
            rule_id="POLICY-002",
            status=ComplianceStatus.INSUFFICIENT_EVIDENCE,
            rationale="Interest rate clause is missing from extracted terms.",
            term_ids=[],
            evidence_ids=[],
            confidence=0.0,
            agent_name="compliance",
        )
        assert result.status == ComplianceStatus.INSUFFICIENT_EVIDENCE

    def test_all_compliance_statuses(self):
        for i, status in enumerate(ComplianceStatus, start=1):
            result = ComplianceResult(
                compliance_id=f"COMP-{i:03d}",
                rule_id="POLICY-001",
                status=status,
                rationale="Test.",
                confidence=0.5,
                agent_name="compliance",
            )
            assert result.status == status


class TestComplianceResultValidation:
    def test_invalid_compliance_id_raises(self):
        with pytest.raises(ValidationError):
            ComplianceResult(
                compliance_id="C-001",  # Wrong prefix.
                rule_id="POLICY-001",
                status=ComplianceStatus.PASS,
                rationale="Test.",
                confidence=0.9,
                agent_name="compliance",
            )

    def test_invalid_rule_id_raises(self):
        with pytest.raises(ValidationError):
            ComplianceResult(
                compliance_id="COMP-001",
                rule_id="RULE-001",  # Wrong prefix.
                status=ComplianceStatus.PASS,
                rationale="Test.",
                confidence=0.9,
                agent_name="compliance",
            )

    def test_confidence_above_1_raises(self):
        with pytest.raises(ValidationError):
            ComplianceResult(
                compliance_id="COMP-001",
                rule_id="POLICY-001",
                status=ComplianceStatus.PASS,
                rationale="Test.",
                confidence=1.5,
                agent_name="compliance",
            )

    def test_confidence_below_0_raises(self):
        with pytest.raises(ValidationError):
            ComplianceResult(
                compliance_id="COMP-001",
                rule_id="POLICY-001",
                status=ComplianceStatus.PASS,
                rationale="Test.",
                confidence=-0.1,
                agent_name="compliance",
            )

    def test_invalid_status_raises(self):
        with pytest.raises(ValidationError):
            ComplianceResult(
                compliance_id="COMP-001",
                rule_id="POLICY-001",
                status="APPROVED",  # type: ignore[arg-type]
                rationale="Test.",
                confidence=0.9,
                agent_name="compliance",
            )

    def test_empty_agent_name_raises(self):
        with pytest.raises(ValidationError):
            ComplianceResult(
                compliance_id="COMP-001",
                rule_id="POLICY-001",
                status=ComplianceStatus.PASS,
                rationale="Test.",
                confidence=0.9,
                agent_name="",
            )

    def test_empty_rationale_raises(self):
        with pytest.raises(ValidationError):
            ComplianceResult(
                compliance_id="COMP-001",
                rule_id="POLICY-001",
                status=ComplianceStatus.PASS,
                rationale="",
                confidence=0.9,
                agent_name="compliance",
            )
