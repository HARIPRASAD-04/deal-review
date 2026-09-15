"""Unit tests for the PolicyRule model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models.policy import PolicyRule, RuleCategory, RuleSeverity


class TestPolicyRuleCreation:
    def test_valid_policy_rule(self):
        rule = PolicyRule(
            rule_id="POLICY-001",
            name="maximum_facility_amount",
            description="Facility must not exceed INR 8 crore.",
            category=RuleCategory.FINANCIAL,
            operator="<=",
            threshold="INR 8 crore",
            severity=RuleSeverity.HIGH,
        )
        assert rule.rule_id == "POLICY-001"
        assert rule.severity == RuleSeverity.HIGH
        assert rule.is_mandatory is True  # Default.

    def test_optional_fields(self):
        rule = PolicyRule(
            rule_id="POLICY-004",
            name="collateral_required",
            description="Collateral must be provided.",
            category=RuleCategory.COLLATERAL,
            operator="exists",
            threshold="true",
            severity=RuleSeverity.CRITICAL,
            applicability="term_loan",
            reference="Credit Policy v1.0 §6.1",
        )
        assert rule.applicability == "term_loan"
        assert rule.reference == "Credit Policy v1.0 §6.1"

    def test_not_mandatory(self):
        rule = PolicyRule(
            rule_id="POLICY-010",
            name="optional_rule",
            description="Optional guideline.",
            category=RuleCategory.GENERAL,
            operator=">=",
            threshold="0",
            severity=RuleSeverity.LOW,
            is_mandatory=False,
        )
        assert rule.is_mandatory is False

    def test_semantic_operator(self):
        """Semantic rules should be expressible."""
        rule = PolicyRule(
            rule_id="POLICY-020",
            name="collateral_tangibility",
            description="Collateral must be tangible fixed assets.",
            category=RuleCategory.COLLATERAL,
            operator="semantic",
            threshold="tangible_fixed_asset",
            severity=RuleSeverity.HIGH,
        )
        assert rule.operator == "semantic"

    def test_policy_is_immutable(self):
        rule = PolicyRule(
            rule_id="POLICY-001",
            name="test",
            description="Test.",
            category=RuleCategory.FINANCIAL,
            operator="<=",
            threshold="100",
            severity=RuleSeverity.MEDIUM,
        )
        with pytest.raises(Exception):
            rule.threshold = "200"  # type: ignore[misc]


class TestPolicyRuleValidation:
    def test_invalid_rule_id_format_raises(self):
        with pytest.raises(ValidationError):
            PolicyRule(
                rule_id="POL-001",  # Wrong prefix.
                name="test",
                description="Test.",
                category=RuleCategory.FINANCIAL,
                operator="<=",
                threshold="100",
                severity=RuleSeverity.HIGH,
            )

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError):
            PolicyRule(
                rule_id="POLICY-001",
                name="",
                description="Test.",
                category=RuleCategory.FINANCIAL,
                operator="<=",
                threshold="100",
                severity=RuleSeverity.HIGH,
            )

    def test_empty_description_raises(self):
        with pytest.raises(ValidationError):
            PolicyRule(
                rule_id="POLICY-001",
                name="test",
                description="",
                category=RuleCategory.FINANCIAL,
                operator="<=",
                threshold="100",
                severity=RuleSeverity.HIGH,
            )

    def test_invalid_category_raises(self):
        with pytest.raises(ValidationError):
            PolicyRule(
                rule_id="POLICY-001",
                name="test",
                description="Test.",
                category="unknown_category",  # type: ignore[arg-type]
                operator="<=",
                threshold="100",
                severity=RuleSeverity.HIGH,
            )

    def test_all_severities(self):
        for severity in RuleSeverity:
            rule = PolicyRule(
                rule_id="POLICY-001",
                name="test",
                description="Test.",
                category=RuleCategory.GENERAL,
                operator="==",
                threshold="true",
                severity=severity,
            )
            assert rule.severity == severity
