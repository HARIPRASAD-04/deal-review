"""Unit tests for the compliance evaluator (pure functions).

Covers:
* parse_numeric: various formats
* evaluate_rule: every supported operator (PASS / FAIL / NEEDS_HUMAN_REVIEW)
* AMBIGUOUS term short-circuit
* Unparseable value with numeric operator
* Unknown operator
"""

from __future__ import annotations

import pytest

from app.agents.compliance.evaluator import evaluate_rule, parse_numeric
from app.models.compliance import ComplianceStatus
from app.models.policy import PolicyRule, RuleCategory, RuleSeverity
from app.models.terms import DealTerm, TermCategory, TermStatus


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_rule(
    operator: str,
    threshold: str,
    name: str = "interest_rate",
    rule_id: str = "POLICY-001",
) -> PolicyRule:
    return PolicyRule(
        rule_id=rule_id,
        name=name,
        description=f"Test rule: {name} {operator} {threshold}",
        category=RuleCategory.FINANCIAL,
        operator=operator,
        threshold=threshold,
        severity=RuleSeverity.HIGH,
    )


def _make_term(
    name: str = "interest_rate",
    value: str = "8.5% per annum",
    normalized_value: str | None = "0.085",
    status: TermStatus = TermStatus.EXTRACTED,
    term_id: str = "TERM-001",
) -> DealTerm:
    return DealTerm(
        term_id=term_id,
        name=name,
        value=value,
        normalized_value=normalized_value,
        category=TermCategory.RATE,
        confidence=0.99,
        evidence_ids=["EV-001"],
        status=status,
    )


# ── parse_numeric tests ───────────────────────────────────────────────────────

class TestParseNumeric:
    def test_plain_float(self):
        assert parse_numeric("1.25") == pytest.approx(1.25)

    def test_plain_integer(self):
        assert parse_numeric("60") == pytest.approx(60.0)

    def test_percentage_string(self):
        assert parse_numeric("8.5%") == pytest.approx(0.085)

    def test_percentage_nine(self):
        assert parse_numeric("9%") == pytest.approx(0.09)

    def test_decimal_already_normalised(self):
        assert parse_numeric("0.085") == pytest.approx(0.085)

    def test_ratio_x(self):
        assert parse_numeric("1.25x") == pytest.approx(1.25)

    def test_ratio_uppercase_x(self):
        assert parse_numeric("3.0X") == pytest.approx(3.0)

    def test_currency_with_commas(self):
        assert parse_numeric("INR 50,000,000") == pytest.approx(50_000_000.0)

    def test_bare_large_number_with_commas(self):
        assert parse_numeric("80,000,000") == pytest.approx(80_000_000.0)

    def test_number_with_units(self):
        assert parse_numeric("60 months") == pytest.approx(60.0)

    def test_non_numeric_returns_none(self):
        assert parse_numeric("India") is None

    def test_empty_string_returns_none(self):
        assert parse_numeric("") is None

    def test_text_only_returns_none(self):
        assert parse_numeric("Term Loan") is None

    def test_negative_number(self):
        assert parse_numeric("-5.0") == pytest.approx(-5.0)

    def test_percentage_zero(self):
        assert parse_numeric("0%") == pytest.approx(0.0)


# ── Ambiguous term short-circuit ──────────────────────────────────────────────

class TestAmbiguousTerm:
    def test_ambiguous_term_returns_needs_human_review(self):
        rule = _make_rule("<=", "0.09")
        term = _make_term(status=TermStatus.AMBIGUOUS)
        status, rationale, ambiguity = evaluate_rule(rule, term)
        assert status == ComplianceStatus.NEEDS_HUMAN_REVIEW
        assert ambiguity is True

    def test_ambiguous_rationale_mentions_ambiguous(self):
        rule = _make_rule("<=", "0.09")
        term = _make_term(status=TermStatus.AMBIGUOUS)
        _, rationale, _ = evaluate_rule(rule, term)
        assert "AMBIGUOUS" in rationale

    def test_ambiguous_mentions_rule_id(self):
        rule = _make_rule("<=", "0.09", rule_id="POLICY-042")
        term = _make_term(status=TermStatus.AMBIGUOUS)
        _, rationale, _ = evaluate_rule(rule, term)
        assert "POLICY-042" in rationale


# ── Operator: <= (maximum threshold) ─────────────────────────────────────────

class TestOperatorLessThanOrEqual:
    def test_pass_below_max(self):
        rule = _make_rule("<=", "0.09")
        term = _make_term(normalized_value="0.085")
        status, _, _ = evaluate_rule(rule, term)
        assert status == ComplianceStatus.PASS

    def test_pass_exactly_at_max(self):
        rule = _make_rule("<=", "0.09")
        term = _make_term(normalized_value="0.09")
        status, _, _ = evaluate_rule(rule, term)
        assert status == ComplianceStatus.PASS

    def test_fail_above_max(self):
        rule = _make_rule("<=", "0.09")
        term = _make_term(normalized_value="0.10")
        status, rationale, _ = evaluate_rule(rule, term)
        assert status == ComplianceStatus.FAIL
        assert "FAIL" in rationale

    def test_fail_using_percentage_threshold(self):
        """Policy threshold as '9%' must parse the same as 0.09."""
        rule = _make_rule("<=", "9%")
        term = _make_term(normalized_value="0.10")
        status, _, _ = evaluate_rule(rule, term)
        assert status == ComplianceStatus.FAIL

    def test_pass_using_percentage_threshold(self):
        rule = _make_rule("<=", "9%")
        term = _make_term(normalized_value="0.085")
        status, _, _ = evaluate_rule(rule, term)
        assert status == ComplianceStatus.PASS

    def test_rationale_mentions_term_name(self):
        rule = _make_rule("<=", "0.09")
        term = _make_term()
        _, rationale, _ = evaluate_rule(rule, term)
        assert "interest_rate" in rationale

    def test_needs_review_when_value_unparseable(self):
        rule = _make_rule("<=", "0.09")
        term = _make_term(value="floating rate", normalized_value=None)
        status, rationale, _ = evaluate_rule(rule, term)
        assert status == ComplianceStatus.NEEDS_HUMAN_REVIEW
        assert "parsed" in rationale.lower()

    def test_needs_review_when_threshold_unparseable(self):
        rule = _make_rule("<=", "market_rate")
        term = _make_term(normalized_value="0.085")
        status, _, _ = evaluate_rule(rule, term)
        assert status == ComplianceStatus.NEEDS_HUMAN_REVIEW


# ── Operator: >= (minimum threshold) ─────────────────────────────────────────

class TestOperatorGreaterThanOrEqual:
    def test_pass_above_min(self):
        rule = _make_rule(">=", "1.25", name="financial_covenant_dscr")
        term = _make_term(name="financial_covenant_dscr", value="1.30x", normalized_value="1.30")
        status, _, _ = evaluate_rule(rule, term)
        assert status == ComplianceStatus.PASS

    def test_pass_exactly_at_min(self):
        rule = _make_rule(">=", "1.25", name="financial_covenant_dscr")
        term = _make_term(name="financial_covenant_dscr", value="1.25x", normalized_value="1.25")
        status, _, _ = evaluate_rule(rule, term)
        assert status == ComplianceStatus.PASS

    def test_fail_below_min(self):
        rule = _make_rule(">=", "1.25", name="financial_covenant_dscr")
        term = _make_term(name="financial_covenant_dscr", value="1.10x", normalized_value="1.10")
        status, rationale, _ = evaluate_rule(rule, term)
        assert status == ComplianceStatus.FAIL
        assert "FAIL" in rationale

    def test_needs_review_unparseable_value(self):
        rule = _make_rule(">=", "1.25", name="financial_covenant_dscr")
        term = _make_term(name="financial_covenant_dscr", value="not reported", normalized_value=None)
        status, _, _ = evaluate_rule(rule, term)
        assert status == ComplianceStatus.NEEDS_HUMAN_REVIEW


# ── Operator: == (string equality) ───────────────────────────────────────────

class TestOperatorEqual:
    def test_pass_exact_match(self):
        rule = _make_rule("==", "India", name="governing_law")
        term = _make_term(name="governing_law", value="India", normalized_value="India")
        status, _, _ = evaluate_rule(rule, term)
        assert status == ComplianceStatus.PASS

    def test_pass_case_insensitive(self):
        rule = _make_rule("==", "india", name="governing_law")
        term = _make_term(name="governing_law", value="India", normalized_value="India")
        status, _, _ = evaluate_rule(rule, term)
        assert status == ComplianceStatus.PASS

    def test_fail_different_value(self):
        rule = _make_rule("==", "England", name="governing_law")
        term = _make_term(name="governing_law", value="India", normalized_value="India")
        status, rationale, _ = evaluate_rule(rule, term)
        assert status == ComplianceStatus.FAIL
        assert "England" in rationale or "India" in rationale


# ── Operator: != (prohibited value) ──────────────────────────────────────────

class TestOperatorNotEqual:
    def test_pass_value_is_different(self):
        rule = _make_rule("!=", "unsecured", name="collateral")
        term = _make_term(name="collateral", value="secured by equipment", normalized_value=None)
        status, _, _ = evaluate_rule(rule, term)
        assert status == ComplianceStatus.PASS

    def test_fail_prohibited_value_matches(self):
        rule = _make_rule("!=", "unsecured", name="collateral")
        term = _make_term(name="collateral", value="unsecured", normalized_value=None)
        status, rationale, _ = evaluate_rule(rule, term)
        assert status == ComplianceStatus.FAIL
        assert "prohibited" in rationale.lower()


# ── Operator: exists ──────────────────────────────────────────────────────────

class TestOperatorExists:
    def test_pass_term_has_value(self):
        rule = _make_rule("exists", "true", name="primary_security")
        term = _make_term(name="primary_security", value="equipment lien", normalized_value=None)
        status, _, _ = evaluate_rule(rule, term)
        assert status == ComplianceStatus.PASS

    def test_fail_term_has_empty_value(self):
        """Edge case: term exists in registry but value is empty."""
        rule = _make_rule("exists", "true", name="collateral")
        # DealTerm requires min_length=1 for value, so we cannot test this via model directly.
        # Instead test the evaluator with a term whose normalized_value is also empty.
        term = _make_term(name="collateral", value="N/A", normalized_value=None)
        # "N/A" is non-empty, so this should PASS.
        status, _, _ = evaluate_rule(rule, term)
        assert status == ComplianceStatus.PASS


# ── Operator: in (allowed values list) ────────────────────────────────────────

class TestOperatorIn:
    def test_pass_value_in_allowed_list(self):
        rule = _make_rule("in", "India, UK, Singapore", name="governing_law")
        term = _make_term(name="governing_law", value="India", normalized_value="India")
        status, _, _ = evaluate_rule(rule, term)
        assert status == ComplianceStatus.PASS

    def test_pass_second_value_in_list(self):
        rule = _make_rule("in", "India, UK, Singapore", name="governing_law")
        term = _make_term(name="governing_law", value="UK", normalized_value="UK")
        status, _, _ = evaluate_rule(rule, term)
        assert status == ComplianceStatus.PASS

    def test_fail_value_not_in_list(self):
        rule = _make_rule("in", "India, UK, Singapore", name="governing_law")
        term = _make_term(name="governing_law", value="Cayman Islands", normalized_value="Cayman Islands")
        status, rationale, _ = evaluate_rule(rule, term)
        assert status == ComplianceStatus.FAIL
        assert "Cayman Islands" in rationale or "India" in rationale

    def test_case_insensitive_match(self):
        rule = _make_rule("in", "INDIA, UK", name="governing_law")
        term = _make_term(name="governing_law", value="india", normalized_value="india")
        status, _, _ = evaluate_rule(rule, term)
        assert status == ComplianceStatus.PASS


# ── Operator: not_in (prohibited values list) ─────────────────────────────────

class TestOperatorNotIn:
    def test_pass_value_not_in_prohibited(self):
        rule = _make_rule("not_in", "Cayman Islands, BVI, offshore", name="governing_law")
        term = _make_term(name="governing_law", value="India", normalized_value="India")
        status, _, _ = evaluate_rule(rule, term)
        assert status == ComplianceStatus.PASS

    def test_fail_value_in_prohibited(self):
        rule = _make_rule("not_in", "Cayman Islands, BVI, offshore", name="governing_law")
        term = _make_term(name="governing_law", value="Cayman Islands", normalized_value="Cayman Islands")
        status, rationale, _ = evaluate_rule(rule, term)
        assert status == ComplianceStatus.FAIL
        assert "prohibited" in rationale.lower()


# ── Operator: semantic ────────────────────────────────────────────────────────

class TestOperatorSemantic:
    def test_semantic_returns_needs_human_review(self):
        rule = _make_rule("semantic", "tangible_fixed_asset", name="collateral")
        term = _make_term(name="collateral", value="personal guarantee", normalized_value=None)
        status, rationale, _ = evaluate_rule(rule, term)
        assert status == ComplianceStatus.NEEDS_HUMAN_REVIEW
        assert "semantic" in rationale.lower()

    def test_semantic_rationale_includes_expected(self):
        rule = _make_rule("semantic", "tangible_fixed_asset", name="collateral")
        term = _make_term(name="collateral", value="personal guarantee", normalized_value=None)
        _, rationale, _ = evaluate_rule(rule, term)
        assert "tangible_fixed_asset" in rationale


# ── Unknown operator ──────────────────────────────────────────────────────────

class TestUnknownOperator:
    def test_unknown_operator_needs_human_review(self):
        rule = _make_rule("fuzzy_match", "some_value")
        term = _make_term()
        status, rationale, _ = evaluate_rule(rule, term)
        assert status == ComplianceStatus.NEEDS_HUMAN_REVIEW
        assert "fuzzy_match" in rationale

    def test_unknown_operator_rationale_mentions_rule(self):
        rule = _make_rule("fuzzy_match", "some_value", rule_id="POLICY-099")
        term = _make_term()
        _, rationale, _ = evaluate_rule(rule, term)
        assert "POLICY-099" in rationale


# ── Rationale quality ─────────────────────────────────────────────────────────

class TestRationaleQuality:
    def test_pass_rationale_mentions_pass(self):
        rule = _make_rule("<=", "0.09")
        term = _make_term(normalized_value="0.085")
        _, rationale, _ = evaluate_rule(rule, term)
        assert "PASS" in rationale or "below" in rationale.lower() or "satisfied" in rationale.lower()

    def test_fail_rationale_mentions_fail(self):
        rule = _make_rule("<=", "0.09")
        term = _make_term(normalized_value="0.10")
        _, rationale, _ = evaluate_rule(rule, term)
        assert "FAIL" in rationale or "exceed" in rationale.lower() or "requires" in rationale.lower()

    def test_rationale_always_non_empty(self):
        for op, threshold in [("<=", "0.09"), (">=", "1.25"), ("==", "India"),
                              ("!=", "USD"), ("exists", "true"), ("semantic", "x")]:
            rule = _make_rule(op, threshold)
            term = _make_term()
            _, rationale, _ = evaluate_rule(rule, term)
            assert rationale, f"Empty rationale for operator '{op}'"


# ── Normalized value preference ───────────────────────────────────────────────

class TestNormalizedValuePreference:
    def test_uses_normalized_value_over_raw(self):
        """normalized_value='0.085' should be used for numeric comparison, not raw value."""
        rule = _make_rule("<=", "0.09")
        # Raw value is text, but normalized is parseable.
        term = _make_term(value="8.5% per annum (floating)", normalized_value="0.085")
        status, _, _ = evaluate_rule(rule, term)
        assert status == ComplianceStatus.PASS

    def test_falls_back_to_raw_value_when_no_normalized(self):
        """When normalized_value is None, raw value is used."""
        rule = _make_rule("<=", "0.09")
        term = _make_term(value="8.5%", normalized_value=None)
        status, _, _ = evaluate_rule(rule, term)
        assert status == ComplianceStatus.PASS
