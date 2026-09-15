"""Unit tests for ComplianceReviewAgent.

Tests the full agent behaviour (not just the evaluator):
* PASS / FAIL / NEEDS_HUMAN_REVIEW / INSUFFICIENT_EVIDENCE results
* Missing terms → INSUFFICIENT_EVIDENCE (not silent skip)
* Ambiguous terms → NEEDS_HUMAN_REVIEW
* Empty policy rules → empty result list
* Empty extracted terms → all rules become INSUFFICIENT_EVIDENCE
* Multiple rules → independent results, sequential COMP-NNN IDs
* Evidence traceability: term_ids + evidence_ids present in results
* Deterministic: same input → same output
* Per-rule fault isolation: one bad rule does not abort the rest
"""

from __future__ import annotations

import pytest

from app.agents.compliance.compliance_review import ComplianceReviewAgent
from app.models.compliance import ComplianceResult, ComplianceStatus
from app.models.evidence import EvidenceSnippet
from app.models.policy import PolicyRule, RuleCategory, RuleSeverity
from app.models.terms import DealTerm, TermCategory, TermStatus


# ── Factories ─────────────────────────────────────────────────────────────────

def _make_rule(
    rule_id: str,
    name: str,
    operator: str,
    threshold: str,
    severity: RuleSeverity = RuleSeverity.HIGH,
) -> PolicyRule:
    return PolicyRule(
        rule_id=rule_id,
        name=name,
        description=f"Test rule {rule_id}: {name} {operator} {threshold}",
        category=RuleCategory.FINANCIAL,
        operator=operator,
        threshold=threshold,
        severity=severity,
    )


def _make_term(
    term_id: str,
    name: str,
    value: str,
    normalized_value: str | None = None,
    status: TermStatus = TermStatus.EXTRACTED,
    evidence_ids: list[str] | None = None,
) -> DealTerm:
    return DealTerm(
        term_id=term_id,
        name=name,
        value=value,
        normalized_value=normalized_value,
        category=TermCategory.FINANCIAL,
        confidence=0.99,
        evidence_ids=evidence_ids or ["EV-001"],
        status=status,
    )


def _make_snippet(evidence_id: str = "EV-001", page: int = 1) -> EvidenceSnippet:
    return EvidenceSnippet(
        evidence_id=evidence_id,
        document_id="DEAL-001",
        page_number=page,
        text="Sample text for evidence.",
    )


def _agent() -> ComplianceReviewAgent:
    return ComplianceReviewAgent()


# ── Empty inputs ──────────────────────────────────────────────────────────────

class TestEmptyInputs:
    def test_empty_policy_rules_returns_empty_list(self):
        results = _agent().review(
            policy_rules=[],
            extracted_terms={"TERM-001": _make_term("TERM-001", "interest_rate", "8.5%", "0.085")},
            evidence_registry={"EV-001": _make_snippet()},
        )
        assert results == []

    def test_empty_extracted_terms_all_rules_become_insufficient(self):
        rules = [
            _make_rule("POLICY-001", "interest_rate", "<=", "0.09"),
            _make_rule("POLICY-002", "financial_covenant_dscr", ">=", "1.25"),
        ]
        results = _agent().review(
            policy_rules=rules,
            extracted_terms={},
            evidence_registry={},
        )
        assert len(results) == 2
        for r in results:
            assert r.status == ComplianceStatus.INSUFFICIENT_EVIDENCE

    def test_empty_extracted_terms_results_have_rationale(self):
        rules = [_make_rule("POLICY-001", "interest_rate", "<=", "0.09")]
        results = _agent().review(
            policy_rules=rules, extracted_terms={}, evidence_registry={}
        )
        assert results[0].rationale  # non-empty
        assert "interest_rate" in results[0].rationale

    def test_both_empty_returns_empty(self):
        results = _agent().review(
            policy_rules=[], extracted_terms={}, evidence_registry={}
        )
        assert results == []


# ── PASS outcomes ─────────────────────────────────────────────────────────────

class TestPassOutcomes:
    def test_max_threshold_satisfied(self):
        rule = _make_rule("POLICY-001", "interest_rate", "<=", "0.09")
        term = _make_term("TERM-001", "interest_rate", "8.5%", "0.085")
        results = _agent().review(
            policy_rules=[rule],
            extracted_terms={"TERM-001": term},
            evidence_registry={"EV-001": _make_snippet()},
        )
        assert results[0].status == ComplianceStatus.PASS

    def test_min_threshold_satisfied(self):
        rule = _make_rule("POLICY-002", "financial_covenant_dscr", ">=", "1.25")
        term = _make_term("TERM-001", "financial_covenant_dscr", "1.30x", "1.30")
        results = _agent().review(
            policy_rules=[rule],
            extracted_terms={"TERM-001": term},
            evidence_registry={"EV-001": _make_snippet()},
        )
        assert results[0].status == ComplianceStatus.PASS

    def test_allowed_value_matched(self):
        rule = _make_rule("POLICY-003", "governing_law", "in", "India, Singapore, UK")
        term = _make_term("TERM-001", "governing_law", "India", "India")
        results = _agent().review(
            policy_rules=[rule],
            extracted_terms={"TERM-001": term},
            evidence_registry={"EV-001": _make_snippet()},
        )
        assert results[0].status == ComplianceStatus.PASS

    def test_exists_pass(self):
        rule = _make_rule("POLICY-004", "primary_security", "exists", "true")
        term = _make_term("TERM-001", "primary_security", "first charge on machinery")
        results = _agent().review(
            policy_rules=[rule],
            extracted_terms={"TERM-001": term},
            evidence_registry={"EV-001": _make_snippet()},
        )
        assert results[0].status == ComplianceStatus.PASS


# ── FAIL outcomes ─────────────────────────────────────────────────────────────

class TestFailOutcomes:
    def test_value_exceeds_maximum(self):
        rule = _make_rule("POLICY-001", "interest_rate", "<=", "0.09")
        term = _make_term("TERM-001", "interest_rate", "10%", "0.10")
        results = _agent().review(
            policy_rules=[rule],
            extracted_terms={"TERM-001": term},
            evidence_registry={"EV-001": _make_snippet()},
        )
        assert results[0].status == ComplianceStatus.FAIL

    def test_value_below_required_minimum(self):
        rule = _make_rule("POLICY-002", "financial_covenant_dscr", ">=", "1.25")
        term = _make_term("TERM-001", "financial_covenant_dscr", "1.10x", "1.10")
        results = _agent().review(
            policy_rules=[rule],
            extracted_terms={"TERM-001": term},
            evidence_registry={"EV-001": _make_snippet()},
        )
        assert results[0].status == ComplianceStatus.FAIL

    def test_prohibited_value_encountered(self):
        rule = _make_rule("POLICY-005", "governing_law", "not_in", "Cayman Islands, BVI")
        term = _make_term("TERM-001", "governing_law", "Cayman Islands", "Cayman Islands")
        results = _agent().review(
            policy_rules=[rule],
            extracted_terms={"TERM-001": term},
            evidence_registry={"EV-001": _make_snippet()},
        )
        assert results[0].status == ComplianceStatus.FAIL

    def test_wrong_required_value(self):
        rule = _make_rule("POLICY-006", "governing_law", "==", "India")
        term = _make_term("TERM-001", "governing_law", "England", "England")
        results = _agent().review(
            policy_rules=[rule],
            extracted_terms={"TERM-001": term},
            evidence_registry={"EV-001": _make_snippet()},
        )
        assert results[0].status == ComplianceStatus.FAIL

    def test_fail_rationale_is_informative(self):
        rule = _make_rule("POLICY-001", "interest_rate", "<=", "0.09")
        term = _make_term("TERM-001", "interest_rate", "10%", "0.10")
        results = _agent().review(
            policy_rules=[rule],
            extracted_terms={"TERM-001": term},
            evidence_registry={"EV-001": _make_snippet()},
        )
        assert "FAIL" in results[0].rationale or "exceed" in results[0].rationale.lower()


# ── Missing term ──────────────────────────────────────────────────────────────

class TestMissingTerm:
    def test_missing_term_is_insufficient_evidence(self):
        rule = _make_rule("POLICY-001", "interest_rate", "<=", "0.09")
        # No interest_rate term in extracted_terms.
        results = _agent().review(
            policy_rules=[rule],
            extracted_terms={},
            evidence_registry={},
        )
        assert results[0].status == ComplianceStatus.INSUFFICIENT_EVIDENCE

    def test_missing_term_not_silent(self):
        """Missing term must produce a result, not be dropped."""
        rule = _make_rule("POLICY-001", "interest_rate", "<=", "0.09")
        results = _agent().review(
            policy_rules=[rule], extracted_terms={}, evidence_registry={}
        )
        assert len(results) == 1

    def test_missing_term_rationale_mentions_term_name(self):
        rule = _make_rule("POLICY-001", "interest_rate", "<=", "0.09")
        results = _agent().review(
            policy_rules=[rule], extracted_terms={}, evidence_registry={}
        )
        assert "interest_rate" in results[0].rationale

    def test_missing_term_has_empty_term_ids(self):
        rule = _make_rule("POLICY-001", "interest_rate", "<=", "0.09")
        results = _agent().review(
            policy_rules=[rule], extracted_terms={}, evidence_registry={}
        )
        assert results[0].term_ids == []
        assert results[0].evidence_ids == []


# ── Ambiguous term ────────────────────────────────────────────────────────────

class TestAmbiguousTerm:
    def test_ambiguous_term_returns_needs_human_review(self):
        rule = _make_rule("POLICY-001", "interest_rate", "<=", "0.09")
        term = _make_term(
            "TERM-001", "interest_rate", "8.5% (section 1) / 10% (section 4)",
            status=TermStatus.AMBIGUOUS,
        )
        results = _agent().review(
            policy_rules=[rule],
            extracted_terms={"TERM-001": term},
            evidence_registry={"EV-001": _make_snippet()},
        )
        assert results[0].status == ComplianceStatus.NEEDS_HUMAN_REVIEW

    def test_ambiguous_term_sets_ambiguity_detected(self):
        rule = _make_rule("POLICY-001", "interest_rate", "<=", "0.09")
        term = _make_term("TERM-001", "interest_rate", "8.5%", status=TermStatus.AMBIGUOUS)
        results = _agent().review(
            policy_rules=[rule],
            extracted_terms={"TERM-001": term},
            evidence_registry={"EV-001": _make_snippet()},
        )
        assert results[0].ambiguity_detected is True

    def test_ambiguous_does_not_arbitrarily_select_value(self):
        """Agent must NOT choose 8.5% and return PASS when term is AMBIGUOUS."""
        rule = _make_rule("POLICY-001", "interest_rate", "<=", "0.09")
        term = _make_term("TERM-001", "interest_rate", "8.5%", "0.085", TermStatus.AMBIGUOUS)
        results = _agent().review(
            policy_rules=[rule],
            extracted_terms={"TERM-001": term},
            evidence_registry={"EV-001": _make_snippet()},
        )
        assert results[0].status != ComplianceStatus.PASS


# ── Evidence traceability ─────────────────────────────────────────────────────

class TestEvidenceTraceability:
    def test_result_contains_term_ids(self):
        rule = _make_rule("POLICY-001", "interest_rate", "<=", "0.09")
        term = _make_term("TERM-001", "interest_rate", "8.5%", "0.085")
        results = _agent().review(
            policy_rules=[rule],
            extracted_terms={"TERM-001": term},
            evidence_registry={"EV-001": _make_snippet()},
        )
        assert "TERM-001" in results[0].term_ids

    def test_result_contains_evidence_ids(self):
        rule = _make_rule("POLICY-001", "interest_rate", "<=", "0.09")
        term = _make_term("TERM-001", "interest_rate", "8.5%", "0.085", evidence_ids=["EV-018"])
        results = _agent().review(
            policy_rules=[rule],
            extracted_terms={"TERM-001": term},
            evidence_registry={"EV-018": _make_snippet("EV-018", page=2)},
        )
        assert "EV-018" in results[0].evidence_ids

    def test_evidence_id_can_be_traced_to_registry(self):
        rule = _make_rule("POLICY-001", "interest_rate", "<=", "0.09")
        snippet = _make_snippet("EV-018", page=2)
        term = _make_term("TERM-001", "interest_rate", "8.5%", "0.085", evidence_ids=["EV-018"])
        evidence_registry = {"EV-018": snippet}
        results = _agent().review(
            policy_rules=[rule],
            extracted_terms={"TERM-001": term},
            evidence_registry=evidence_registry,
        )
        for ev_id in results[0].evidence_ids:
            assert ev_id in evidence_registry

    def test_multiple_evidence_ids_all_present(self):
        rule = _make_rule("POLICY-001", "interest_rate", "<=", "0.09")
        term = _make_term("TERM-001", "interest_rate", "8.5%", "0.085",
                          evidence_ids=["EV-001", "EV-018"])
        evidence_registry = {
            "EV-001": _make_snippet("EV-001"),
            "EV-018": _make_snippet("EV-018", page=2),
        }
        results = _agent().review(
            policy_rules=[rule],
            extracted_terms={"TERM-001": term},
            evidence_registry=evidence_registry,
        )
        assert "EV-001" in results[0].evidence_ids
        assert "EV-018" in results[0].evidence_ids


# ── Multiple policy rules ─────────────────────────────────────────────────────

class TestMultiplePolicyRules:
    def test_each_rule_produces_one_result(self):
        rules = [
            _make_rule("POLICY-001", "interest_rate", "<=", "0.09"),
            _make_rule("POLICY-002", "financial_covenant_dscr", ">=", "1.25"),
        ]
        terms = {
            "TERM-001": _make_term("TERM-001", "interest_rate", "8.5%", "0.085"),
            "TERM-002": _make_term("TERM-002", "financial_covenant_dscr", "1.30x", "1.30"),
        }
        results = _agent().review(
            policy_rules=rules,
            extracted_terms=terms,
            evidence_registry={"EV-001": _make_snippet()},
        )
        assert len(results) == 2

    def test_rule_ids_match_input_order(self):
        rules = [
            _make_rule("POLICY-001", "interest_rate", "<=", "0.09"),
            _make_rule("POLICY-002", "financial_covenant_dscr", ">=", "1.25"),
        ]
        terms = {
            "TERM-001": _make_term("TERM-001", "interest_rate", "8.5%", "0.085"),
            "TERM-002": _make_term("TERM-002", "financial_covenant_dscr", "1.30x", "1.30"),
        }
        results = _agent().review(
            policy_rules=rules,
            extracted_terms=terms,
            evidence_registry={"EV-001": _make_snippet()},
        )
        assert results[0].rule_id == "POLICY-001"
        assert results[1].rule_id == "POLICY-002"

    def test_comp_ids_are_sequential(self):
        rules = [
            _make_rule("POLICY-001", "interest_rate", "<=", "0.09"),
            _make_rule("POLICY-002", "financial_covenant_dscr", ">=", "1.25"),
            _make_rule("POLICY-003", "governing_law", "in", "India"),
        ]
        results = _agent().review(
            policy_rules=rules, extracted_terms={}, evidence_registry={}
        )
        assert results[0].compliance_id == "COMP-001"
        assert results[1].compliance_id == "COMP-002"
        assert results[2].compliance_id == "COMP-003"

    def test_independent_results_no_bleed(self):
        """FAIL on one rule must not affect the result of another rule."""
        rules = [
            _make_rule("POLICY-001", "interest_rate", "<=", "0.09"),  # FAIL
            _make_rule("POLICY-002", "financial_covenant_dscr", ">=", "1.25"),  # PASS
        ]
        terms = {
            "TERM-001": _make_term("TERM-001", "interest_rate", "10%", "0.10"),
            "TERM-002": _make_term("TERM-002", "financial_covenant_dscr", "1.30x", "1.30"),
        }
        results = _agent().review(
            policy_rules=rules,
            extracted_terms=terms,
            evidence_registry={"EV-001": _make_snippet()},
        )
        assert results[0].status == ComplianceStatus.FAIL
        assert results[1].status == ComplianceStatus.PASS


# ── Multiple matching terms ───────────────────────────────────────────────────

class TestMultipleMatchingTerms:
    def test_all_matching_terms_evaluated(self):
        """Both terms named 'interest_rate' must be evaluated."""
        rule = _make_rule("POLICY-001", "interest_rate", "<=", "0.09")
        # Two terms with the same name but different term_ids (ambiguous scenario).
        terms = {
            "TERM-001": _make_term("TERM-001", "interest_rate", "8.5%", "0.085",
                                   status=TermStatus.AMBIGUOUS),
            "TERM-002": _make_term("TERM-002", "interest_rate", "10%", "0.10",
                                   status=TermStatus.AMBIGUOUS, evidence_ids=["EV-002"]),
        }
        results = _agent().review(
            policy_rules=[rule],
            extracted_terms=terms,
            evidence_registry={"EV-001": _make_snippet(), "EV-002": _make_snippet("EV-002")},
        )
        # Both are AMBIGUOUS, so result must be NEEDS_HUMAN_REVIEW.
        assert results[0].status == ComplianceStatus.NEEDS_HUMAN_REVIEW

    def test_correct_term_selected_for_rule(self):
        """Rule for 'interest_rate' must not match 'facility_amount' term."""
        rule = _make_rule("POLICY-001", "interest_rate", "<=", "0.09")
        terms = {
            "TERM-001": _make_term("TERM-001", "facility_amount", "INR 50,000,000", "50000000"),
            "TERM-002": _make_term("TERM-002", "interest_rate", "8.5%", "0.085"),
        }
        results = _agent().review(
            policy_rules=[rule],
            extracted_terms=terms,
            evidence_registry={"EV-001": _make_snippet()},
        )
        assert results[0].status == ComplianceStatus.PASS
        assert "TERM-002" in results[0].term_ids
        assert "TERM-001" not in results[0].term_ids


# ── Determinism ───────────────────────────────────────────────────────────────

class TestDeterminism:
    def test_same_input_same_output(self):
        rule = _make_rule("POLICY-001", "interest_rate", "<=", "0.09")
        term = _make_term("TERM-001", "interest_rate", "8.5%", "0.085")
        ev = {"EV-001": _make_snippet()}
        terms = {"TERM-001": term}

        results_1 = _agent().review([rule], terms, ev)
        results_2 = _agent().review([rule], terms, ev)

        assert results_1[0].status == results_2[0].status
        assert results_1[0].rationale == results_2[0].rationale
        assert results_1[0].confidence == results_2[0].confidence


# ── Agent name ────────────────────────────────────────────────────────────────

class TestAgentMetadata:
    def test_agent_name_on_result(self):
        rule = _make_rule("POLICY-001", "interest_rate", "<=", "0.09")
        term = _make_term("TERM-001", "interest_rate", "8.5%", "0.085")
        results = _agent().review(
            policy_rules=[rule],
            extracted_terms={"TERM-001": term},
            evidence_registry={"EV-001": _make_snippet()},
        )
        assert results[0].agent_name == "compliance"

    def test_compliance_id_pattern(self):
        import re
        rule = _make_rule("POLICY-001", "interest_rate", "<=", "0.09")
        results = _agent().review(
            policy_rules=[rule], extracted_terms={}, evidence_registry={}
        )
        pattern = re.compile(r"^COMP-\d{3,}$")
        assert pattern.match(results[0].compliance_id)


# ── Confidence values ─────────────────────────────────────────────────────────

class TestConfidence:
    def test_pass_has_high_confidence(self):
        rule = _make_rule("POLICY-001", "interest_rate", "<=", "0.09")
        term = _make_term("TERM-001", "interest_rate", "8.5%", "0.085")
        results = _agent().review(
            policy_rules=[rule],
            extracted_terms={"TERM-001": term},
            evidence_registry={"EV-001": _make_snippet()},
        )
        assert results[0].confidence == 1.0

    def test_fail_has_high_confidence(self):
        rule = _make_rule("POLICY-001", "interest_rate", "<=", "0.09")
        term = _make_term("TERM-001", "interest_rate", "10%", "0.10")
        results = _agent().review(
            policy_rules=[rule],
            extracted_terms={"TERM-001": term},
            evidence_registry={"EV-001": _make_snippet()},
        )
        assert results[0].confidence == 1.0

    def test_insufficient_evidence_has_zero_confidence(self):
        rule = _make_rule("POLICY-001", "interest_rate", "<=", "0.09")
        results = _agent().review(
            policy_rules=[rule], extracted_terms={}, evidence_registry={}
        )
        assert results[0].confidence == 0.0

    def test_needs_human_review_has_mid_confidence(self):
        rule = _make_rule("POLICY-001", "interest_rate", "<=", "0.09")
        term = _make_term("TERM-001", "interest_rate", "8.5%", status=TermStatus.AMBIGUOUS)
        results = _agent().review(
            policy_rules=[rule],
            extracted_terms={"TERM-001": term},
            evidence_registry={"EV-001": _make_snippet()},
        )
        assert results[0].confidence == 0.5
