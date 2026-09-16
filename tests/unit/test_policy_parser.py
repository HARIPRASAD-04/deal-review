"""Unit tests for app/ingestion/policy_parser.py — Module 8."""

from __future__ import annotations

import pathlib
import tempfile
from unittest.mock import MagicMock

import pymupdf as fitz
import pytest

from app.ingestion.policy_parser import PolicyParseResult, parse_policy_pdf
from app.llm.client import FakeLLMClient
from app.llm.schemas import ExtractedPolicyRuleSchema, PolicyExtractionOutput
from app.models.policy import RuleCategory, RuleSeverity


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_simple_pdf(text: str, path: pathlib.Path) -> None:
    """Write a one-page PDF with the given text using PyMuPDF."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=10)
    doc.save(str(path))
    doc.close()


def _make_policy_llm(rules: list[ExtractedPolicyRuleSchema], ambiguous: list[str] | None = None):
    """FakeLLMClient that returns a PolicyExtractionOutput."""
    output = PolicyExtractionOutput(rules=rules, ambiguous_statements=ambiguous or [])
    return FakeLLMClient(response=output)


def _one_valid_rule_schema(
    rule_id: str | None = "POLICY-001",
    name: str = "interest_rate",
    category: str = "financial",
    operator: str = "<=",
    threshold: str = "0.09",
    severity: str = "high",
) -> ExtractedPolicyRuleSchema:
    return ExtractedPolicyRuleSchema(
        rule_id=rule_id,
        name=name,
        description=f"Test rule: {name}",
        category=category,
        operator=operator,
        threshold=threshold,
        severity=severity,
    )


# ── No LLM client ─────────────────────────────────────────────────────────────

class TestPolicyParserNoLLM:
    def test_returns_failure_when_no_llm(self, tmp_path):
        pdf = tmp_path / "policy.pdf"
        _write_simple_pdf("Rule 1: Interest ≤ 9%", pdf)

        result = parse_policy_pdf(pdf, llm_client=None)

        assert result.success is False
        assert result.error is not None
        assert "LLM" in result.error or "llm" in result.error.lower()
        assert result.rules == []

    def test_returns_failure_not_bundled_rules(self, tmp_path):
        """Ensures no rules are fabricated or substituted when LLM is absent."""
        pdf = tmp_path / "policy.pdf"
        _write_simple_pdf("Policy text here.", pdf)

        result = parse_policy_pdf(pdf, llm_client=None)

        assert result.success is False
        assert len(result.rules) == 0


# ── PDF not found ─────────────────────────────────────────────────────────────

class TestPolicyParserMissingPDF:
    def test_returns_failure_when_pdf_not_found(self, tmp_path):
        llm = _make_policy_llm([])
        result = parse_policy_pdf(tmp_path / "nonexistent.pdf", llm_client=llm)

        assert result.success is False
        assert result.error is not None
        assert result.rules == []

    def test_returns_failure_not_a_pdf(self, tmp_path):
        bad_file = tmp_path / "not_a_pdf.txt"
        bad_file.write_text("not a pdf")
        llm = _make_policy_llm([])
        result = parse_policy_pdf(bad_file, llm_client=llm)
        # Should fail — either PDF load fails or empty text.
        assert result.rules == [] or result.success is False


# ── Valid extraction ──────────────────────────────────────────────────────────

class TestPolicyParserValidExtraction:
    def test_returns_valid_policy_rules(self, tmp_path):
        pdf = tmp_path / "policy.pdf"
        _write_simple_pdf("POLICY-001: Interest rate ≤ 9% per annum.", pdf)

        schema = _one_valid_rule_schema(rule_id="POLICY-001")
        llm = _make_policy_llm([schema])
        result = parse_policy_pdf(pdf, llm_client=llm)

        assert result.success is True
        assert len(result.rules) == 1
        assert result.rules[0].name == "interest_rate"
        assert result.rules[0].operator == "<="
        assert result.rules[0].threshold == "0.09"

    def test_rule_category_mapped_correctly(self, tmp_path):
        pdf = tmp_path / "policy.pdf"
        _write_simple_pdf("Rule: collateral required.", pdf)

        schema = _one_valid_rule_schema(category="collateral", rule_id="POLICY-010", name="collateral_check", operator="exists", threshold="true")
        llm = _make_policy_llm([schema])
        result = parse_policy_pdf(pdf, llm_client=llm)

        assert result.success is True
        assert result.rules[0].category == RuleCategory.COLLATERAL

    def test_rule_severity_mapped_correctly(self, tmp_path):
        pdf = tmp_path / "policy.pdf"
        _write_simple_pdf("Critical rule text.", pdf)

        schema = _one_valid_rule_schema(severity="critical", rule_id="POLICY-011")
        llm = _make_policy_llm([schema])
        result = parse_policy_pdf(pdf, llm_client=llm)

        assert result.success is True
        assert result.rules[0].severity == RuleSeverity.CRITICAL

    def test_multiple_rules_all_returned(self, tmp_path):
        pdf = tmp_path / "policy.pdf"
        _write_simple_pdf("Two rules here.", pdf)

        schemas = [
            _one_valid_rule_schema(rule_id="POLICY-001", name="interest_rate"),
            _one_valid_rule_schema(rule_id="POLICY-002", name="dscr", operator=">=", threshold="1.25"),
        ]
        llm = _make_policy_llm(schemas)
        result = parse_policy_pdf(pdf, llm_client=llm)

        assert result.success is True
        assert len(result.rules) == 2
        names = {r.name for r in result.rules}
        assert "interest_rate" in names
        assert "dscr" in names


# ── Rule ID handling ──────────────────────────────────────────────────────────

class TestPolicyParserRuleIDs:
    def test_explicit_rule_id_preserved(self, tmp_path):
        pdf = tmp_path / "policy.pdf"
        _write_simple_pdf("POLICY-042: some rule.", pdf)

        schema = _one_valid_rule_schema(rule_id="POLICY-042")
        llm = _make_policy_llm([schema])
        result = parse_policy_pdf(pdf, llm_client=llm)

        assert result.success is True
        assert result.rules[0].rule_id == "POLICY-042"

    def test_no_rule_id_gets_deterministic_auto_id(self, tmp_path):
        pdf = tmp_path / "policy.pdf"
        _write_simple_pdf("Rule: interest capped at 9%.", pdf)

        schema = _one_valid_rule_schema(rule_id=None)
        llm = _make_policy_llm([schema])
        result = parse_policy_pdf(pdf, llm_client=llm)

        assert result.success is True
        # Auto-generated ID starts with POLICY-EXT-
        assert result.rules[0].rule_id.startswith("POLICY-")

    def test_two_rules_without_ids_get_sequential_ids(self, tmp_path):
        pdf = tmp_path / "policy.pdf"
        _write_simple_pdf("Two implicit rules.", pdf)

        schemas = [
            _one_valid_rule_schema(rule_id=None, name="rule_a"),
            _one_valid_rule_schema(rule_id=None, name="rule_b", threshold="1.0", operator=">="),
        ]
        llm = _make_policy_llm(schemas)
        result = parse_policy_pdf(pdf, llm_client=llm)

        assert result.success is True
        ids = [r.rule_id for r in result.rules]
        # IDs must be distinct.
        assert len(set(ids)) == 2


# ── Invalid / ambiguous rules ─────────────────────────────────────────────────

class TestPolicyParserInvalidRules:
    def test_bad_category_rule_skipped_not_raised(self, tmp_path):
        pdf = tmp_path / "policy.pdf"
        _write_simple_pdf("Rule with bad category.", pdf)

        bad_schema = ExtractedPolicyRuleSchema(
            rule_id="POLICY-001",
            name="bad_rule",
            description="Bad category rule",
            category="INVALID_CATEGORY_XYZ",
            operator="<=",
            threshold="0.09",
            severity="high",
        )
        llm = _make_policy_llm([bad_schema])
        result = parse_policy_pdf(pdf, llm_client=llm)

        # Bad rule skipped → no valid rules → success=False (not an exception).
        assert result.success is False
        assert len(result.rules) == 0

    def test_bad_category_skipped_good_rule_accepted(self, tmp_path):
        """One bad + one good rule → good rule accepted, bad one skipped."""
        pdf = tmp_path / "policy.pdf"
        _write_simple_pdf("Two rules, one bad.", pdf)

        bad = ExtractedPolicyRuleSchema(
            rule_id="POLICY-001",
            name="bad",
            description="bad",
            category="NONEXISTENT",
            operator="<=",
            threshold="0.5",
            severity="medium",
        )
        good = _one_valid_rule_schema(rule_id="POLICY-002", name="good_rule")
        llm = _make_policy_llm([bad, good])
        result = parse_policy_pdf(pdf, llm_client=llm)

        assert result.success is True
        assert len(result.rules) == 1
        assert result.rules[0].name == "good_rule"

    def test_ambiguous_statements_preserved(self, tmp_path):
        pdf = tmp_path / "policy.pdf"
        _write_simple_pdf("Rule plus ambiguous text.", pdf)

        schema = _one_valid_rule_schema()
        ambiguous = ["Collateral should be adequate — no specific ratio stated."]
        llm = _make_policy_llm([schema], ambiguous=ambiguous)
        result = parse_policy_pdf(pdf, llm_client=llm)

        assert result.success is True
        assert len(result.ambiguous_statements) == 1
        assert "Collateral" in result.ambiguous_statements[0]

    def test_only_ambiguous_statements_returns_failure(self, tmp_path):
        """Only ambiguous statements, no valid rules → failure (no rules to enforce)."""
        pdf = tmp_path / "policy.pdf"
        _write_simple_pdf("Ambiguous policy language.", pdf)

        llm = _make_policy_llm([], ambiguous=["This statement is vague."])
        result = parse_policy_pdf(pdf, llm_client=llm)

        assert result.success is False
        assert len(result.ambiguous_statements) == 1
        assert result.error is not None

    def test_llm_raises_exception_returns_failure(self, tmp_path):
        pdf = tmp_path / "policy.pdf"
        _write_simple_pdf("Some policy text.", pdf)

        llm = FakeLLMClient(raise_error=RuntimeError("LLM timeout"))
        result = parse_policy_pdf(pdf, llm_client=llm)

        assert result.success is False
        assert "LLM" in result.error or "timeout" in result.error.lower() or "failed" in result.error.lower()
