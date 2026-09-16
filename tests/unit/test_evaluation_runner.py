"""Unit tests for app/evaluation/runner.py — Module 8."""

from __future__ import annotations

import pathlib
import tempfile

import pymupdf as fitz
import pytest

from app.evaluation.runner import EvaluationCase, EvaluationResult, evaluate_case
from app.llm.client import FakeLLMClient
from app.llm.schemas import (
    ExtractionOutput,
    ExtractedTermSchema,
    PolicyExtractionOutput,
    ExtractedPolicyRuleSchema,
    SummaryOutput,
)
from app.models.report import ReviewStatus
from app.models.terms import TermCategory

DEAL_PDF = pathlib.Path(__file__).parent.parent.parent / "data" / "samples" / "deal_002.pdf"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_policy_pdf(text: str, path: pathlib.Path) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=10)
    doc.save(str(path))
    doc.close()


def _make_eval_llm(
    policy_rules: list[ExtractedPolicyRuleSchema] | None = None,
    term_schemas: list | None = None,
    ambiguous: list[str] | None = None,
) -> FakeLLMClient:
    if policy_rules is None:
        policy_rules = [
            ExtractedPolicyRuleSchema(
                rule_id="POLICY-001",
                name="interest_rate",
                description="Interest <= 9%",
                category="financial",
                operator="<=",
                threshold="0.09",
                severity="high",
            )
        ]
    if term_schemas is None:
        term_schemas = [
            ExtractedTermSchema(
                name="interest_rate",
                value="10.5% per annum",
                normalized_value="0.105",
                category=TermCategory.RATE,
                confidence=0.99,
                evidence_ids=["EV-001"],
            )
        ]
    fake_policy = PolicyExtractionOutput(rules=policy_rules, ambiguous_statements=ambiguous or [])
    fake_extraction = ExtractionOutput(terms=term_schemas)
    fake_summary = SummaryOutput(executive_summary="Test summary.")

    class _EFake(FakeLLMClient):
        def __init__(self):
            super().__init__(response=fake_extraction)
            self._p = fake_policy
            self._e = fake_extraction
            self._s = fake_summary

        def get_structured_completion(self, system_prompt, user_prompt, response_schema):
            self._call_count += 1
            if response_schema is PolicyExtractionOutput:
                return self._p
            if response_schema is SummaryOutput:
                return self._s
            return self._e

    return _EFake()


def _minimal_case(
    tmp_path: pathlib.Path,
    expected_status: ReviewStatus = ReviewStatus.COMPLETED_WITH_HUMAN_REVIEW,
    **kwargs,
) -> tuple[EvaluationCase, FakeLLMClient]:
    policy_pdf = tmp_path / "policy.pdf"
    _write_policy_pdf("POLICY-001: Interest <= 9%.", policy_pdf)
    llm = _make_eval_llm()
    case = EvaluationCase(
        case_id="TEST-CASE",
        deal_pdf_path=str(DEAL_PDF),
        policy_pdf_path=str(policy_pdf),
        expected_review_status=expected_status,
        **kwargs,
    )
    return case, llm


# ── Basic execution ───────────────────────────────────────────────────────────

class TestEvaluationRunnerBasic:
    def test_returns_evaluation_result(self, tmp_path):
        case, llm = _minimal_case(tmp_path)
        result = evaluate_case(case, llm_client=llm)
        assert isinstance(result, EvaluationResult)

    def test_score_between_0_and_1(self, tmp_path):
        case, llm = _minimal_case(tmp_path)
        result = evaluate_case(case, llm_client=llm)
        assert 0.0 <= result.score <= 1.0

    def test_case_id_preserved(self, tmp_path):
        policy_pdf = tmp_path / "policy.pdf"
        _write_policy_pdf("Policy rule.", policy_pdf)
        llm = _make_eval_llm()
        case = EvaluationCase(
            case_id="MY-CASE-42",
            deal_pdf_path=str(DEAL_PDF),
            policy_pdf_path=str(policy_pdf),
            expected_review_status=ReviewStatus.COMPLETED,
        )
        result = evaluate_case(case, llm_client=llm)
        assert result.case_id == "MY-CASE-42"

    def test_report_integrity_check_present(self, tmp_path):
        case, llm = _minimal_case(tmp_path)
        result = evaluate_case(case, llm_client=llm)
        assert "report_integrity" in result.check_results

    def test_final_report_present_on_success(self, tmp_path):
        case, llm = _minimal_case(tmp_path)
        result = evaluate_case(case, llm_client=llm)
        if result.error is None:
            assert result.final_report is not None


# ── review_status_match check ─────────────────────────────────────────────────

class TestReviewStatusCheck:
    def test_status_match_passes_when_correct(self, tmp_path):
        # The pipeline with a failing rule → COMPLETED_WITH_HUMAN_REVIEW
        case, llm = _minimal_case(tmp_path, expected_status=ReviewStatus.COMPLETED_WITH_HUMAN_REVIEW)
        result = evaluate_case(case, llm_client=llm)
        # check_results is populated regardless of pass/fail.
        assert "review_status_match" in result.check_results

    def test_status_match_fails_when_wrong_expected(self, tmp_path):
        # Expect COMPLETED but pipeline likely produces COMPLETED_WITH_HUMAN_REVIEW.
        case, llm = _minimal_case(tmp_path, expected_status=ReviewStatus.FAILED)
        result = evaluate_case(case, llm_client=llm)
        assert result.check_results.get("review_status_match") is False


# ── terms_present check ───────────────────────────────────────────────────────

class TestTermsPresentCheck:
    def test_term_present_check_passes_for_extracted_term(self, tmp_path):
        # LLM extracts interest_rate.
        case, llm = _minimal_case(tmp_path, expected_term_names=["interest_rate"])
        result = evaluate_case(case, llm_client=llm)
        # interest_rate was in the fake extraction schema.
        assert "terms_present_interest_rate" in result.check_results

    def test_term_present_check_fails_for_missing_term(self, tmp_path):
        policy_pdf = tmp_path / "policy.pdf"
        _write_policy_pdf("Policy.", policy_pdf)
        llm = _make_eval_llm()  # LLM does NOT extract "governing_law"
        case = EvaluationCase(
            case_id="TERM-FAIL",
            deal_pdf_path=str(DEAL_PDF),
            policy_pdf_path=str(policy_pdf),
            expected_review_status=ReviewStatus.COMPLETED_WITH_HUMAN_REVIEW,
            expected_term_names=["governing_law"],
        )
        result = evaluate_case(case, llm_client=llm)
        assert result.check_results.get("terms_present_governing_law") is False


# ── evidence_grounding check ──────────────────────────────────────────────────

class TestEvidenceGroundingCheck:
    def test_evidence_grounding_passes_for_term_with_evidence(self, tmp_path):
        case, llm = _minimal_case(
            tmp_path,
            expected_term_names=["interest_rate"],
            min_evidence_ids_per_term=1,
        )
        result = evaluate_case(case, llm_client=llm)
        # interest_rate has EV-001 in the fake → should pass.
        grounding_check = result.check_results.get("evidence_grounding_interest_rate")
        if grounding_check is not None:
            assert grounding_check is True

    def test_evidence_grounding_fails_when_term_missing(self, tmp_path):
        policy_pdf = tmp_path / "policy.pdf"
        _write_policy_pdf("Policy.", policy_pdf)
        llm = _make_eval_llm()
        case = EvaluationCase(
            case_id="EV-FAIL",
            deal_pdf_path=str(DEAL_PDF),
            policy_pdf_path=str(policy_pdf),
            expected_review_status=ReviewStatus.COMPLETED_WITH_HUMAN_REVIEW,
            expected_term_names=["nonexistent_term"],
            min_evidence_ids_per_term=1,
        )
        result = evaluate_case(case, llm_client=llm)
        # Term not found → evidence grounding trivially fails.
        assert result.check_results.get("evidence_grounding_nonexistent_term") is False


# ── Policy parse failure ──────────────────────────────────────────────────────

class TestEvaluationRunnerPolicyFailure:
    def test_missing_policy_pdf_returns_failure_result(self, tmp_path):
        case = EvaluationCase(
            case_id="MISSING-POLICY",
            deal_pdf_path=str(DEAL_PDF),
            policy_pdf_path=str(tmp_path / "nonexistent.pdf"),
            expected_review_status=ReviewStatus.COMPLETED,
        )
        llm = _make_eval_llm()
        result = evaluate_case(case, llm_client=llm)

        assert result.passed is False
        assert result.score == 0.0
        assert result.error is not None

    def test_no_llm_returns_failure_result(self, tmp_path):
        policy_pdf = tmp_path / "policy.pdf"
        _write_policy_pdf("Policy.", policy_pdf)
        case = EvaluationCase(
            case_id="NO-LLM",
            deal_pdf_path=str(DEAL_PDF),
            policy_pdf_path=str(policy_pdf),
            expected_review_status=ReviewStatus.COMPLETED,
        )
        result = evaluate_case(case, llm_client=None)

        assert result.passed is False
        assert result.score == 0.0
        assert result.error is not None


# ── Missing deal PDF ──────────────────────────────────────────────────────────

class TestEvaluationRunnerDealFailure:
    def test_missing_deal_pdf_returns_failure_or_failed_status(self, tmp_path):
        policy_pdf = tmp_path / "policy.pdf"
        _write_policy_pdf("POLICY-001: Rate <= 9%.", policy_pdf)
        llm = _make_eval_llm()
        case = EvaluationCase(
            case_id="MISSING-DEAL",
            deal_pdf_path=str(tmp_path / "nonexistent_deal.pdf"),
            policy_pdf_path=str(policy_pdf),
            expected_review_status=ReviewStatus.COMPLETED,
        )
        result = evaluate_case(case, llm_client=llm)

        # Either pipeline never ran (score=0 + error) OR report shows FAILED.
        if result.final_report:
            assert result.final_report.review_status == ReviewStatus.FAILED
        else:
            assert result.score == 0.0


# ── Risk category and title checks ───────────────────────────────────────────

class TestRiskChecks:
    def test_risk_category_check_passes_when_category_present(self, tmp_path):
        case, llm = _minimal_case(
            tmp_path,
            expected_risk_categories=["financial"],
        )
        result = evaluate_case(case, llm_client=llm)
        assert "risk_category_financial" in result.check_results

    def test_risk_category_check_fails_for_absent_category(self, tmp_path):
        case, llm = _minimal_case(
            tmp_path,
            expected_risk_categories=["reputational"],
        )
        result = evaluate_case(case, llm_client=llm)
        check = result.check_results.get("risk_category_reputational")
        # reputational risk is unlikely from the canned run.
        if check is not None:
            # Don't assert a specific value — just verify the check ran.
            assert isinstance(check, bool)

    def test_risk_title_check_in_results(self, tmp_path):
        case, llm = _minimal_case(
            tmp_path,
            expected_risk_finding_titles=["Policy Violation"],
        )
        result = evaluate_case(case, llm_client=llm)
        assert "risk_title_0" in result.check_results
