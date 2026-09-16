"""CLI evaluation script — Module 8.

Runs the deterministic evaluation suite against sample cases and reports
per-check pass/fail status and overall scores.

Usage:
    python scripts/run_evaluation.py

This script:
1. Generates runtime policy PDFs for two evaluation cases using PyMuPDF.
2. Executes `evaluate_case()` for Case 1 (Standard Deal) and Case 2 (Strict Policy Breach).
3. Prints a clean formatted summary of evaluation results to the console.
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

import pymupdf as fitz

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

ROOT_DIR = pathlib.Path(__file__).parent.parent
DEAL_001 = ROOT_DIR / "data" / "samples" / "deal_001.pdf"
DEAL_002 = ROOT_DIR / "data" / "samples" / "deal_002.pdf"


def _write_pdf(text: str, path: pathlib.Path) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=10)
    doc.save(str(path))
    doc.close()


def _make_eval_llm_case_1() -> FakeLLMClient:
    rules = [
        ExtractedPolicyRuleSchema(
            rule_id="POLICY-001",
            name="interest_rate",
            description="Interest rate must not exceed 9.0% per annum.",
            category="financial",
            operator="<=",
            threshold="0.09",
            severity="high",
        ),
        ExtractedPolicyRuleSchema(
            rule_id="POLICY-002",
            name="financial_covenant_dscr",
            description="Minimum Debt Service Coverage Ratio of 1.25x.",
            category="financial",
            operator=">=",
            threshold="1.25",
            severity="high",
        ),
    ]
    terms = [
        ExtractedTermSchema(
            name="interest_rate",
            value="10.5% per annum",
            normalized_value="0.105",
            category=TermCategory.RATE,
            confidence=0.99,
            evidence_ids=["EV-001"],
        ),
        ExtractedTermSchema(
            name="financial_covenant_dscr",
            value="minimum DSCR of 1.25x",
            normalized_value="1.25",
            category=TermCategory.COVENANT,
            confidence=0.95,
            evidence_ids=["EV-002"],
        ),
    ]

    fake_policy = PolicyExtractionOutput(rules=rules, ambiguous_statements=[])
    fake_extraction = ExtractionOutput(terms=terms)
    fake_summary = SummaryOutput(
        executive_summary="The deal exceeds the 9% interest rate cap (actual: 10.5%). DSCR covenant meets requirement at 1.25x."
    )

    class _Case1LLM(FakeLLMClient):
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

    return _Case1LLM()


def _make_eval_llm_case_2() -> FakeLLMClient:
    rules = [
        ExtractedPolicyRuleSchema(
            rule_id="POLICY-001",
            name="facility_amount",
            description="Facility amount cannot exceed INR 50,000,000.",
            category="financial",
            operator="<=",
            threshold="50000000",
            severity="critical",
        ),
    ]
    terms = [
        ExtractedTermSchema(
            name="facility_amount",
            value="INR 80,000,000",
            normalized_value="80000000",
            category=TermCategory.FINANCIAL,
            confidence=0.98,
            evidence_ids=["EV-001"],
        ),
    ]

    fake_policy = PolicyExtractionOutput(rules=rules, ambiguous_statements=[])
    fake_extraction = ExtractionOutput(terms=terms)
    fake_summary = SummaryOutput(
        executive_summary="Critical breach: Facility amount INR 80M exceeds policy cap of INR 50M."
    )

    class _Case2LLM(FakeLLMClient):
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

    return _Case2LLM()


def run_evaluation() -> bool:
    print("=" * 80)
    print("  MODULE 8 — DETERMINISTIC EVALUATION RUNNER")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)

        policy_1 = tmp_path / "policy_case1.pdf"
        _write_pdf(
            "POLICY-001: Maximum interest rate 9.0% per annum.\n"
            "POLICY-002: Minimum DSCR 1.25x.",
            policy_1,
        )

        policy_2 = tmp_path / "policy_case2.pdf"
        _write_pdf(
            "POLICY-001: Facility amount limit INR 50,000,000.",
            policy_2,
        )

        case1 = EvaluationCase(
            case_id="EVAL-CASE-001",
            deal_pdf_path=str(DEAL_002 if DEAL_002.exists() else DEAL_001),
            policy_pdf_path=str(policy_1),
            expected_review_status=ReviewStatus.COMPLETED,
            expected_term_names=["interest_rate", "financial_covenant_dscr"],
            expected_compliance={"POLICY-001": "FAIL", "POLICY-002": "PASS"},
            expected_risk_categories=["financial"],
            min_evidence_ids_per_term=1,
            llm_client=_make_eval_llm_case_1(),
        )

        case2 = EvaluationCase(
            case_id="EVAL-CASE-002",
            deal_pdf_path=str(DEAL_002 if DEAL_002.exists() else DEAL_001),
            policy_pdf_path=str(policy_2),
            expected_review_status=ReviewStatus.COMPLETED,
            expected_term_names=["facility_amount"],
            expected_compliance={"POLICY-001": "FAIL"},
            expected_risk_categories=["financial"],
            min_evidence_ids_per_term=1,
            llm_client=_make_eval_llm_case_2(),
        )

        cases = [case1, case2]
        all_passed = True

        for idx, case in enumerate(cases, 1):
            print(f"\n--- Case {idx}: {case.case_id} ---")
            result: EvaluationResult = evaluate_case(case)

            print(f"Status: {'PASS' if result.passed else 'FAIL'}")
            print(f"Score:  {result.score:.2f} ({sum(1 for v in result.check_results.values() if v)}/{len(result.check_results)} checks passed)")

            if result.error:
                print(f"Error:  {result.error}")
                all_passed = False
                continue

            print("\nCheck breakdown:")
            for check_name, check_pass in result.check_results.items():
                icon = "[PASS]" if check_pass else "[FAIL]"
                detail = result.details.get(check_name, {})
                print(f"  {icon} {check_name:<35} {detail}")

            if not result.passed:
                all_passed = False

        print("\n" + "=" * 80)
        if all_passed:
            print("  ALL EVALUATION CASES PASSED PERFECTLY!")
        else:
            print("  SOME EVALUATION CASES FAILED.")
        print("=" * 80)

        return all_passed


if __name__ == "__main__":
    success = run_evaluation()
    sys.exit(0 if success else 1)
