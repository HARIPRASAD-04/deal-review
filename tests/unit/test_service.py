"""Unit tests for app/service.py — DealReviewService (Module 8)."""

from __future__ import annotations

import pathlib
import tempfile

import pymupdf as fitz
import pytest

from app.llm.client import FakeLLMClient
from app.llm.schemas import (
    ExtractionOutput,
    ExtractedTermSchema,
    PolicyExtractionOutput,
    ExtractedPolicyRuleSchema,
    SummaryOutput,
)
from app.models.report import FinalDealReviewReport, ReviewStatus
from app.models.terms import TermCategory
from app.service import DealReviewRequest, DealReviewResponse, DealReviewService

DEAL_PDF = pathlib.Path(__file__).parent.parent.parent / "data" / "samples" / "deal_002.pdf"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_policy_pdf(text: str, path: pathlib.Path) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=10)
    doc.save(str(path))
    doc.close()


def _make_full_llm(
    policy_rules: list[ExtractedPolicyRuleSchema] | None = None,
    term_schemas: list | None = None,
) -> FakeLLMClient:
    """Multi-response FakeLLMClient for service tests."""
    if policy_rules is None:
        policy_rules = [
            ExtractedPolicyRuleSchema(
                rule_id="POLICY-001",
                name="interest_rate",
                description="Interest rate <= 9%",
                category="financial",
                operator="<=",
                threshold="0.09",
                severity="high",
            )
        ]
    if term_schemas is None:
        ev_ids = [f"EV-{i:03d}" for i in range(1, 10)]
        term_schemas = [
            ExtractedTermSchema(
                name="interest_rate",
                value="10.5% per annum",
                normalized_value="0.105",
                category=TermCategory.RATE,
                confidence=0.99,
                evidence_ids=[ev_ids[0]],
            )
        ]

    fake_policy = PolicyExtractionOutput(rules=policy_rules, ambiguous_statements=[])
    fake_extraction = ExtractionOutput(terms=term_schemas)
    fake_summary = SummaryOutput(executive_summary="Test summary.")

    class _ServiceFake(FakeLLMClient):
        def __init__(self) -> None:
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

    return _ServiceFake()


# ── Policy parse failure is a hard stop ──────────────────────────────────────

class TestDealReviewServicePolicyFailure:
    def test_nonexistent_policy_pdf_returns_failure(self, tmp_path):
        service = DealReviewService()
        llm = _make_full_llm()

        response = service.run(DealReviewRequest(
            deal_pdf_path=str(DEAL_PDF),
            policy_pdf_path=str(tmp_path / "nonexistent.pdf"),
            llm_client=llm,
        ))

        assert response.success is False
        assert response.report is None
        assert response.final_state is None
        assert "policy" in (response.error or "").lower() or "ingestion" in (response.error or "").lower()

    def test_no_llm_client_stops_at_policy_stage(self, tmp_path):
        policy_pdf = tmp_path / "policy.pdf"
        _write_policy_pdf("Rule 1: Interest <= 9%.", policy_pdf)

        service = DealReviewService()
        response = service.run(DealReviewRequest(
            deal_pdf_path=str(DEAL_PDF),
            policy_pdf_path=str(policy_pdf),
            llm_client=None,  # No LLM — policy parse will fail
        ))

        assert response.success is False
        assert response.report is None
        assert response.final_state is None

    def test_policy_failure_does_not_use_bundled_rules(self, tmp_path):
        """Verify no bundled/default rules are substituted on policy failure."""
        service = DealReviewService()
        response = service.run(DealReviewRequest(
            deal_pdf_path=str(DEAL_PDF),
            policy_pdf_path=str(tmp_path / "bad.pdf"),
            llm_client=None,
        ))
        # Pipeline must not have run at all.
        assert response.final_state is None
        assert response.report is None


# ── Deal PDF failure ──────────────────────────────────────────────────────────

class TestDealReviewServiceDealFailure:
    def test_nonexistent_deal_pdf_returns_failure_or_failed_status(self, tmp_path):
        policy_pdf = tmp_path / "policy.pdf"
        _write_policy_pdf("POLICY-001: Interest <= 9%.", policy_pdf)
        llm = _make_full_llm()

        service = DealReviewService()
        response = service.run(DealReviewRequest(
            deal_pdf_path=str(tmp_path / "nonexistent_deal.pdf"),
            policy_pdf_path=str(policy_pdf),
            llm_client=llm,
        ))

        # Either the service returns success=False or the report shows FAILED status.
        if response.success and response.report:
            assert response.report.review_status == ReviewStatus.FAILED
        else:
            assert response.success is False


# ── Successful run ────────────────────────────────────────────────────────────

class TestDealReviewServiceSuccess:
    def test_full_run_returns_report(self, tmp_path):
        policy_pdf = tmp_path / "policy.pdf"
        _write_policy_pdf("POLICY-001: Interest rate must not exceed 9% per annum.", policy_pdf)
        llm = _make_full_llm()

        service = DealReviewService()
        response = service.run(DealReviewRequest(
            deal_pdf_path=str(DEAL_PDF),
            policy_pdf_path=str(policy_pdf),
            document_id="TEST-DEAL",
            llm_client=llm,
        ))

        assert response.success is True
        assert response.report is not None
        assert isinstance(response.report, FinalDealReviewReport)

    def test_report_has_required_fields(self, tmp_path):
        policy_pdf = tmp_path / "policy.pdf"
        _write_policy_pdf("POLICY-001: Interest <= 9%.", policy_pdf)
        llm = _make_full_llm()

        service = DealReviewService()
        response = service.run(DealReviewRequest(
            deal_pdf_path=str(DEAL_PDF),
            policy_pdf_path=str(policy_pdf),
            document_id="TEST-DEAL",
            llm_client=llm,
        ))

        assert response.success is True
        report = response.report
        assert report.run_id
        assert report.document_id == "TEST-DEAL"
        assert report.review_status in list(ReviewStatus)
        assert report.audit_event_count > 0

    def test_policy_warnings_surfaced_in_response(self, tmp_path):
        policy_pdf = tmp_path / "policy.pdf"
        _write_policy_pdf("Policy text with ambiguous clause.", policy_pdf)

        # LLM returns one valid rule + one ambiguous statement.
        good_rule = ExtractedPolicyRuleSchema(
            rule_id="POLICY-001",
            name="interest_rate",
            description="Interest <= 9%",
            category="financial",
            operator="<=",
            threshold="0.09",
            severity="high",
        )
        fake_policy = PolicyExtractionOutput(
            rules=[good_rule],
            ambiguous_statements=["Collateral coverage should be adequate — not quantified."],
        )
        fake_extraction = ExtractionOutput(terms=[])
        fake_summary = SummaryOutput(executive_summary="Summary.")

        class _WarnFake(FakeLLMClient):
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

        service = DealReviewService()
        response = service.run(DealReviewRequest(
            deal_pdf_path=str(DEAL_PDF),
            policy_pdf_path=str(policy_pdf),
            llm_client=_WarnFake(),
        ))

        assert response.success is True
        assert len(response.policy_parse_warnings) >= 1
        assert "Collateral" in response.policy_parse_warnings[0]

    def test_report_json_round_trips(self, tmp_path):
        policy_pdf = tmp_path / "policy.pdf"
        _write_policy_pdf("POLICY-001: Rate <= 9%.", policy_pdf)
        llm = _make_full_llm()

        service = DealReviewService()
        response = service.run(DealReviewRequest(
            deal_pdf_path=str(DEAL_PDF),
            policy_pdf_path=str(policy_pdf),
            llm_client=llm,
        ))

        assert response.success is True
        json_str = response.report.model_dump_json()
        reconstructed = FinalDealReviewReport.model_validate_json(json_str)
        assert reconstructed.report_id == response.report.report_id
