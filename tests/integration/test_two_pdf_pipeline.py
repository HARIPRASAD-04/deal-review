"""Integration tests: two-PDF pipeline → FinalDealReviewReport (Module 8).

These tests exercise the full DealReviewService boundary:
    deal PDF + policy PDF
    → PolicyRule parsing (FakeLLMClient)
    → 4-agent pipeline (LangGraph)
    → FinalDealReviewReport

No mocks of internal components — this is a black-box integration test.
Policy PDFs are generated at runtime using PyMuPDF; none are committed.
"""

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


# ── Shared helpers ────────────────────────────────────────────────────────────

def _write_policy_pdf(text: str, path: pathlib.Path) -> None:
    """Write a minimal one-page policy PDF using PyMuPDF."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=10, color=(0, 0, 0))
    doc.save(str(path))
    doc.close()


def _make_service_llm(
    policy_rules: list[ExtractedPolicyRuleSchema],
    term_schemas: list[ExtractedTermSchema],
    summary_text: str = "Integration test executive summary.",
    ambiguous: list[str] | None = None,
) -> FakeLLMClient:
    """Build a multi-response FakeLLMClient for integration tests."""
    fake_policy = PolicyExtractionOutput(rules=policy_rules, ambiguous_statements=ambiguous or [])
    fake_extraction = ExtractionOutput(terms=term_schemas)
    fake_summary = SummaryOutput(executive_summary=summary_text)

    class _IntFake(FakeLLMClient):
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

    return _IntFake()


# ── Test 1: Full pipeline — deal + policy PDFs ────────────────────────────────

class TestFullTwoPdfPipeline:
    """Full end-to-end integration: two PDFs → FinalDealReviewReport."""

    def test_full_pipeline_produces_report(self, tmp_path):
        """The primary Module 8 integration test."""
        policy_pdf = tmp_path / "policy.pdf"
        _write_policy_pdf(
            "POLICY-001: The interest rate must not exceed 9% per annum.\n"
            "POLICY-002: Minimum DSCR of 1.25x required.",
            policy_pdf,
        )

        policy_rules = [
            ExtractedPolicyRuleSchema(
                rule_id="POLICY-001",
                name="interest_rate",
                description="Interest rate <= 9% per annum.",
                category="financial",
                operator="<=",
                threshold="0.09",
                severity="high",
            ),
            ExtractedPolicyRuleSchema(
                rule_id="POLICY-002",
                name="financial_covenant_dscr",
                description="Minimum DSCR 1.25x.",
                category="financial",
                operator=">=",
                threshold="1.25",
                severity="high",
            ),
        ]
        term_schemas = [
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

        llm = _make_service_llm(policy_rules, term_schemas)
        service = DealReviewService()
        response = service.run(DealReviewRequest(
            deal_pdf_path=str(DEAL_PDF),
            policy_pdf_path=str(policy_pdf),
            document_id="INTTEST-001",
            llm_client=llm,
        ))

        # Core assertion: a report was produced.
        assert response.success is True, f"Service failed: {response.error}"
        assert response.report is not None

        report: FinalDealReviewReport = response.report
        assert isinstance(report, FinalDealReviewReport)

    def test_report_has_valid_review_status(self, tmp_path):
        policy_pdf = tmp_path / "policy.pdf"
        _write_policy_pdf("POLICY-001: Interest <= 9%.", policy_pdf)
        policy_rules = [
            ExtractedPolicyRuleSchema(
                rule_id="POLICY-001",
                name="interest_rate",
                description="Rate cap.",
                category="financial",
                operator="<=",
                threshold="0.09",
                severity="high",
            )
        ]
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
        llm = _make_service_llm(policy_rules, term_schemas)
        service = DealReviewService()
        response = service.run(DealReviewRequest(
            deal_pdf_path=str(DEAL_PDF),
            policy_pdf_path=str(policy_pdf),
            document_id="INTTEST-STATUS",
            llm_client=llm,
        ))

        assert response.success is True
        assert response.report.review_status in list(ReviewStatus)

    def test_report_compliance_summary_reflects_rules(self, tmp_path):
        policy_pdf = tmp_path / "policy.pdf"
        _write_policy_pdf("Two rules.", policy_pdf)
        policy_rules = [
            ExtractedPolicyRuleSchema(
                rule_id="POLICY-001",
                name="interest_rate",
                description="Rate cap.",
                category="financial",
                operator="<=",
                threshold="0.09",
                severity="high",
            ),
            ExtractedPolicyRuleSchema(
                rule_id="POLICY-002",
                name="dscr",
                description="DSCR minimum.",
                category="financial",
                operator=">=",
                threshold="1.25",
                severity="high",
            ),
        ]
        term_schemas = [
            ExtractedTermSchema(
                name="interest_rate",
                value="10.5%",
                normalized_value="0.105",
                category=TermCategory.RATE,
                confidence=0.99,
                evidence_ids=["EV-001"],
            )
        ]
        llm = _make_service_llm(policy_rules, term_schemas)
        service = DealReviewService()
        response = service.run(DealReviewRequest(
            deal_pdf_path=str(DEAL_PDF),
            policy_pdf_path=str(policy_pdf),
            document_id="INTTEST-COMPLIANCE",
            llm_client=llm,
        ))

        assert response.success is True
        cs = response.report.compliance_summary
        assert cs.total_rules > 0

    def test_report_has_run_id_and_document_id(self, tmp_path):
        policy_pdf = tmp_path / "policy.pdf"
        _write_policy_pdf("POLICY-001: Rate <= 9%.", policy_pdf)
        policy_rules = [
            ExtractedPolicyRuleSchema(
                rule_id="POLICY-001", name="interest_rate",
                description="Rate cap.", category="financial",
                operator="<=", threshold="0.09", severity="high",
            )
        ]
        term_schemas = [
            ExtractedTermSchema(
                name="interest_rate", value="10%", normalized_value="0.10",
                category=TermCategory.RATE, confidence=0.9,
                evidence_ids=["EV-001"],
            )
        ]
        llm = _make_service_llm(policy_rules, term_schemas)
        service = DealReviewService()
        response = service.run(DealReviewRequest(
            deal_pdf_path=str(DEAL_PDF),
            policy_pdf_path=str(policy_pdf),
            document_id="MY-DOCUMENT-ID",
            llm_client=llm,
        ))

        assert response.success is True
        assert response.report.run_id
        assert response.report.document_id == "MY-DOCUMENT-ID"

    def test_report_metadata_has_evidence_count(self, tmp_path):
        policy_pdf = tmp_path / "policy.pdf"
        _write_policy_pdf("POLICY-001: Rate <= 9%.", policy_pdf)
        policy_rules = [
            ExtractedPolicyRuleSchema(
                rule_id="POLICY-001", name="interest_rate",
                description="Rate cap.", category="financial",
                operator="<=", threshold="0.09", severity="high",
            )
        ]
        term_schemas = [
            ExtractedTermSchema(
                name="interest_rate", value="10%", normalized_value="0.10",
                category=TermCategory.RATE, confidence=0.9,
                evidence_ids=["EV-001"],
            )
        ]
        llm = _make_service_llm(policy_rules, term_schemas)
        service = DealReviewService()
        response = service.run(DealReviewRequest(
            deal_pdf_path=str(DEAL_PDF),
            policy_pdf_path=str(policy_pdf),
            document_id="INTTEST-META",
            llm_client=llm,
        ))

        assert response.success is True
        assert "evidence_count" in response.report.metadata
        assert response.report.metadata["evidence_count"] > 0

    def test_report_audit_event_count_positive(self, tmp_path):
        policy_pdf = tmp_path / "policy.pdf"
        _write_policy_pdf("POLICY-001: Rate <= 9%.", policy_pdf)
        policy_rules = [
            ExtractedPolicyRuleSchema(
                rule_id="POLICY-001", name="interest_rate",
                description="Rate cap.", category="financial",
                operator="<=", threshold="0.09", severity="high",
            )
        ]
        term_schemas = [
            ExtractedTermSchema(
                name="interest_rate", value="10%", normalized_value="0.10",
                category=TermCategory.RATE, confidence=0.9,
                evidence_ids=["EV-001"],
            )
        ]
        llm = _make_service_llm(policy_rules, term_schemas)
        service = DealReviewService()
        response = service.run(DealReviewRequest(
            deal_pdf_path=str(DEAL_PDF),
            policy_pdf_path=str(policy_pdf),
            document_id="INTTEST-AUDIT",
            llm_client=llm,
        ))

        assert response.success is True
        assert response.report.audit_event_count > 0

    def test_report_round_trips_json(self, tmp_path):
        policy_pdf = tmp_path / "policy.pdf"
        _write_policy_pdf("POLICY-001: Rate <= 9%.", policy_pdf)
        policy_rules = [
            ExtractedPolicyRuleSchema(
                rule_id="POLICY-001", name="interest_rate",
                description="Rate cap.", category="financial",
                operator="<=", threshold="0.09", severity="high",
            )
        ]
        term_schemas = [
            ExtractedTermSchema(
                name="interest_rate", value="10%", normalized_value="0.10",
                category=TermCategory.RATE, confidence=0.9,
                evidence_ids=["EV-001"],
            )
        ]
        llm = _make_service_llm(policy_rules, term_schemas)
        service = DealReviewService()
        response = service.run(DealReviewRequest(
            deal_pdf_path=str(DEAL_PDF),
            policy_pdf_path=str(policy_pdf),
            document_id="INTTEST-ROUNDTRIP",
            llm_client=llm,
        ))

        assert response.success is True
        json_str = response.report.model_dump_json()
        reconstructed = FinalDealReviewReport.model_validate_json(json_str)
        assert reconstructed.report_id == response.report.report_id
        assert reconstructed.review_status == response.report.review_status


# ── Test 2: Policy parse failure stops the pipeline ──────────────────────────

class TestPolicyParseFailureStopsPipeline:
    def test_nonexistent_policy_pdf_stops_pipeline(self, tmp_path):
        policy_rules = [
            ExtractedPolicyRuleSchema(
                rule_id="POLICY-001", name="rate",
                description="Rate.", category="financial",
                operator="<=", threshold="0.09", severity="high",
            )
        ]
        llm = _make_service_llm(policy_rules, [])

        service = DealReviewService()
        response = service.run(DealReviewRequest(
            deal_pdf_path=str(DEAL_PDF),
            policy_pdf_path=str(tmp_path / "nonexistent_policy.pdf"),
            llm_client=llm,
        ))

        assert response.success is False
        assert "policy" in (response.error or "").lower() or "ingestion" in (response.error or "").lower()
        assert response.report is None
        # Pipeline was never invoked.
        assert response.final_state is None

    def test_no_llm_stops_at_policy_stage(self, tmp_path):
        policy_pdf = tmp_path / "policy.pdf"
        _write_policy_pdf("POLICY-001: Rate <= 9%.", policy_pdf)

        service = DealReviewService()
        response = service.run(DealReviewRequest(
            deal_pdf_path=str(DEAL_PDF),
            policy_pdf_path=str(policy_pdf),
            llm_client=None,
        ))

        assert response.success is False
        assert response.report is None
        assert response.final_state is None


# ── Test 3: Deal PDF not found ────────────────────────────────────────────────

class TestDealPdfNotFound:
    def test_missing_deal_pdf_produces_failure_or_failed_status(self, tmp_path):
        policy_pdf = tmp_path / "policy.pdf"
        _write_policy_pdf("POLICY-001: Rate <= 9%.", policy_pdf)
        policy_rules = [
            ExtractedPolicyRuleSchema(
                rule_id="POLICY-001", name="interest_rate",
                description="Rate cap.", category="financial",
                operator="<=", threshold="0.09", severity="high",
            )
        ]
        llm = _make_service_llm(policy_rules, [])

        service = DealReviewService()
        response = service.run(DealReviewRequest(
            deal_pdf_path=str(tmp_path / "nonexistent_deal.pdf"),
            policy_pdf_path=str(policy_pdf),
            llm_client=llm,
        ))

        if response.success and response.report:
            assert response.report.review_status == ReviewStatus.FAILED
        else:
            assert response.success is False


# ── Test 4: Policy warnings surfaced ─────────────────────────────────────────

class TestPolicyWarningsSurfaced:
    def test_ambiguous_statements_appear_in_response_warnings(self, tmp_path):
        policy_pdf = tmp_path / "policy.pdf"
        _write_policy_pdf(
            "POLICY-001: Rate <= 9%.\n"
            "Note: Collateral should be adequate — no ratio specified.",
            policy_pdf,
        )
        policy_rules = [
            ExtractedPolicyRuleSchema(
                rule_id="POLICY-001", name="interest_rate",
                description="Rate cap.", category="financial",
                operator="<=", threshold="0.09", severity="high",
            )
        ]
        ambiguous = ["Collateral should be adequate — no ratio specified."]
        term_schemas = [
            ExtractedTermSchema(
                name="interest_rate", value="10%", normalized_value="0.10",
                category=TermCategory.RATE, confidence=0.9,
                evidence_ids=["EV-001"],
            )
        ]
        llm = _make_service_llm(policy_rules, term_schemas, ambiguous=ambiguous)

        service = DealReviewService()
        response = service.run(DealReviewRequest(
            deal_pdf_path=str(DEAL_PDF),
            policy_pdf_path=str(policy_pdf),
            document_id="INTTEST-WARNINGS",
            llm_client=llm,
        ))

        assert response.success is True
        assert len(response.policy_parse_warnings) >= 1
        assert any("Collateral" in w for w in response.policy_parse_warnings)

    def test_warnings_do_not_block_successful_run(self, tmp_path):
        """Ambiguous statements in the policy PDF must not prevent a successful run."""
        policy_pdf = tmp_path / "policy.pdf"
        _write_policy_pdf("POLICY-001: Rate <= 9%.\nAmbiguous clause here.", policy_pdf)
        policy_rules = [
            ExtractedPolicyRuleSchema(
                rule_id="POLICY-001", name="interest_rate",
                description="Rate cap.", category="financial",
                operator="<=", threshold="0.09", severity="high",
            )
        ]
        term_schemas = [
            ExtractedTermSchema(
                name="interest_rate", value="10%", normalized_value="0.10",
                category=TermCategory.RATE, confidence=0.9,
                evidence_ids=["EV-001"],
            )
        ]
        llm = _make_service_llm(
            policy_rules, term_schemas,
            ambiguous=["Ambiguous clause — not codified."],
        )

        service = DealReviewService()
        response = service.run(DealReviewRequest(
            deal_pdf_path=str(DEAL_PDF),
            policy_pdf_path=str(policy_pdf),
            llm_client=llm,
        ))

        assert response.success is True
        assert response.report is not None
