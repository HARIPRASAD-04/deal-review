"""Unit tests for the evidence segmenter (segmenter.py)."""

from __future__ import annotations

import pytest

from app.ingestion.segmenter import (
    MAX_BLOCK_CHARS,
    MIN_BLOCK_CHARS,
    _detect_section,
    _normalize_text,
    _split_large_block,
    segment_page,
)
from app.models.document import DocumentPage
from app.models.evidence import SourceType


# ── _detect_section ────────────────────────────────────────────────────────────

class TestDetectSection:
    def test_section_keyword(self):
        assert _detect_section("Section 4.2 — Interest Rate\nsome text") is not None

    def test_clause_keyword(self):
        assert _detect_section("Clause 3 — Covenants\ntext") is not None

    def test_schedule(self):
        assert _detect_section("Schedule A — Security Documents\ntext") is not None

    def test_numeric_prefix_with_separator(self):
        assert _detect_section("4.2 — Interest Rate\ntext") is not None

    def test_standalone_numeric(self):
        result = _detect_section("4.2\ntext")
        assert result is not None

    def test_plain_paragraph_returns_none(self):
        assert _detect_section("The interest rate shall be 8.5% per annum.") is None

    def test_body_text_returns_none(self):
        assert _detect_section("Borrower shall maintain a minimum DSCR of 1.25x.") is None

    def test_empty_string_returns_none(self):
        assert _detect_section("") is None

    def test_section_value_is_trimmed(self):
        result = _detect_section("Section 5.2 — Financial Covenants (Ongoing)\nmore text")
        assert result is not None
        assert result == result.strip()


# ── _normalize_text ────────────────────────────────────────────────────────────

class TestNormalizeText:
    def test_strips_surrounding_whitespace(self):
        assert _normalize_text("  hello  ") == "hello"

    def test_preserves_percentage(self):
        assert "8.5%" in _normalize_text("Interest rate: 8.5% per annum.")

    def test_preserves_currency(self):
        assert "INR 50,000,000" in _normalize_text("  INR 50,000,000  ")

    def test_preserves_qualifier_subject_to(self):
        text = "Disbursement is subject to completion of documentation."
        assert "subject to" in _normalize_text(text)

    def test_preserves_qualifier_not_less_than(self):
        text = "Amount shall be not less than INR 1,000,000."
        assert "not less than" in _normalize_text(text)

    def test_collapses_excessive_blank_lines(self):
        text = "line one\n\n\n\nline two"
        result = _normalize_text(text)
        assert "\n\n\n" not in result

    def test_does_not_remove_legal_qualifiers(self):
        qualifiers = ["except", "unless", "provided that", "not exceeding", "at least"]
        for q in qualifiers:
            assert q in _normalize_text(f"The loan is valid {q} further notice.")


# ── _split_large_block ─────────────────────────────────────────────────────────

class TestSplitLargeBlock:
    def test_small_block_not_split(self):
        text = "Short paragraph."
        chunks = _split_large_block(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_large_block_split_into_multiple(self):
        # Create text longer than MAX_BLOCK_CHARS
        para = "This is a sentence about interest rates and covenants. " * 10
        text = "\n\n".join([para] * 5)  # ~2750+ chars total
        chunks = _split_large_block(text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= MAX_BLOCK_CHARS

    def test_all_content_preserved_after_split(self):
        unique_marker = "UNIQUE-TERM-XYZ-999"
        text = ("Filler sentence for padding. " * 60) + unique_marker
        chunks = _split_large_block(text)
        combined = " ".join(chunks)
        assert unique_marker in combined


# ── segment_page ───────────────────────────────────────────────────────────────

class TestSegmentPage:
    def _make_page(self, text: str, page_number: int = 1, has_text: bool = True) -> DocumentPage:
        return DocumentPage(
            document_id="DEAL-002",
            page_number=page_number,
            raw_text=text,
            block_count=text.count("\n\n") + 1 if text.strip() else 0,
            has_extractable_text=has_text,
        )

    def test_single_block_produces_one_snippet(self):
        page = self._make_page("Interest rate: 8.5% per annum, subject to revision.")
        snippets = segment_page(page, evidence_id_start=1)
        assert len(snippets) == 1

    def test_two_blocks_produce_two_snippets(self):
        page = self._make_page(
            "Principal Amount: INR 50,000,000.\n\n"
            "Interest Rate: 8.5% per annum."
        )
        snippets = segment_page(page, evidence_id_start=1)
        assert len(snippets) == 2

    def test_evidence_ids_are_sequential(self):
        page = self._make_page(
            "Block A contains important financial terms.\n\n"
            "Block B contains covenant obligations here.\n\n"
            "Block C contains security and collateral details."
        )
        snippets = segment_page(page, evidence_id_start=5)
        ids = [s.evidence_id for s in snippets]
        assert ids == ["EV-005", "EV-006", "EV-007"]

    def test_evidence_ids_match_ev_pattern(self):
        page = self._make_page("Some important financial term: 8.5% per annum.")
        snippets = segment_page(page, evidence_id_start=1)
        for snippet in snippets:
            assert snippet.evidence_id.startswith("EV-")

    def test_page_number_preserved(self):
        page = self._make_page("Text on page 2.", page_number=2)
        snippets = segment_page(page, evidence_id_start=10)
        for snippet in snippets:
            assert snippet.page_number == 2

    def test_document_id_preserved(self):
        page = self._make_page("Some financial text.")
        snippets = segment_page(page, evidence_id_start=1)
        for snippet in snippets:
            assert snippet.document_id == "DEAL-002"

    def test_empty_page_returns_no_snippets(self):
        page = self._make_page("", has_text=False)
        snippets = segment_page(page, evidence_id_start=1)
        assert snippets == []

    def test_whitespace_only_page_returns_no_snippets(self):
        page = self._make_page("   \n\n\t\n  ", has_text=False)
        snippets = segment_page(page, evidence_id_start=1)
        assert snippets == []

    def test_blank_blocks_are_skipped(self):
        page = self._make_page("Real text block.\n\n   \n\nAnother real block.")
        snippets = segment_page(page, evidence_id_start=1)
        # Should produce 2 snippets (blank block in middle skipped).
        assert len(snippets) == 2

    def test_trivially_short_blocks_skipped(self):
        # "Hi" is fewer than MIN_BLOCK_CHARS
        page = self._make_page("Hi\n\nThis is a meaningful paragraph with enough content.")
        snippets = segment_page(page, evidence_id_start=1)
        assert len(snippets) == 1  # Only the substantial block.

    def test_section_detected_when_present(self):
        page = self._make_page(
            "4.2 — Interest Rate\nThe rate shall be 8.5% per annum."
        )
        snippets = segment_page(page, evidence_id_start=1)
        assert len(snippets) >= 1
        # At least one snippet should have section detected.
        sections = [s.section for s in snippets]
        assert any(sec is not None for sec in sections)

    def test_clause_is_none_when_no_section_detected(self):
        page = self._make_page(
            "The Borrower shall maintain a minimum DSCR of 1.25x at all times."
        )
        snippets = segment_page(page, evidence_id_start=1)
        # Plain body text — no section heading detectable.
        for snippet in snippets:
            assert snippet.clause is None

    def test_source_type_default_is_deal_document(self):
        page = self._make_page("Some deal text here to be extracted.")
        snippets = segment_page(page, evidence_id_start=1)
        for snippet in snippets:
            assert snippet.source_type == SourceType.DEAL_DOCUMENT

    def test_custom_source_type_passed_through(self):
        page = self._make_page("Policy rule text for testing purposes.")
        snippets = segment_page(page, evidence_id_start=1, source_type=SourceType.POLICY_DOCUMENT)
        for snippet in snippets:
            assert snippet.source_type == SourceType.POLICY_DOCUMENT

    def test_financial_text_not_stripped(self):
        critical_text = "Interest rate: 8.5% p.a., subject to quarterly revision."
        page = self._make_page(critical_text)
        snippets = segment_page(page, evidence_id_start=1)
        assert len(snippets) == 1
        assert "8.5%" in snippets[0].text
        assert "subject to" in snippets[0].text

    def test_large_block_split_respects_max_chars(self):
        long_para = "This is a long sentence about financial covenants and obligations. " * 30
        page = self._make_page(long_para)
        snippets = segment_page(page, evidence_id_start=1)
        for snippet in snippets:
            assert len(snippet.text) <= MAX_BLOCK_CHARS + 100  # small tolerance for splitting logic

    def test_evidence_id_start_respected_at_page_boundary(self):
        """When called for page 2 with start=5, first ID should be EV-005."""
        page = self._make_page("Text on page two.", page_number=2)
        snippets = segment_page(page, evidence_id_start=5)
        assert snippets[0].evidence_id == "EV-005"

    def test_has_extractable_text_false_returns_empty(self):
        page = self._make_page("Some text", has_text=False)
        snippets = segment_page(page, evidence_id_start=1)
        assert snippets == []
