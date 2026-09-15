"""Determinism tests — same PDF must produce identical evidence twice."""

from __future__ import annotations

import pathlib

import pytest

from app.ingestion.evidence_registry import EvidenceRegistry
from app.ingestion.pdf_loader import load_pdf
from app.ingestion.segmenter import segment_page

FIXTURE_PDF = pathlib.Path(__file__).parents[2] / "data" / "samples" / "deal_002.pdf"
PDF_EXISTS = FIXTURE_PDF.exists()


def _full_ingest(pdf_path: pathlib.Path, doc_id: str = "DEAL-002") -> EvidenceRegistry:
    """Run the full ingestion pipeline and return the populated registry."""
    result = load_pdf(pdf_path, doc_id)
    assert result.success

    registry = EvidenceRegistry()
    counter = 1
    for page in result.pages:
        snippets = segment_page(page, evidence_id_start=counter)
        for s in snippets:
            registry.add(s)
        counter += len(snippets)
    return registry


@pytest.mark.skipif(not PDF_EXISTS, reason="deal_002.pdf fixture not found")
class TestIngestionDeterminism:
    def test_same_evidence_count_on_two_runs(self):
        r1 = _full_ingest(FIXTURE_PDF)
        r2 = _full_ingest(FIXTURE_PDF)
        assert len(r1) == len(r2)

    def test_same_evidence_ids_on_two_runs(self):
        r1 = _full_ingest(FIXTURE_PDF)
        r2 = _full_ingest(FIXTURE_PDF)
        assert set(r1) == set(r2)

    def test_same_ordering_on_two_runs(self):
        r1 = _full_ingest(FIXTURE_PDF)
        r2 = _full_ingest(FIXTURE_PDF)
        ids1 = [s.evidence_id for s in r1.list_all()]
        ids2 = [s.evidence_id for s in r2.list_all()]
        assert ids1 == ids2

    def test_same_page_numbers_on_two_runs(self):
        r1 = _full_ingest(FIXTURE_PDF)
        r2 = _full_ingest(FIXTURE_PDF)
        pages1 = [s.page_number for s in r1.list_all()]
        pages2 = [s.page_number for s in r2.list_all()]
        assert pages1 == pages2

    def test_same_text_for_each_snippet_on_two_runs(self):
        r1 = _full_ingest(FIXTURE_PDF)
        r2 = _full_ingest(FIXTURE_PDF)
        for ev_id in r1:
            assert r1.get(ev_id).text == r2.get(ev_id).text

    def test_same_sections_on_two_runs(self):
        r1 = _full_ingest(FIXTURE_PDF)
        r2 = _full_ingest(FIXTURE_PDF)
        for ev_id in r1:
            assert r1.get(ev_id).section == r2.get(ev_id).section

    def test_same_document_id_on_two_runs(self):
        r1 = _full_ingest(FIXTURE_PDF)
        r2 = _full_ingest(FIXTURE_PDF)
        for ev_id in r1:
            assert r1.get(ev_id).document_id == r2.get(ev_id).document_id

    def test_no_random_ids_introduced(self):
        """Ensure IDs always start at EV-001 and are sequential."""
        r1 = _full_ingest(FIXTURE_PDF)
        ids = sorted(r1)
        assert ids[0] == "EV-001"
        for i, ev_id in enumerate(ids, start=1):
            assert ev_id == f"EV-{i:03d}"

    def test_five_consecutive_runs_produce_identical_results(self):
        """Extra thoroughness: 5 consecutive runs must all match run 1."""
        baseline = _full_ingest(FIXTURE_PDF)
        baseline_ids = [s.evidence_id for s in baseline.list_all()]
        baseline_texts = [s.text for s in baseline.list_all()]

        for _ in range(4):
            run = _full_ingest(FIXTURE_PDF)
            assert [s.evidence_id for s in run.list_all()] == baseline_ids
            assert [s.text for s in run.list_all()] == baseline_texts
