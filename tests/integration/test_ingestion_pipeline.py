"""Integration tests — full PDF → EvidenceRegistry → WorkflowState pipeline."""

from __future__ import annotations

import pathlib

import pytest

from app.ingestion.evidence_registry import EvidenceRegistry
from app.ingestion.pdf_loader import load_pdf
from app.ingestion.segmenter import segment_page
from app.models.audit import AuditEventType
from app.models.evidence import EvidenceSnippet
from app.models.state import WorkflowState
from app.orchestration.graph import deal_review_graph, ingest_document

FIXTURE_PDF = pathlib.Path(__file__).parents[2] / "data" / "samples" / "deal_002.pdf"
PDF_EXISTS = FIXTURE_PDF.exists()


def _run_ingestion(pdf_path: pathlib.Path, doc_id: str = "DEAL-002"):
    """Helper: load PDF, segment, build registry."""
    result = load_pdf(pdf_path, doc_id)
    assert result.success, f"Ingestion failed: {result.error}"

    registry = EvidenceRegistry()
    counter = 1
    for page in result.pages:
        snippets = segment_page(page, evidence_id_start=counter)
        for s in snippets:
            registry.add(s)
        counter += len(snippets)
    return result, registry


# ── Full pipeline ──────────────────────────────────────────────────────────────

@pytest.mark.skipif(not PDF_EXISTS, reason="deal_002.pdf fixture not found")
class TestFullIngestionPipeline:
    def test_pipeline_produces_snippets(self):
        _, registry = _run_ingestion(FIXTURE_PDF)
        assert len(registry) > 0

    def test_evidence_count_reasonable(self):
        """A 3-page deal document should produce a meaningful number of snippets."""
        _, registry = _run_ingestion(FIXTURE_PDF)
        assert len(registry) >= 5

    def test_all_snippets_have_non_empty_text(self):
        _, registry = _run_ingestion(FIXTURE_PDF)
        for snippet in registry.list_all():
            assert len(snippet.text.strip()) > 0

    def test_all_snippets_have_valid_ev_id(self):
        import re
        _, registry = _run_ingestion(FIXTURE_PDF)
        pattern = re.compile(r"^EV-\d{3,}$")
        for snippet in registry.list_all():
            assert pattern.match(snippet.evidence_id), f"Bad ID: {snippet.evidence_id}"

    def test_all_snippets_have_correct_document_id(self):
        _, registry = _run_ingestion(FIXTURE_PDF, doc_id="DEAL-002")
        for snippet in registry.list_all():
            assert snippet.document_id == "DEAL-002"

    def test_all_page_numbers_are_positive_and_in_range(self):
        result, registry = _run_ingestion(FIXTURE_PDF)
        max_page = result.metadata.page_count
        for snippet in registry.list_all():
            assert 1 <= snippet.page_number <= max_page

    def test_ev_001_is_first_snippet(self):
        _, registry = _run_ingestion(FIXTURE_PDF)
        assert registry.has("EV-001")

    def test_ev_ids_are_sequential_no_gaps(self):
        """EV-001, EV-002, ... EV-N with no gaps."""
        _, registry = _run_ingestion(FIXTURE_PDF)
        count = len(registry)
        expected_ids = {f"EV-{i:03d}" for i in range(1, count + 1)}
        actual_ids = set(registry)
        assert actual_ids == expected_ids

    def test_key_financial_terms_present_in_evidence(self):
        """Critical financial text must survive the ingestion pipeline."""
        _, registry = _run_ingestion(FIXTURE_PDF)
        all_text = " ".join(s.text for s in registry.list_all())
        # These are in the synthetic PDF fixture.
        assert "8.5%" in all_text
        assert "50,000,000" in all_text
        assert "1.25" in all_text

    def test_qualifiers_preserved(self):
        _, registry = _run_ingestion(FIXTURE_PDF)
        all_text = " ".join(s.text for s in registry.list_all()).lower()
        assert "subject to" in all_text

    def test_traceability_chain(self):
        """Each snippet can be traced: ev_id → doc_id → page_number → text."""
        _, registry = _run_ingestion(FIXTURE_PDF)
        snippet = registry.get("EV-001")
        assert snippet.document_id == "DEAL-002"
        assert snippet.page_number >= 1
        assert len(snippet.text) > 0

    def test_sections_extracted_from_pdf(self):
        """Sections with numeric prefixes (e.g. 4.1 - Interest Rate) are extracted."""
        _, registry = _run_ingestion(FIXTURE_PDF)
        snippets = registry.list_all()
        detected_sections = [s.section for s in snippets if s.section is not None]
        assert len(detected_sections) > 0
        assert any("4.1" in sec for sec in detected_sections)

    def test_evidence_integrates_with_workflow_state(self):
        """Registry.to_dict() can be assigned to WorkflowState.evidence_registry."""
        result, registry = _run_ingestion(FIXTURE_PDF)
        state = WorkflowState(
            document_id="DEAL-002",
            evidence_registry=registry.to_dict(),
            document_metadata=result.metadata,
        )
        assert len(state.evidence_registry) == len(registry)
        assert state.document_metadata is not None
        assert state.document_metadata.page_count == 3

    def test_registry_lookup_by_id_works_after_state_roundtrip(self):
        _, registry = _run_ingestion(FIXTURE_PDF)
        state = WorkflowState(
            document_id="DEAL-002",
            evidence_registry=registry.to_dict(),
        )
        # Direct dict access on state.evidence_registry
        ev = state.evidence_registry.get("EV-001")
        assert ev is not None
        assert isinstance(ev, EvidenceSnippet)


# ── LangGraph node integration ─────────────────────────────────────────────────

@pytest.mark.skipif(not PDF_EXISTS, reason="deal_002.pdf fixture not found")
class TestIngestDocumentNode:
    def _initial_state(self) -> WorkflowState:
        return WorkflowState(
            document_id="DEAL-002",
            source_pdf_path=str(FIXTURE_PDF),
        )

    def test_node_returns_evidence_registry(self):
        state = self._initial_state()
        result = ingest_document(state)
        assert "evidence_registry" in result
        assert len(result["evidence_registry"]) > 0

    def test_node_returns_document_metadata(self):
        state = self._initial_state()
        result = ingest_document(state)
        assert "document_metadata" in result
        assert result["document_metadata"].page_count == 3

    def test_node_emits_ingestion_completed_audit_event(self):
        state = self._initial_state()
        result = ingest_document(state)
        event_types = [e.event_type for e in result["audit_events"]]
        assert AuditEventType.DOCUMENT_INGESTION_COMPLETED in event_types

    def test_node_missing_path_returns_error(self):
        state = WorkflowState(document_id="DEAL-002")  # no source_pdf_path
        result = ingest_document(state)
        assert "errors" in result
        assert "ingest_document" in result["errors"]

    def test_node_missing_path_emits_failed_event(self):
        state = WorkflowState(document_id="DEAL-002")
        result = ingest_document(state)
        event_types = [e.event_type for e in result["audit_events"]]
        assert AuditEventType.DOCUMENT_INGESTION_FAILED in event_types

    def test_node_invalid_path_returns_error(self):
        state = WorkflowState(
            document_id="DEAL-002",
            source_pdf_path="/nonexistent/path/file.pdf",
        )
        result = ingest_document(state)
        assert "errors" in result

    def test_full_graph_with_valid_pdf(self):
        """End-to-end: run the full compiled graph with a real PDF."""
        state = self._initial_state()
        raw_result = deal_review_graph.invoke(state)
        final = WorkflowState(**raw_result) if isinstance(raw_result, dict) else raw_result
        assert len(final.evidence_registry) > 0
        assert final.document_metadata is not None

    def test_full_graph_audit_trail_contains_ingestion_events(self):
        state = self._initial_state()
        raw_result = deal_review_graph.invoke(state)
        final = WorkflowState(**raw_result) if isinstance(raw_result, dict) else raw_result
        event_types = {e.event_type for e in final.audit_events}
        assert AuditEventType.RUN_STARTED in event_types
        assert AuditEventType.DOCUMENT_INGESTION_STARTED in event_types
        assert AuditEventType.DOCUMENT_INGESTION_COMPLETED in event_types

    def test_full_graph_no_pdf_produces_error_in_state(self):
        """Graph should complete without crashing even when PDF is missing."""
        state = WorkflowState(document_id="DEAL-002")
        raw_result = deal_review_graph.invoke(state)
        final = WorkflowState(**raw_result) if isinstance(raw_result, dict) else raw_result
        # No crash, but error recorded.
        assert "ingest_document" in final.errors
