"""Integration tests — Term Extraction pipeline.

Tests the full path:
    deal_002.pdf
        → PyMuPDF
        → EvidenceRegistry
        → TermExtractionAgent (FakeLLMClient)
        → DealTerm[]
        → WorkflowState.extracted_terms

All LLM calls use FakeLLMClient — no network, no API key.
"""

from __future__ import annotations

import pathlib

import pytest

from app.agents.extraction.term_extraction import TermExtractionAgent
from app.ingestion.evidence_registry import EvidenceRegistry
from app.ingestion.pdf_loader import load_pdf
from app.ingestion.segmenter import segment_page
from app.llm.client import FakeLLMClient
from app.llm.schemas import ExtractionOutput, ExtractedTermSchema
from app.models.audit import AuditEventType
from app.models.state import WorkflowState
from app.models.terms import DealTerm, TermCategory, TermStatus
from app.orchestration.graph import build_graph

FIXTURE_PDF = pathlib.Path(__file__).parents[2] / "data" / "samples" / "deal_002.pdf"
PDF_EXISTS = FIXTURE_PDF.exists()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ingest_pdf() -> dict[str, object]:
    """Run the ingestion pipeline over the real PDF fixture."""
    result = load_pdf(str(FIXTURE_PDF), "DEAL-002")
    assert result.success, f"PDF ingestion failed: {result.error}"
    registry = EvidenceRegistry()
    counter = 1
    for page in result.pages:
        snippets = segment_page(page, evidence_id_start=counter)
        for s in snippets:
            registry.add(s)
        counter += len(snippets)
    return registry.to_dict()


def _fake_deal_terms(ev_registry: dict) -> ExtractionOutput:
    """Build a realistic fake LLM response referencing EV IDs that actually exist."""
    # Pick first three real EV IDs from the registry for grounded terms.
    real_ids = sorted(ev_registry.keys())[:4]
    terms = [
        ExtractedTermSchema(
            name="principal_amount",
            value="INR 50,000,000 (Indian Rupees Five Crore only)",
            normalized_value="50000000 INR",
            category=TermCategory.FINANCIAL,
            confidence=0.99,
            evidence_ids=[real_ids[0]],
        ),
        ExtractedTermSchema(
            name="interest_rate",
            value="8.5% per annum, subject to revision",
            normalized_value="0.085",
            category=TermCategory.RATE,
            confidence=0.99,
            evidence_ids=[real_ids[1]],
        ),
        ExtractedTermSchema(
            name="dscr_covenant",
            value="minimum DSCR of 1.25x, measured on a trailing 12-month basis",
            category=TermCategory.COVENANT,
            confidence=0.97,
            evidence_ids=[real_ids[2]],
        ),
    ]
    if len(real_ids) >= 4:
        terms.append(
            ExtractedTermSchema(
                name="tenure",
                value="60 months (5 years) from the date of first disbursement",
                normalized_value="60 months",
                category=TermCategory.DATE,
                confidence=0.98,
                evidence_ids=[real_ids[3]],
            )
        )
    return ExtractionOutput(terms=terms)


# ── Full Agent Integration (Agent + Registry, no LangGraph) ───────────────────

@pytest.mark.skipif(not PDF_EXISTS, reason="deal_002.pdf fixture not found")
class TestAgentWithRealRegistry:
    def test_agent_extracts_terms_from_real_evidence(self):
        ev_registry = _ingest_pdf()
        fake_output = _fake_deal_terms(ev_registry)
        agent = TermExtractionAgent(FakeLLMClient(response=fake_output))
        terms, rejections = agent.extract(ev_registry)
        assert len(terms) >= 3
        assert rejections == []

    def test_all_extracted_terms_are_deal_terms(self):
        ev_registry = _ingest_pdf()
        agent = TermExtractionAgent(FakeLLMClient(response=_fake_deal_terms(ev_registry)))
        terms, _ = agent.extract(ev_registry)
        assert all(isinstance(t, DealTerm) for t in terms)

    def test_all_evidence_ids_exist_in_registry(self):
        """Core traceability requirement: every EV-NNN in a term must exist."""
        ev_registry = _ingest_pdf()
        agent = TermExtractionAgent(FakeLLMClient(response=_fake_deal_terms(ev_registry)))
        terms, _ = agent.extract(ev_registry)
        for term in terms:
            for ev_id in term.evidence_ids:
                assert ev_id in ev_registry, (
                    f"Term '{term.name}' references '{ev_id}' which is not in the registry."
                )

    def test_term_ids_are_sequential_term_nnn(self):
        ev_registry = _ingest_pdf()
        agent = TermExtractionAgent(FakeLLMClient(response=_fake_deal_terms(ev_registry)))
        terms, _ = agent.extract(ev_registry)
        import re
        pattern = re.compile(r"^TERM-\d{3,}$")
        for term in terms:
            assert pattern.match(term.term_id), f"Unexpected term_id format: {term.term_id}"

    def test_traceability_chain(self):
        """DealTerm → evidence_ids → EvidenceSnippet → page_number + text."""
        ev_registry = _ingest_pdf()
        agent = TermExtractionAgent(FakeLLMClient(response=_fake_deal_terms(ev_registry)))
        terms, _ = agent.extract(ev_registry)
        for term in terms:
            for ev_id in term.evidence_ids:
                snippet = ev_registry[ev_id]
                assert snippet.page_number >= 1
                assert len(snippet.text.strip()) > 0

    def test_hallucinated_ev_rejected_from_real_registry(self):
        """A fake EV-999 not in the real registry must be rejected."""
        ev_registry = _ingest_pdf()
        hallucinated = ExtractionOutput(terms=[
            ExtractedTermSchema(
                name="fake_term",
                value="invented value",
                category=TermCategory.GENERAL,
                confidence=0.5,
                evidence_ids=["EV-999"],  # Does not exist in real registry.
            )
        ])
        agent = TermExtractionAgent(FakeLLMClient(response=hallucinated))
        terms, rejections = agent.extract(ev_registry)
        assert terms == []
        assert len(rejections) == 1
        assert "EV-999" in rejections[0]

    def test_qualifiers_in_extracted_value_preserved(self):
        ev_registry = _ingest_pdf()
        real_id = sorted(ev_registry.keys())[0]
        output = ExtractionOutput(terms=[
            ExtractedTermSchema(
                name="rate",
                value="8.5% per annum, subject to quarterly revision",
                category=TermCategory.RATE,
                confidence=0.95,
                evidence_ids=[real_id],
            )
        ])
        agent = TermExtractionAgent(FakeLLMClient(response=output))
        terms, _ = agent.extract(ev_registry)
        assert "subject to" in terms[0].value


# ── Full LangGraph Pipeline Integration ───────────────────────────────────────

@pytest.mark.skipif(not PDF_EXISTS, reason="deal_002.pdf fixture not found")
class TestLangGraphWithFakeLLM:
    def _initial_state(self) -> WorkflowState:
        return WorkflowState(
            document_id="DEAL-002",
            source_pdf_path=str(FIXTURE_PDF),
        )

    def _fake_llm(self) -> FakeLLMClient:
        """Build a FakeLLMClient — registry not known yet, populated lazily."""
        # We use a placeholder response; the real EV IDs will be set per-test
        # via a registry-aware factory.
        return FakeLLMClient(response=ExtractionOutput(terms=[]))

    def test_graph_completes_without_raising(self):
        ev_registry = _ingest_pdf()
        fake_output = _fake_deal_terms(ev_registry)
        graph = build_graph(llm_client=FakeLLMClient(response=fake_output))
        result = graph.invoke(self._initial_state())
        assert result is not None

    def test_graph_populates_extracted_terms(self):
        ev_registry = _ingest_pdf()
        fake_output = _fake_deal_terms(ev_registry)
        graph = build_graph(llm_client=FakeLLMClient(response=fake_output))
        result = graph.invoke(self._initial_state())
        final = WorkflowState(**result) if isinstance(result, dict) else result
        assert len(final.extracted_terms) >= 3

    def test_extracted_terms_keyed_by_term_nnn(self):
        ev_registry = _ingest_pdf()
        fake_output = _fake_deal_terms(ev_registry)
        graph = build_graph(llm_client=FakeLLMClient(response=fake_output))
        result = graph.invoke(self._initial_state())
        final = WorkflowState(**result) if isinstance(result, dict) else result
        import re
        pattern = re.compile(r"^TERM-\d{3,}$")
        for key in final.extracted_terms:
            assert pattern.match(key)

    def test_all_ev_ids_in_terms_exist_in_state_registry(self):
        ev_registry = _ingest_pdf()
        fake_output = _fake_deal_terms(ev_registry)
        graph = build_graph(llm_client=FakeLLMClient(response=fake_output))
        result = graph.invoke(self._initial_state())
        final = WorkflowState(**result) if isinstance(result, dict) else result
        for term in final.extracted_terms.values():
            for ev_id in term.evidence_ids:
                assert ev_id in final.evidence_registry, (
                    f"Term '{term.name}' references '{ev_id}' not in evidence_registry."
                )

    def test_term_extraction_completed_audit_event(self):
        ev_registry = _ingest_pdf()
        fake_output = _fake_deal_terms(ev_registry)
        graph = build_graph(llm_client=FakeLLMClient(response=fake_output))
        result = graph.invoke(self._initial_state())
        final = WorkflowState(**result) if isinstance(result, dict) else result
        event_types = {e.event_type for e in final.audit_events}
        assert AuditEventType.TERM_EXTRACTION_COMPLETED in event_types

    def test_term_extraction_started_audit_event(self):
        ev_registry = _ingest_pdf()
        fake_output = _fake_deal_terms(ev_registry)
        graph = build_graph(llm_client=FakeLLMClient(response=fake_output))
        result = graph.invoke(self._initial_state())
        final = WorkflowState(**result) if isinstance(result, dict) else result
        event_types = [e.event_type for e in final.audit_events]
        assert AuditEventType.TERM_EXTRACTION_STARTED in event_types

    def test_hallucinated_ev_produces_diagnostic_audit_event(self):
        """Rejected candidate → GENERIC audit event with warning."""
        ev_registry = _ingest_pdf()
        hallucinated = ExtractionOutput(terms=[
            ExtractedTermSchema(
                name="bad_term",
                value="invented",
                category=TermCategory.GENERAL,
                confidence=0.5,
                evidence_ids=["EV-999"],
            )
        ])
        graph = build_graph(llm_client=FakeLLMClient(response=hallucinated))
        result = graph.invoke(self._initial_state())
        final = WorkflowState(**result) if isinstance(result, dict) else result
        # Term must NOT be in extracted_terms.
        assert "bad_term" not in {t.name for t in final.extracted_terms.values()}
        # GENERIC diagnostic event must be present.
        generic_events = [
            e for e in final.audit_events
            if e.event_type == AuditEventType.GENERIC
        ]
        assert len(generic_events) >= 1
        assert any("EV-999" in e.message for e in generic_events)

    def test_zero_extracted_terms_no_extraction_failure(self):
        """Zero terms from LLM → COMPLETED (not FAILED)."""
        ev_registry = _ingest_pdf()
        empty_output = ExtractionOutput(terms=[])
        graph = build_graph(llm_client=FakeLLMClient(response=empty_output))
        result = graph.invoke(self._initial_state())
        final = WorkflowState(**result) if isinstance(result, dict) else result
        event_types = {e.event_type for e in final.audit_events}
        assert AuditEventType.TERM_EXTRACTION_COMPLETED in event_types
        assert AuditEventType.TERM_EXTRACTION_FAILED not in event_types

    def test_full_traceability_chain_through_state(self):
        """DealTerm → evidence_ids → state.evidence_registry → page + text."""
        ev_registry = _ingest_pdf()
        fake_output = _fake_deal_terms(ev_registry)
        graph = build_graph(llm_client=FakeLLMClient(response=fake_output))
        result = graph.invoke(self._initial_state())
        final = WorkflowState(**result) if isinstance(result, dict) else result
        for term in final.extracted_terms.values():
            for ev_id in term.evidence_ids:
                snippet = final.evidence_registry[ev_id]
                assert snippet.page_number >= 1
                assert snippet.document_id == "DEAL-002"
                assert len(snippet.text.strip()) > 0

    def test_no_llm_client_records_extraction_failure(self):
        """Module-level deal_review_graph (no LLM) → TERM_EXTRACTION_FAILED."""
        from app.orchestration.graph import deal_review_graph
        state = WorkflowState(
            document_id="DEAL-002",
            source_pdf_path=str(FIXTURE_PDF),
        )
        result = deal_review_graph.invoke(state)
        final = WorkflowState(**result) if isinstance(result, dict) else result
        event_types = {e.event_type for e in final.audit_events}
        assert AuditEventType.TERM_EXTRACTION_FAILED in event_types
        assert "extract_terms" in final.errors
