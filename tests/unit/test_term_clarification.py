"""Unit tests for TermExtractionAgent.clarify() targeted re-extraction — Module 5."""

from __future__ import annotations

import pytest

from app.agents.extraction.term_extraction import TermExtractionAgent
from app.llm.client import FakeLLMClient
from app.llm.schemas import ExtractionOutput, ExtractedTermSchema
from app.models.evidence import EvidenceSnippet
from app.models.terms import TermCategory, TermStatus


@pytest.fixture
def evidence_registry() -> dict[str, EvidenceSnippet]:
    return {
        "EV-001": EvidenceSnippet(
            evidence_id="EV-001",
            document_id="DEAL-001",
            page_number=1,
            text="The Interest Rate shall be 8.5% per annum.",
            char_start=0,
            char_end=42,
        ),
    }


class TestTermClarification:
    def test_clarify_empty_registry_returns_empty(self):
        client = FakeLLMClient(response=ExtractionOutput(terms=[]))
        agent = TermExtractionAgent(client)

        terms, rejections = agent.clarify(
            evidence_registry={},
            requested_term_names=["interest_rate"],
        )
        assert terms == []
        assert rejections == []

    def test_clarify_successful_extraction_respects_start_counter(
        self, evidence_registry: dict[str, EvidenceSnippet]
    ):
        fake_output = ExtractionOutput(
            terms=[
                ExtractedTermSchema(
                    name="interest_rate",
                    value="8.5%",
                    category=TermCategory.RATE,
                    confidence=0.95,
                    evidence_ids=["EV-001"],
                )
            ]
        )
        client = FakeLLMClient(response=fake_output)
        agent = TermExtractionAgent(client)

        terms, rejections = agent.clarify(
            evidence_registry=evidence_registry,
            requested_term_names=["interest_rate"],
            reason="Missing interest_rate term.",
            start_counter=6,
        )

        assert len(terms) == 1
        assert rejections == []
        term = terms[0]
        assert term.term_id == "TERM-006"
        assert term.name == "interest_rate"
        assert term.value == "8.5%"
        assert term.evidence_ids == ["EV-001"]
        assert term.status == TermStatus.EXTRACTED

    def test_clarify_rejects_invalid_evidence_id(
        self, evidence_registry: dict[str, EvidenceSnippet]
    ):
        fake_output = ExtractionOutput(
            terms=[
                ExtractedTermSchema(
                    name="interest_rate",
                    value="8.5%",
                    category=TermCategory.RATE,
                    confidence=0.9,
                    evidence_ids=["EV-999"],  # Invalid / not in registry
                )
            ]
        )
        client = FakeLLMClient(response=fake_output)
        agent = TermExtractionAgent(client)

        terms, rejections = agent.clarify(
            evidence_registry=evidence_registry,
            requested_term_names=["interest_rate"],
            start_counter=1,
        )

        assert terms == []
        assert len(rejections) == 1
        assert "EV-999" in rejections[0]
