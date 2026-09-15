"""Unit tests for TermExtractionAgent.

All tests use FakeLLMClient — no network calls, no API key required.

Coverage targets (from spec §20 and user amendments):
- Successful extraction (single / multiple terms)
- Valid evidence grounding
- Hallucinated EV-999 rejected + reason recorded
- Invalid (well-formatted but absent) EV ID rejected + reason recorded
- Empty evidence registry → ([], [])
- Zero terms returned by LLM → valid, no error
- Malformed LLM output (confidence string / invalid category / missing field)
- Qualifiers ("subject to", "approximately", "up to", "unless") preserved
- Multiple evidence IDs on one term
- Deterministic TERM-NNN IDs for identical inputs
- Conflicting values for same term name → both marked AMBIGUOUS
- Duplicate terms (same name, same value) → merged, evidence IDs combined
- Rejection reason is human-readable and non-empty
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.agents.extraction.term_extraction import TermExtractionAgent
from app.llm.client import FakeLLMClient
from app.llm.schemas import ExtractionOutput, ExtractedTermSchema
from app.models.evidence import EvidenceSnippet
from app.models.terms import DealTerm, TermCategory, TermStatus


# ── Fixtures / helpers ─────────────────────────────────────────────────────────

def _snippet(ev_id: str, page: int = 1, text: str = "Some deal text.") -> EvidenceSnippet:
    return EvidenceSnippet(
        evidence_id=ev_id,
        document_id="DEAL-002",
        page_number=page,
        text=text,
    )


def _registry(*ev_ids_and_texts: tuple[str, str]) -> dict[str, EvidenceSnippet]:
    """Build a minimal evidence registry from (ev_id, text) pairs."""
    return {
        ev_id: _snippet(ev_id, text=text)
        for ev_id, text in ev_ids_and_texts
    }


def _term_schema(
    name: str,
    value: str,
    ev_ids: list[str],
    category: TermCategory = TermCategory.FINANCIAL,
    confidence: float = 0.95,
    normalized_value: str | None = None,
    notes: str | None = None,
) -> ExtractedTermSchema:
    return ExtractedTermSchema(
        name=name,
        value=value,
        normalized_value=normalized_value,
        category=category,
        confidence=confidence,
        evidence_ids=ev_ids,
        notes=notes,
    )


def _fake_output(*schemas: ExtractedTermSchema) -> ExtractionOutput:
    return ExtractionOutput(terms=list(schemas))


def _agent(response: ExtractionOutput | None = None, error: Exception | None = None) -> TermExtractionAgent:
    client = FakeLLMClient(response=response, raise_error=error)
    return TermExtractionAgent(llm_client=client)


# ── Empty evidence registry ────────────────────────────────────────────────────

class TestEmptyRegistry:
    def test_empty_registry_returns_empty_lists(self):
        agent = _agent(response=_fake_output())
        terms, rejections = agent.extract({})
        assert terms == []
        assert rejections == []

    def test_empty_registry_does_not_call_llm(self):
        client = FakeLLMClient(response=_fake_output())
        agent = TermExtractionAgent(llm_client=client)
        agent.extract({})
        assert client.call_count == 0  # LLM not called for empty registry.


# ── Zero terms from LLM ───────────────────────────────────────────────────────

class TestZeroTermsExtracted:
    def test_zero_terms_is_valid_result(self):
        reg = _registry(("EV-001", "Some text."))
        agent = _agent(response=_fake_output())  # LLM returns no terms.
        terms, rejections = agent.extract(reg)
        assert terms == []
        assert rejections == []

    def test_zero_terms_not_treated_as_failure(self):
        """The agent returns ([], []) — caller decides if that's a problem."""
        reg = _registry(("EV-001", "Some text."))
        agent = _agent(response=_fake_output())
        terms, rejections = agent.extract(reg)
        assert isinstance(terms, list)
        assert isinstance(rejections, list)


# ── Successful extraction ──────────────────────────────────────────────────────

class TestSuccessfulExtraction:
    def test_single_term_extracted(self):
        reg = _registry(("EV-001", "Principal Amount: INR 50,000,000"))
        output = _fake_output(
            _term_schema("principal_amount", "INR 50,000,000", ["EV-001"])
        )
        terms, rejections = _agent(output).extract(reg)
        assert len(terms) == 1
        assert rejections == []

    def test_multiple_terms_extracted(self):
        reg = _registry(
            ("EV-001", "Principal: INR 50,000,000"),
            ("EV-002", "Interest: 8.5% p.a."),
            ("EV-003", "Tenure: 60 months"),
        )
        output = _fake_output(
            _term_schema("principal_amount", "INR 50,000,000", ["EV-001"]),
            _term_schema("interest_rate", "8.5% per annum", ["EV-002"], category=TermCategory.RATE),
            _term_schema("tenure", "60 months", ["EV-003"], category=TermCategory.DATE),
        )
        terms, rejections = _agent(output).extract(reg)
        assert len(terms) == 3
        assert rejections == []

    def test_returned_objects_are_deal_terms(self):
        reg = _registry(("EV-001", "Borrower: Apex Mfg"))
        output = _fake_output(_term_schema("borrower", "Apex Mfg", ["EV-001"], TermCategory.PARTY))
        terms, _ = _agent(output).extract(reg)
        assert all(isinstance(t, DealTerm) for t in terms)

    def test_term_status_is_extracted_by_default(self):
        reg = _registry(("EV-001", "Some text."))
        output = _fake_output(_term_schema("some_term", "some value", ["EV-001"]))
        terms, _ = _agent(output).extract(reg)
        assert terms[0].status == TermStatus.EXTRACTED

    def test_normalized_value_passed_through(self):
        reg = _registry(("EV-001", "Interest: 8.5% p.a."))
        output = _fake_output(
            _term_schema("interest_rate", "8.5% per annum", ["EV-001"],
                         normalized_value="0.085")
        )
        terms, _ = _agent(output).extract(reg)
        assert terms[0].normalized_value == "0.085"

    def test_normalized_value_none_when_not_provided(self):
        reg = _registry(("EV-001", "Some clause."))
        output = _fake_output(_term_schema("some_term", "some value", ["EV-001"]))
        terms, _ = _agent(output).extract(reg)
        assert terms[0].normalized_value is None

    def test_confidence_stored_as_provided(self):
        reg = _registry(("EV-001", "text"))
        output = _fake_output(_term_schema("term", "value", ["EV-001"], confidence=0.73))
        terms, _ = _agent(output).extract(reg)
        assert terms[0].confidence == pytest.approx(0.73)


# ── Evidence grounding validation ─────────────────────────────────────────────

class TestEvidenceGrounding:
    def test_valid_ev_ids_accepted(self):
        reg = _registry(("EV-001", "text"), ("EV-002", "more text"))
        output = _fake_output(_term_schema("term", "value", ["EV-001", "EV-002"]))
        terms, rejections = _agent(output).extract(reg)
        assert len(terms) == 1
        assert rejections == []
        assert "EV-001" in terms[0].evidence_ids
        assert "EV-002" in terms[0].evidence_ids

    def test_hallucinated_ev999_rejected(self):
        reg = _registry(("EV-001", "text"))
        output = _fake_output(_term_schema("bad_term", "bad value", ["EV-999"]))
        terms, rejections = _agent(output).extract(reg)
        assert terms == []
        assert len(rejections) == 1

    def test_hallucinated_ev_id_not_added_to_terms(self):
        """Most important test: hallucinated EV is never in extracted_terms."""
        reg = _registry(("EV-001", "real text"))
        output = _fake_output(
            _term_schema("good_term", "good value", ["EV-001"]),
            _term_schema("bad_term", "bad value", ["EV-999"]),
        )
        terms, rejections = _agent(output).extract(reg)
        term_names = [t.name for t in terms]
        assert "bad_term" not in term_names
        assert "good_term" in term_names

    def test_rejection_reason_is_non_empty_string(self):
        reg = _registry(("EV-001", "text"))
        output = _fake_output(_term_schema("bad_term", "val", ["EV-999"]))
        _, rejections = _agent(output).extract(reg)
        assert len(rejections) == 1
        assert isinstance(rejections[0], str)
        assert len(rejections[0]) > 10  # non-trivial message

    def test_rejection_reason_mentions_invalid_ev_id(self):
        reg = _registry(("EV-001", "text"))
        output = _fake_output(_term_schema("bad", "val", ["EV-999"]))
        _, rejections = _agent(output).extract(reg)
        assert "EV-999" in rejections[0]

    def test_rejection_reason_mentions_term_name(self):
        reg = _registry(("EV-001", "text"))
        output = _fake_output(_term_schema("interest_rate", "8.5%", ["EV-999"]))
        _, rejections = _agent(output).extract(reg)
        assert "interest_rate" in rejections[0]

    def test_mix_valid_and_invalid_ev_ids_rejects_whole_term(self):
        """If ANY evidence ID in a term is invalid, the whole term is rejected."""
        reg = _registry(("EV-001", "text"))
        output = _fake_output(
            _term_schema("mixed_term", "value", ["EV-001", "EV-999"])
        )
        terms, rejections = _agent(output).extract(reg)
        assert terms == []
        assert len(rejections) == 1

    def test_well_formatted_but_absent_ev_id_rejected(self):
        """EV-042 exists in the registry; EV-099 does not — same rejection logic."""
        reg = _registry(("EV-042", "text"))
        output = _fake_output(_term_schema("term", "val", ["EV-099"]))
        terms, rejections = _agent(output).extract(reg)
        assert terms == []
        assert "EV-099" in rejections[0]

    def test_multiple_valid_evidence_ids_on_one_term(self):
        reg = _registry(("EV-018", "interest clause"), ("EV-029", "another mention"))
        output = _fake_output(
            _term_schema("interest_rate", "8.5% per annum", ["EV-018", "EV-029"])
        )
        terms, _ = _agent(output).extract(reg)
        assert len(terms) == 1
        assert set(terms[0].evidence_ids) == {"EV-018", "EV-029"}


# ── Qualifier preservation ────────────────────────────────────────────────────

class TestQualifierPreservation:
    def test_subject_to_preserved(self):
        text = "8.5% per annum, subject to revision at quarterly intervals."
        reg = _registry(("EV-001", text))
        output = _fake_output(_term_schema("interest_rate", text, ["EV-001"]))
        terms, _ = _agent(output).extract(reg)
        assert "subject to" in terms[0].value

    def test_approximately_preserved(self):
        text = "approximately INR 50,000,000"
        reg = _registry(("EV-001", text))
        output = _fake_output(_term_schema("amount", text, ["EV-001"]))
        terms, _ = _agent(output).extract(reg)
        assert "approximately" in terms[0].value

    def test_up_to_preserved(self):
        text = "up to INR 75,000,000"
        reg = _registry(("EV-001", text))
        output = _fake_output(_term_schema("limit", text, ["EV-001"]))
        terms, _ = _agent(output).extract(reg)
        assert "up to" in terms[0].value

    def test_unless_preserved(self):
        text = "no premium shall be charged unless the borrower defaults."
        reg = _registry(("EV-001", text))
        output = _fake_output(_term_schema("prepayment", text, ["EV-001"]))
        terms, _ = _agent(output).extract(reg)
        assert "unless" in terms[0].value

    def test_not_less_than_preserved(self):
        text = "valued at not less than INR 70,000,000"
        reg = _registry(("EV-001", text))
        output = _fake_output(_term_schema("security_value", text, ["EV-001"]))
        terms, _ = _agent(output).extract(reg)
        assert "not less than" in terms[0].value


# ── Deterministic TERM-NNN IDs ────────────────────────────────────────────────

class TestDeterministicIDs:
    def test_ids_start_at_term_001(self):
        reg = _registry(("EV-001", "text"))
        output = _fake_output(_term_schema("term", "value", ["EV-001"]))
        terms, _ = _agent(output).extract(reg)
        assert terms[0].term_id == "TERM-001"

    def test_ids_are_sequential(self):
        reg = _registry(("EV-001", "t1"), ("EV-002", "t2"), ("EV-003", "t3"))
        output = _fake_output(
            _term_schema("a", "v1", ["EV-001"]),
            _term_schema("b", "v2", ["EV-002"]),
            _term_schema("c", "v3", ["EV-003"]),
        )
        terms, _ = _agent(output).extract(reg)
        ids = [t.term_id for t in terms]
        assert ids == ["TERM-001", "TERM-002", "TERM-003"]

    def test_same_input_same_ids(self):
        reg = _registry(("EV-001", "t"), ("EV-002", "t2"))
        output = _fake_output(
            _term_schema("a", "v", ["EV-001"]),
            _term_schema("b", "v2", ["EV-002"]),
        )
        client = FakeLLMClient(response=output)
        agent = TermExtractionAgent(llm_client=client)
        terms1, _ = agent.extract(reg)
        # Re-create agent with same fake output.
        client2 = FakeLLMClient(response=output)
        agent2 = TermExtractionAgent(llm_client=client2)
        terms2, _ = agent2.extract(reg)
        assert [t.term_id for t in terms1] == [t.term_id for t in terms2]

    def test_ids_follow_ev_nnn_pattern(self):
        import re
        reg = _registry(("EV-001", "text"))
        output = _fake_output(_term_schema("t", "v", ["EV-001"]))
        terms, _ = _agent(output).extract(reg)
        assert re.match(r"^TERM-\d{3,}$", terms[0].term_id)


# ── Conflict detection ────────────────────────────────────────────────────────

class TestConflictingEvidence:
    def test_conflicting_values_both_preserved(self):
        reg = _registry(("EV-018", "rate 8.5%"), ("EV-029", "rate 9.0%"))
        output = _fake_output(
            _term_schema("interest_rate", "8.5% per annum", ["EV-018"]),
            _term_schema("interest_rate", "9.0% per annum", ["EV-029"]),
        )
        terms, rejections = _agent(output).extract(reg)
        assert len(terms) == 2
        assert rejections == []

    def test_conflicting_terms_marked_ambiguous(self):
        reg = _registry(("EV-018", "8.5%"), ("EV-029", "9.0%"))
        output = _fake_output(
            _term_schema("interest_rate", "8.5% per annum", ["EV-018"]),
            _term_schema("interest_rate", "9.0% per annum", ["EV-029"]),
        )
        terms, _ = _agent(output).extract(reg)
        assert all(t.status == TermStatus.AMBIGUOUS for t in terms)

    def test_conflicting_values_not_silently_collapsed(self):
        reg = _registry(("EV-018", "8.5%"), ("EV-029", "9.0%"))
        output = _fake_output(
            _term_schema("interest_rate", "8.5% per annum", ["EV-018"]),
            _term_schema("interest_rate", "9.0% per annum", ["EV-029"]),
        )
        terms, _ = _agent(output).extract(reg)
        values = {t.value for t in terms}
        assert "8.5% per annum" in values
        assert "9.0% per annum" in values

    def test_non_conflicting_different_names_not_ambiguous(self):
        reg = _registry(("EV-001", "t1"), ("EV-002", "t2"))
        output = _fake_output(
            _term_schema("principal", "50,000,000", ["EV-001"]),
            _term_schema("interest_rate", "8.5%", ["EV-002"]),
        )
        terms, _ = _agent(output).extract(reg)
        assert all(t.status == TermStatus.EXTRACTED for t in terms)


# ── Duplicate / same-value merging ────────────────────────────────────────────

class TestDuplicateTermMerging:
    def test_same_name_same_value_merged_into_one_term(self):
        reg = _registry(("EV-001", "rate 8.5%"), ("EV-002", "rate 8.5% again"))
        output = _fake_output(
            _term_schema("interest_rate", "8.5% per annum", ["EV-001"]),
            _term_schema("interest_rate", "8.5% per annum", ["EV-002"]),
        )
        terms, _ = _agent(output).extract(reg)
        assert len(terms) == 1

    def test_merged_term_has_all_evidence_ids(self):
        reg = _registry(("EV-001", "rate 8.5%"), ("EV-002", "rate 8.5% again"))
        output = _fake_output(
            _term_schema("interest_rate", "8.5% per annum", ["EV-001"]),
            _term_schema("interest_rate", "8.5% per annum", ["EV-002"]),
        )
        terms, _ = _agent(output).extract(reg)
        assert set(terms[0].evidence_ids) == {"EV-001", "EV-002"}

    def test_merged_term_is_extracted_status(self):
        reg = _registry(("EV-001", "a"), ("EV-002", "a"))
        output = _fake_output(
            _term_schema("t", "same value", ["EV-001"]),
            _term_schema("t", "same value", ["EV-002"]),
        )
        terms, _ = _agent(output).extract(reg)
        assert terms[0].status == TermStatus.EXTRACTED


# ── Malformed LLM output (schema validation) ──────────────────────────────────

class TestMalformedLLMOutput:
    def test_confidence_string_raises_validation_error(self):
        """The LLM schema rejects non-float confidence at construction time."""
        with pytest.raises((ValidationError, TypeError, ValueError)):
            ExtractedTermSchema(
                name="term",
                value="val",
                category=TermCategory.FINANCIAL,
                confidence="very high",   # invalid
                evidence_ids=["EV-001"],
            )

    def test_confidence_above_1_raises_validation_error(self):
        with pytest.raises(ValidationError):
            ExtractedTermSchema(
                name="term",
                value="val",
                category=TermCategory.FINANCIAL,
                confidence=1.5,
                evidence_ids=["EV-001"],
            )

    def test_confidence_below_0_raises_validation_error(self):
        with pytest.raises(ValidationError):
            ExtractedTermSchema(
                name="term",
                value="val",
                category=TermCategory.FINANCIAL,
                confidence=-0.1,
                evidence_ids=["EV-001"],
            )

    def test_invalid_category_raises_validation_error(self):
        with pytest.raises(ValidationError):
            ExtractedTermSchema(
                name="term",
                value="val",
                category="not_a_category",   # invalid
                confidence=0.9,
                evidence_ids=["EV-001"],
            )

    def test_missing_name_raises_validation_error(self):
        with pytest.raises(ValidationError):
            ExtractedTermSchema(
                name="",    # min_length=1
                value="val",
                category=TermCategory.FINANCIAL,
                confidence=0.9,
                evidence_ids=["EV-001"],
            )

    def test_missing_value_raises_validation_error(self):
        with pytest.raises(ValidationError):
            ExtractedTermSchema(
                name="term",
                value="",   # min_length=1
                category=TermCategory.FINANCIAL,
                confidence=0.9,
                evidence_ids=["EV-001"],
            )

    def test_empty_evidence_ids_raises_validation_error(self):
        """evidence_ids has min_length=1 in ExtractedTermSchema."""
        with pytest.raises(ValidationError):
            ExtractedTermSchema(
                name="term",
                value="val",
                category=TermCategory.FINANCIAL,
                confidence=0.9,
                evidence_ids=[],  # must have at least one
            )

    def test_llm_api_failure_propagates(self):
        reg = _registry(("EV-001", "text"))
        agent = _agent(error=RuntimeError("LLM timeout"))
        with pytest.raises(RuntimeError, match="LLM timeout"):
            agent.extract(reg)
