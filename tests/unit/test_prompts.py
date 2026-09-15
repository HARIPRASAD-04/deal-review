"""Unit tests for app/agents/extraction/prompts.py."""

from __future__ import annotations

from app.agents.extraction.prompts import SYSTEM_PROMPT, serialize_evidence
from app.models.evidence import EvidenceSnippet, SourceType


# ── Helpers ────────────────────────────────────────────────────────────────────

def _snippet(ev_id: str, page: int, text: str, section: str | None = None) -> EvidenceSnippet:
    return EvidenceSnippet(
        evidence_id=ev_id,
        document_id="DEAL-002",
        page_number=page,
        section=section,
        text=text,
    )


# ── SYSTEM_PROMPT content ─────────────────────────────────────────────────────

class TestSystemPrompt:
    def test_system_prompt_is_non_empty(self):
        assert len(SYSTEM_PROMPT.strip()) > 0

    def test_prompt_mentions_evidence_id(self):
        assert "EV-" in SYSTEM_PROMPT or "Evidence ID" in SYSTEM_PROMPT

    def test_prompt_forbids_compliance_analysis(self):
        lower = SYSTEM_PROMPT.lower()
        assert "compliance" in lower

    def test_prompt_forbids_risk_analysis(self):
        lower = SYSTEM_PROMPT.lower()
        assert "risk" in lower

    def test_prompt_mentions_qualifiers(self):
        lower = SYSTEM_PROMPT.lower()
        assert "subject to" in lower or "qualifier" in lower

    def test_prompt_instructs_not_to_invent_ev_ids(self):
        assert "Do NOT invent evidence IDs" in SYSTEM_PROMPT or "not invent" in SYSTEM_PROMPT.lower()


# ── serialize_evidence ─────────────────────────────────────────────────────────

class TestSerializeEvidence:
    def test_empty_list_returns_empty_string(self):
        result = serialize_evidence([])
        assert result == ""

    def test_ev_id_appears_as_first_token_in_block(self):
        snippet = _snippet("EV-001", page=1, text="Principal Amount: INR 50,000,000")
        result = serialize_evidence([snippet])
        assert result.startswith("[EV-001]")

    def test_page_number_in_output(self):
        snippet = _snippet("EV-018", page=2, text="Interest rate: 8.5% per annum")
        result = serialize_evidence([snippet])
        assert "Page 2" in result

    def test_section_included_when_present(self):
        snippet = _snippet("EV-018", page=2, text="Interest: 8.5%", section="4.1 - Interest Rate")
        result = serialize_evidence([snippet])
        assert "4.1 - Interest Rate" in result

    def test_section_none_not_rendered_as_none_string(self):
        snippet = _snippet("EV-001", page=1, text="Some text", section=None)
        result = serialize_evidence([snippet])
        assert "None" not in result

    def test_text_included_in_output(self):
        snippet = _snippet("EV-001", page=1, text="Borrower: Apex Manufacturing Pvt. Ltd.")
        result = serialize_evidence([snippet])
        assert "Apex Manufacturing" in result

    def test_multiple_snippets_separated_by_divider(self):
        snippets = [
            _snippet("EV-001", page=1, text="Principal Amount: INR 50,000,000"),
            _snippet("EV-002", page=1, text="Tenure: 60 months"),
        ]
        result = serialize_evidence(snippets)
        assert "---" in result
        assert "[EV-001]" in result
        assert "[EV-002]" in result

    def test_all_snippet_ids_appear_in_output(self):
        snippets = [
            _snippet(f"EV-{i:03d}", page=1, text=f"Text block {i}")
            for i in range(1, 6)
        ]
        result = serialize_evidence(snippets)
        for i in range(1, 6):
            assert f"EV-{i:03d}" in result

    def test_text_qualifiers_preserved_in_output(self):
        text = "Rate: 8.5% per annum, subject to revision at quarterly intervals."
        snippet = _snippet("EV-018", page=2, text=text)
        result = serialize_evidence([snippet])
        assert "subject to" in result
        assert "8.5%" in result
