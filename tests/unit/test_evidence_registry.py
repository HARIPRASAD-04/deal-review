"""Unit tests for EvidenceRegistry."""

from __future__ import annotations

import pytest

from app.ingestion.evidence_registry import EvidenceRegistry
from app.models.evidence import EvidenceSnippet, SourceType


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_snippet(ev_id: str, page: int = 1, text: str = "Sample text for testing.") -> EvidenceSnippet:
    return EvidenceSnippet(
        evidence_id=ev_id,
        document_id="DEAL-002",
        page_number=page,
        text=text,
    )


# ── Creation ───────────────────────────────────────────────────────────────────

class TestEvidenceRegistryCreation:
    def test_empty_registry(self):
        reg = EvidenceRegistry()
        assert len(reg) == 0

    def test_repr(self):
        reg = EvidenceRegistry()
        assert "0 snippets" in repr(reg)


# ── Add ────────────────────────────────────────────────────────────────────────

class TestEvidenceRegistryAdd:
    def test_add_single_snippet(self):
        reg = EvidenceRegistry()
        reg.add(_make_snippet("EV-001"))
        assert len(reg) == 1

    def test_add_multiple_snippets(self):
        reg = EvidenceRegistry()
        reg.add(_make_snippet("EV-001"))
        reg.add(_make_snippet("EV-002", page=2))
        reg.add(_make_snippet("EV-003", page=3))
        assert len(reg) == 3

    def test_duplicate_id_raises(self):
        reg = EvidenceRegistry()
        reg.add(_make_snippet("EV-001"))
        with pytest.raises(ValueError, match="Duplicate evidence_id"):
            reg.add(_make_snippet("EV-001"))

    def test_invalid_id_format_raises(self):
        reg = EvidenceRegistry()
        with pytest.raises((ValueError, Exception)):
            reg.add(_make_snippet("INVALID-001"))


# ── Get ────────────────────────────────────────────────────────────────────────

class TestEvidenceRegistryGet:
    def test_get_existing_snippet(self):
        reg = EvidenceRegistry()
        snippet = _make_snippet("EV-042", text="Principal Amount: INR 50,000,000.")
        reg.add(snippet)
        retrieved = reg.get("EV-042")
        assert retrieved.text == "Principal Amount: INR 50,000,000."

    def test_get_missing_id_raises_key_error(self):
        reg = EvidenceRegistry()
        with pytest.raises(KeyError):
            reg.get("EV-999")

    def test_get_invalid_format_raises_value_error(self):
        reg = EvidenceRegistry()
        with pytest.raises(ValueError):
            reg.get("BAD-ID")

    def test_get_or_none_existing(self):
        reg = EvidenceRegistry()
        reg.add(_make_snippet("EV-001"))
        result = reg.get_or_none("EV-001")
        assert result is not None

    def test_get_or_none_missing_returns_none(self):
        reg = EvidenceRegistry()
        result = reg.get_or_none("EV-999")
        assert result is None

    def test_get_or_none_invalid_format_returns_none(self):
        reg = EvidenceRegistry()
        result = reg.get_or_none("GARBAGE")
        assert result is None


# ── Has ────────────────────────────────────────────────────────────────────────

class TestEvidenceRegistryHas:
    def test_has_existing(self):
        reg = EvidenceRegistry()
        reg.add(_make_snippet("EV-001"))
        assert reg.has("EV-001") is True

    def test_has_missing(self):
        reg = EvidenceRegistry()
        assert reg.has("EV-001") is False

    def test_contains_protocol(self):
        reg = EvidenceRegistry()
        reg.add(_make_snippet("EV-001"))
        assert "EV-001" in reg
        assert "EV-999" not in reg


# ── List ───────────────────────────────────────────────────────────────────────

class TestEvidenceRegistryList:
    def test_list_all_returns_in_insertion_order(self):
        reg = EvidenceRegistry()
        reg.add(_make_snippet("EV-001"))
        reg.add(_make_snippet("EV-002"))
        reg.add(_make_snippet("EV-003"))
        ids = [s.evidence_id for s in reg.list_all()]
        assert ids == ["EV-001", "EV-002", "EV-003"]

    def test_list_all_empty(self):
        reg = EvidenceRegistry()
        assert reg.list_all() == []

    def test_list_for_document(self):
        reg = EvidenceRegistry()
        reg.add(_make_snippet("EV-001"))
        reg.add(
            EvidenceSnippet(
                evidence_id="EV-002",
                document_id="DEAL-003",
                page_number=1,
                text="Different document.",
            )
        )
        result = reg.list_for_document("DEAL-002")
        assert len(result) == 1
        assert result[0].evidence_id == "EV-001"

    def test_list_for_document_no_match(self):
        reg = EvidenceRegistry()
        reg.add(_make_snippet("EV-001"))
        assert reg.list_for_document("DEAL-999") == []


# ── to_dict ────────────────────────────────────────────────────────────────────

class TestEvidenceRegistryToDict:
    def test_to_dict_structure(self):
        reg = EvidenceRegistry()
        reg.add(_make_snippet("EV-001"))
        reg.add(_make_snippet("EV-002"))
        d = reg.to_dict()
        assert isinstance(d, dict)
        assert "EV-001" in d
        assert "EV-002" in d
        assert isinstance(d["EV-001"], EvidenceSnippet)

    def test_to_dict_is_copy(self):
        reg = EvidenceRegistry()
        reg.add(_make_snippet("EV-001"))
        d = reg.to_dict()
        d["EV-999"] = _make_snippet("EV-999")  # mutate copy
        assert not reg.has("EV-999")  # original unaffected


# ── Iteration ──────────────────────────────────────────────────────────────────

class TestEvidenceRegistryIteration:
    def test_iter_yields_ids(self):
        reg = EvidenceRegistry()
        reg.add(_make_snippet("EV-001"))
        reg.add(_make_snippet("EV-002"))
        ids = list(reg)
        assert "EV-001" in ids
        assert "EV-002" in ids

    def test_len_matches_add_count(self):
        reg = EvidenceRegistry()
        for i in range(1, 6):
            reg.add(_make_snippet(f"EV-{i:03d}"))
        assert len(reg) == 5
