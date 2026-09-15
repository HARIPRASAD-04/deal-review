"""Unit tests for the EvidenceSnippet model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models.evidence import EvidenceSnippet, SourceType


# ── Valid creation ─────────────────────────────────────────────────────────────

class TestEvidenceSnippetCreation:
    def test_valid_evidence_creation(self):
        ev = EvidenceSnippet(
            evidence_id="EV-001",
            document_id="DEAL-001",
            page_number=2,
            section="1.1",
            clause="Facility Amount",
            text="The facility amount shall be INR 10 crore.",
            source_type=SourceType.DEAL_DOCUMENT,
        )
        assert ev.evidence_id == "EV-001"
        assert ev.document_id == "DEAL-001"
        assert ev.page_number == 2
        assert ev.section == "1.1"
        assert ev.clause == "Facility Amount"
        assert ev.source_type == SourceType.DEAL_DOCUMENT

    def test_section_is_optional(self):
        ev = EvidenceSnippet(
            evidence_id="EV-002",
            document_id="DEAL-001",
            page_number=1,
            clause="Preamble",
            text="Sample preamble text.",
        )
        assert ev.section is None

    def test_default_source_type_is_deal_document(self):
        ev = EvidenceSnippet(
            evidence_id="EV-003",
            document_id="DEAL-001",
            page_number=3,
            clause="Interest",
            text="Interest at 8.5%.",
        )
        assert ev.source_type == SourceType.DEAL_DOCUMENT

    def test_high_numbered_evidence_id(self):
        ev = EvidenceSnippet(
            evidence_id="EV-999",
            document_id="DEAL-002",
            page_number=10,
            clause="Covenant",
            text="Some covenant text.",
        )
        assert ev.evidence_id == "EV-999"

    def test_policy_source_type(self):
        ev = EvidenceSnippet(
            evidence_id="EV-100",
            document_id="POLICY-DOC-001",
            page_number=1,
            clause="Rule 1",
            text="Maximum facility shall not exceed INR 8 crore.",
            source_type=SourceType.POLICY_DOCUMENT,
        )
        assert ev.source_type == SourceType.POLICY_DOCUMENT

    def test_evidence_is_immutable(self):
        """EvidenceSnippet is frozen — mutations must raise an error."""
        ev = EvidenceSnippet(
            evidence_id="EV-005",
            document_id="DEAL-001",
            page_number=5,
            clause="Collateral",
            text="First charge on property.",
        )
        with pytest.raises(Exception):
            ev.page_number = 99  # type: ignore[misc]


# ── Invalid data — negative tests ──────────────────────────────────────────────

class TestEvidenceSnippetValidation:
    def test_missing_evidence_id_raises(self):
        with pytest.raises(ValidationError):
            EvidenceSnippet(
                document_id="DEAL-001",
                page_number=1,
                clause="Test",
                text="Some text.",
            )

    def test_invalid_evidence_id_format_raises(self):
        """ID must match EV-NNN pattern."""
        with pytest.raises(ValidationError):
            EvidenceSnippet(
                evidence_id="EVIDENCE-001",  # Wrong prefix.
                document_id="DEAL-001",
                page_number=1,
                clause="Test",
                text="Some text.",
            )

    def test_evidence_id_without_prefix_raises(self):
        with pytest.raises(ValidationError):
            EvidenceSnippet(
                evidence_id="001",
                document_id="DEAL-001",
                page_number=1,
                clause="Test",
                text="Some text.",
            )

    def test_page_number_zero_raises(self):
        """Page numbers must be >= 1."""
        with pytest.raises(ValidationError):
            EvidenceSnippet(
                evidence_id="EV-001",
                document_id="DEAL-001",
                page_number=0,
                clause="Test",
                text="Some text.",
            )

    def test_page_number_negative_raises(self):
        with pytest.raises(ValidationError):
            EvidenceSnippet(
                evidence_id="EV-001",
                document_id="DEAL-001",
                page_number=-1,
                clause="Test",
                text="Some text.",
            )

    def test_empty_clause_raises(self):
        with pytest.raises(ValidationError):
            EvidenceSnippet(
                evidence_id="EV-001",
                document_id="DEAL-001",
                page_number=1,
                clause="",
                text="Some text.",
            )

    def test_empty_text_raises(self):
        with pytest.raises(ValidationError):
            EvidenceSnippet(
                evidence_id="EV-001",
                document_id="DEAL-001",
                page_number=1,
                clause="Test",
                text="",
            )

    def test_invalid_source_type_raises(self):
        with pytest.raises(ValidationError):
            EvidenceSnippet(
                evidence_id="EV-001",
                document_id="DEAL-001",
                page_number=1,
                clause="Test",
                text="Some text.",
                source_type="not_a_valid_source",  # type: ignore[arg-type]
            )


# ── Serialisation ─────────────────────────────────────────────────────────────

class TestEvidenceSnippetSerialisation:
    def test_model_dump_round_trip(self):
        ev = EvidenceSnippet(
            evidence_id="EV-014",
            document_id="DEAL-001",
            page_number=4,
            section="2.1",
            clause="Facility Amount",
            text="The facility amount shall be INR 10 crore.",
        )
        data = ev.model_dump()
        restored = EvidenceSnippet(**data)
        assert restored == ev

    def test_model_dump_json(self):
        ev = EvidenceSnippet(
            evidence_id="EV-014",
            document_id="DEAL-001",
            page_number=4,
            clause="Facility Amount",
            text="The facility amount shall be INR 10 crore.",
        )
        json_str = ev.model_dump_json()
        assert "EV-014" in json_str
        assert "DEAL-001" in json_str
