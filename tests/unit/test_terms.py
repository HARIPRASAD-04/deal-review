"""Unit tests for the DealTerm model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models.terms import DealTerm, TermCategory, TermStatus


class TestDealTermCreation:
    def test_valid_deal_term(self):
        term = DealTerm(
            term_id="TERM-001",
            name="facility_amount",
            value="INR 10 crore",
            category=TermCategory.FINANCIAL,
            confidence=0.97,
            evidence_ids=["EV-001"],
        )
        assert term.term_id == "TERM-001"
        assert term.name == "facility_amount"
        assert term.status == TermStatus.EXTRACTED  # Default.
        assert term.confidence == 0.97

    def test_normalized_value_optional(self):
        term = DealTerm(
            term_id="TERM-002",
            name="interest_rate",
            value="8.5% per annum",
            category=TermCategory.RATE,
            confidence=0.95,
            evidence_ids=["EV-002"],
        )
        assert term.normalized_value is None

    def test_normalized_value_set(self):
        term = DealTerm(
            term_id="TERM-002",
            name="interest_rate",
            value="8.5% per annum",
            normalized_value="0.085",
            category=TermCategory.RATE,
            confidence=0.95,
            evidence_ids=["EV-002"],
        )
        assert term.normalized_value == "0.085"

    def test_all_term_categories(self):
        for cat in TermCategory:
            term = DealTerm(
                term_id="TERM-001",
                name="test",
                value="test",
                category=cat,
                confidence=0.5,
                evidence_ids=[],
            )
            assert term.category == cat

    def test_all_term_statuses(self):
        for status in TermStatus:
            term = DealTerm(
                term_id="TERM-001",
                name="test",
                value="test",
                category=TermCategory.GENERAL,
                confidence=0.5,
                status=status,
            )
            assert term.status == status

    def test_multiple_evidence_ids(self):
        term = DealTerm(
            term_id="TERM-001",
            name="facility_amount",
            value="INR 10 crore",
            category=TermCategory.FINANCIAL,
            confidence=0.9,
            evidence_ids=["EV-001", "EV-002", "EV-003"],
        )
        assert len(term.evidence_ids) == 3

    def test_empty_evidence_ids_allowed(self):
        """Terms can have no evidence yet (e.g. MISSING status)."""
        term = DealTerm(
            term_id="TERM-001",
            name="guarantor",
            value="Unknown",
            category=TermCategory.OBLIGATION,
            confidence=0.0,
            evidence_ids=[],
            status=TermStatus.MISSING,
        )
        assert term.evidence_ids == []


class TestDealTermValidation:
    def test_invalid_term_id_format_raises(self):
        with pytest.raises(ValidationError):
            DealTerm(
                term_id="T-001",  # Wrong prefix.
                name="test",
                value="test",
                category=TermCategory.FINANCIAL,
                confidence=0.5,
            )

    def test_confidence_above_1_raises(self):
        with pytest.raises(ValidationError):
            DealTerm(
                term_id="TERM-001",
                name="test",
                value="test",
                category=TermCategory.FINANCIAL,
                confidence=1.01,
            )

    def test_confidence_below_0_raises(self):
        with pytest.raises(ValidationError):
            DealTerm(
                term_id="TERM-001",
                name="test",
                value="test",
                category=TermCategory.FINANCIAL,
                confidence=-0.01,
            )

    def test_invalid_evidence_id_format_raises(self):
        """evidence_ids must match EV-NNN pattern."""
        with pytest.raises(ValidationError):
            DealTerm(
                term_id="TERM-001",
                name="test",
                value="test",
                category=TermCategory.FINANCIAL,
                confidence=0.5,
                evidence_ids=["EVIDENCE-001"],  # Wrong format.
            )

    def test_missing_name_raises(self):
        with pytest.raises(ValidationError):
            DealTerm(
                term_id="TERM-001",
                name="",
                value="test",
                category=TermCategory.FINANCIAL,
                confidence=0.5,
            )

    def test_missing_value_raises(self):
        with pytest.raises(ValidationError):
            DealTerm(
                term_id="TERM-001",
                name="test",
                value="",
                category=TermCategory.FINANCIAL,
                confidence=0.5,
            )


class TestDealTermSerialisation:
    def test_round_trip(self):
        term = DealTerm(
            term_id="TERM-001",
            name="facility_amount",
            value="INR 10 crore",
            category=TermCategory.FINANCIAL,
            confidence=0.97,
            evidence_ids=["EV-001"],
        )
        restored = DealTerm(**term.model_dump())
        assert restored == term
