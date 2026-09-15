"""Deal term model.

A DealTerm represents a single material term extracted from the deal document
by the Term Extraction Agent.  Each term:

* has a stable ``term_id`` (``TERM-001``, …)
* references one or more EvidenceSnippet IDs so the value can be traced
* carries a ``confidence`` score so downstream agents know how reliable the
  extraction was
* exposes a ``status`` field so the Orchestrator and Compliance Agent can
  track whether a term has been verified, challenged, or superseded

The model is intentionally kept open to allow future agents to attach
``normalized_value`` (e.g. converting "INR 10 crore" → 100_000_000.0) without
breaking the existing schema.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class TermCategory(str, Enum):
    """Broad category classifying what kind of deal term this is."""

    FINANCIAL = "financial"
    COVENANT = "covenant"
    OBLIGATION = "obligation"
    RATE = "rate"
    COLLATERAL = "collateral"
    CONDITION = "condition"
    EXCLUSION = "exclusion"
    PARTY = "party"
    DATE = "date"
    GENERAL = "general"


class TermStatus(str, Enum):
    """Lifecycle state of a deal term within the pipeline run."""

    EXTRACTED = "extracted"          # Initial extraction, not yet verified.
    VERIFIED = "verified"            # Confirmed correct by re-check.
    CHALLENGED = "challenged"        # Flagged as potentially incorrect.
    SUPERSEDED = "superseded"        # Replaced by a newer extraction.
    MISSING = "missing"              # Expected term not found in document.
    AMBIGUOUS = "ambiguous"          # Extracted but value is unclear.


class DealTerm(BaseModel):
    """A single material deal term extracted from the source document.

    Attributes:
        term_id:          Stable identifier in the format ``TERM-NNN``.
        name:             Machine-friendly key, e.g. ``"facility_amount"``.
        value:            Raw extracted value as a string (preserves original
                          wording).
        normalized_value: Optional normalised representation for comparison
                          (e.g. numeric amount in base units).
        category:         Broad classification of the term.
        confidence:       Extraction confidence score in ``[0.0, 1.0]``.
        evidence_ids:     List of ``EV-NNN`` IDs grounding this term.
        status:           Current lifecycle status of the term.
        notes:            Optional free-text notes from the extracting agent.
    """

    term_id: str = Field(
        ...,
        pattern=r"^TERM-\d{3,}$",
        description="Stable term identifier, e.g. TERM-001.",
        examples=["TERM-001"],
    )
    name: str = Field(
        ...,
        min_length=1,
        description="Machine-friendly term key, e.g. 'facility_amount'.",
    )
    value: str = Field(
        ...,
        min_length=1,
        description="Raw extracted value preserving original wording.",
    )
    normalized_value: Optional[str] = Field(
        default=None,
        description="Optional normalised value for deterministic comparison.",
    )
    category: TermCategory = Field(
        ...,
        description="Broad classification of this deal term.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Extraction confidence score in [0.0, 1.0].",
    )
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="EV-NNN IDs of the evidence grounding this term.",
    )
    status: TermStatus = Field(
        default=TermStatus.EXTRACTED,
        description="Current lifecycle status of this term.",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Optional free-text notes from the extracting agent.",
    )

    @field_validator("evidence_ids")
    @classmethod
    def validate_evidence_id_format(cls, ids: list[str]) -> list[str]:
        """Ensure every referenced evidence ID follows the EV-NNN pattern."""
        import re
        pattern = re.compile(r"^EV-\d{3,}$")
        for eid in ids:
            if not pattern.match(eid):
                raise ValueError(
                    f"Invalid evidence_id format: '{eid}'. Expected EV-NNN."
                )
        return ids
