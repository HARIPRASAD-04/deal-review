"""Evidence model.

An EvidenceSnippet represents a grounded, traceable reference back to a
specific location in the source document.  Every material fact extracted by
the Term Extraction Agent must cite at least one EvidenceSnippet, so the
compliance and risk decisions can always be traced back to the original text.

Evidence IDs (``EV-001``, ``EV-002``, …) are assigned once during document
parsing and remain stable for the duration of a pipeline run.  Downstream
agents must never invent new IDs; they should reference existing ones.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class SourceType(str, Enum):
    """Where the evidence snippet originates from."""

    DEAL_DOCUMENT = "deal_document"
    POLICY_DOCUMENT = "policy_document"
    SUPPORTING_ANNEX = "supporting_annex"


class EvidenceSnippet(BaseModel):
    """A single grounded evidence reference extracted from a source document.

    Attributes:
        evidence_id:  Stable unique identifier in the format ``EV-NNN``.
        document_id:  Identifier of the source document (e.g. ``DEAL-001``).
        page_number:  1-indexed page number within the source document.
        section:      Section heading or number (e.g. ``"2.1"``).  Optional
                      because not all documents have explicit section markers.
        clause:       Human-readable clause label (e.g. ``"Facility Amount"``).
        text:         Verbatim text snippet from the document.
        source_type:  Enum distinguishing deal documents from policy documents.
    """

    evidence_id: str = Field(
        ...,
        pattern=r"^EV-\d{3,}$",
        description="Stable evidence identifier, e.g. EV-014.",
        examples=["EV-014"],
    )
    document_id: str = Field(
        ...,
        description="Identifier of the source document.",
        examples=["DEAL-001"],
    )
    page_number: int = Field(
        ...,
        ge=1,
        description="1-indexed page number in the source document.",
    )
    section: Optional[str] = Field(
        default=None,
        description="Section identifier, e.g. '2.1' or 'Schedule A'.",
    )
    clause: str = Field(
        ...,
        min_length=1,
        description="Clause or heading label for the snippet.",
        examples=["Facility Amount"],
    )
    text: str = Field(
        ...,
        min_length=1,
        description="Verbatim text extracted from the source document.",
    )
    source_type: SourceType = Field(
        default=SourceType.DEAL_DOCUMENT,
        description="Origin type of the evidence.",
    )

    @field_validator("evidence_id")
    @classmethod
    def evidence_id_must_be_uppercase(cls, v: str) -> str:
        """Ensure the prefix is always upper-case EV."""
        if not v.startswith("EV-"):
            raise ValueError("evidence_id must start with 'EV-'")
        return v

    model_config = {"frozen": True}   # Evidence is immutable once created.
