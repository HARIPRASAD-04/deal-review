"""LLM-facing extraction schemas — Module 3.

These Pydantic models define the structured output contract between the LLM
and the Term Extraction Agent.  They are intentionally separate from the
domain models (``DealTerm``) because:

* The LLM-facing schema is a prompt/response contract.
* ``DealTerm`` is the validated application domain model.
* Fields such as ``notes`` are useful for the LLM's reasoning trace but do
  not belong on every ``DealTerm`` in the workflow state.

The mapping from ``ExtractedTermSchema`` → ``DealTerm`` is explicit and
performed by ``TermExtractionAgent``, not by this module.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from app.models.terms import TermCategory


class ExtractedTermSchema(BaseModel):
    """A single term candidate returned by the LLM.

    Attributes:
        name:             Machine-friendly term key (e.g. ``"interest_rate"``).
        value:            Raw extracted value, preserving original wording and
                          qualifiers (e.g. ``"8.5% per annum, subject to revision"``).
        normalized_value: Optional normalized form for deterministic comparison
                          (e.g. ``"0.085"``).  None when reliable normalization
                          is not possible.
        category:         Broad term classification from ``TermCategory`` enum.
        confidence:       Extraction confidence in ``[0.0, 1.0]``.  Treated as a
                          signal, NOT a calibrated probability.
        evidence_ids:     One or more ``EV-NNN`` IDs the LLM used as source.
        notes:            Optional free-text from the LLM explaining its
                          extraction reasoning.  Not stored in ``DealTerm``.
    """

    name: str = Field(
        ...,
        min_length=1,
        description="Machine-friendly term key, e.g. 'interest_rate'.",
    )
    value: str = Field(
        ...,
        min_length=1,
        description=(
            "Raw extracted value preserving original wording and qualifiers. "
            "Do not strip qualifiers such as 'subject to', 'approximately', 'up to'."
        ),
    )
    normalized_value: Optional[str] = Field(
        default=None,
        description=(
            "Optional normalized form for deterministic comparison. "
            "Leave None when reliable normalization is not possible."
        ),
    )
    category: TermCategory = Field(
        ...,
        description="Broad classification of this deal term.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Extraction confidence score in [0.0, 1.0]. "
            "Treat as a signal, not a calibrated probability."
        ),
    )
    evidence_ids: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "One or more EV-NNN identifiers from the supplied evidence. "
            "Every ID must correspond to a snippet in the evidence registry. "
            "Do NOT invent evidence IDs."
        ),
    )
    notes: Optional[str] = Field(
        default=None,
        description=(
            "Optional LLM reasoning note. Not stored in the domain model. "
            "Use this to explain ambiguity or conflicting information."
        ),
    )


class ExtractionOutput(BaseModel):
    """Top-level structured response from the Term Extraction LLM call.

    The LLM must return a single ``ExtractionOutput`` containing all extracted
    term candidates.  If no material terms are found, ``terms`` is empty.
    """

    terms: list[ExtractedTermSchema] = Field(
        default_factory=list,
        description=(
            "List of extracted term candidates. "
            "Empty list is valid when no material terms are found."
        ),
    )
