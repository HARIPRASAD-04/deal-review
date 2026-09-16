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


# ── Module 6: Summary generation schema ───────────────────────────────────────

class SummaryOutput(BaseModel):
    """Structured response from the LLM when generating the executive summary.

    The LLM receives structured risk data (not raw PDF text) and must return
    a concise summary grounded only in the information supplied.  Key_risks and
    recommended_actions are capped to keep the output actionable and scannable.

    Attributes:
        executive_summary:    2-4 sentence narrative for a credit officer.
        key_risks:            Up to 5 bullet-point risk titles (critical/high first).
        recommended_actions:  Up to 5 specific, actionable next steps.
    """

    executive_summary: str = Field(
        ...,
        min_length=1,
        description=(
            "2-4 sentence narrative executive summary for a credit officer. "
            "Ground every statement in the supplied risk data. "
            "Do NOT invent information not present in the input."
        ),
    )
    key_risks: list[str] = Field(
        default_factory=list,
        description=(
            "Up to 5 concise bullet-point risk titles. "
            "Order critical and high severity risks first."
        ),
    )
    recommended_actions: list[str] = Field(
        default_factory=list,
        description=(
            "Up to 5 specific, actionable next steps. "
            "Be concrete — avoid generic advice like 'review the document'."
        ),
    )


# ── Module 8: Policy extraction schema ────────────────────────────────────────

class ExtractedPolicyRuleSchema(BaseModel):
    """A single policy rule candidate returned by the LLM.

    The LLM must only extract rules that are clearly and unambiguously stated
    in the policy document.  If a statement is vague, incomplete, or cannot be
    reliably codified, it must be placed in ``PolicyExtractionOutput.ambiguous_statements``
    instead.

    Attributes:
        rule_id:       Verbatim rule identifier from the document (e.g. "Rule 3",
                       "POLICY-003").  Set to ``None`` if no explicit ID appears
                       in the document — the parser will assign a deterministic
                       ``POLICY-EXT-NNN`` identifier.  Do NOT invent an ID.
        name:          Short machine-friendly key for the rule (snake_case).
        description:   Full verbatim or paraphrased description of the rule.
        category:      Domain category.  Must be one of: financial, legal,
                       operational, credit, regulatory, collateral, general.
        operator:      Comparison operator: ``<=``, ``>=``, ``==``, ``!=``,
                       ``exists``, ``in``, or ``semantic``.
        threshold:     The required limit or value as a string.
        severity:      Violation severity: critical, high, medium, low.
        applicability: Optional scope (e.g. ``"term_loan"``).
        reference:     Verbatim section/clause reference from the document.
        is_mandatory:  Whether the rule is always applied.
    """

    rule_id: Optional[str] = Field(
        default=None,
        description=(
            "Verbatim rule ID from the document. "
            "Set to None if no explicit ID is present — do NOT invent one."
        ),
    )
    name: str = Field(
        ...,
        min_length=1,
        description="Short machine-friendly rule name (snake_case).",
    )
    description: str = Field(
        ...,
        min_length=1,
        description="Full description of what this rule checks.",
    )
    category: str = Field(
        ...,
        description=(
            "Domain category. Must be one of: "
            "financial, legal, operational, credit, regulatory, collateral, general."
        ),
    )
    operator: str = Field(
        ...,
        min_length=1,
        description=(
            "Comparison operator: '<=', '>=', '==', '!=', 'exists', 'in', or 'semantic'. "
            "Use 'semantic' when no numeric threshold is clearly stated."
        ),
    )
    threshold: str = Field(
        ...,
        min_length=1,
        description=(
            "Required limit or value as a string. "
            "Use 'semantic' when the rule requires qualitative judgement."
        ),
    )
    severity: str = Field(
        default="medium",
        description=(
            "Violation severity. Must be one of: critical, high, medium, low. "
            "Default to 'medium' when not explicitly stated."
        ),
    )
    applicability: Optional[str] = Field(
        default=None,
        description="Optional scope of the rule (deal type, region, product).",
    )
    reference: Optional[str] = Field(
        default=None,
        description="Verbatim section/clause reference from the source document.",
    )
    is_mandatory: bool = Field(
        default=True,
        description="Whether the rule is always applied.",
    )


class PolicyExtractionOutput(BaseModel):
    """Top-level structured response from the Policy Extraction LLM call.

    The LLM must return all clearly-stated policy rules in ``rules``, and
    place any vague or uncodifiable statements in ``ambiguous_statements``.
    It must NOT fabricate rules for ambiguous content.
    """

    rules: list[ExtractedPolicyRuleSchema] = Field(
        default_factory=list,
        description=(
            "List of clearly-stated, unambiguous policy rules extracted from the document. "
            "Each rule must be fully supported by the document text."
        ),
    )
    ambiguous_statements: list[str] = Field(
        default_factory=list,
        description=(
            "Verbatim or closely-paraphrased policy statements that could not be safely "
            "codified as rules (vague thresholds, conditional language, incomplete rules). "
            "Include the source text. Do NOT fabricate rules for these statements."
        ),
    )
