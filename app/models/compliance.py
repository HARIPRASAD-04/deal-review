"""Compliance result model.

A ComplianceResult records the outcome of evaluating a single PolicyRule
against the extracted deal terms.  Every decision must carry:

* the rule that was checked (``rule_id``)
* the terms that were evaluated (``term_ids``)
* the evidence that supports the decision (``evidence_ids``)
* a human-readable ``rationale``
* the agent that produced the result (``agent_name``)

The ``status`` field uses an enum so conditional routing logic in the
Orchestrator can make decisions without string comparisons.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ComplianceStatus(str, Enum):
    """Outcome of a compliance evaluation."""

    PASS = "PASS"                          # Rule satisfied.
    FAIL = "FAIL"                          # Rule violated — escalation required.
    NEEDS_HUMAN_REVIEW = "NEEDS_HUMAN_REVIEW"  # Ambiguous, human must decide.
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"  # Can't decide; missing data.
    SKIPPED = "SKIPPED"                    # Rule not applicable to this deal.


class ComplianceResult(BaseModel):
    """Outcome of evaluating one PolicyRule against extracted deal terms.

    Attributes:
        compliance_id:      Stable identifier in the format ``COMP-NNN``.
        rule_id:            The ``POLICY-NNN`` rule that was evaluated.
        status:             Outcome of the evaluation.
        rationale:          Human-readable explanation of the decision.
        term_ids:           ``TERM-NNN`` IDs of terms examined.
        evidence_ids:       ``EV-NNN`` IDs grounding the decision.
        confidence:         Agent's confidence in the decision ``[0.0, 1.0]``.
        agent_name:         Name of the agent that produced this result.
        ambiguity_detected: Whether the agent detected conflicting signals.
        notes:              Optional additional notes.
    """

    compliance_id: str = Field(
        ...,
        pattern=r"^COMP-\d{3,}$",
        description="Stable compliance result identifier, e.g. COMP-001.",
        examples=["COMP-001"],
    )
    rule_id: str = Field(
        ...,
        pattern=r"^POLICY-\d{3,}$",
        description="The policy rule that was evaluated.",
    )
    status: ComplianceStatus = Field(
        ...,
        description="Outcome of the compliance evaluation.",
    )
    rationale: str = Field(
        ...,
        min_length=1,
        description="Human-readable explanation of the compliance decision.",
    )
    term_ids: list[str] = Field(
        default_factory=list,
        description="TERM-NNN IDs of the deal terms examined.",
    )
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="EV-NNN IDs grounding this compliance decision.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Agent confidence in this decision [0.0, 1.0].",
    )
    agent_name: str = Field(
        ...,
        min_length=1,
        description="Name of the agent that produced this compliance result.",
    )
    ambiguity_detected: bool = Field(
        default=False,
        description="Whether conflicting signals were detected during evaluation.",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Optional additional context from the compliance agent.",
    )
