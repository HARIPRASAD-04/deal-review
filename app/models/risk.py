"""Risk finding model.

A RiskFinding represents a single identified risk produced by the Risk &
Summary Agent.  Risks are synthesised from:

* compliance failures (a FAIL maps to at least one risk)
* ambiguous or missing information
* deal term anomalies

Each risk traces back to the underlying compliance decisions, deal terms, and
evidence, maintaining the full traceability chain:

    Document → Evidence → Term → Compliance → Risk → Report
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class RiskCategory(str, Enum):
    """Domain category of the identified risk."""

    FINANCIAL = "financial"
    LEGAL = "legal"
    OPERATIONAL = "operational"
    COMPLIANCE = "compliance"
    CREDIT = "credit"
    REPUTATIONAL = "reputational"


class RiskSeverity(str, Enum):
    """Severity level of the identified risk."""

    CRITICAL = "critical"    # Potential deal-breaker.
    HIGH = "high"            # Requires immediate senior attention.
    MEDIUM = "medium"        # Requires review before approval.
    LOW = "low"              # Document and monitor.


class RiskLikelihood(str, Enum):
    """Estimated likelihood of the risk materialising."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class RiskFinding(BaseModel):
    """A single identified risk within the deal review pipeline.

    Attributes:
        risk_id:               Stable identifier in the format ``RISK-NNN``.
        category:              Domain classification of the risk.
        title:                 Short title summarising the risk.
        description:           Full description of the risk.
        severity:              Severity level.
        likelihood:            Estimated likelihood of materialisation.
        evidence_ids:          ``EV-NNN`` IDs grounding this risk.
        related_term_ids:      ``TERM-NNN`` IDs of relevant deal terms.
        related_compliance_ids:``COMP-NNN`` IDs of compliance decisions that
                               triggered or informed this risk.
        recommended_action:    Suggested mitigation or next step.
        requires_human_review: Whether a human must review this risk before
                               the deal can proceed.
        notes:                 Optional additional context.
    """

    risk_id: str = Field(
        ...,
        pattern=r"^RISK-\d{3,}$",
        description="Stable risk identifier, e.g. RISK-001.",
        examples=["RISK-001"],
    )
    category: RiskCategory = Field(
        ...,
        description="Domain classification of the risk.",
    )
    title: str = Field(
        ...,
        min_length=1,
        description="Short descriptive title for the risk.",
    )
    description: str = Field(
        ...,
        min_length=1,
        description="Full description of the risk and its potential impact.",
    )
    severity: RiskSeverity = Field(
        ...,
        description="Severity level of the risk.",
    )
    likelihood: RiskLikelihood = Field(
        default=RiskLikelihood.UNKNOWN,
        description="Estimated likelihood of this risk materialising.",
    )
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="EV-NNN IDs grounding this risk.",
    )
    related_term_ids: list[str] = Field(
        default_factory=list,
        description="TERM-NNN IDs of deal terms related to this risk.",
    )
    related_compliance_ids: list[str] = Field(
        default_factory=list,
        description="COMP-NNN IDs of compliance decisions that informed this risk.",
    )
    recommended_action: str = Field(
        ...,
        min_length=1,
        description="Suggested mitigation or escalation action.",
    )
    requires_human_review: bool = Field(
        default=False,
        description="True if a human must review before the deal can proceed.",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Optional additional notes from the risk agent.",
    )
