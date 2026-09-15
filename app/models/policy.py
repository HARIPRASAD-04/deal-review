"""Policy rule model.

A PolicyRule represents a single rule from the company's compliance / policy
rule set.  Rules are loaded at pipeline startup and remain immutable for the
duration of a run.

Design decisions:
* ``operator`` and ``threshold`` are kept as strings so the model can represent
  both deterministic rules (``"<= INR 8 crore"``) and future semantic rules
  (``"collateral must be tangible fixed asset"``).
* ``severity`` is an enum so downstream agents can consistently prioritise
  violations.
* ``applicability`` allows rules to be scoped to deal types or regions, making
  the rule set extensible without schema changes.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class RuleSeverity(str, Enum):
    """How serious a rule violation is."""

    CRITICAL = "critical"    # Deal must be blocked.
    HIGH = "high"            # Requires senior approval / immediate escalation.
    MEDIUM = "medium"        # Requires review before proceeding.
    LOW = "low"              # Informational — note in report.


class RuleCategory(str, Enum):
    """Domain category of the policy rule."""

    FINANCIAL = "financial"
    LEGAL = "legal"
    OPERATIONAL = "operational"
    CREDIT = "credit"
    REGULATORY = "regulatory"
    COLLATERAL = "collateral"
    GENERAL = "general"


class PolicyRule(BaseModel):
    """A single compliance / policy rule.

    Attributes:
        rule_id:       Stable identifier in the format ``POLICY-NNN``.
        name:          Short human-readable rule name.
        description:   Full description of what the rule checks.
        category:      Domain category of the rule.
        operator:      Comparison operator (e.g. ``"<="``).  Use ``"semantic"``
                       for rules that require LLM judgement.
        threshold:     The limit or required value as a string.
        applicability: Optional scope (deal type, region, product).
        reference:     Source of truth for this rule (regulation, policy doc).
        severity:      How serious a violation is.
        is_mandatory:  Whether this rule is always applied (True) or
                       context-dependent (False).
    """

    rule_id: str = Field(
        ...,
        pattern=r"^POLICY-\d{3,}$",
        description="Stable policy rule identifier, e.g. POLICY-003.",
        examples=["POLICY-003"],
    )
    name: str = Field(
        ...,
        min_length=1,
        description="Short human-readable rule name.",
    )
    description: str = Field(
        ...,
        min_length=1,
        description="Full description of what this rule checks.",
    )
    category: RuleCategory = Field(
        ...,
        description="Domain category of the rule.",
    )
    operator: str = Field(
        ...,
        min_length=1,
        description=(
            "Comparison operator: '<=', '>=', '==', '!=', 'exists', 'semantic', …"
        ),
    )
    threshold: str = Field(
        ...,
        min_length=1,
        description="Limit or required value as a string for portability.",
    )
    applicability: Optional[str] = Field(
        default=None,
        description="Scope of the rule, e.g. 'term_loan' or 'India'.",
    )
    reference: Optional[str] = Field(
        default=None,
        description="Source of truth for this rule (regulation, policy doc).",
    )
    severity: RuleSeverity = Field(
        ...,
        description="Severity of a rule violation.",
    )
    is_mandatory: bool = Field(
        default=True,
        description="Whether the rule is always applied.",
    )

    model_config = {"frozen": True}   # Rules are immutable during a run.
