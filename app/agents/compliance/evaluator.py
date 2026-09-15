"""Deterministic policy rule evaluation logic for the Compliance Review Agent.

This module contains *pure functions only* — no side effects, no I/O.
The goal is maximum testability and predictability.

Value normalisation strategy
-----------------------------
Both the ``threshold`` on the ``PolicyRule`` and the ``value`` /
``normalized_value`` on the ``DealTerm`` are stored as free-form strings.
Before comparing, we run ``parse_numeric()`` to extract a float where possible.

``parse_numeric`` handles the most common formats seen in financial deal sheets:
* Bare numbers:        ``"1.25"``
* Percentage strings:  ``"8.5%"``  → ``0.085`` (divided by 100)
* Ratio strings:       ``"1.25x"`` → ``1.25``
* Currency strings:    ``"INR 50,000,000"`` → ``50_000_000.0``
* Decimal threshold:   ``"0.085"`` → ``0.085``

Note on percentages
-------------------
``"8.5%"`` becomes ``0.085`` and ``"0.085"`` also becomes ``0.085``, so they
compare correctly.  The threshold in the PolicyRule should use the same format
as the normalized_value if present (e.g. ``"0.09"`` for 9 %).  If the policy
author writes ``"9%"`` as the threshold, that too becomes ``0.09``.

Supported operators
--------------------
``<=``     numeric less-than-or-equal
``>=``     numeric greater-than-or-equal
``==``     string equality (case-insensitive, stripped)
``!=``     string inequality
``exists`` term must be present and have a non-empty value
``in``     term value must appear in comma-separated threshold list
``not_in`` term value must NOT appear in comma-separated threshold list
``semantic`` always NEEDS_HUMAN_REVIEW — requires LLM / human judgement
*(other)*   NEEDS_HUMAN_REVIEW with explanation
"""

from __future__ import annotations

import re
from typing import Optional

from app.models.compliance import ComplianceStatus
from app.models.policy import PolicyRule
from app.models.terms import DealTerm, TermStatus

# ── Constants ─────────────────────────────────────────────────────────────────

AGENT_NAME = "compliance"

# Operators that require numeric parsing.
_NUMERIC_OPERATORS: frozenset[str] = frozenset({"<=", ">=", "<", ">"})

# Regex patterns for numeric extraction.
_RE_PERCENT = re.compile(r"(-?\d[\d,]*\.?\d*)\s*%")
_RE_RATIO   = re.compile(r"(-?\d[\d,]*\.?\d*)\s*[xX]")
_RE_PLAIN   = re.compile(r"(-?\d[\d,]*\.?\d*)")


# ── Value Parsing ─────────────────────────────────────────────────────────────

def parse_numeric(value: str) -> Optional[float]:
    """Extract a float from a free-form financial value string.

    Returns ``None`` if no numeric value can be extracted.

    Rules (applied in priority order):
    1. If the string contains ``%``, parse the leading number and divide by 100.
    2. If the string contains a trailing ``x`` or ``X`` (ratio), return the number.
    3. Strip currency codes / commas and parse the first numeric substring.

    Examples::

        parse_numeric("8.5%")           -> 0.085
        parse_numeric("0.085")          -> 0.085
        parse_numeric("1.25x")          -> 1.25
        parse_numeric("INR 50,000,000") -> 50000000.0
        parse_numeric("60 months")      -> 60.0
        parse_numeric("India")          -> None
    """
    if not value:
        return None

    cleaned = value.strip()

    # 1. Percentage — divide by 100 to normalise to decimal.
    m = _RE_PERCENT.search(cleaned)
    if m:
        try:
            return float(m.group(1).replace(",", "")) / 100.0
        except ValueError:
            pass

    # 2. Ratio (trailing x / X).
    m = _RE_RATIO.search(cleaned)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass

    # 3. Plain numeric — strip commas, currency codes, etc.
    m = _RE_PLAIN.search(cleaned.replace(",", ""))
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass

    return None


# ── Operator Evaluation ───────────────────────────────────────────────────────

def _get_comparison_value(term: DealTerm) -> str:
    """Return the best string to compare against the policy threshold.

    Prefers ``normalized_value`` when present, falls back to ``value``.
    """
    if term.normalized_value and term.normalized_value.strip():
        return term.normalized_value.strip()
    return term.value.strip()


def evaluate_rule(
    rule: PolicyRule,
    term: DealTerm,
) -> tuple[ComplianceStatus, str, bool]:
    """Evaluate a single ``PolicyRule`` against a single ``DealTerm``.

    Args:
        rule: The policy rule to evaluate.
        term: The deal term to check.

    Returns:
        A 3-tuple ``(status, rationale, ambiguity_detected)`` where:
        * ``status``             — ``ComplianceStatus`` enum value.
        * ``rationale``          — Human-readable explanation.
        * ``ambiguity_detected`` — True when conflicting signals were detected.
    """
    # ── Ambiguous term — never guess ────────────────────────────────────────
    if term.status == TermStatus.AMBIGUOUS:
        rationale = (
            f"Rule '{rule.rule_id}' ({rule.name}): "
            f"Term '{term.term_id}' ({term.name}) is marked AMBIGUOUS — "
            f"conflicting values were detected during extraction. "
            f"A human must review which value is authoritative before "
            f"compliance can be determined."
        )
        return ComplianceStatus.NEEDS_HUMAN_REVIEW, rationale, True

    op = rule.operator.strip().lower()
    threshold_raw = rule.threshold.strip()
    term_value = _get_comparison_value(term)
    original_value = term.value.strip()

    # ── exists ───────────────────────────────────────────────────────────────
    if op == "exists":
        if term_value:
            rationale = (
                f"Rule '{rule.rule_id}' ({rule.name}): "
                f"Term '{term.name}' is present with value '{original_value}'. "
                f"The 'exists' requirement is satisfied."
            )
            return ComplianceStatus.PASS, rationale, False
        else:
            rationale = (
                f"Rule '{rule.rule_id}' ({rule.name}): "
                f"Term '{term.name}' exists but has an empty value. "
                f"The 'exists' requirement is NOT satisfied."
            )
            return ComplianceStatus.FAIL, rationale, False

    # ── == (string equality, case-insensitive) ───────────────────────────────
    if op == "==":
        if term_value.lower() == threshold_raw.lower():
            rationale = (
                f"Rule '{rule.rule_id}' ({rule.name}): "
                f"Term '{term.name}' value '{original_value}' matches "
                f"required value '{threshold_raw}'."
            )
            return ComplianceStatus.PASS, rationale, False
        else:
            rationale = (
                f"Rule '{rule.rule_id}' ({rule.name}): "
                f"Term '{term.name}' value '{original_value}' does not match "
                f"required value '{threshold_raw}'."
            )
            return ComplianceStatus.FAIL, rationale, False

    # ── != (string inequality) ───────────────────────────────────────────────
    if op == "!=":
        if term_value.lower() != threshold_raw.lower():
            rationale = (
                f"Rule '{rule.rule_id}' ({rule.name}): "
                f"Term '{term.name}' value '{original_value}' is not "
                f"'{threshold_raw}', as required."
            )
            return ComplianceStatus.PASS, rationale, False
        else:
            rationale = (
                f"Rule '{rule.rule_id}' ({rule.name}): "
                f"Term '{term.name}' value '{original_value}' is "
                f"'{threshold_raw}', which is prohibited."
            )
            return ComplianceStatus.FAIL, rationale, False

    # ── in (allowed values list) ─────────────────────────────────────────────
    if op == "in":
        allowed = [v.strip().lower() for v in threshold_raw.split(",")]
        if term_value.lower() in allowed:
            rationale = (
                f"Rule '{rule.rule_id}' ({rule.name}): "
                f"Term '{term.name}' value '{original_value}' is in the "
                f"allowed list [{threshold_raw}]."
            )
            return ComplianceStatus.PASS, rationale, False
        else:
            rationale = (
                f"Rule '{rule.rule_id}' ({rule.name}): "
                f"Term '{term.name}' value '{original_value}' is NOT in the "
                f"allowed list [{threshold_raw}]."
            )
            return ComplianceStatus.FAIL, rationale, False

    # ── not_in (prohibited values list) ─────────────────────────────────────
    if op == "not_in":
        prohibited = [v.strip().lower() for v in threshold_raw.split(",")]
        if term_value.lower() not in prohibited:
            rationale = (
                f"Rule '{rule.rule_id}' ({rule.name}): "
                f"Term '{term.name}' value '{original_value}' is not in the "
                f"prohibited list [{threshold_raw}], as required."
            )
            return ComplianceStatus.PASS, rationale, False
        else:
            rationale = (
                f"Rule '{rule.rule_id}' ({rule.name}): "
                f"Term '{term.name}' value '{original_value}' appears in the "
                f"prohibited list [{threshold_raw}]."
            )
            return ComplianceStatus.FAIL, rationale, False

    # ── Numeric operators: <=, >=, <, > ─────────────────────────────────────
    if op in _NUMERIC_OPERATORS:
        term_num = parse_numeric(term_value) if term_value else None
        threshold_num = parse_numeric(threshold_raw)

        if term_num is None or threshold_num is None:
            rationale = (
                f"Rule '{rule.rule_id}' ({rule.name}): "
                f"Cannot perform numeric comparison '{op}' between "
                f"term value '{original_value}' and threshold '{threshold_raw}'. "
                f"One or both values could not be parsed as a number. "
                f"Human review is required."
            )
            return ComplianceStatus.NEEDS_HUMAN_REVIEW, rationale, False

        # Perform the comparison.
        if op == "<=":
            passed = term_num <= threshold_num
        elif op == ">=":
            passed = term_num >= threshold_num
        elif op == "<":
            passed = term_num < threshold_num
        else:  # ">"
            passed = term_num > threshold_num

        # Format a human-friendly representation for the rationale.
        fmt = _format_numeric_pair(term_num, threshold_num)

        if passed:
            rationale = (
                f"Rule '{rule.rule_id}' ({rule.name}): "
                f"Extracted {term.name} is {fmt['term']} "
                f"({op} threshold of {fmt['threshold']}). "
                f"PASS."
            )
        else:
            rationale = (
                f"Rule '{rule.rule_id}' ({rule.name}): "
                f"Extracted {term.name} is {fmt['term']} "
                f"(threshold requires {op} {fmt['threshold']}). "
                f"FAIL."
            )
        return (
            ComplianceStatus.PASS if passed else ComplianceStatus.FAIL,
            rationale,
            False,
        )

    # ── semantic — requires LLM / human ─────────────────────────────────────
    if op == "semantic":
        rationale = (
            f"Rule '{rule.rule_id}' ({rule.name}): "
            f"Operator 'semantic' requires human or LLM judgement. "
            f"Term '{term.name}' value is '{original_value}'; "
            f"expected: '{threshold_raw}'. "
            f"Automatic evaluation is not supported for semantic rules."
        )
        return ComplianceStatus.NEEDS_HUMAN_REVIEW, rationale, False

    # ── Unknown operator ─────────────────────────────────────────────────────
    rationale = (
        f"Rule '{rule.rule_id}' ({rule.name}): "
        f"Unknown operator '{rule.operator}'. "
        f"Cannot evaluate deterministically. Human review required."
    )
    return ComplianceStatus.NEEDS_HUMAN_REVIEW, rationale, False


# ── Internal Helpers ──────────────────────────────────────────────────────────

def _format_numeric_pair(
    term_num: float, threshold_num: float
) -> dict[str, str]:
    """Format numeric values consistently for rationale strings."""

    def fmt(n: float) -> str:
        if n == int(n) and abs(n) >= 1000:
            return f"{int(n):,}"
        if abs(n) < 1 and n != 0:
            # Likely a decimal representation of a percentage.
            return f"{n:.4f} ({n * 100:.2f}%)"
        return f"{n:g}"

    return {"term": fmt(term_num), "threshold": fmt(threshold_num)}
