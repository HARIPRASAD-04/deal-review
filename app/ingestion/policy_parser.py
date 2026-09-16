"""Policy PDF parser — Module 8.

Responsibility: Convert a policy PDF into a list of ``PolicyRule`` objects
by (a) loading the PDF with the existing ``load_pdf()`` infrastructure and
(b) calling the LLM with a structured ``PolicyExtractionOutput`` prompt.

Design decisions
----------------
* **No rule fabrication.** The LLM prompt explicitly instructs the model to
  place ambiguous or incomplete statements in ``ambiguous_statements`` rather
  than inventing rules.  Ambiguous statements are surfaced to callers via
  ``PolicyParseResult.ambiguous_statements``.
* **Deterministic IDs.** Where a document contains an explicit rule ID
  (e.g. "POLICY-003", "Rule 3"), it is preserved verbatim.  Where no ID is
  present, the parser assigns ``POLICY-EXT-001``, ``POLICY-EXT-002``, … in
  order of appearance — always the same for the same document.
* **No silent fallbacks.** On any failure (missing file, corrupt PDF, LLM
  unavailable), the function returns ``PolicyParseResult(success=False, ...)``.
  It never returns bundled/default rules to compensate.
* **Graceful validation.** Each extracted rule is validated through
  ``PolicyRule(...)``; rules that fail validation are logged and skipped.
  A rule is not a blocker for the rest of the extraction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from app.ingestion.pdf_loader import load_pdf
from app.llm.client import LLMClient
from app.llm.schemas import PolicyExtractionOutput
from app.models.policy import PolicyRule, RuleCategory, RuleSeverity

logger = logging.getLogger(__name__)

# Prefix for auto-generated rule IDs when none is found in the document.
_AUTO_ID_PREFIX = "POLICY-EXT"

# Maps raw LLM category strings to RuleCategory enum values.
_CATEGORY_MAP: dict[str, RuleCategory] = {
    "financial": RuleCategory.FINANCIAL,
    "legal": RuleCategory.LEGAL,
    "operational": RuleCategory.OPERATIONAL,
    "credit": RuleCategory.CREDIT,
    "regulatory": RuleCategory.REGULATORY,
    "collateral": RuleCategory.COLLATERAL,
    "general": RuleCategory.GENERAL,
}

# Maps raw LLM severity strings to RuleSeverity enum values.
_SEVERITY_MAP: dict[str, RuleSeverity] = {
    "critical": RuleSeverity.CRITICAL,
    "high": RuleSeverity.HIGH,
    "medium": RuleSeverity.MEDIUM,
    "low": RuleSeverity.LOW,
}

_POLICY_EXTRACTION_SYSTEM_PROMPT = """\
You are a precise policy extraction assistant.  Your task is to read a
financial institution's policy document and extract each compliance rule as a
structured record.

RULES FOR EXTRACTION:
1. Only extract rules that are CLEARLY AND UNAMBIGUOUSLY stated in the document.
2. If a statement is vague, conditional, or lacks a clear threshold, add it to
   `ambiguous_statements` instead of fabricating a rule.
3. Preserve explicit rule IDs (e.g. "Rule 3", "POLICY-003") verbatim in
   `rule_id`.  Set `rule_id` to null if no explicit ID is present.
4. Use `operator="semantic"` and `threshold="see document"` when a rule
   requires qualitative judgement rather than a numeric comparison.
5. For numeric thresholds, use the decimal/numeric form (e.g. "0.09" for 9%,
   "80000000" for INR 8 crore).
6. Do NOT invent rule semantics.  Ground every field in document text.
7. Severity defaults to "medium" if not stated.
8. Category must be exactly one of: financial, legal, operational, credit,
   regulatory, collateral, general.
"""


@dataclass
class PolicyParseResult:
    """Result of a policy PDF parsing attempt.

    Attributes:
        success:              True when at least one valid rule was extracted
                              and no fatal error occurred.
        rules:                Valid, validated ``PolicyRule`` objects extracted
                              from the document.
        ambiguous_statements: Statements the LLM identified but could not
                              safely codify as rules.
        error:                Human-readable error description; ``None`` on
                              success.
    """

    success: bool
    rules: list[PolicyRule] = field(default_factory=list)
    ambiguous_statements: list[str] = field(default_factory=list)
    error: Optional[str] = None


def parse_policy_pdf(
    pdf_path: str | Path,
    llm_client: Optional[LLMClient],
) -> PolicyParseResult:
    """Parse a policy PDF into a list of ``PolicyRule`` objects.

    Args:
        pdf_path:   Path to the policy PDF file.
        llm_client: A configured ``LLMClient`` instance.  If ``None``,
                    returns ``PolicyParseResult(success=False, error="...")``.
                    No bundled/default rules are substituted.

    Returns:
        A ``PolicyParseResult``.  Check ``result.success`` before using
        ``result.rules``.  Inspect ``result.ambiguous_statements`` for
        statements that were found but could not be codified.
    """
    pdf_path = Path(pdf_path)

    # ── Guard: no LLM ─────────────────────────────────────────────────────────
    if llm_client is None:
        msg = (
            "No LLM client configured. Policy PDF parsing requires an LLM client. "
            "Configure one via GOOGLE_API_KEY / MODEL_NAME in your .env file."
        )
        logger.warning(msg)
        return PolicyParseResult(success=False, error=msg)

    # ── Load PDF ───────────────────────────────────────────────────────────────
    result = load_pdf(pdf_path, document_id="policy-doc")
    if not result.success:
        msg = f"Policy PDF loading failed: {result.error}"
        logger.warning(msg)
        return PolicyParseResult(success=False, error=msg)

    # ── Build prompt context from page text ───────────────────────────────────
    page_texts = [p.raw_text for p in result.pages if p.raw_text.strip()]
    if not page_texts:
        msg = "Policy PDF contains no extractable text (may be image-only or empty)."
        logger.warning(msg)
        return PolicyParseResult(success=False, error=msg)

    user_prompt = (
        "Extract all compliance rules from the following policy document.\n\n"
        "=== POLICY DOCUMENT ===\n"
        + "\n\n".join(page_texts)
        + "\n=== END OF DOCUMENT ==="
    )

    # ── Call LLM ───────────────────────────────────────────────────────────────
    try:
        llm_output: PolicyExtractionOutput = llm_client.get_structured_completion(
            system_prompt=_POLICY_EXTRACTION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_schema=PolicyExtractionOutput,
        )
    except Exception as exc:  # noqa: BLE001
        msg = f"LLM call failed during policy extraction: {exc}"
        logger.error(msg, exc_info=True)
        return PolicyParseResult(success=False, error=msg)

    # ── Validate and convert extracted rules ──────────────────────────────────
    valid_rules: list[PolicyRule] = []
    auto_id_counter = 1
    seen_explicit_ids: set[str] = set()

    for raw_rule in llm_output.rules:
        # Resolve rule_id: preserve explicit POLICY-NNN or generate valid POLICY-9NN IDs.
        import re
        if raw_rule.rule_id and raw_rule.rule_id.strip():
            raw_id = raw_rule.rule_id.strip().upper().replace(" ", "-")
            if re.match(r"^POLICY-\d{3,}$", raw_id):
                rule_id = raw_id
            else:
                digits = "".join(c for c in raw_id if c.isdigit())
                if digits:
                    rule_id = f"POLICY-{int(digits):03d}"
                else:
                    rule_id = f"POLICY-{auto_id_counter + 900:03d}"
                    auto_id_counter += 1
            if rule_id in seen_explicit_ids:
                rule_id = f"POLICY-{auto_id_counter + 900:03d}"
                auto_id_counter += 1
            seen_explicit_ids.add(rule_id)
        else:
            rule_id = f"POLICY-{auto_id_counter + 900:03d}"
            auto_id_counter += 1

        # Map category.
        category = _CATEGORY_MAP.get(raw_rule.category.lower().strip())
        if category is None:
            logger.warning(
                "Policy rule '%s' has unrecognised category '%s'. Skipping.",
                raw_rule.name,
                raw_rule.category,
            )
            continue

        # Map severity.
        severity = _SEVERITY_MAP.get(raw_rule.severity.lower().strip())
        if severity is None:
            logger.warning(
                "Policy rule '%s' has unrecognised severity '%s'. Defaulting to MEDIUM.",
                raw_rule.name,
                raw_rule.severity,
            )
            severity = RuleSeverity.MEDIUM

        try:
            policy_rule = PolicyRule(
                rule_id=rule_id,
                name=raw_rule.name,
                description=raw_rule.description,
                category=category,
                operator=raw_rule.operator,
                threshold=raw_rule.threshold,
                applicability=raw_rule.applicability,
                reference=raw_rule.reference,
                severity=severity,
                is_mandatory=raw_rule.is_mandatory,
            )
            valid_rules.append(policy_rule)
            logger.debug("Accepted policy rule: %s (%s)", rule_id, raw_rule.name)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Policy rule '%s' failed validation and was skipped: %s",
                raw_rule.name,
                exc,
            )

    logger.info(
        "Policy PDF parsed: %d valid rule(s) accepted, %d raw rule(s) from LLM, "
        "%d ambiguous statement(s).",
        len(valid_rules),
        len(llm_output.rules),
        len(llm_output.ambiguous_statements),
    )

    if not valid_rules and not llm_output.ambiguous_statements:
        # LLM returned nothing — treat as a failure.
        msg = "Policy PDF parsed but no rules or ambiguous statements were returned by the LLM."
        logger.warning(msg)
        return PolicyParseResult(success=False, error=msg)

    if not valid_rules:
        # Only ambiguous statements found — still a failure since we have no rules to enforce.
        msg = (
            f"Policy PDF produced no valid rules. "
            f"{len(llm_output.ambiguous_statements)} ambiguous statement(s) require review."
        )
        logger.warning(msg)
        return PolicyParseResult(
            success=False,
            rules=[],
            ambiguous_statements=list(llm_output.ambiguous_statements),
            error=msg,
        )

    return PolicyParseResult(
        success=True,
        rules=valid_rules,
        ambiguous_statements=list(llm_output.ambiguous_statements),
        error=None,
    )
