"""Compliance Review Agent — Module 4.

The Compliance Review Agent evaluates every ``PolicyRule`` in the pipeline
state against the structured ``DealTerm`` objects produced by the Term
Extraction Agent and produces ``ComplianceResult`` objects.

Design principles
-----------------
* **No LLM** — all supported operators are evaluated deterministically via
  Python comparison logic in ``evaluator.py``.
* **One result per rule** — if multiple terms share the same name, each is
  evaluated against the rule and the most restrictive outcome wins (FAIL >
  NEEDS_HUMAN_REVIEW > INSUFFICIENT_EVIDENCE > PASS).
* **Missing term → INSUFFICIENT_EVIDENCE** (not silently skipped).
* **Ambiguous term → NEEDS_HUMAN_REVIEW** (never arbitrarily choose one value).
* **Per-rule fault isolation** — an exception evaluating one rule is captured
  and recorded as NEEDS_HUMAN_REVIEW without aborting the other rules.
* **Full evidence traceability** — every result carries the ``term_ids`` and
  ``evidence_ids`` of the evaluated terms.

What this agent does NOT do
----------------------------
* It does not read the source PDF.
* It does not assess risk or produce a risk summary.
* It does not invent policy rules or modify thresholds.
* It does not select between conflicting (AMBIGUOUS) term values.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.agents.compliance.evaluator import evaluate_rule
from app.models.compliance import ComplianceResult, ComplianceStatus
from app.models.evidence import EvidenceSnippet
from app.models.policy import PolicyRule
from app.models.terms import DealTerm

logger = logging.getLogger(__name__)

# ── Status ordering for multi-term conflict resolution ────────────────────────
# When multiple terms match a rule, the most severe non-PASS status wins.
_STATUS_RANK: dict[ComplianceStatus, int] = {
    ComplianceStatus.FAIL: 4,
    ComplianceStatus.NEEDS_HUMAN_REVIEW: 3,
    ComplianceStatus.INSUFFICIENT_EVIDENCE: 2,
    ComplianceStatus.SKIPPED: 1,
    ComplianceStatus.PASS: 0,
}


class ComplianceReviewAgent:
    """Evaluates policy rules against extracted deal terms.

    Args:
        None — this agent is fully deterministic and requires no external
               dependencies beyond the data passed to ``review()``.

    Usage::

        agent = ComplianceReviewAgent()
        results = agent.review(
            policy_rules=state.policy_rules,
            extracted_terms=state.extracted_terms,
            evidence_registry=state.evidence_registry,
        )
    """

    AGENT_NAME: str = "compliance"

    # ── Public interface ───────────────────────────────────────────────────────

    def review(
        self,
        policy_rules: list[PolicyRule],
        extracted_terms: dict[str, DealTerm],
        evidence_registry: dict[str, EvidenceSnippet],
    ) -> list[ComplianceResult]:
        """Evaluate all policy rules against the extracted deal terms.

        Args:
            policy_rules:      List of ``PolicyRule`` objects to evaluate.
            extracted_terms:   Dict of ``TERM-NNN → DealTerm`` from
                               ``WorkflowState.extracted_terms``.
            evidence_registry: Dict of ``EV-NNN → EvidenceSnippet`` for
                               traceability (not directly evaluated here, but
                               used to validate evidence chains).

        Returns:
            An ordered list of ``ComplianceResult`` objects, one per policy
            rule.  Sequential ``COMP-NNN`` IDs are assigned starting at
            ``COMP-001``.
        """
        results: list[ComplianceResult] = []
        counter = 1

        # Build a name → [DealTerm] lookup (case-insensitive) once.
        terms_by_name = self._build_name_index(extracted_terms)

        for rule in policy_rules:
            comp_id = f"COMP-{counter:03d}"
            try:
                result = self._evaluate_rule(
                    comp_id=comp_id,
                    rule=rule,
                    terms_by_name=terms_by_name,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "[%s] Unexpected error evaluating rule %s: %s",
                    self.AGENT_NAME, rule.rule_id, exc, exc_info=True,
                )
                result = ComplianceResult(
                    compliance_id=comp_id,
                    rule_id=rule.rule_id,
                    status=ComplianceStatus.NEEDS_HUMAN_REVIEW,
                    rationale=(
                        f"Rule '{rule.rule_id}' ({rule.name}): "
                        f"An unexpected error occurred during evaluation: {exc}. "
                        f"Human review is required."
                    ),
                    confidence=0.0,
                    agent_name=self.AGENT_NAME,
                    notes=f"Exception: {exc}",
                )

            results.append(result)
            counter += 1
            logger.debug(
                "[%s] Rule %s evaluated → %s.",
                self.AGENT_NAME, rule.rule_id, result.status.value,
            )

        logger.info(
            "[%s] Review complete: %d rule(s) evaluated. "
            "PASS=%d, FAIL=%d, NEEDS_HUMAN_REVIEW=%d, INSUFFICIENT_EVIDENCE=%d.",
            self.AGENT_NAME,
            len(results),
            sum(1 for r in results if r.status == ComplianceStatus.PASS),
            sum(1 for r in results if r.status == ComplianceStatus.FAIL),
            sum(1 for r in results if r.status == ComplianceStatus.NEEDS_HUMAN_REVIEW),
            sum(1 for r in results if r.status == ComplianceStatus.INSUFFICIENT_EVIDENCE),
        )
        return results

    def review_rule(
        self,
        rule: PolicyRule,
        extracted_terms: dict[str, DealTerm],
        evidence_registry: dict[str, EvidenceSnippet],
        comp_id: str,
    ) -> ComplianceResult:
        """Evaluate a single policy rule against extracted deal terms.

        Args:
            rule: The ``PolicyRule`` to evaluate.
            extracted_terms: Dict of ``TERM-NNN → DealTerm``.
            evidence_registry: Dict of ``EV-NNN → EvidenceSnippet``.
            comp_id: Canonical ``COMP-NNN`` identifier for the result.

        Returns:
            A ``ComplianceResult`` for the single specified rule.
        """
        terms_by_name = self._build_name_index(extracted_terms)
        return self._evaluate_rule(
            comp_id=comp_id,
            rule=rule,
            terms_by_name=terms_by_name,
        )

    def generate_clarification_handoffs(
        self,
        results: list[ComplianceResult],
        policy_rules: list[PolicyRule],
        evidence_registry: Optional[dict[str, EvidenceSnippet]] = None,
    ) -> list[Handoff]:
        """Generate targeted handoffs for INSUFFICIENT_EVIDENCE results.

        If the evidence registry contains snippets with keywords related to the
        missing term, generates a CLARIFICATION_REQUIRED handoff to term_extraction.
        If the evidence registry contains no trace of the term (or is empty),
        generates an ESCALATION_REQUIRED handoff directly to human review.

        Args:
            results: List of evaluated ``ComplianceResult`` objects.
            policy_rules: List of ``PolicyRule`` objects to resolve rule names.
            evidence_registry: Optional dict of ``EV-NNN → EvidenceSnippet``.

        Returns:
            List of ``Handoff`` objects.
        """
        from app.models.handoff import Handoff, HandoffPriority, HandoffType

        rules_by_id = {r.rule_id: r for r in policy_rules}
        handoffs: list[Handoff] = []

        for result in results:
            if result.status == ComplianceStatus.INSUFFICIENT_EVIDENCE:
                rule = rules_by_id.get(result.rule_id)
                term_name = rule.name if rule else result.rule_id

                has_evidence = (
                    self._evidence_has_relevant_snippets(term_name, rule, evidence_registry)
                    if evidence_registry is not None
                    else True
                )

                if has_evidence:
                    reason = (
                        f"Compliance review for rule '{result.rule_id}' failed due to missing term '{term_name}'. "
                        f"Matching keywords exist in evidence registry; targeted clarification re-extraction requested."
                    )
                    handoff = Handoff(
                        source_agent=self.AGENT_NAME,
                        target_agent="term_extraction",
                        handoff_type=HandoffType.CLARIFICATION_REQUIRED,
                        reason=reason,
                        rule_ids=[result.rule_id],
                        requested_term_names=[term_name],
                        priority=HandoffPriority.HIGH,
                    )
                else:
                    reason = (
                        f"Compliance review for rule '{result.rule_id}' failed due to missing term '{term_name}'. "
                        f"No supporting keywords found in evidence registry; escalated directly to human review."
                    )
                    handoff = Handoff(
                        source_agent=self.AGENT_NAME,
                        target_agent="human",
                        handoff_type=HandoffType.ESCALATION_REQUIRED,
                        reason=reason,
                        rule_ids=[result.rule_id],
                        requested_term_names=[term_name],
                        priority=HandoffPriority.HIGH,
                    )

                handoffs.append(handoff)

        logger.info(
            "[%s] Generated %d handoff(s) for INSUFFICIENT_EVIDENCE rules.",
            self.AGENT_NAME,
            len(handoffs),
        )
        return handoffs

    @staticmethod
    def _evidence_has_relevant_snippets(
        term_name: str,
        rule: Optional[PolicyRule],
        evidence_registry: dict[str, EvidenceSnippet],
    ) -> bool:
        """Check if any snippet in the evidence registry contains keywords related to term_name."""
        if not evidence_registry:
            return False

        import re

        search_words = set(re.findall(r"\w+", term_name.lower()))
        if rule and rule.description:
            desc_words = set(re.findall(r"\w+", rule.description.lower()))
            stop_words = {"must", "than", "equal", "shall", "with", "from", "that", "this", "have", "more", "less", "rule"}
            search_words.update(w for w in desc_words if len(w) > 3 and w not in stop_words)

        for snippet in evidence_registry.values():
            snippet_text_lower = snippet.text.lower()
            if any(w in snippet_text_lower for w in search_words):
                return True

        return False

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _build_name_index(
        extracted_terms: dict[str, DealTerm],
    ) -> dict[str, list[DealTerm]]:
        """Build a case-insensitive name → [DealTerm] index."""
        index: dict[str, list[DealTerm]] = {}
        for term in extracted_terms.values():
            key = term.name.strip().lower()
            index.setdefault(key, []).append(term)
        return index

    def _evaluate_rule(
        self,
        comp_id: str,
        rule: PolicyRule,
        terms_by_name: dict[str, list[DealTerm]],
    ) -> ComplianceResult:
        """Evaluate one rule and return a single ComplianceResult."""
        rule_term_key = rule.name.strip().lower()
        matching_terms = terms_by_name.get(rule_term_key, [])

        # ── No matching term extracted ────────────────────────────────────────
        if not matching_terms:
            rationale = (
                f"Rule '{rule.rule_id}' ({rule.name}): "
                f"No term named '{rule.name}' was extracted from the deal "
                f"document. Compliance cannot be established without the "
                f"required value."
            )
            logger.info(
                "[%s] Rule %s: no term named '%s' found.",
                self.AGENT_NAME, rule.rule_id, rule.name,
            )
            return ComplianceResult(
                compliance_id=comp_id,
                rule_id=rule.rule_id,
                status=ComplianceStatus.INSUFFICIENT_EVIDENCE,
                rationale=rationale,
                term_ids=[],
                evidence_ids=[],
                confidence=0.0,
                agent_name=self.AGENT_NAME,
            )

        # ── One or more matching terms — evaluate each, pick worst outcome ────
        evaluated: list[tuple[ComplianceStatus, str, bool, DealTerm]] = []
        for term in matching_terms:
            status, rat, ambiguity = evaluate_rule(rule, term)
            evaluated.append((status, rat, ambiguity, term))

        # Pick the most restrictive result by status rank.
        best = max(evaluated, key=lambda x: _STATUS_RANK[x[0]])
        final_status, final_rationale, ambiguity_detected, primary_term = best

        # If there are multiple results and they disagree, note it.
        if len(evaluated) > 1:
            statuses = {e[0] for e in evaluated}
            if len(statuses) > 1:
                ambiguity_detected = True
                final_rationale += (
                    f" [Note: {len(evaluated)} terms matched rule '{rule.name}'. "
                    f"Statuses were: {sorted(s.value for s in statuses)}. "
                    f"Most restrictive outcome selected.]"
                )

        # Collect all term_ids and evidence_ids from matching terms.
        all_term_ids = [t.term_id for t in matching_terms]
        all_evidence_ids: list[str] = []
        for term in matching_terms:
            for ev_id in term.evidence_ids:
                if ev_id not in all_evidence_ids:
                    all_evidence_ids.append(ev_id)

        # Confidence is 1.0 for deterministic outcomes, 0.5 for human review.
        confidence = self._confidence_for(final_status)

        return ComplianceResult(
            compliance_id=comp_id,
            rule_id=rule.rule_id,
            status=final_status,
            rationale=final_rationale,
            term_ids=all_term_ids,
            evidence_ids=all_evidence_ids,
            confidence=confidence,
            agent_name=self.AGENT_NAME,
            ambiguity_detected=ambiguity_detected,
        )

    @staticmethod
    def _confidence_for(status: ComplianceStatus) -> float:
        """Return an appropriate confidence score for the given status."""
        return {
            ComplianceStatus.PASS: 1.0,
            ComplianceStatus.FAIL: 1.0,
            ComplianceStatus.INSUFFICIENT_EVIDENCE: 0.0,
            ComplianceStatus.NEEDS_HUMAN_REVIEW: 0.5,
            ComplianceStatus.SKIPPED: 1.0,
        }.get(status, 0.5)
