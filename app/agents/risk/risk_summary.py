"""Risk & Summary Agent — Module 6.

The Risk & Summary Agent is the fourth and final specialized agent in the
pipeline.  It consumes the outputs of Modules 2–5 and synthesises them into:

* A prioritised risk register (``list[RiskFinding]``)
* A missing / ambiguous information list (``list[str]``)
* Recommended follow-ups (``list[str]``)
* An executive summary (``str``)

Design principles
-----------------
* **No LLM for risk derivation** — all risk findings are derived deterministically
  from existing structured state (compliance results, extracted terms, escalations,
  evidence coverage).  This makes findings fully reproducible and traceable.
* **LLM for narrative summary only** — and with a deterministic fallback so the
  agent works without any LLM client configured.
* **Zero hallucination** — every ``evidence_ids``, ``related_term_ids``, and
  ``related_compliance_ids`` is taken from existing state objects and validated
  before assignment.  No IDs are inferred or fabricated.
* **Four risk sources** — compliance failures, term anomalies, escalations, and
  evidence coverage gaps — so the register reflects the full deal review picture,
  not just rule violations.
* **Does not touch compliance logic** — the agent reads ``ComplianceResult``
  objects but never re-evaluates policy rules.

What this agent does NOT do
----------------------------
* It does not re-read the source PDF.
* It does not call the Term Extraction or Compliance Review Agents.
* It does not write to ``state.final_report`` — that is Module 7's output.
* It does not invent risk findings not grounded in the supplied state.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.llm.client import LLMClient
from app.llm.schemas import SummaryOutput
from app.models.compliance import ComplianceResult, ComplianceStatus
from app.models.evidence import EvidenceSnippet
from app.models.policy import PolicyRule, RuleSeverity
from app.models.risk import RiskCategory, RiskFinding, RiskLikelihood, RiskSeverity
from app.models.terms import DealTerm, TermStatus

logger = logging.getLogger(__name__)


# ── Severity mapping from PolicyRule severity → RiskSeverity ─────────────────

_RULE_TO_RISK_SEVERITY: dict[RuleSeverity, RiskSeverity] = {
    RuleSeverity.CRITICAL: RiskSeverity.CRITICAL,
    RuleSeverity.HIGH: RiskSeverity.HIGH,
    RuleSeverity.MEDIUM: RiskSeverity.MEDIUM,
    RuleSeverity.LOW: RiskSeverity.LOW,
}

_SEVERITY_RANK: dict[RiskSeverity, int] = {
    RiskSeverity.CRITICAL: 4,
    RiskSeverity.HIGH: 3,
    RiskSeverity.MEDIUM: 2,
    RiskSeverity.LOW: 1,
}


class RiskSummaryAgent:
    """Synthesises risk findings and executive summary from structured pipeline output.

    Args:
        llm_client: Optional LLM client used solely for generating the narrative
                    executive summary.  When ``None``, a deterministic template
                    summary is produced instead.

    Usage::

        agent = RiskSummaryAgent(llm_client=None)
        findings, missing_info, follow_ups, summary = agent.analyze(
            compliance_results=state.compliance_results,
            extracted_terms=state.extracted_terms,
            evidence_registry=state.evidence_registry,
            policy_rules=state.policy_rules,
            escalations=state.escalations,
        )
    """

    AGENT_NAME: str = "risk_summary"

    def __init__(self, llm_client: Optional[LLMClient] = None) -> None:
        self._llm = llm_client

    # ── Public interface ───────────────────────────────────────────────────────

    def analyze(
        self,
        compliance_results: dict[str, ComplianceResult],
        extracted_terms: dict[str, DealTerm],
        evidence_registry: dict[str, EvidenceSnippet],
        policy_rules: list[PolicyRule],
        escalations: list[dict],
    ) -> tuple[list[RiskFinding], list[str], list[str], str]:
        """Derive risk findings and produce the executive summary.

        Args:
            compliance_results: Dict of ``COMP-NNN → ComplianceResult``.
            extracted_terms:    Dict of ``TERM-NNN → DealTerm``.
            evidence_registry:  Dict of ``EV-NNN → EvidenceSnippet``.
            policy_rules:       Ordered list of ``PolicyRule`` objects.
            escalations:        Human-escalation entries from ``WorkflowState``.

        Returns:
            A 4-tuple of:
            - ``risk_findings`` — prioritised ``list[RiskFinding]``
            - ``missing_information`` — ``list[str]``
            - ``follow_ups`` — ``list[str]``
            - ``executive_summary`` — ``str``
        """
        rules_by_id: dict[str, PolicyRule] = {r.rule_id: r for r in policy_rules}
        missing_information: list[str] = []
        raw_findings: list[_RawFinding] = []

        # ── Source 1: Compliance results ───────────────────────────────────────
        compliance_rule_ids_covered: set[str] = set()
        for comp_result in compliance_results.values():
            finding, mi_entry = self._finding_from_compliance(
                result=comp_result,
                rules_by_id=rules_by_id,
                evidence_registry=evidence_registry,
            )
            if finding is not None:
                raw_findings.append(finding)
                compliance_rule_ids_covered.add(comp_result.rule_id)
            if mi_entry is not None:
                missing_information.append(mi_entry)

        # ── Source 2: Term anomalies ───────────────────────────────────────────
        # Only for terms NOT already captured by a compliance finding above.
        term_names_in_compliance = self._term_names_in_compliance(
            compliance_results, extracted_terms
        )
        for term in extracted_terms.values():
            findings_and_mi = self._findings_from_term_anomaly(
                term=term,
                evidence_registry=evidence_registry,
                already_covered=term_names_in_compliance,
            )
            for f, mi in findings_and_mi:
                if f is not None:
                    raw_findings.append(f)
                if mi is not None:
                    missing_information.append(mi)

        # ── Source 3: Escalations ──────────────────────────────────────────────
        for escalation in escalations:
            finding = self._finding_from_escalation(
                escalation=escalation,
                evidence_registry=evidence_registry,
                compliance_rule_ids_covered=compliance_rule_ids_covered,
                rules_by_id=rules_by_id,
            )
            if finding is not None:
                raw_findings.append(finding)

        # ── Source 4: Evidence / extraction coverage gaps ─────────────────────
        coverage_findings = self._coverage_gap_findings(
            evidence_registry=evidence_registry,
            extracted_terms=extracted_terms,
        )
        raw_findings.extend(coverage_findings)

        # ── Assign RISK-NNN IDs (sorted by severity, then human_review flag) ───
        risk_findings = self._assign_risk_ids(raw_findings)

        # ── Deduplicate missing_information ────────────────────────────────────
        missing_information = _deduplicate(missing_information)

        # ── Build follow_ups ───────────────────────────────────────────────────
        follow_ups = self._build_follow_ups(risk_findings)

        # ── Executive summary ──────────────────────────────────────────────────
        executive_summary = self._generate_summary(
            risk_findings=risk_findings,
            missing_information=missing_information,
            follow_ups=follow_ups,
        )

        logger.info(
            "[%s] Analysis complete: %d finding(s), %d missing info item(s).",
            self.AGENT_NAME,
            len(risk_findings),
            len(missing_information),
        )
        return risk_findings, missing_information, follow_ups, executive_summary

    # ── Risk derivation — Source 1: Compliance ────────────────────────────────

    def _finding_from_compliance(
        self,
        result: ComplianceResult,
        rules_by_id: dict[str, PolicyRule],
        evidence_registry: dict[str, EvidenceSnippet],
    ) -> tuple[Optional["_RawFinding"], Optional[str]]:
        """Derive a risk finding and/or missing-info entry from a compliance result."""
        rule = rules_by_id.get(result.rule_id)
        rule_label = f"{result.rule_id} ({rule.name})" if rule else result.rule_id

        if result.status == ComplianceStatus.PASS:
            return None, None

        if result.status == ComplianceStatus.SKIPPED:
            return None, None

        if result.status == ComplianceStatus.FAIL:
            severity = self._rule_severity_to_risk(
                rule.severity if rule else RuleSeverity.HIGH
            )
            term_name = rule.name if rule else result.rule_id
            finding = _RawFinding(
                category=self._category_from_rule(rule),
                title=f"Policy Violation: {rule_label}",
                description=(
                    f"Compliance rule '{rule_label}' was evaluated and failed. "
                    f"Rationale: {result.rationale}"
                ),
                severity=severity,
                likelihood=RiskLikelihood.HIGH,
                evidence_ids=self._safe_evidence_ids(result.evidence_ids, evidence_registry),
                related_term_ids=result.term_ids,
                related_compliance_ids=[result.compliance_id],
                recommended_action=(
                    f"Resolve the policy violation for '{rule_label}'. "
                    f"{self._default_action_for_severity(severity)}"
                ),
                requires_human_review=False,
                notes=result.notes,
            )
            return finding, None

        if result.status == ComplianceStatus.NEEDS_HUMAN_REVIEW:
            term_name = rule.name if rule else result.rule_id
            severity = RiskSeverity.HIGH
            if rule and _RULE_TO_RISK_SEVERITY.get(rule.severity, RiskSeverity.HIGH) == RiskSeverity.CRITICAL:
                severity = RiskSeverity.CRITICAL
            mi_entry = None
            if result.ambiguity_detected:
                mi_entry = (
                    f"{rule_label}: ambiguous value for '{term_name}' — "
                    f"human judgment required."
                )
            finding = _RawFinding(
                category=self._category_from_rule(rule),
                title=f"Human Review Required: {rule_label}",
                description=(
                    f"Compliance rule '{rule_label}' could not be evaluated deterministically. "
                    f"Rationale: {result.rationale}"
                ),
                severity=severity,
                likelihood=RiskLikelihood.MEDIUM,
                evidence_ids=self._safe_evidence_ids(result.evidence_ids, evidence_registry),
                related_term_ids=result.term_ids,
                related_compliance_ids=[result.compliance_id],
                recommended_action=(
                    f"Submit '{rule_label}' to credit committee or senior reviewer for manual assessment."
                ),
                requires_human_review=True,
                notes=result.notes,
            )
            return finding, mi_entry

        if result.status == ComplianceStatus.INSUFFICIENT_EVIDENCE:
            term_name = rule.name if rule else result.rule_id
            mi_entry = (
                f"{rule_label}: term '{term_name}' not found in document — "
                f"compliance could not be established."
            )
            finding = _RawFinding(
                category=RiskCategory.COMPLIANCE,
                title=f"Missing Required Term: {rule_label}",
                description=(
                    f"Compliance rule '{rule_label}' could not be evaluated because "
                    f"the required term '{term_name}' was not extracted from the deal document."
                ),
                severity=RiskSeverity.HIGH,
                likelihood=RiskLikelihood.UNKNOWN,
                evidence_ids=self._safe_evidence_ids(result.evidence_ids, evidence_registry),
                related_term_ids=result.term_ids,
                related_compliance_ids=[result.compliance_id],
                recommended_action=(
                    f"Obtain the missing '{term_name}' value from the deal document "
                    f"or directly from the counterparty before proceeding."
                ),
                requires_human_review=True,
                notes=result.notes,
            )
            return finding, mi_entry

        # Unknown status — be conservative
        return None, None

    # ── Risk derivation — Source 2: Term anomalies ────────────────────────────

    @staticmethod
    def _term_names_in_compliance(
        compliance_results: dict[str, ComplianceResult],
        extracted_terms: dict[str, DealTerm],
    ) -> set[str]:
        """Return the set of term names already covered by compliance findings."""
        covered_term_ids: set[str] = set()
        for r in compliance_results.values():
            covered_term_ids.update(r.term_ids)
        names: set[str] = set()
        for tid, term in extracted_terms.items():
            if tid in covered_term_ids:
                names.add(term.name.lower())
        return names

    def _findings_from_term_anomaly(
        self,
        term: DealTerm,
        evidence_registry: dict[str, EvidenceSnippet],
        already_covered: set[str],
    ) -> list[tuple[Optional["_RawFinding"], Optional[str]]]:
        """Derive findings from term status anomalies not already in compliance results."""
        results = []
        name_key = term.name.lower()

        if term.status == TermStatus.AMBIGUOUS and name_key not in already_covered:
            mi = f"TERM '{term.name}' (id={term.term_id}): extracted value is ambiguous — multiple conflicting values found."
            finding = _RawFinding(
                category=RiskCategory.OPERATIONAL,
                title=f"Ambiguous Term: '{term.name}'",
                description=(
                    f"Term '{term.name}' was extracted with ambiguous or conflicting values. "
                    f"Raw value: '{term.value}'. "
                    f"Downstream compliance evaluation may have used an incorrect value."
                ),
                severity=RiskSeverity.MEDIUM,
                likelihood=RiskLikelihood.MEDIUM,
                evidence_ids=self._safe_evidence_ids(term.evidence_ids, evidence_registry),
                related_term_ids=[term.term_id],
                related_compliance_ids=[],
                recommended_action=(
                    f"Obtain a definitive value for '{term.name}' from the deal document "
                    f"or the counterparty before compliance can be confirmed."
                ),
                requires_human_review=True,
                notes=term.notes,
            )
            results.append((finding, mi))

        elif term.status == TermStatus.MISSING:
            mi = f"TERM '{term.name}' (id={term.term_id}): expected term marked MISSING."
            results.append((None, mi))

        if term.confidence < 0.5 and term.status not in (TermStatus.AMBIGUOUS, TermStatus.MISSING):
            mi = (
                f"TERM '{term.name}' (id={term.term_id}): "
                f"low-confidence extraction (confidence={term.confidence:.2f}) — "
                f"value may be unreliable."
            )
            results.append((None, mi))

        return results

    # ── Risk derivation — Source 3: Escalations ───────────────────────────────

    def _finding_from_escalation(
        self,
        escalation: dict,
        evidence_registry: dict[str, EvidenceSnippet],
        compliance_rule_ids_covered: set[str],
        rules_by_id: dict[str, PolicyRule],
    ) -> Optional["_RawFinding"]:
        """Create a risk finding from an escalation entry if not already covered."""
        rule_id = escalation.get("rule_id", "UNKNOWN")
        if rule_id in compliance_rule_ids_covered:
            # The rule is already represented by a compliance finding.
            # Avoid a duplicate entry — the compliance finding has more detail.
            return None

        reason = escalation.get("reason", "No reason provided.")
        rule = rules_by_id.get(rule_id)
        rule_label = f"{rule_id} ({rule.name})" if rule else rule_id

        return _RawFinding(
            category=RiskCategory.COMPLIANCE,
            title=f"Human Escalation Required: {rule_label}",
            description=(
                f"Automated resolution was not possible for rule '{rule_label}'. "
                f"Escalation reason: {reason}"
            ),
            severity=RiskSeverity.HIGH,
            likelihood=RiskLikelihood.UNKNOWN,
            evidence_ids=[],
            related_term_ids=[],
            related_compliance_ids=[],
            recommended_action=(
                f"Route rule '{rule_label}' to a senior reviewer or credit committee "
                f"for manual determination."
            ),
            requires_human_review=True,
            notes=f"Escalation handoff ID: {escalation.get('handoff_id', 'N/A')}",
        )

    # ── Risk derivation — Source 4: Coverage gaps ─────────────────────────────

    @staticmethod
    def _coverage_gap_findings(
        evidence_registry: dict[str, EvidenceSnippet],
        extracted_terms: dict[str, DealTerm],
    ) -> list["_RawFinding"]:
        """Generate critical findings when document content or extraction is absent."""
        findings = []

        if not evidence_registry:
            findings.append(_RawFinding(
                category=RiskCategory.OPERATIONAL,
                title="No Document Content Extracted",
                description=(
                    "No evidence snippets were extracted from the deal document. "
                    "All compliance and risk conclusions are unreliable."
                ),
                severity=RiskSeverity.CRITICAL,
                likelihood=RiskLikelihood.HIGH,
                evidence_ids=[],
                related_term_ids=[],
                related_compliance_ids=[],
                recommended_action=(
                    "Verify the deal document is readable and re-run the pipeline. "
                    "Obtain a valid document from the counterparty."
                ),
                requires_human_review=True,
                notes=None,
            ))
        elif not extracted_terms:
            findings.append(_RawFinding(
                category=RiskCategory.COMPLIANCE,
                title="No Deal Terms Extracted",
                description=(
                    "Evidence was extracted from the document but no deal terms "
                    "could be identified. Policy rules could not be evaluated."
                ),
                severity=RiskSeverity.HIGH,
                likelihood=RiskLikelihood.HIGH,
                evidence_ids=[],
                related_term_ids=[],
                related_compliance_ids=[],
                recommended_action=(
                    "Review the extraction output and verify the document contains "
                    "structured deal terms. Re-run with an updated extraction prompt if needed."
                ),
                requires_human_review=True,
                notes=None,
            ))

        return findings

    # ── RISK-NNN ID assignment ─────────────────────────────────────────────────

    @staticmethod
    def _assign_risk_ids(raw_findings: list["_RawFinding"]) -> list[RiskFinding]:
        """Sort by severity (desc) then human_review flag, assign RISK-NNN IDs."""
        sorted_findings = sorted(
            raw_findings,
            key=lambda f: (
                -_SEVERITY_RANK.get(f.severity, 0),
                -int(f.requires_human_review),
                f.category.value,   # deterministic tie-break
                f.title,
            ),
        )
        results = []
        for idx, raw in enumerate(sorted_findings, start=1):
            results.append(RiskFinding(
                risk_id=f"RISK-{idx:03d}",
                category=raw.category,
                title=raw.title,
                description=raw.description,
                severity=raw.severity,
                likelihood=raw.likelihood,
                evidence_ids=raw.evidence_ids,
                related_term_ids=raw.related_term_ids,
                related_compliance_ids=raw.related_compliance_ids,
                recommended_action=raw.recommended_action,
                requires_human_review=raw.requires_human_review,
                notes=raw.notes,
            ))
        return results

    # ── Follow-ups ─────────────────────────────────────────────────────────────

    @staticmethod
    def _build_follow_ups(risk_findings: list[RiskFinding]) -> list[str]:
        """Derive deduplicated, prioritised follow-up actions from risk findings."""
        seen: set[str] = set()
        follow_ups: list[str] = []

        for finding in risk_findings:  # Already sorted critical-first.
            action = finding.recommended_action.strip()
            if action and action not in seen:
                seen.add(action)
                follow_ups.append(action)
            if len(follow_ups) >= 10:
                break

        return follow_ups

    # ── Executive summary ──────────────────────────────────────────────────────

    def _generate_summary(
        self,
        risk_findings: list[RiskFinding],
        missing_information: list[str],
        follow_ups: list[str],
    ) -> str:
        """Generate the executive summary using LLM or a deterministic fallback."""
        if self._llm is not None:
            try:
                return self._llm_summary(risk_findings, missing_information, follow_ups)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[%s] LLM summary generation failed (%s). "
                    "Falling back to deterministic template.",
                    self.AGENT_NAME, exc,
                )

        return self._deterministic_summary(risk_findings, missing_information, follow_ups)

    def _llm_summary(
        self,
        risk_findings: list[RiskFinding],
        missing_information: list[str],
        follow_ups: list[str],
    ) -> str:
        """Call the LLM for a structured narrative summary."""
        from app.llm.schemas import SummaryOutput

        # Build a compact text representation of the risk data for the prompt.
        counts = self._severity_counts(risk_findings)
        human_review_count = sum(1 for f in risk_findings if f.requires_human_review)

        findings_text = "\n".join(
            f"- [{f.severity.value.upper()}] {f.title}: {f.description[:120]}..."
            if len(f.description) > 120 else
            f"- [{f.severity.value.upper()}] {f.title}: {f.description}"
            for f in risk_findings[:10]  # Cap to avoid excessive prompt length.
        ) or "None."

        missing_text = "\n".join(f"- {m}" for m in missing_information[:5]) or "None."
        follow_up_text = "\n".join(f"- {a}" for a in follow_ups[:5]) or "None."

        system_prompt = (
            "You are a credit risk officer summarising a deal review pipeline output. "
            "Write a concise executive summary based ONLY on the structured data provided. "
            "Do NOT invent information. Do NOT reference the raw deal document. "
            "Be precise, factual, and professionally toned."
        )

        user_prompt = (
            f"Deal Review Risk Summary\n"
            f"========================\n"
            f"Total findings: {len(risk_findings)}\n"
            f"CRITICAL: {counts.get(RiskSeverity.CRITICAL, 0)} | "
            f"HIGH: {counts.get(RiskSeverity.HIGH, 0)} | "
            f"MEDIUM: {counts.get(RiskSeverity.MEDIUM, 0)} | "
            f"LOW: {counts.get(RiskSeverity.LOW, 0)}\n"
            f"Requires human review: {human_review_count}\n\n"
            f"Key findings:\n{findings_text}\n\n"
            f"Missing information:\n{missing_text}\n\n"
            f"Recommended actions:\n{follow_up_text}\n"
        )

        result = self._llm.get_structured_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=SummaryOutput,
        )
        if isinstance(result, SummaryOutput):
            return result.executive_summary
        if hasattr(result, "executive_summary"):
            return str(getattr(result, "executive_summary"))
        raise TypeError(
            f"LLM client returned '{type(result).__name__}' instead of SummaryOutput "
            f"(missing 'executive_summary' attribute)."
        )

    @staticmethod
    def _deterministic_summary(
        risk_findings: list[RiskFinding],
        missing_information: list[str],
        follow_ups: list[str],
    ) -> str:
        """Generate a template-based executive summary without an LLM."""
        if not risk_findings:
            return (
                "Deal review completed. "
                "No risk findings were identified. "
                "All evaluated policy rules passed or were skipped. "
                f"Missing information items: {len(missing_information)}."
            )

        counts = RiskSummaryAgent._severity_counts(risk_findings)
        human_count = sum(1 for f in risk_findings if f.requires_human_review)
        critical_count = counts.get(RiskSeverity.CRITICAL, 0)
        high_count = counts.get(RiskSeverity.HIGH, 0)

        lines = [
            f"Deal review identified {len(risk_findings)} risk finding(s): "
            f"{critical_count} CRITICAL, {high_count} HIGH, "
            f"{counts.get(RiskSeverity.MEDIUM, 0)} MEDIUM, "
            f"{counts.get(RiskSeverity.LOW, 0)} LOW.",
        ]

        if critical_count > 0 or high_count > 0:
            top = [
                f for f in risk_findings
                if f.severity in (RiskSeverity.CRITICAL, RiskSeverity.HIGH)
            ][:3]
            top_titles = "; ".join(f.title for f in top)
            lines.append(f"Top priority items: {top_titles}.")

        if human_count > 0:
            lines.append(
                f"{human_count} finding(s) require human review before the deal can proceed."
            )

        if missing_information:
            lines.append(
                f"{len(missing_information)} information gap(s) identified that could not be resolved automatically."
            )

        if follow_ups:
            lines.append(f"Immediate recommended action: {follow_ups[0]}")

        return " ".join(lines)

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _severity_counts(findings: list[RiskFinding]) -> dict[RiskSeverity, int]:
        counts: dict[RiskSeverity, int] = {}
        for f in findings:
            counts[f.severity] = counts.get(f.severity, 0) + 1
        return counts

    @staticmethod
    def _rule_severity_to_risk(rule_severity: RuleSeverity) -> RiskSeverity:
        return _RULE_TO_RISK_SEVERITY.get(rule_severity, RiskSeverity.HIGH)

    @staticmethod
    def _category_from_rule(rule: Optional[PolicyRule]) -> RiskCategory:
        """Map a PolicyRule's category to a RiskCategory."""
        if rule is None:
            return RiskCategory.COMPLIANCE
        mapping = {
            "financial": RiskCategory.FINANCIAL,
            "legal": RiskCategory.LEGAL,
            "operational": RiskCategory.OPERATIONAL,
            "credit": RiskCategory.CREDIT,
            "regulatory": RiskCategory.COMPLIANCE,
            "collateral": RiskCategory.FINANCIAL,
            "general": RiskCategory.COMPLIANCE,
        }
        return mapping.get(rule.category.value, RiskCategory.COMPLIANCE)

    @staticmethod
    def _safe_evidence_ids(
        ids: list[str],
        evidence_registry: dict[str, EvidenceSnippet],
    ) -> list[str]:
        """Return only IDs that genuinely exist in the evidence registry."""
        return [eid for eid in ids if eid in evidence_registry]

    @staticmethod
    def _default_action_for_severity(severity: RiskSeverity) -> str:
        return {
            RiskSeverity.CRITICAL: "Escalate to credit committee immediately. Deal must be blocked pending resolution.",
            RiskSeverity.HIGH: "Escalate to senior credit officer for exception approval before proceeding.",
            RiskSeverity.MEDIUM: "Flag for review before deal approval.",
            RiskSeverity.LOW: "Document and monitor. No immediate escalation required.",
        }.get(severity, "Review with senior stakeholders.")


# ── Internal dataclass (not part of public API) ───────────────────────────────

class _RawFinding:
    """Intermediate representation before RISK-NNN IDs are assigned."""

    __slots__ = (
        "category", "title", "description", "severity", "likelihood",
        "evidence_ids", "related_term_ids", "related_compliance_ids",
        "recommended_action", "requires_human_review", "notes",
    )

    def __init__(
        self,
        category: RiskCategory,
        title: str,
        description: str,
        severity: RiskSeverity,
        likelihood: RiskLikelihood,
        evidence_ids: list[str],
        related_term_ids: list[str],
        related_compliance_ids: list[str],
        recommended_action: str,
        requires_human_review: bool,
        notes: Optional[str],
    ) -> None:
        self.category = category
        self.title = title
        self.description = description
        self.severity = severity
        self.likelihood = likelihood
        self.evidence_ids = evidence_ids
        self.related_term_ids = related_term_ids
        self.related_compliance_ids = related_compliance_ids
        self.recommended_action = recommended_action
        self.requires_human_review = requires_human_review
        self.notes = notes


def _deduplicate(items: list[str]) -> list[str]:
    """Return a deduplicated list preserving original insertion order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
