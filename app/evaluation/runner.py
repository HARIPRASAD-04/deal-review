"""Evaluation runner — Module 8.

Deterministic, non-agent evaluation infrastructure for the deal review pipeline.

``evaluate_case()`` runs the full four-agent pipeline against a labelled
``EvaluationCase`` and scores the result across a rich set of per-check
assertions.  It is NOT an agent — it is scoring infrastructure.

Scoring model
-------------
Each check is worth 1 point.  ``score = passed_checks / total_checks``.
``passed = (score == 1.0)``.

Checks:
    review_status_match            Expected vs actual ReviewStatus.
    terms_present_{name}           Each expected term name found in extracted_terms.
    evidence_grounding_{name}      Each extracted term has ≥ min_evidence_ids_per_term.
    compliance_status_{rule_id}    Actual status matches expected per rule.
    risk_category_{cat}            At least one finding per expected category.
    risk_title_{idx}               Each title substring found in ≥ 1 finding title.
    report_integrity               final_report is non-None and round-trips cleanly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from app.llm.client import LLMClient
from app.models.report import FinalDealReviewReport, ReviewStatus
from app.service import DealReviewRequest, DealReviewResponse, DealReviewService

logger = logging.getLogger(__name__)


# ── Evaluation case ───────────────────────────────────────────────────────────

@dataclass
class EvaluationCase:
    """Specification for a single evaluation run.

    Attributes:
        case_id:                    Unique identifier for this case.
        deal_pdf_path:              Path to the deal PDF file.
        policy_pdf_path:            Path to the policy PDF file (required; not bundled rules).
        expected_review_status:     Expected ``ReviewStatus`` value.
        expected_term_names:        Term names (snake_case) that MUST appear in
                                    ``extracted_terms``.
        expected_compliance:        ``{rule_id: status_string}`` — expected compliance
                                    outcome per rule.  Status strings match
                                    ``ComplianceStatus`` values (e.g. "PASS", "FAIL",
                                    "INSUFFICIENT_EVIDENCE").
        expected_risk_categories:   At least one ``RiskFinding`` per listed category
                                    (e.g. ["financial", "compliance"]).
        expected_risk_finding_titles: Substring matches against risk finding titles.
                                    At least one finding title must contain each string.
        min_evidence_ids_per_term:  Each extracted term must cite at least this many
                                    ``EV-NNN`` IDs.
        document_id:                Optional document ID override.
        llm_client:                 Optional LLM client override for this case.
    """

    case_id: str
    deal_pdf_path: str
    policy_pdf_path: str
    expected_review_status: ReviewStatus
    expected_term_names: list[str] = field(default_factory=list)
    expected_compliance: dict[str, str] = field(default_factory=dict)
    expected_risk_categories: list[str] = field(default_factory=list)
    expected_risk_finding_titles: list[str] = field(default_factory=list)
    min_evidence_ids_per_term: int = 1
    document_id: str = ""
    llm_client: Optional[LLMClient] = None


# ── Evaluation result ─────────────────────────────────────────────────────────

@dataclass
class EvaluationResult:
    """Outcome of running a single ``EvaluationCase``.

    Attributes:
        case_id:       Matches ``EvaluationCase.case_id``.
        passed:        True only when ``score == 1.0``.
        score:         Fraction of checks that passed (0.0 – 1.0).
        check_results: ``{check_name: bool}`` — per-check pass/fail.
        details:       ``{check_name: {expected, actual, ...}}`` — detail per check.
        final_report:  The assembled report, or ``None`` if pipeline failed.
        error:         Top-level error message; ``None`` on success.
    """

    case_id: str
    passed: bool
    score: float
    check_results: dict[str, bool] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    final_report: Optional[FinalDealReviewReport] = None
    error: Optional[str] = None


# ── Core evaluate_case function ───────────────────────────────────────────────

def evaluate_case(
    case: EvaluationCase,
    llm_client: Optional[LLMClient] = None,
) -> EvaluationResult:
    """Run the full pipeline against a case and score the result.

    The ``llm_client`` argument takes precedence over ``case.llm_client``.
    Both are ``None``-safe — the service handles FakeLLMClient fallback in
    demo mode but policy parsing will fail if no LLM is configured.

    Args:
        case:       The labelled evaluation case.
        llm_client: LLM client to use.  Overrides ``case.llm_client``.

    Returns:
        An ``EvaluationResult`` with per-check scores.
    """
    effective_llm = llm_client if llm_client is not None else case.llm_client

    # ── Run pipeline via service ───────────────────────────────────────────────
    service = DealReviewService()
    try:
        response: DealReviewResponse = service.run(
            DealReviewRequest(
                deal_pdf_path=case.deal_pdf_path,
                policy_pdf_path=case.policy_pdf_path,
                document_id=case.document_id or case.case_id,
                llm_client=effective_llm,
            )
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected error running case '%s': %s", case.case_id, exc, exc_info=True)
        return EvaluationResult(
            case_id=case.case_id,
            passed=False,
            score=0.0,
            error=f"Unexpected exception: {exc}",
        )

    if not response.success or response.report is None:
        return EvaluationResult(
            case_id=case.case_id,
            passed=False,
            score=0.0,
            error=response.error or "Pipeline did not produce a report.",
            final_report=response.report,
        )

    report = response.report
    final_state = response.final_state

    check_results: dict[str, bool] = {}
    details: dict[str, Any] = {}

    # ── Check 1: report_integrity ─────────────────────────────────────────────
    try:
        FinalDealReviewReport.model_validate_json(report.model_dump_json())
        check_results["report_integrity"] = True
        details["report_integrity"] = {"status": "round-trip OK"}
    except Exception as exc:
        check_results["report_integrity"] = False
        details["report_integrity"] = {"error": str(exc)}

    # ── Check 2: review_status_match ─────────────────────────────────────────
    actual_status = report.review_status
    expected_status = case.expected_review_status
    check_results["review_status_match"] = actual_status == expected_status
    details["review_status_match"] = {
        "expected": expected_status.value,
        "actual": actual_status.value,
    }

    # ── Checks 3+4: terms_present + evidence_grounding ───────────────────────
    extracted_terms = {}
    if final_state and hasattr(final_state, "extracted_terms"):
        extracted_terms = final_state.extracted_terms or {}

    for term_name in case.expected_term_names:
        # Find by name match.
        found_term = next(
            (t for t in extracted_terms.values() if t.name == term_name),
            None,
        )
        present = found_term is not None
        check_results[f"terms_present_{term_name}"] = present
        details[f"terms_present_{term_name}"] = {
            "expected_name": term_name,
            "found": present,
        }

        # Evidence grounding check (only if term was found).
        ev_check_name = f"evidence_grounding_{term_name}"
        if found_term is not None:
            ev_count = len(found_term.evidence_ids)
            ok = ev_count >= case.min_evidence_ids_per_term
            check_results[ev_check_name] = ok
            details[ev_check_name] = {
                "expected_min": case.min_evidence_ids_per_term,
                "actual_count": ev_count,
                "evidence_ids": list(found_term.evidence_ids),
            }
        else:
            # Term not found → evidence grounding trivially fails.
            check_results[ev_check_name] = False
            details[ev_check_name] = {
                "expected_min": case.min_evidence_ids_per_term,
                "actual_count": 0,
                "note": "Term not extracted",
            }

    # ── Check 5: compliance_status per rule ───────────────────────────────────
    compliance_results = {}
    if final_state and hasattr(final_state, "compliance_results"):
        compliance_results = final_state.compliance_results or {}

    for rule_id, expected_comp_status in case.expected_compliance.items():
        check_name = f"compliance_status_{rule_id}"
        # Find compliance result by rule_id.
        matching = [
            r for r in compliance_results.values()
            if r.rule_id == rule_id
        ]
        if matching:
            actual_comp_status = matching[0].status.value
            ok = actual_comp_status == expected_comp_status
            check_results[check_name] = ok
            details[check_name] = {
                "rule_id": rule_id,
                "expected": expected_comp_status,
                "actual": actual_comp_status,
            }
        else:
            check_results[check_name] = False
            details[check_name] = {
                "rule_id": rule_id,
                "expected": expected_comp_status,
                "actual": "NOT_FOUND",
            }

    # ── Check 6: risk_categories ──────────────────────────────────────────────
    risk_categories_found = {
        f.get("category", "").lower()
        for f in report.risk_findings
    }
    for cat in case.expected_risk_categories:
        check_name = f"risk_category_{cat}"
        found = cat.lower() in risk_categories_found
        check_results[check_name] = found
        details[check_name] = {
            "expected_category": cat,
            "found": found,
            "all_categories": sorted(risk_categories_found),
        }

    # ── Check 7: risk_finding_titles (substring match) ────────────────────────
    all_titles = [f.get("title", "").lower() for f in report.risk_findings]
    for idx, title_substr in enumerate(case.expected_risk_finding_titles):
        check_name = f"risk_title_{idx}"
        found = any(title_substr.lower() in t for t in all_titles)
        check_results[check_name] = found
        details[check_name] = {
            "expected_substring": title_substr,
            "found": found,
            "all_titles": [f.get("title", "") for f in report.risk_findings],
        }

    # ── Score ─────────────────────────────────────────────────────────────────
    total = len(check_results)
    passed_count = sum(1 for v in check_results.values() if v)
    score = round(passed_count / total, 4) if total > 0 else 0.0
    passed = score == 1.0

    logger.info(
        "EvaluationCase '%s': score=%.2f (%d/%d checks), passed=%s.",
        case.case_id,
        score,
        passed_count,
        total,
        passed,
    )

    return EvaluationResult(
        case_id=case.case_id,
        passed=passed,
        score=score,
        check_results=check_results,
        details=details,
        final_report=report,
        error=None,
    )
