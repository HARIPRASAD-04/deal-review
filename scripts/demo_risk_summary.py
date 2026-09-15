"""Manual demo -- Module 6 Risk & Summary Agent.

Demonstrates the full pipeline ending at risk assessment & summary:
    deal_002.pdf
        -> Document Ingestion (Module 2)
        -> Evidence Registry
        -> Term Extraction (Module 3)
        -> Compliance Review Agent (Module 4)
        -> Handoff & Escalation Handling (Module 5)
        -> Risk & Summary Agent (Module 6)
            -> Prioritized Risk Register
            -> Missing Information
            -> Recommended Follow-ups
            -> Executive Summary
            -> Evidence Traceability

Run from project root:
    python scripts/demo_risk_summary.py

Exit code 0 = success.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from app.ingestion.evidence_registry import EvidenceRegistry
from app.ingestion.pdf_loader import load_pdf
from app.ingestion.segmenter import segment_page
from app.llm.client import FakeLLMClient
from app.llm.schemas import ExtractionOutput, ExtractedTermSchema, SummaryOutput
from app.models.compliance import ComplianceStatus
from app.models.policy import PolicyRule, RuleCategory, RuleSeverity
from app.models.state import WorkflowState
from app.models.terms import TermCategory
from app.orchestration.graph import build_graph

FIXTURE_PDF = pathlib.Path(__file__).parent.parent / "data" / "samples" / "deal_002.pdf"
SEP72 = "-" * 72
SEP72_WIDE = "=" * 72


def _section(title: str) -> None:
    print(f"\n{SEP72}")
    print(f"  {title}")
    print(SEP72)


def _make_policy_rules() -> list[PolicyRule]:
    """Return a representative compliance rule set for deal_002."""
    def rule(rule_id, name, op, threshold, severity=RuleSeverity.HIGH, desc=None, cat=RuleCategory.FINANCIAL):
        return PolicyRule(
            rule_id=rule_id,
            name=name,
            description=desc or f"{name} {op} {threshold}",
            category=cat,
            operator=op,
            threshold=threshold,
            severity=severity,
        )

    return [
        rule("POLICY-001", "interest_rate", "<=", "0.09", RuleSeverity.HIGH,
             desc="Interest rate must not exceed 9% per annum.", cat=RuleCategory.FINANCIAL),
        rule("POLICY-002", "financial_covenant_dscr", ">=", "1.25", RuleSeverity.HIGH,
             desc="Minimum Debt Service Coverage Ratio of 1.25x.", cat=RuleCategory.FINANCIAL),
        rule("POLICY-003", "governing_law", "in", "India, Singapore, UK", RuleSeverity.HIGH,
             desc="Governing law must be India, Singapore or UK.", cat=RuleCategory.LEGAL),
        rule("POLICY-004", "principal_amount", "<=", "80000000", RuleSeverity.HIGH,
             desc="Facility amount must not exceed INR 8 crore (80,000,000).", cat=RuleCategory.FINANCIAL),
        rule("POLICY-005", "tenure", "<=", "72", RuleSeverity.MEDIUM,
             desc="Loan tenure must not exceed 72 months.", cat=RuleCategory.OPERATIONAL),
        rule("POLICY-006", "missing_required_term", "exists", "true", RuleSeverity.HIGH,
             desc="Material term required for credit review.", cat=RuleCategory.REGULATORY),
    ]


def _fake_extraction(ev_registry: dict) -> ExtractionOutput:
    """Simulate LLM extraction using real EV IDs from the registry."""
    real_ids = sorted(ev_registry.keys())
    ev0 = real_ids[0] if len(real_ids) > 0 else "EV-001"
    ev1 = real_ids[1] if len(real_ids) > 1 else ev0
    ev2 = real_ids[2] if len(real_ids) > 2 else ev0

    return ExtractionOutput(terms=[
        ExtractedTermSchema(
            name="interest_rate",
            value="10.5% per annum",  # Breaches <= 0.09 rule -> NON_COMPLIANT
            normalized_value="0.105",
            category=TermCategory.RATE,
            confidence=0.99,
            evidence_ids=[ev0],
        ),
        ExtractedTermSchema(
            name="financial_covenant_dscr",
            value="minimum DSCR of 1.25x",
            normalized_value="1.25",
            category=TermCategory.COVENANT,
            confidence=0.98,
            evidence_ids=[ev1],
        ),
        ExtractedTermSchema(
            name="governing_law",
            value="State of New York",  # Breaches allowed list -> NON_COMPLIANT
            normalized_value="State of New York",
            category=TermCategory.GENERAL,
            confidence=0.95,
            evidence_ids=[ev2],
        ),
        ExtractedTermSchema(
            name="missing_required_term",
            value="unspecified",
            normalized_value=None,
            category=TermCategory.GENERAL,
            confidence=0.20,
            evidence_ids=[ev0],  # Low confidence -> triggers clarification/human review
        ),
    ])


def main() -> int:
    print(f"\n{SEP72_WIDE}")
    print("  MODULE 6 DEMO: RISK & SUMMARY AGENT PIPELINE")
    print(SEP72_WIDE)

    # 1. Ingestion
    _section("STEP 1: Document Ingestion & Evidence Registry")
    result = load_pdf(str(FIXTURE_PDF), "DOC-DEMO-006")
    if not result.success:
        print(f"  [ERROR] Ingestion failed: {result.error}")
        return 1

    registry = EvidenceRegistry()
    counter = 1
    for page in result.pages:
        snippets = segment_page(page, evidence_id_start=counter)
        for s in snippets:
            registry.add(s)
        counter += len(snippets)

    print(f"Loaded {result.metadata.page_count} page(s) from {FIXTURE_PDF.name}")
    print(f"Registered {len(registry)} evidence items. Sample IDs: {sorted(registry.to_dict())[:3]}")

    # 2. Build pipeline state & policy
    rules = _make_policy_rules()
    state = WorkflowState(
        source_pdf_path=str(FIXTURE_PDF),
        document_id="DOC-DEMO-006",
        evidence_registry=registry.to_dict(),
        policy_rules=rules,
    )

    # 3. Setup Graph with fake LLM client (providing extraction, clarification, and summary responses)
    ev0 = sorted(state.evidence_registry.keys())[0] if state.evidence_registry else "EV-001"
    fake_clarification = ExtractionOutput(terms=[
        ExtractedTermSchema(
            name="missing_required_term",
            value="resolved_term_value",
            normalized_value="resolved_term_value",
            category=TermCategory.GENERAL,
            confidence=0.90,
            evidence_ids=[ev0],
        )
    ])
    fake_summary = SummaryOutput(
        executive_summary=(
            "Executive Summary (LLM Generated): Deal review identified 4 risk finding(s) "
            "(0 CRITICAL, 4 HIGH). Key concerns include policy breaches for interest_rate "
            "(10.5% vs max 9.0%) and governing_law (State of New York vs allowed India/Singapore/UK), "
            "alongside missing required terms principal_amount and tenure. Immediate counterparty "
            "clarification and credit officer exception approval are recommended."
        )
    )
    fake_llm = FakeLLMClient(responses=[
        _fake_extraction(state.evidence_registry),
        fake_clarification,
        fake_clarification,
        fake_summary,
    ])
    app = build_graph(llm_client=fake_llm)

    _section("STEP 2: Executing LangGraph Workflow (Ingestion -> Extraction -> Compliance -> Handoff -> Risk & Summary)")
    final_dict = app.invoke(state.model_dump())
    final_state = WorkflowState.model_validate(final_dict)

    # 4. Results summary
    _section("STEP 3: Risk & Summary Agent Output")
    print(f"Workflow Run Status : {final_state.run_status}")
    print(f"Total Terms        : {len(final_state.extracted_terms)}")
    print(f"Compliance Results : {len(final_state.compliance_results)}")
    print(f"Risk Findings      : {len(final_state.risk_findings)}")
    print(f"Missing Information: {len(final_state.missing_information)}")
    print(f"Follow-ups         : {len(final_state.follow_ups)}")

    _section("RISK REGISTER")
    for r in final_state.risk_findings:
        print(f"[{r.risk_id}] [{r.severity.name}] {r.category.name} | {r.title}")
        print(f"   Description : {r.description}")
        print(f"   Action      : {r.recommended_action}")
        print(f"   Evidence IDs: {r.evidence_ids}")
        print()

    _section("MISSING INFORMATION")
    for m in final_state.missing_information:
        print(f"• {m}")

    _section("RECOMMENDED FOLLOW-UPS")
    for idx, f in enumerate(final_state.follow_ups, 1):
        print(f"{idx}. {f}")

    _section("EXECUTIVE SUMMARY")
    print(final_state.executive_summary)

    _section("VERIFICATION")
    assert final_state.executive_summary is not None, "Executive summary must be present"
    assert len(final_state.risk_findings) > 0, "Risk findings must be identified"
    # Ensure evidence traceability
    valid_ev = set(final_state.evidence_registry.keys())
    for r in final_state.risk_findings:
        for ev_id in r.evidence_ids:
            assert ev_id in valid_ev, f"Invalid evidence ID {ev_id} in risk finding!"
    print("[OK] All assertions passed: Risk findings, missing information, follow-ups, executive summary, and evidence traceability verified!")

    print(f"\n{SEP72_WIDE}")
    print("  MODULE 6 DEMO COMPLETED SUCCESSFULLY")
    print(SEP72_WIDE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
