"""Manual demo -- Module 7 Orchestrator Agent & End-to-End Deal Review Pipeline.

Demonstrates the full orchestrated 4-agent deal review pipeline:
    deal_002.pdf
        -> 1. Ingestion (Infrastructure)
        -> 2. Term Extraction Agent
        -> 3. Compliance Review Agent
        -> 4. HandoffRouter (Clarification feedback loop)
        -> 5. Risk & Summary Agent
        -> 6. OrchestratorAgent Report Assembly
            -> FinalDealReviewReport (JSON output)

Run from project root:
    python scripts/demo_orchestrator.py

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
from app.models.policy import PolicyRule, RuleCategory, RuleSeverity
from app.models.report import FinalDealReviewReport
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
    """Return representative compliance rules for deal_002."""
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
    """Simulate LLM extraction using real EV IDs."""
    real_ids = sorted(ev_registry.keys())
    ev0 = real_ids[0] if len(real_ids) > 0 else "EV-001"
    ev1 = real_ids[1] if len(real_ids) > 1 else ev0
    ev2 = real_ids[2] if len(real_ids) > 2 else ev0

    return ExtractionOutput(terms=[
        ExtractedTermSchema(
            name="interest_rate",
            value="10.5% per annum",
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
            confidence=0.95,
            evidence_ids=[ev1],
        ),
        ExtractedTermSchema(
            name="governing_law",
            value="Laws of India",
            normalized_value="India",
            category=TermCategory.GENERAL,
            confidence=0.98,
            evidence_ids=[ev2],
        ),
        ExtractedTermSchema(
            name="principal_amount",
            value="INR 50,000,000",
            normalized_value="50000000",
            category=TermCategory.FINANCIAL,
            confidence=0.97,
            evidence_ids=[ev0],
        ),
        ExtractedTermSchema(
            name="tenure",
            value="60 months",
            normalized_value="60",
            category=TermCategory.CONDITION,
            confidence=0.96,
            evidence_ids=[ev1],
        ),
    ])


def main() -> int:
    print(SEP72_WIDE)
    print("  MODULE 7 DEMO: ORCHESTRATOR AGENT & FINAL REPORT PIPELINE")
    print(SEP72_WIDE)

    # ── STEP 1: Ingest Document ─────────────────────────────────────────────
    _section("STEP 1: Document Ingestion & Evidence Registry")
    if not FIXTURE_PDF.exists():
        print(f"[ERROR] Sample PDF not found at: {FIXTURE_PDF}")
        return 1

    result = load_pdf(str(FIXTURE_PDF), "deal_002.pdf")
    if not result.success:
        print(f"[ERROR] PDF loading failed: {result.error}")
        return 1

    registry = EvidenceRegistry()
    counter = 1
    for page in result.pages:
        snippets = segment_page(page, evidence_id_start=counter)
        for snip in snippets:
            registry.add(snip)
        counter += len(snippets)

    print(f"Ingested {result.metadata.page_count} page(s) from {result.metadata.filename}")
    print(f"Registered {len(registry)} evidence items. Sample IDs: {sorted(registry.to_dict().keys())[:3]}")

    # ── STEP 2: Configure Graph & LLM ───────────────────────────────────────
    _section("STEP 2: Executing End-to-End 4-Agent Pipeline Graph")
    policy_rules = _make_policy_rules()
    fake_extraction = _fake_extraction(registry.to_dict())
    fake_summary = SummaryOutput(
        executive_summary=(
            "Executive Summary (Module 7 Demo):\n"
            "The borrower requests a facility of INR 5.0 crore at 10.5% interest per annum. "
            "Compliance evaluation identified a policy breach on interest rate (10.5% vs 9.0% threshold). "
            "Financial covenant DSCR (1.25x), governing law (India), amount, and tenure are fully compliant. "
            "One missing required term was flagged for human review."
        )
    )

    class MultiResponseFakeLLM(FakeLLMClient):
        def __init__(self):
            super().__init__(response=fake_extraction)
            self._extraction_output = fake_extraction
            self._summary_output = fake_summary

        def get_structured_completion(self, system_prompt: str, user_prompt: str, response_schema: type):
            self._call_count += 1
            if response_schema is SummaryOutput:
                return self._summary_output
            return self._extraction_output

    llm = MultiResponseFakeLLM()
    graph = build_graph(llm_client=llm)

    initial_state = WorkflowState(
        document_id="deal_002.pdf",
        source_pdf_path=str(FIXTURE_PDF),
        policy_rules=policy_rules,
        evidence_registry=registry.to_dict(),
    )

    res = graph.invoke(initial_state)
    final_state = WorkflowState(**res) if isinstance(res, dict) else res

    print(f"Pipeline Graph Run Completed.")
    print(f"Workflow Run Status: {final_state.run_status.value}")
    statuses = {k: v.status.value for k, v in final_state.agent_statuses.items()}
    print(f"Agent Statuses: {statuses}")

    # ── STEP 3: Final Report Deserialisation & Inspection ───────────────────
    _section("STEP 3: Orchestrator Agent Final Report Output")

    if not final_state.final_report:
        print("[ERROR] final_report was not set by OrchestratorAgent!")
        return 1

    report = FinalDealReviewReport.model_validate_json(final_state.final_report)

    print(f"Report ID:               {report.report_id}")
    print(f"Run ID:                  {report.run_id}")
    print(f"Document ID:             {report.document_id}")
    print(f"Generated At:            {report.generated_at.isoformat()}")
    print(f"Pipeline Review Status:  {report.review_status.value}")
    print(f"Review Rationale:        {report.review_status_rationale}")

    _section("STEP 4: Executive Summary (Narrative)")
    print(report.executive_summary)

    _section("STEP 5: Compliance Summary")
    cs = report.compliance_summary
    print(f"Total Rules Evaluated:       {cs.total_rules}")
    print(f"Pass Count:                  {cs.pass_count}")
    print(f"Fail Count:                  {cs.fail_count}")
    print(f"Needs Human Review Count:    {cs.needs_review_count}")
    print(f"Insufficient Evidence Count: {cs.insufficient_evidence_count}")
    print(f"Overall Compliance Status:   {cs.overall_status}")

    _section("STEP 6: Prioritised Risk Register")
    if not report.risk_findings:
        print("No risk findings.")
    else:
        for idx, finding in enumerate(report.risk_findings, start=1):
            sev = finding.get("severity", "").upper()
            title = finding.get("title", "")
            action = finding.get("recommended_action", "")
            human = finding.get("requires_human_review", False)
            print(f"[{idx}] [{sev}] {finding.get('risk_id')} — {title}")
            print(f"    Category: {finding.get('category')}")
            print(f"    Description: {finding.get('description')}")
            print(f"    Recommended Action: {action}")
            print(f"    Requires Human Review: {human}")

    _section("STEP 7: Information Gaps & Follow-ups")
    print(f"Missing Information Items ({len(report.missing_information)}):")
    for gap in report.missing_information:
        print(f"  * {gap}")
    print(f"Recommended Follow-ups ({len(report.follow_ups)}):")
    for fu in report.follow_ups:
        print(f"  * {fu}")

    _section("STEP 8: Escalations & Audit Trail")
    print(f"Escalated Items Count: {len(report.escalations)}")
    for esc in report.escalations:
        print(f"  * [{esc.get('severity', 'UNKNOWN')}] {esc.get('escalation_id')}: {esc.get('reason')}")
    print(f"Total Audit Events Emitted: {report.audit_event_count}")

    print(f"\n{SEP72_WIDE}")
    print("  MODULE 7 DEMO COMPLETED SUCCESSFULLY!")
    print(SEP72_WIDE)

    return 0


if __name__ == "__main__":
    sys.exit(main())
