"""Manual demo -- Module 4 Compliance Review Agent.

Demonstrates the full pipeline ending at compliance review:
    deal_002.pdf
        -> Document Ingestion (Module 2)
        -> Evidence Registry
        -> Term Extraction (FakeLLMClient -- offline, fast)
        -> Compliance Review Agent (deterministic -- no LLM)
        -> ComplianceResult[]

The fake LLM output is built from real EV-NNN IDs that exist in the
registry, so evidence traceability is fully exercised.

Run from project root:
    python scripts/demo_compliance.py

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
from app.llm.schemas import ExtractionOutput, ExtractedTermSchema
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
    def rule(rule_id, name, op, threshold, severity=RuleSeverity.HIGH, desc=None):
        return PolicyRule(
            rule_id=rule_id,
            name=name,
            description=desc or f"{name} {op} {threshold}",
            category=RuleCategory.FINANCIAL,
            operator=op,
            threshold=threshold,
            severity=severity,
        )

    return [
        rule("POLICY-001", "interest_rate",          "<=", "0.09",
             desc="Interest rate must not exceed 9% per annum."),
        rule("POLICY-002", "financial_covenant_dscr", ">=", "1.25",
             desc="Minimum Debt Service Coverage Ratio of 1.25x."),
        rule("POLICY-003", "governing_law",           "in", "India, Singapore, UK",
             desc="Governing law must be India, Singapore or UK."),
        rule("POLICY-004", "principal_amount",        "<=", "80000000",
             desc="Facility amount must not exceed INR 8 crore (80,000,000)."),
        rule("POLICY-005", "tenure",                  "<=", "72",
             desc="Loan tenure must not exceed 72 months."),
        rule("POLICY-006", "missing_term",            "exists", "true",
             severity=RuleSeverity.MEDIUM,
             desc="A hypothetical required term not present in this document."),
    ]


def _fake_extraction(ev_registry: dict) -> ExtractionOutput:
    """Simulate LLM extraction using real EV IDs from the registry."""
    real_ids = sorted(ev_registry.keys())
    return ExtractionOutput(terms=[
        ExtractedTermSchema(
            name="interest_rate",
            value="8.5% per annum",
            normalized_value="0.085",
            category=TermCategory.RATE,
            confidence=0.99,
            evidence_ids=[real_ids[0]],
        ),
        ExtractedTermSchema(
            name="financial_covenant_dscr",
            value="minimum DSCR of 1.25x",
            normalized_value="1.25",
            category=TermCategory.COVENANT,
            confidence=0.98,
            evidence_ids=[real_ids[1]],
        ),
        ExtractedTermSchema(
            name="governing_law",
            value="India",
            normalized_value="India",
            category=TermCategory.GENERAL,
            confidence=0.99,
            evidence_ids=[real_ids[2]],
        ),
        ExtractedTermSchema(
            name="principal_amount",
            value="INR 50,000,000",
            normalized_value="50000000 INR",
            category=TermCategory.FINANCIAL,
            confidence=0.99,
            evidence_ids=[real_ids[3]],
        ),
        ExtractedTermSchema(
            name="tenure",
            value="60 months (5 years)",
            normalized_value="60",
            category=TermCategory.FINANCIAL,
            confidence=0.97,
            evidence_ids=[real_ids[4]],
        ),
    ])


def _status_marker(status: ComplianceStatus) -> str:
    mapping = {
        ComplianceStatus.PASS:                 "[PASS]",
        ComplianceStatus.FAIL:                 "[FAIL]",
        ComplianceStatus.NEEDS_HUMAN_REVIEW:   "[REVIEW]",
        ComplianceStatus.INSUFFICIENT_EVIDENCE:"[NO_DATA]",
        ComplianceStatus.SKIPPED:              "[SKIP]",
    }
    return mapping.get(status, f"[{status.value}]")


def main() -> int:
    print(f"\n{SEP72_WIDE}")
    print("  DEAL REVIEW PIPELINE -- Module 4 Compliance Review Demo")
    print("  LLM:        FakeLLMClient (offline -- no API key needed)")
    print("  Compliance: Deterministic (no LLM)")
    print(SEP72_WIDE)

    # ── Step 1: Document Ingestion ────────────────────────────────────────────
    _section("STEP 1 -- Document Ingestion")
    if not FIXTURE_PDF.exists():
        print(f"  [ERROR] Fixture not found: {FIXTURE_PDF}")
        return 1

    result = load_pdf(str(FIXTURE_PDF), "DEAL-002")
    if not result.success:
        print(f"  [ERROR] PDF ingestion failed: {result.error}")
        return 1

    registry = EvidenceRegistry()
    counter = 1
    for page in result.pages:
        snippets = segment_page(page, evidence_id_start=counter)
        for s in snippets:
            registry.add(s)
        counter += len(snippets)

    print(f"  Document  : {FIXTURE_PDF.name}")
    print(f"  Pages     : {result.metadata.page_count}")
    print(f"  Evidence  : {len(registry)} snippets ({sorted(registry.to_dict())[:3]} ...)")

    # ── Step 2: Term Extraction (Fake LLM) ───────────────────────────────────
    _section("STEP 2 -- Term Extraction (FakeLLMClient)")
    ev_dict = registry.to_dict()
    fake_response = _fake_extraction(ev_dict)
    policy_rules = _make_policy_rules()
    print(f"  Fake LLM  : {len(fake_response.terms)} terms prepared")

    # ── Step 3: Build graph and invoke ───────────────────────────────────────
    _section("STEP 3 -- Graph Execution (extract_terms -> review_compliance)")
    state = WorkflowState(
        document_id="DEAL-002",
        source_pdf_path=str(FIXTURE_PDF),
        policy_rules=policy_rules,
    )
    graph = build_graph(llm_client=FakeLLMClient(response=fake_response))
    raw_result = graph.invoke(state)
    final = WorkflowState(**raw_result) if isinstance(raw_result, dict) else raw_result

    print(f"  Extracted terms     : {len(final.extracted_terms)}")
    print(f"  Policy rules        : {len(final.policy_rules)}")
    print(f"  Compliance results  : {len(final.compliance_results)}")
    print(f"  Audit events        : {len(final.audit_events)}")

    # ── Step 4: Compliance Report ─────────────────────────────────────────────
    _section(f"STEP 4 -- Compliance Report ({len(final.compliance_results)} results)")

    pass_count = 0
    fail_count = 0
    review_count = 0
    insuff_count = 0

    for comp_id, comp in sorted(final.compliance_results.items()):
        marker = _status_marker(comp.status)
        print(f"\n  {marker} {comp.compliance_id}  |  {comp.rule_id}")
        print(f"      Rationale : {comp.rationale}")

        if comp.term_ids:
            for term_id in comp.term_ids:
                term = final.extracted_terms.get(term_id)
                if term:
                    print(f"      Term      : {term_id} = {term.value!r}")
                    for ev_id in term.evidence_ids:
                        snippet = final.evidence_registry.get(ev_id)
                        if snippet:
                            import textwrap
                            preview = textwrap.shorten(snippet.text.replace("\n", " "), width=65)
                            print(f"      Evidence  : {ev_id} | Page {snippet.page_number} | \"{preview}\"")
        else:
            print("      Term      : (no matching term extracted)")

        if comp.status == ComplianceStatus.PASS:
            pass_count += 1
        elif comp.status == ComplianceStatus.FAIL:
            fail_count += 1
        elif comp.status == ComplianceStatus.NEEDS_HUMAN_REVIEW:
            review_count += 1
        elif comp.status == ComplianceStatus.INSUFFICIENT_EVIDENCE:
            insuff_count += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    _section("SUMMARY")
    print(f"  Total rules evaluated   : {len(final.compliance_results)}")
    print(f"  PASS                    : {pass_count}")
    print(f"  FAIL                    : {fail_count}")
    print(f"  NEEDS_HUMAN_REVIEW      : {review_count}")
    print(f"  INSUFFICIENT_EVIDENCE   : {insuff_count}")
    print()

    if fail_count > 0:
        print("  [!] Deal has COMPLIANCE FAILURES that require escalation.")
    elif review_count > 0 or insuff_count > 0:
        print("  [!] Deal requires human review before approval.")
    else:
        print("  [+] All evaluated rules PASSED. No compliance issues detected.")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
