"""Module 5 Demo -- Agent Communication & Compliance → Extraction Feedback Loop.

Demonstrates:
1. Initial term extraction pass leaving a required term missing.
2. Compliance Review Agent evaluating policy rules and detecting INSUFFICIENT_EVIDENCE.
3. Compliance agent creating a structured Handoff (CLARIFICATION_REQUIRED) to term_extraction.
4. HandoffRouter routing the handoff, triggering targeted clarification re-extraction.
5. Re-evaluation of compliance leading to PASS.
6. Loop-prevention and audit trail visualization.
"""

from __future__ import annotations

import logging
from app.agents.compliance.compliance_review import ComplianceReviewAgent
from app.agents.extraction.term_extraction import TermExtractionAgent
from app.agents.orchestrator.router import HandoffRouter
from app.llm.client import FakeLLMClient
from app.llm.schemas import ExtractionOutput, ExtractedTermSchema
from app.models.compliance import ComplianceStatus
from app.models.evidence import EvidenceSnippet
from app.models.handoff import HandoffType
from app.models.policy import PolicyRule, RuleCategory, RuleSeverity
from app.models.state import WorkflowState
from app.models.terms import TermCategory

# Configure logging for demo output
logging.basicConfig(level=logging.INFO, format="  %(message)s")


def run_demo():
    print("\n" + "=" * 76)
    print("  DEAL REVIEW PIPELINE -- Module 5 Feedback Loop & Handoff Routing Demo")
    print("========================================================================\n")

    # 1. Setup Policy Rules
    policy_rules = [
        PolicyRule(
            rule_id="POLICY-001",
            name="interest_rate",
            description="Interest rate must not exceed 10.0% p.a.",
            category=RuleCategory.FINANCIAL,
            operator="<=",
            threshold="10.0%",
            severity=RuleSeverity.HIGH,
        ),
        PolicyRule(
            rule_id="POLICY-002",
            name="leverage_ratio",
            description="Leverage ratio must not exceed 3.5x",
            category=RuleCategory.FINANCIAL,
            operator="<=",
            threshold="3.5x",
            severity=RuleSeverity.HIGH,
        ),
    ]

    # 2. Setup Evidence Registry
    evidence_registry = {
        "EV-001": EvidenceSnippet(
            evidence_id="EV-001",
            document_id="DEAL-002",
            page_number=1,
            text="The Borrower's Leverage Ratio shall not exceed 2.8x at any time.",
            char_start=0,
            char_end=62,
        ),
        "EV-002": EvidenceSnippet(
            evidence_id="EV-002",
            document_id="DEAL-002",
            page_number=2,
            text="The Loan carries a fixed Interest Rate of 8.25% per annum.",
            char_start=0,
            char_end=58,
        ),
    }

    print("------------------------------------------------------------------------")
    print("  STEP 1 -- Initial Term Extraction (Simulated Initial Miss)")
    print("------------------------------------------------------------------------")

    # Initial extraction pass only finds leverage_ratio (misses interest_rate)
    initial_output = ExtractionOutput(
        terms=[
            ExtractedTermSchema(
                name="leverage_ratio",
                value="2.8x",
                category=TermCategory.FINANCIAL,
                confidence=0.95,
                evidence_ids=["EV-001"],
            )
        ]
    )
    extraction_agent = TermExtractionAgent(FakeLLMClient(response=initial_output))
    initial_terms, _ = extraction_agent.extract(evidence_registry)
    extracted_dict = {t.term_id: t for t in initial_terms}

    print(f"  Extracted {len(initial_terms)} term(s): {[t.name for t in initial_terms]}")
    print(f"  Missing term: 'interest_rate'\n")

    # Initialize WorkflowState
    state = WorkflowState(
        document_id="DEAL-002",
        policy_rules=policy_rules,
        evidence_registry=evidence_registry,
        extracted_terms=extracted_dict,
    )

    print("------------------------------------------------------------------------")
    print("  STEP 2 -- Initial Compliance Review & Handoff Generation")
    print("------------------------------------------------------------------------")

    compliance_agent = ComplianceReviewAgent()
    initial_results = compliance_agent.review(
        policy_rules=state.policy_rules,
        extracted_terms=state.extracted_terms,
        evidence_registry=state.evidence_registry,
    )
    comp_dict = {r.compliance_id: r for r in initial_results}

    handoffs = compliance_agent.generate_clarification_handoffs(initial_results, policy_rules)
    state = state.model_copy(
        update={
            "compliance_results": comp_dict,
            "pending_handoffs": handoffs,
        }
    )

    for r in initial_results:
        print(f"  Rule '{r.rule_id}' ({r.status.value}): {r.rationale}")

    print(f"\n  Generated {len(handoffs)} handoff(s) to 'term_extraction':")
    for h in handoffs:
        print(f"    - ID: {h.handoff_id} | Type: {h.handoff_type.value} | Target Terms: {h.requested_term_names}")

    print("\n------------------------------------------------------------------------")
    print("  STEP 3 -- HandoffRouter Dispatch & Targeted Clarification Re-extraction")
    print("------------------------------------------------------------------------")

    # Clarification LLM pass finds the missing interest_rate term
    clarification_output = ExtractionOutput(
        terms=[
            ExtractedTermSchema(
                name="interest_rate",
                value="8.25%",
                category=TermCategory.RATE,
                confidence=0.92,
                evidence_ids=["EV-002"],
            )
        ]
    )
    clarification_client = FakeLLMClient(response=clarification_output)

    router = HandoffRouter()
    final_state = router.route(state, llm_client=clarification_client, max_attempts=1)

    print(f"  Pending handoffs after routing : {len(final_state.pending_handoffs)}")
    print(f"  Resolved handoffs after routing: {len(final_state.resolved_handoffs)}")
    print(f"  Total Extracted terms          : {len(final_state.extracted_terms)} ({[t.name for t in final_state.extracted_terms.values()]})")

    print("\n------------------------------------------------------------------------")
    print("  STEP 4 — Final Re-evaluated Compliance Results")
    print("------------------------------------------------------------------------")

    for r in final_state.compliance_results.values():
        print(f"  Rule '{r.rule_id}' ({r.status.value}): {r.rationale}")

    print("\n------------------------------------------------------------------------")
    print("  STEP 5 — Audit Trail Lifecycle Events")
    print("------------------------------------------------------------------------")

    for event in final_state.audit_events:
        safe_msg = event.message.replace("→", "->")
        print(f"  [{event.event_type.value:<28}] {safe_msg}")

    print("\n========================================================================")
    print("  DEMO COMPLETE -- Compliance -> Extraction Feedback Loop Succeeded!")
    print("========================================================================\n")


if __name__ == "__main__":
    run_demo()
