"""Streamlit UI — Module 8.

Human-readable interface for the Multi-Agent Deal Review Pipeline.

Architecture compliance:
- Presentation only: calls `DealReviewService().run()`.
- Enforces Deal PDF + Policy PDF (no fallback to bundled rules).
- Surfaces policy parse warnings prominently.
- Renders structured report across intuitive, high-density tabs.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st

from app.config.settings import settings
from app.llm.client import FakeLLMClient, get_llm_client
from app.llm.schemas import (
    ExtractionOutput,
    ExtractedTermSchema,
    PolicyExtractionOutput,
    ExtractedPolicyRuleSchema,
    SummaryOutput,
)
from app.models.report import FinalDealReviewReport, ReviewStatus
from app.models.terms import TermCategory
from app.service import DealReviewRequest, DealReviewResponse, DealReviewService


def _get_demo_llm() -> FakeLLMClient:
    """Fallback FakeLLMClient for demo mode when no GOOGLE_API_KEY is present."""
    policy_rules = [
        ExtractedPolicyRuleSchema(
            rule_id="POLICY-001",
            name="interest_rate",
            description="Interest rate must not exceed 9.0% per annum.",
            category="financial",
            operator="<=",
            threshold="0.09",
            severity="high",
        ),
        ExtractedPolicyRuleSchema(
            rule_id="POLICY-002",
            name="financial_covenant_dscr",
            description="Minimum Debt Service Coverage Ratio of 1.25x.",
            category="financial",
            operator=">=",
            threshold="1.25",
            severity="high",
        ),
    ]
    term_schemas = [
        ExtractedTermSchema(
            name="interest_rate",
            value="10.5% per annum",
            normalized_value="0.105",
            category=TermCategory.RATE,
            confidence=0.99,
            evidence_ids=["EV-001"],
        ),
        ExtractedTermSchema(
            name="financial_covenant_dscr",
            value="minimum DSCR of 1.25x",
            normalized_value="1.25",
            category=TermCategory.COVENANT,
            confidence=0.95,
            evidence_ids=["EV-002"],
        ),
    ]

    fake_policy = PolicyExtractionOutput(
        rules=policy_rules,
        ambiguous_statements=["Collateral valuation should be verified independently — no frequency specified."],
    )
    fake_extraction = ExtractionOutput(terms=term_schemas)
    fake_summary = SummaryOutput(
        executive_summary=(
            "The proposed deal terms breach Policy Rule POLICY-001 (Interest Rate Cap). "
            "The extracted interest rate of 10.5% per annum exceeds the maximum allowed threshold of 9.0%. "
            "Financial covenant DSCR meets the required 1.25x minimum."
        )
    )

    class _DemoLLM(FakeLLMClient):
        def __init__(self):
            super().__init__(response=fake_extraction)
            self._p = fake_policy
            self._e = fake_extraction
            self._s = fake_summary

        def get_structured_completion(self, system_prompt, user_prompt, response_schema):
            self._call_count += 1
            if response_schema is PolicyExtractionOutput:
                return self._p
            if response_schema is SummaryOutput:
                return self._s
            return self._e

    return _DemoLLM()


def main() -> None:
    st.set_page_config(
        page_title="Deal Review — Multi-Agent AI",
        page_icon="📋",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS for dark glassmorphism styling
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            color: #f8fafc;
        }
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        .status-badge-completed {
            background-color: #065f46;
            color: #34d399;
            padding: 0.35rem 0.85rem;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.9rem;
        }
        .status-badge-review {
            background-color: #9a3412;
            color: #fb923c;
            padding: 0.35rem 0.85rem;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.9rem;
        }
        .status-badge-failed {
            background-color: #991b1b;
            color: #fca5a5;
            padding: 0.35rem 0.85rem;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.9rem;
        }
        .metric-card {
            background: rgba(30, 41, 59, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='main-header'>📋 Multi-Agent Deal Review System</div>", unsafe_allow_html=True)
    st.caption("Evidence-Grounded Credit Agreement & Compliance Review (Module 8 Final)")

    # Sidebar inputs
    with st.sidebar:
        st.header("Document Ingestion")
        st.info("Both Deal PDF and Policy PDF are **REQUIRED** for review.")

        deal_file = st.file_uploader("Upload Deal PDF (Credit Agreement)", type=["pdf"])
        policy_file = st.file_uploader("Upload Policy PDF (Institutional Rules)", type=["pdf"])
        document_id = st.text_input("Document ID", value="DEAL-001")

        use_demo_llm = st.checkbox("Use Demo Mode (Offline Synthetic LLM)", value=True)

        run_button = st.button("🚀 Run Deal Review", type="primary", use_container_width=True)

    if not run_button:
        st.markdown(
            """
            ### Welcome to the Deal Review System
            Upload a **Deal PDF** and a **Policy PDF** in the sidebar to run an automated, evidence-grounded compliance review.
            
            #### Core Agent Pipeline Architecture:
            1. **Term Extraction Agent** — Extracts material deal terms with evidence grounding.
            2. **Compliance Review Agent** — Evaluates deal terms against custom extracted policy rules.
            3. **Risk & Summary Agent** — Prioritises risk findings and synthesises executive narrative.
            4. **Orchestrator Agent** — Coordinates routing, clarification loops, and report assembly.
            """
        )
        return

    # Form Validation
    if deal_file is None or policy_file is None:
        st.error("⚠️ Both **Deal PDF** and **Policy PDF** must be uploaded before running review.")
        return

    # Save uploaded files temporarily for processing
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        deal_path = tmp_path / deal_file.name
        policy_path = tmp_path / policy_file.name

        deal_path.write_bytes(deal_file.getvalue())
        policy_path.write_bytes(policy_file.getvalue())

        if use_demo_llm:
            llm_client = _get_demo_llm()
        else:
            try:
                llm_client = get_llm_client(settings)
            except Exception as exc:
                st.error(f"❌ Failed to initialize LLM client from .env: {exc}")
                return

        with st.spinner("Running Multi-Agent Deal Review Pipeline..."):
            service = DealReviewService()
            response: DealReviewResponse = service.run(
                DealReviewRequest(
                    deal_pdf_path=str(deal_path),
                    policy_pdf_path=str(policy_path),
                    document_id=document_id,
                    llm_client=llm_client,
                )
            )

    if not response.success or response.report is None:
        st.error(f"❌ Deal Review Failed: {response.error}")
        if response.policy_parse_warnings:
            st.warning("Policy Parsing Warnings:\n" + "\n".join(f"- {w}" for w in response.policy_parse_warnings))
        return

    report: FinalDealReviewReport = response.report
    final_state = response.final_state

    # Policy parse warnings
    if response.policy_parse_warnings:
        with st.expander("⚠️ Policy Parser Warnings (Ambiguous Clauses)", expanded=False):
            for w in response.policy_parse_warnings:
                st.write(f"- {w}")

    # Status Header
    st.subheader(f"Review Report: {report.document_id}")
    status_enum = report.review_status

    if status_enum == ReviewStatus.COMPLETED:
        badge_html = "<span class='status-badge-completed'>COMPLETED — FULL COMPLIANCE</span>"
    elif status_enum == ReviewStatus.COMPLETED_WITH_HUMAN_REVIEW:
        badge_html = "<span class='status-badge-review'>COMPLETED — HUMAN REVIEW REQUIRED</span>"
    else:
        badge_html = "<span class='status-badge-failed'>FAILED — INCOMPLETE REVIEW</span>"

    st.markdown(f"**Overall Pipeline Status:** {badge_html}", unsafe_allow_html=True)
    st.write(f"**Status Rationale:** {report.review_status_rationale}")

    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Rules Evaluated", report.compliance_summary.total_rules)
    with m2:
        st.metric("Passed Rules", report.compliance_summary.pass_count)
    with m3:
        st.metric("Failed Rules", report.compliance_summary.fail_count)
    with m4:
        st.metric("Risk Findings", len(report.risk_findings))

    st.markdown("---")

    # 5 Main Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📄 Executive Summary",
            "⚖️ Compliance Matrix",
            "🔍 Extracted Terms & Evidence",
            "🚨 Risk Findings & Escalations",
            "📊 Audit & Raw JSON",
        ]
    )

    with tab1:
        st.subheader("Executive Summary")
        st.write(report.executive_summary)

        st.subheader("Key Recommendations & Next Steps")
        if report.risk_findings:
            for rf in report.risk_findings:
                st.write(f"- **[{rf.get('severity', 'HIGH').upper()}] {rf.get('title')}**: {rf.get('recommended_action')}")
        else:
            st.success("No high-priority risk findings identified.")

    with tab2:
        st.subheader("Compliance Evaluation Matrix")
        rules_data = []
        if final_state and final_state.compliance_results:
            policy_map = {r.rule_id: r for r in (final_state.policy_rules or [])}
            terms_map = final_state.extracted_terms or {}
            for cr in final_state.compliance_results.values():
                p_rule = policy_map.get(cr.rule_id)
                actual_vals = [terms_map[tid].value for tid in cr.term_ids if tid in terms_map]
                rules_data.append(
                    {
                        "Rule ID": cr.rule_id,
                        "Rule Name": p_rule.name if p_rule else cr.rule_id,
                        "Operator": p_rule.operator if p_rule else "",
                        "Threshold": p_rule.threshold if p_rule else "",
                        "Extracted Value": ", ".join(actual_vals) if actual_vals else "N/A",
                        "Status": cr.status.value if hasattr(cr.status, "value") else str(cr.status),
                        "Severity": p_rule.severity.value if p_rule and hasattr(p_rule.severity, "value") else (str(p_rule.severity) if p_rule else ""),
                        "Rationale": cr.rationale,
                    }
                )
        st.dataframe(rules_data, use_container_width=True)

    with tab3:
        st.subheader("Extracted Deal Terms")
        terms_data = []
        if final_state and final_state.extracted_terms:
            for t in final_state.extracted_terms.values():
                terms_data.append(
                    {
                        "Term ID": t.term_id,
                        "Name": t.name,
                        "Category": t.category.value if hasattr(t.category, "value") else str(t.category),
                        "Raw Value": t.value,
                        "Normalized Value": t.normalized_value or "",
                        "Confidence": t.confidence,
                        "Evidence IDs": ", ".join(t.evidence_ids),
                    }
                )
        st.dataframe(terms_data, use_container_width=True)

    with tab4:
        st.subheader("Prioritised Risk Findings")
        if not report.risk_findings:
            st.info("No risk findings flagged.")
        else:
            for rf in report.risk_findings:
                sev = rf.get("severity", "medium").upper()
                color = "red" if sev in ["CRITICAL", "HIGH"] else "orange"
                st.markdown(f"#### :{color}[{sev}] {rf.get('title')}")
                st.write(f"**Description:** {rf.get('description')}")
                st.write(f"**Recommended Action:** {rf.get('recommended_action')}")
                st.write(f"**Requires Human Review:** `{rf.get('requires_human_review')}`")
                st.markdown("---")

    with tab5:
        st.subheader("Audit Event Trail & Metadata")
        st.write(f"**Run ID:** `{report.run_id}`")
        st.write(f"**Generated At:** `{report.generated_at}`")
        st.write(f"**Audit Event Count:** `{report.audit_event_count}`")

        report_json = report.model_dump_json(indent=2)
        st.download_button(
            label="💾 Download Final Deal Review Report (JSON)",
            data=report_json,
            file_name=f"deal_review_report_{report.document_id}.json",
            mime="application/json",
            type="primary",
        )

        with st.expander("View Raw JSON Report"):
            st.json(report.model_dump())


if __name__ == "__main__":
    main()
