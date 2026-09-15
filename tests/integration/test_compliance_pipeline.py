"""Integration tests — Compliance Review pipeline.

Tests the full LangGraph pipeline ending at review_compliance:
    deal_002.pdf
        -> PyMuPDF
        -> EvidenceRegistry
        -> TermExtractionAgent (FakeLLMClient)
        -> DealTerm[]
        -> ComplianceReviewAgent
        -> ComplianceResult[]
        -> WorkflowState.compliance_results

All LLM calls use FakeLLMClient -- no network, no API key.
Compliance evaluation is fully deterministic (no LLM).
"""

from __future__ import annotations

import pathlib

import pytest

from app.ingestion.evidence_registry import EvidenceRegistry
from app.ingestion.pdf_loader import load_pdf
from app.ingestion.segmenter import segment_page
from app.llm.client import FakeLLMClient
from app.llm.schemas import ExtractionOutput, ExtractedTermSchema
from app.models.audit import AuditEventType
from app.models.compliance import ComplianceStatus
from app.models.evidence import EvidenceSnippet
from app.models.policy import PolicyRule, RuleCategory, RuleSeverity
from app.models.state import AgentStatusEnum, WorkflowState
from app.models.terms import TermCategory
from app.orchestration.graph import build_graph, make_compliance_node

FIXTURE_PDF = pathlib.Path(__file__).parents[2] / "data" / "samples" / "deal_002.pdf"
PDF_EXISTS = FIXTURE_PDF.exists()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_rule(rule_id: str, name: str, operator: str, threshold: str) -> PolicyRule:
    return PolicyRule(
        rule_id=rule_id,
        name=name,
        description=f"Test rule {rule_id}: {name} {operator} {threshold}",
        category=RuleCategory.FINANCIAL,
        operator=operator,
        threshold=threshold,
        severity=RuleSeverity.HIGH,
    )


def _ingest_pdf() -> dict:
    result = load_pdf(str(FIXTURE_PDF), "DEAL-002")
    assert result.success
    registry = EvidenceRegistry()
    counter = 1
    for page in result.pages:
        snippets = segment_page(page, evidence_id_start=counter)
        for s in snippets:
            registry.add(s)
        counter += len(snippets)
    return registry.to_dict()


def _fake_terms(ev_registry: dict) -> ExtractionOutput:
    """Build realistic fake LLM extraction output referencing real EV IDs."""
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
    ])


def _sample_rules() -> list[PolicyRule]:
    """A small but representative set of policy rules for deal_002."""
    return [
        _make_rule("POLICY-001", "interest_rate", "<=", "0.09"),           # PASS (8.5% < 9%)
        _make_rule("POLICY-002", "financial_covenant_dscr", ">=", "1.25"), # PASS (1.25 >= 1.25)
        _make_rule("POLICY-003", "governing_law", "in", "India, Singapore, UK"),  # PASS
        _make_rule("POLICY-004", "principal_amount", "<=", "80000000"),    # PASS (5cr < 8cr)
        _make_rule("POLICY-005", "missing_term", "exists", "true"),        # INSUFFICIENT_EVIDENCE
    ]


# ── Compliance node unit tests (node in isolation) ────────────────────────────

class TestComplianceNodeIsolated:
    """Test make_compliance_node() in isolation — no LangGraph overhead."""

    def _make_state(
        self,
        policy_rules: list[PolicyRule] | None = None,
        extracted_terms: dict | None = None,
    ) -> WorkflowState:
        return WorkflowState(
            document_id="DEAL-TEST",
            policy_rules=policy_rules or [],
            extracted_terms=extracted_terms or {},
        )

    def test_empty_rules_completes_without_failure(self):
        node = make_compliance_node()
        state = self._make_state(policy_rules=[])
        result = node(state)
        assert "compliance_results" in result
        assert result["compliance_results"] == {}

    def test_empty_rules_emits_completed_event(self):
        node = make_compliance_node()
        state = self._make_state(policy_rules=[])
        result = node(state)
        event_types = {e.event_type for e in result["audit_events"]}
        assert AuditEventType.COMPLIANCE_REVIEW_COMPLETED in event_types

    def test_empty_rules_does_not_emit_failed_event(self):
        node = make_compliance_node()
        state = self._make_state(policy_rules=[])
        result = node(state)
        event_types = {e.event_type for e in result["audit_events"]}
        assert AuditEventType.COMPLIANCE_REVIEW_FAILED not in event_types

    def test_compliance_started_event_always_emitted(self):
        node = make_compliance_node()
        state = self._make_state(policy_rules=[])
        result = node(state)
        event_types = [e.event_type for e in result["audit_events"]]
        assert AuditEventType.COMPLIANCE_REVIEW_STARTED in event_types

    def test_compliance_agent_status_completed(self):
        from app.models.handoff import AGENT_COMPLIANCE
        node = make_compliance_node()
        state = self._make_state(policy_rules=[])
        result = node(state)
        compliance_status = result["agent_statuses"].get(AGENT_COMPLIANCE)
        assert compliance_status is not None
        assert compliance_status.status == AgentStatusEnum.COMPLETED


# ── Full LangGraph Pipeline ───────────────────────────────────────────────────

@pytest.mark.skipif(not PDF_EXISTS, reason="deal_002.pdf fixture not found")
class TestFullCompliancePipeline:
    def _initial_state(self, policy_rules: list[PolicyRule] | None = None) -> WorkflowState:
        return WorkflowState(
            document_id="DEAL-002",
            source_pdf_path=str(FIXTURE_PDF),
            policy_rules=policy_rules or [],
        )

    def test_graph_with_zero_policy_rules_completes(self):
        ev_registry = _ingest_pdf()
        fake_llm = FakeLLMClient(response=_fake_terms(ev_registry))
        graph = build_graph(llm_client=fake_llm)
        result = graph.invoke(self._initial_state(policy_rules=[]))
        final = WorkflowState(**result) if isinstance(result, dict) else result
        assert final.compliance_results == {}

    def test_graph_with_rules_populates_compliance_results(self):
        ev_registry = _ingest_pdf()
        fake_llm = FakeLLMClient(response=_fake_terms(ev_registry))
        graph = build_graph(llm_client=fake_llm)
        result = graph.invoke(self._initial_state(policy_rules=_sample_rules()))
        final = WorkflowState(**result) if isinstance(result, dict) else result
        assert len(final.compliance_results) == len(_sample_rules())

    def test_compliance_results_keyed_by_comp_nnn(self):
        ev_registry = _ingest_pdf()
        fake_llm = FakeLLMClient(response=_fake_terms(ev_registry))
        graph = build_graph(llm_client=fake_llm)
        result = graph.invoke(self._initial_state(policy_rules=_sample_rules()))
        final = WorkflowState(**result) if isinstance(result, dict) else result
        import re
        pattern = re.compile(r"^COMP-\d{3,}$")
        for key in final.compliance_results:
            assert pattern.match(key)

    def test_interest_rate_rule_passes(self):
        """8.5% interest rate should PASS the <= 9% rule."""
        ev_registry = _ingest_pdf()
        fake_llm = FakeLLMClient(response=_fake_terms(ev_registry))
        rules = [_make_rule("POLICY-001", "interest_rate", "<=", "0.09")]
        graph = build_graph(llm_client=fake_llm)
        result = graph.invoke(self._initial_state(policy_rules=rules))
        final = WorkflowState(**result) if isinstance(result, dict) else result
        comp = list(final.compliance_results.values())[0]
        assert comp.status == ComplianceStatus.PASS
        assert comp.rule_id == "POLICY-001"

    def test_dscr_rule_passes(self):
        """DSCR of 1.25 should PASS the >= 1.25 rule."""
        ev_registry = _ingest_pdf()
        fake_llm = FakeLLMClient(response=_fake_terms(ev_registry))
        rules = [_make_rule("POLICY-002", "financial_covenant_dscr", ">=", "1.25")]
        graph = build_graph(llm_client=fake_llm)
        result = graph.invoke(self._initial_state(policy_rules=rules))
        final = WorkflowState(**result) if isinstance(result, dict) else result
        comp = list(final.compliance_results.values())[0]
        assert comp.status == ComplianceStatus.PASS

    def test_interest_rate_fails_when_above_max(self):
        """8.5% interest rate should FAIL the <= 8% rule."""
        ev_registry = _ingest_pdf()
        fake_llm = FakeLLMClient(response=_fake_terms(ev_registry))
        rules = [_make_rule("POLICY-001", "interest_rate", "<=", "0.08")]
        graph = build_graph(llm_client=fake_llm)
        result = graph.invoke(self._initial_state(policy_rules=rules))
        final = WorkflowState(**result) if isinstance(result, dict) else result
        comp = list(final.compliance_results.values())[0]
        assert comp.status == ComplianceStatus.FAIL

    def test_missing_term_becomes_insufficient_evidence(self):
        ev_registry = _ingest_pdf()
        fake_llm = FakeLLMClient(response=_fake_terms(ev_registry))
        rules = [_make_rule("POLICY-010", "missing_term", "exists", "true")]
        graph = build_graph(llm_client=fake_llm)
        result = graph.invoke(self._initial_state(policy_rules=rules))
        final = WorkflowState(**result) if isinstance(result, dict) else result
        comp = list(final.compliance_results.values())[0]
        assert comp.status == ComplianceStatus.INSUFFICIENT_EVIDENCE

    def test_compliance_review_audit_events_emitted(self):
        ev_registry = _ingest_pdf()
        fake_llm = FakeLLMClient(response=_fake_terms(ev_registry))
        graph = build_graph(llm_client=fake_llm)
        result = graph.invoke(self._initial_state(policy_rules=_sample_rules()))
        final = WorkflowState(**result) if isinstance(result, dict) else result
        event_types = {e.event_type for e in final.audit_events}
        assert AuditEventType.COMPLIANCE_REVIEW_STARTED in event_types
        assert AuditEventType.COMPLIANCE_REVIEW_COMPLETED in event_types

    def test_full_traceability_chain(self):
        """ComplianceResult -> term_ids -> DealTerm -> evidence_ids -> EvidenceSnippet."""
        ev_registry = _ingest_pdf()
        fake_llm = FakeLLMClient(response=_fake_terms(ev_registry))
        rules = [_make_rule("POLICY-001", "interest_rate", "<=", "0.09")]
        graph = build_graph(llm_client=fake_llm)
        result = graph.invoke(self._initial_state(policy_rules=rules))
        final = WorkflowState(**result) if isinstance(result, dict) else result

        for comp in final.compliance_results.values():
            # Every term_id must be in extracted_terms.
            for term_id in comp.term_ids:
                assert term_id in final.extracted_terms, (
                    f"Compliance result references term '{term_id}' not in extracted_terms."
                )
                term = final.extracted_terms[term_id]
                # Every evidence_id from the term must be in evidence_registry.
                for ev_id in term.evidence_ids:
                    assert ev_id in final.evidence_registry, (
                        f"Term '{term_id}' references evidence '{ev_id}' not in registry."
                    )
                    snippet = final.evidence_registry[ev_id]
                    assert snippet.page_number >= 1
                    assert len(snippet.text.strip()) > 0

    def test_all_compliance_results_have_agent_name(self):
        ev_registry = _ingest_pdf()
        fake_llm = FakeLLMClient(response=_fake_terms(ev_registry))
        graph = build_graph(llm_client=fake_llm)
        result = graph.invoke(self._initial_state(policy_rules=_sample_rules()))
        final = WorkflowState(**result) if isinstance(result, dict) else result
        for comp in final.compliance_results.values():
            assert comp.agent_name == "compliance"

    def test_completed_audit_event_has_summary_metadata(self):
        ev_registry = _ingest_pdf()
        fake_llm = FakeLLMClient(response=_fake_terms(ev_registry))
        graph = build_graph(llm_client=fake_llm)
        result = graph.invoke(self._initial_state(policy_rules=_sample_rules()))
        final = WorkflowState(**result) if isinstance(result, dict) else result
        completed_events = [
            e for e in final.audit_events
            if e.event_type == AuditEventType.COMPLIANCE_REVIEW_COMPLETED
        ]
        assert len(completed_events) == 1
        metadata = completed_events[0].metadata
        assert metadata is not None
        assert "pass_count" in metadata
        assert "fail_count" in metadata
        assert "needs_review_count" in metadata

    def test_no_llm_client_extracts_zero_terms_all_rules_insufficient(self):
        """Without an LLM client, no terms are extracted, so all rules -> INSUFFICIENT_EVIDENCE."""
        rules = [
            _make_rule("POLICY-001", "interest_rate", "<=", "0.09"),
            _make_rule("POLICY-002", "financial_covenant_dscr", ">=", "1.25"),
        ]
        state = WorkflowState(
            document_id="DEAL-002",
            source_pdf_path=str(FIXTURE_PDF),
            policy_rules=rules,
        )
        from app.orchestration.graph import deal_review_graph
        result = deal_review_graph.invoke(state)
        final = WorkflowState(**result) if isinstance(result, dict) else result
        # With no LLM, extracted_terms is empty, so all rules are INSUFFICIENT_EVIDENCE.
        assert len(final.compliance_results) == 2
        for comp in final.compliance_results.values():
            assert comp.status == ComplianceStatus.INSUFFICIENT_EVIDENCE
