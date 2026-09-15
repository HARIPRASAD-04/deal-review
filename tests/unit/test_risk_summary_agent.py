"""Unit tests for the Risk & Summary Agent — Module 6."""

from __future__ import annotations

import pytest

from app.agents.risk.risk_summary import RiskSummaryAgent
from app.llm.client import FakeLLMClient
from app.llm.schemas import SummaryOutput
from app.models.compliance import ComplianceResult, ComplianceStatus
from app.models.evidence import EvidenceSnippet
from app.models.policy import PolicyRule, RuleCategory, RuleSeverity
from app.models.risk import RiskCategory, RiskSeverity
from app.models.terms import DealTerm, TermCategory, TermStatus


# ── Fixtures ────────────────────────────────────────────────────────────────────

def _make_snippet(ev_id: str = "EV-001", text: str = "Sample evidence text.") -> EvidenceSnippet:
    return EvidenceSnippet(
        evidence_id=ev_id,
        document_id="DEAL-001",
        page_number=1,
        text=text,
        char_start=0,
        char_end=len(text),
    )


def _make_term(
    term_id: str = "TERM-001",
    name: str = "interest_rate",
    value: str = "8.5%",
    status: TermStatus = TermStatus.EXTRACTED,
    confidence: float = 0.9,
    evidence_ids: list[str] | None = None,
) -> DealTerm:
    return DealTerm(
        term_id=term_id,
        name=name,
        value=value,
        category=TermCategory.RATE,
        confidence=confidence,
        status=status,
        evidence_ids=evidence_ids or ["EV-001"],
    )


def _make_rule(
    rule_id: str = "POLICY-001",
    name: str = "interest_rate",
    severity: RuleSeverity = RuleSeverity.HIGH,
    category: RuleCategory = RuleCategory.FINANCIAL,
) -> PolicyRule:
    return PolicyRule(
        rule_id=rule_id,
        name=name,
        description=f"Checks {name}.",
        category=category,
        operator="<=",
        threshold="10%",
        severity=severity,
    )


def _make_comp_result(
    comp_id: str = "COMP-001",
    rule_id: str = "POLICY-001",
    status: ComplianceStatus = ComplianceStatus.FAIL,
    term_ids: list[str] | None = None,
    evidence_ids: list[str] | None = None,
    ambiguity: bool = False,
) -> ComplianceResult:
    return ComplianceResult(
        compliance_id=comp_id,
        rule_id=rule_id,
        status=status,
        rationale=f"Rule {rule_id} evaluated.",
        term_ids=term_ids or [],
        evidence_ids=evidence_ids or [],
        confidence=1.0 if status in (ComplianceStatus.PASS, ComplianceStatus.FAIL) else 0.5,
        agent_name="compliance",
        ambiguity_detected=ambiguity,
    )


def _agent(llm_client=None) -> RiskSummaryAgent:
    return RiskSummaryAgent(llm_client=llm_client)


# ── Source 1: Compliance-based findings ─────────────────────────────────────────

class TestComplianceBasedFindings:
    def test_fail_compliance_produces_risk_finding(self):
        rule = _make_rule(severity=RuleSeverity.HIGH)
        comp = _make_comp_result(status=ComplianceStatus.FAIL, term_ids=["TERM-001"], evidence_ids=["EV-001"])
        ev = {"EV-001": _make_snippet()}
        agent = _agent()
        findings, _, _, _ = agent.analyze(
            compliance_results={"COMP-001": comp},
            extracted_terms={"TERM-001": _make_term()},
            evidence_registry=ev,
            policy_rules=[rule],
            escalations=[],
        )
        assert len(findings) >= 1
        fail_finding = next(f for f in findings if "COMP-001" in f.related_compliance_ids)
        assert fail_finding.severity == RiskSeverity.HIGH
        assert fail_finding.requires_human_review is False

    def test_fail_critical_rule_maps_to_critical_severity(self):
        rule = _make_rule(severity=RuleSeverity.CRITICAL)
        comp = _make_comp_result(status=ComplianceStatus.FAIL)
        agent = _agent()
        findings, _, _, _ = agent.analyze(
            compliance_results={"COMP-001": comp},
            extracted_terms={},
            evidence_registry={},
            policy_rules=[rule],
            escalations=[],
        )
        fail_finding = next((f for f in findings if "COMP-001" in f.related_compliance_ids), None)
        assert fail_finding is not None
        assert fail_finding.severity == RiskSeverity.CRITICAL

    def test_needs_human_review_produces_finding_with_flag(self):
        rule = _make_rule()
        comp = _make_comp_result(status=ComplianceStatus.NEEDS_HUMAN_REVIEW)
        agent = _agent()
        findings, _, _, _ = agent.analyze(
            compliance_results={"COMP-001": comp},
            extracted_terms={},
            evidence_registry={},
            policy_rules=[rule],
            escalations=[],
        )
        assert any(f.requires_human_review for f in findings)

    def test_needs_human_review_ambiguous_adds_missing_info(self):
        rule = _make_rule()
        comp = _make_comp_result(status=ComplianceStatus.NEEDS_HUMAN_REVIEW, ambiguity=True)
        agent = _agent()
        _, missing, _, _ = agent.analyze(
            compliance_results={"COMP-001": comp},
            extracted_terms={},
            evidence_registry={},
            policy_rules=[rule],
            escalations=[],
        )
        assert any("ambiguous" in m.lower() for m in missing)

    def test_insufficient_evidence_produces_finding_and_missing_info(self):
        rule = _make_rule()
        comp = _make_comp_result(status=ComplianceStatus.INSUFFICIENT_EVIDENCE)
        agent = _agent()
        findings, missing, _, _ = agent.analyze(
            compliance_results={"COMP-001": comp},
            extracted_terms={},
            evidence_registry={},
            policy_rules=[rule],
            escalations=[],
        )
        insuf_finding = next((f for f in findings if "COMP-001" in f.related_compliance_ids), None)
        assert insuf_finding is not None
        assert insuf_finding.requires_human_review is True
        assert len(missing) >= 1
        assert any("not found" in m.lower() for m in missing)

    def test_pass_produces_no_finding(self):
        rule = _make_rule()
        comp = _make_comp_result(status=ComplianceStatus.PASS)
        agent = _agent()
        findings, missing, _, _ = agent.analyze(
            compliance_results={"COMP-001": comp},
            extracted_terms={"TERM-001": _make_term()},
            evidence_registry={"EV-001": _make_snippet()},
            policy_rules=[rule],
            escalations=[],
        )
        assert not any("COMP-001" in (f.related_compliance_ids or []) for f in findings)

    def test_skipped_produces_no_finding(self):
        rule = _make_rule()
        comp = _make_comp_result(status=ComplianceStatus.SKIPPED)
        agent = _agent()
        findings, _, _, _ = agent.analyze(
            compliance_results={"COMP-001": comp},
            extracted_terms={},
            evidence_registry={},
            policy_rules=[rule],
            escalations=[],
        )
        assert not any("COMP-001" in (f.related_compliance_ids or []) for f in findings)


# ── Source 2: Term anomaly findings ─────────────────────────────────────────────

class TestTermAnomalyFindings:
    def test_ambiguous_term_not_in_compliance_produces_risk_finding(self):
        # Term not referenced in any compliance result → should appear as anomaly
        term = _make_term(term_id="TERM-001", name="interest_rate", status=TermStatus.AMBIGUOUS)
        agent = _agent()
        findings, missing, _, _ = agent.analyze(
            compliance_results={},
            extracted_terms={"TERM-001": term},
            evidence_registry={"EV-001": _make_snippet()},
            policy_rules=[],
            escalations=[],
        )
        ambig_finding = next((f for f in findings if "TERM-001" in f.related_term_ids), None)
        assert ambig_finding is not None
        assert ambig_finding.requires_human_review is True
        assert any("ambiguous" in m.lower() for m in missing)

    def test_ambiguous_term_already_in_compliance_not_duplicated(self):
        # Term IS referenced in compliance result → anomaly finding should NOT be created (already covered)
        rule = _make_rule()
        comp = _make_comp_result(
            status=ComplianceStatus.NEEDS_HUMAN_REVIEW,
            term_ids=["TERM-001"],
        )
        term = _make_term(term_id="TERM-001", name="interest_rate", status=TermStatus.AMBIGUOUS)
        agent = _agent()
        findings, _, _, _ = agent.analyze(
            compliance_results={"COMP-001": comp},
            extracted_terms={"TERM-001": term},
            evidence_registry={"EV-001": _make_snippet()},
            policy_rules=[rule],
            escalations=[],
        )
        # Only one finding per term — no duplicate
        term_findings = [f for f in findings if "TERM-001" in (f.related_term_ids or [])]
        assert len(term_findings) == 1

    def test_missing_term_status_adds_missing_info_no_finding(self):
        term = _make_term(term_id="TERM-001", status=TermStatus.MISSING)
        agent = _agent()
        _, missing, _, _ = agent.analyze(
            compliance_results={},
            extracted_terms={"TERM-001": term},
            evidence_registry={},
            policy_rules=[],
            escalations=[],
        )
        assert any("MISSING" in m.upper() or "missing" in m.lower() for m in missing)

    def test_low_confidence_term_adds_missing_info(self):
        term = _make_term(term_id="TERM-001", confidence=0.3)
        agent = _agent()
        _, missing, _, _ = agent.analyze(
            compliance_results={},
            extracted_terms={"TERM-001": term},
            evidence_registry={"EV-001": _make_snippet()},
            policy_rules=[],
            escalations=[],
        )
        assert any("low-confidence" in m.lower() or "confidence" in m.lower() for m in missing)


# ── Source 3: Escalation-based findings ─────────────────────────────────────────

class TestEscalationFindings:
    def test_escalation_not_in_compliance_produces_finding(self):
        escalation = {
            "handoff_id": "HDOFF-001",
            "rule_id": "POLICY-999",
            "requested_terms": ["unknown_term"],
            "reason": "Cannot be resolved automatically.",
            "status": "ESCALATION_REQUIRED",
        }
        agent = _agent()
        findings, _, _, _ = agent.analyze(
            compliance_results={},
            extracted_terms={},
            evidence_registry={},
            policy_rules=[],
            escalations=[escalation],
        )
        assert any(f.requires_human_review for f in findings)

    def test_escalation_for_covered_rule_not_duplicated(self):
        rule = _make_rule()
        comp = _make_comp_result(status=ComplianceStatus.INSUFFICIENT_EVIDENCE)
        escalation = {
            "handoff_id": "HDOFF-001",
            "rule_id": "POLICY-001",
            "requested_terms": ["interest_rate"],
            "reason": "Term not found.",
            "status": "ESCALATION_REQUIRED",
        }
        agent = _agent()
        findings, _, _, _ = agent.analyze(
            compliance_results={"COMP-001": comp},
            extracted_terms={},
            evidence_registry={},
            policy_rules=[rule],
            escalations=[escalation],
        )
        policy_findings = [f for f in findings if "COMP-001" in (f.related_compliance_ids or [])]
        assert len(policy_findings) == 1  # Not doubled

    def test_escalation_finding_has_high_severity(self):
        escalation = {
            "handoff_id": "HDOFF-001",
            "rule_id": "POLICY-888",
            "requested_terms": ["some_term"],
            "reason": "Not resolvable.",
            "status": "ESCALATION_REQUIRED",
        }
        agent = _agent()
        findings, _, _, _ = agent.analyze(
            compliance_results={},
            extracted_terms={},
            evidence_registry={},
            policy_rules=[],
            escalations=[escalation],
        )
        esc_finding = next((f for f in findings if f.requires_human_review), None)
        assert esc_finding is not None
        assert esc_finding.severity in (RiskSeverity.HIGH, RiskSeverity.CRITICAL)


# ── Source 4: Coverage gap findings ─────────────────────────────────────────────

class TestCoverageGapFindings:
    def test_empty_evidence_registry_produces_critical_finding(self):
        agent = _agent()
        findings, _, _, _ = agent.analyze(
            compliance_results={},
            extracted_terms={},
            evidence_registry={},
            policy_rules=[],
            escalations=[],
        )
        critical = [f for f in findings if f.severity == RiskSeverity.CRITICAL]
        assert len(critical) >= 1

    def test_evidence_present_but_no_terms_produces_high_finding(self):
        agent = _agent()
        findings, _, _, _ = agent.analyze(
            compliance_results={},
            extracted_terms={},
            evidence_registry={"EV-001": _make_snippet()},
            policy_rules=[],
            escalations=[],
        )
        high_or_above = [f for f in findings if f.severity in (RiskSeverity.HIGH, RiskSeverity.CRITICAL)]
        assert len(high_or_above) >= 1

    def test_evidence_and_terms_present_no_coverage_gap_finding(self):
        term = _make_term()
        agent = _agent()
        findings, _, _, _ = agent.analyze(
            compliance_results={},
            extracted_terms={"TERM-001": term},
            evidence_registry={"EV-001": _make_snippet()},
            policy_rules=[],
            escalations=[],
        )
        # No coverage-gap findings expected when data is present
        assert not any("no document content" in f.description.lower() for f in findings)
        assert not any("no deal terms" in f.description.lower() for f in findings)


# ── Evidence ID validation (no hallucination) ────────────────────────────────────

class TestEvidenceIDValidation:
    def test_evidence_ids_from_compliance_validated_against_registry(self):
        """Only EV IDs that exist in evidence_registry appear in risk findings."""
        ev = {"EV-001": _make_snippet("EV-001")}
        comp = _make_comp_result(
            status=ComplianceStatus.FAIL,
            evidence_ids=["EV-001", "EV-999"],  # EV-999 does not exist
        )
        agent = _agent()
        findings, _, _, _ = agent.analyze(
            compliance_results={"COMP-001": comp},
            extracted_terms={},
            evidence_registry=ev,
            policy_rules=[_make_rule()],
            escalations=[],
        )
        for finding in findings:
            for eid in finding.evidence_ids:
                assert eid in ev, f"Hallucinated evidence ID: {eid}"

    def test_no_evidence_ids_if_none_in_registry(self):
        comp = _make_comp_result(
            status=ComplianceStatus.FAIL,
            evidence_ids=["EV-001"],
        )
        agent = _agent()
        findings, _, _, _ = agent.analyze(
            compliance_results={"COMP-001": comp},
            extracted_terms={},
            evidence_registry={},  # Empty — EV-001 cannot be validated
            policy_rules=[_make_rule()],
            escalations=[],
        )
        fail_finding = next((f for f in findings if "COMP-001" in f.related_compliance_ids), None)
        if fail_finding:
            assert fail_finding.evidence_ids == []


# ── RISK-NNN ID assignment ───────────────────────────────────────────────────────

class TestRiskIDAssignment:
    def test_risk_ids_follow_nnn_pattern(self):
        comp = _make_comp_result(status=ComplianceStatus.FAIL)
        agent = _agent()
        findings, _, _, _ = agent.analyze(
            compliance_results={"COMP-001": comp},
            extracted_terms={},
            evidence_registry={},
            policy_rules=[_make_rule()],
            escalations=[],
        )
        for f in findings:
            assert f.risk_id.startswith("RISK-")
            assert f.risk_id[5:].isdigit()

    def test_critical_findings_assigned_first(self):
        rule_c = _make_rule(rule_id="POLICY-001", severity=RuleSeverity.CRITICAL)
        rule_l = _make_rule(rule_id="POLICY-002", name="other_term", severity=RuleSeverity.LOW)
        comp_c = _make_comp_result(comp_id="COMP-001", rule_id="POLICY-001", status=ComplianceStatus.FAIL)
        comp_l = _make_comp_result(comp_id="COMP-002", rule_id="POLICY-002", status=ComplianceStatus.FAIL)
        agent = _agent()
        findings, _, _, _ = agent.analyze(
            compliance_results={"COMP-001": comp_c, "COMP-002": comp_l},
            extracted_terms={},
            evidence_registry={},
            policy_rules=[rule_c, rule_l],
            escalations=[],
        )
        severities = [f.severity for f in findings]
        assert severities[0] == RiskSeverity.CRITICAL or severities[0] == RiskSeverity.HIGH

    def test_risk_ids_are_sequential_from_001(self):
        comps = {
            f"COMP-{i:03d}": _make_comp_result(
                comp_id=f"COMP-{i:03d}",
                rule_id=f"POLICY-{i:03d}",
                status=ComplianceStatus.FAIL,
            )
            for i in range(1, 4)
        }
        rules = [_make_rule(rule_id=f"POLICY-{i:03d}", name=f"term_{i}") for i in range(1, 4)]
        agent = _agent()
        findings, _, _, _ = agent.analyze(
            compliance_results=comps,
            extracted_terms={},
            evidence_registry={},
            policy_rules=rules,
            escalations=[],
        )
        risk_ids = sorted(f.risk_id for f in findings)
        # IDs must be sequential (though exact order depends on sort)
        for i, rid in enumerate(risk_ids, start=1):
            assert rid == f"RISK-{i:03d}"


# ── Follow-ups ───────────────────────────────────────────────────────────────────

class TestFollowUps:
    def test_follow_ups_non_empty_when_findings_exist(self):
        comp = _make_comp_result(status=ComplianceStatus.FAIL)
        agent = _agent()
        _, _, follow_ups, _ = agent.analyze(
            compliance_results={"COMP-001": comp},
            extracted_terms={},
            evidence_registry={},
            policy_rules=[_make_rule()],
            escalations=[],
        )
        assert len(follow_ups) >= 1

    def test_follow_ups_deduplicated(self):
        # Two rules with same name → same action → deduplicated
        comp1 = _make_comp_result(comp_id="COMP-001", rule_id="POLICY-001", status=ComplianceStatus.FAIL)
        comp2 = _make_comp_result(comp_id="COMP-002", rule_id="POLICY-002", status=ComplianceStatus.FAIL)
        rule1 = _make_rule(rule_id="POLICY-001")
        rule2 = _make_rule(rule_id="POLICY-002", name="interest_rate2")  # Different name, same severity
        agent = _agent()
        _, _, follow_ups, _ = agent.analyze(
            compliance_results={"COMP-001": comp1, "COMP-002": comp2},
            extracted_terms={},
            evidence_registry={},
            policy_rules=[rule1, rule2],
            escalations=[],
        )
        assert len(follow_ups) == len(set(follow_ups))

    def test_empty_findings_means_empty_follow_ups(self):
        comp = _make_comp_result(status=ComplianceStatus.PASS)
        term = _make_term()
        agent = _agent()
        _, _, follow_ups, _ = agent.analyze(
            compliance_results={"COMP-001": comp},
            extracted_terms={"TERM-001": term},
            evidence_registry={"EV-001": _make_snippet()},
            policy_rules=[_make_rule()],
            escalations=[],
        )
        assert follow_ups == []


# ── Executive summary ────────────────────────────────────────────────────────────

class TestExecutiveSummary:
    def test_no_llm_returns_non_empty_deterministic_summary(self):
        comp = _make_comp_result(status=ComplianceStatus.FAIL)
        agent = _agent(llm_client=None)
        _, _, _, summary = agent.analyze(
            compliance_results={"COMP-001": comp},
            extracted_terms={},
            evidence_registry={},
            policy_rules=[_make_rule()],
            escalations=[],
        )
        assert isinstance(summary, str)
        assert len(summary) > 10

    def test_no_findings_deterministic_summary_says_no_risks(self):
        comp = _make_comp_result(status=ComplianceStatus.PASS)
        term = _make_term()
        agent = _agent(llm_client=None)
        _, _, _, summary = agent.analyze(
            compliance_results={"COMP-001": comp},
            extracted_terms={"TERM-001": term},
            evidence_registry={"EV-001": _make_snippet()},
            policy_rules=[_make_rule()],
            escalations=[],
        )
        assert "no risk" in summary.lower() or "no finding" in summary.lower()

    def test_llm_client_summary_used_when_valid(self):
        llm_response = SummaryOutput(
            executive_summary="This deal has one HIGH risk requiring immediate review.",
            key_risks=["Policy violation: interest_rate"],
            recommended_actions=["Escalate to credit committee."],
        )
        fake_llm = FakeLLMClient(response=llm_response)
        comp = _make_comp_result(status=ComplianceStatus.FAIL)
        agent = _agent(llm_client=fake_llm)
        _, _, _, summary = agent.analyze(
            compliance_results={"COMP-001": comp},
            extracted_terms={},
            evidence_registry={},
            policy_rules=[_make_rule()],
            escalations=[],
        )
        assert "HIGH risk" in summary

    def test_llm_failure_falls_back_to_deterministic_summary(self):
        fake_llm = FakeLLMClient(raise_error=RuntimeError("LLM is unavailable"))
        comp = _make_comp_result(status=ComplianceStatus.FAIL)
        agent = _agent(llm_client=fake_llm)
        # Must not raise — should produce fallback summary
        _, _, _, summary = agent.analyze(
            compliance_results={"COMP-001": comp},
            extracted_terms={},
            evidence_registry={},
            policy_rules=[_make_rule()],
            escalations=[],
        )
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_empty_pipeline_no_crash(self):
        agent = _agent()
        findings, missing, follow_ups, summary = agent.analyze(
            compliance_results={},
            extracted_terms={},
            evidence_registry={},
            policy_rules=[],
            escalations=[],
        )
        # Must not raise even with empty everything
        assert isinstance(findings, list)
        assert isinstance(missing, list)
        assert isinstance(follow_ups, list)
        assert isinstance(summary, str)


# ── Missing information deduplication ────────────────────────────────────────────

class TestMissingInformation:
    def test_missing_info_deduplicated(self):
        """Identical missing info entries should not be duplicated."""
        rule = _make_rule()
        comp = _make_comp_result(status=ComplianceStatus.INSUFFICIENT_EVIDENCE)
        agent = _agent()
        _, missing, _, _ = agent.analyze(
            compliance_results={"COMP-001": comp},
            extracted_terms={},
            evidence_registry={},
            policy_rules=[rule],
            escalations=[],
        )
        assert len(missing) == len(set(missing))


# ── LLM Summary Schema Regression Tests ──────────────────────────────────────────

class TestLLMSummarySchemaRegression:
    def test_correct_schema_used_by_fake_llm(self):
        """With FakeLLMClient, Risk & Summary Agent receives/uses SummaryOutput schema."""
        summary_resp = SummaryOutput(
            executive_summary="Deal review identified two material risks requiring human review."
        )
        fake_llm = FakeLLMClient(response=summary_resp)
        agent = RiskSummaryAgent(llm_client=fake_llm)
        rule = _make_rule()
        comp = _make_comp_result(status=ComplianceStatus.FAIL)

        _, _, _, summary = agent.analyze(
            compliance_results={"COMP-001": comp},
            extracted_terms={},
            evidence_registry={},
            policy_rules=[rule],
            escalations=[],
        )

        assert fake_llm.call_count == 1
        assert summary == "Deal review identified two material risks requiring human review."

    def test_llm_summary_reaches_workflow_state(self):
        """Given a successful LLM summary, WorkflowState.executive_summary receives it."""
        from app.models.state import WorkflowRunStatus, WorkflowState
        from app.orchestration.graph import build_graph

        expected_summary = "Deal review identified two material risks requiring human review."
        fake_summary = SummaryOutput(executive_summary=expected_summary)
        fake_llm = FakeLLMClient(response=fake_summary)

        graph = build_graph(llm_client=fake_llm)
        initial_state = WorkflowState(document_id="DEAL-001")

        result = graph.invoke(initial_state)
        final_state = WorkflowState(**result) if isinstance(result, dict) else result

        assert final_state.executive_summary == expected_summary
        assert final_state.run_status == WorkflowRunStatus.COMPLETED

    def test_final_report_remains_untouched(self):
        """state.final_report must remain None after Module 6 completes."""
        from app.models.state import WorkflowState
        from app.orchestration.graph import build_graph

        fake_summary = SummaryOutput(executive_summary="Executive summary test.")
        fake_llm = FakeLLMClient(response=fake_summary)

        graph = build_graph(llm_client=fake_llm)
        initial_state = WorkflowState(document_id="DEAL-001")

        result = graph.invoke(initial_state)
        final_state = WorkflowState(**result) if isinstance(result, dict) else result

        assert final_state.final_report is None

    def test_existing_fallback_still_works_on_llm_exception(self):
        """When LLM raises exception, pipeline does not crash, summary falls back, final_report is None."""
        from app.models.state import WorkflowState
        from app.orchestration.graph import build_graph

        fake_llm = FakeLLMClient(raise_error=RuntimeError("LLM API rate limit exceeded"))
        graph = build_graph(llm_client=fake_llm)
        initial_state = WorkflowState(document_id="DEAL-001")

        result = graph.invoke(initial_state)
        final_state = WorkflowState(**result) if isinstance(result, dict) else result

        assert final_state.executive_summary is not None
        assert isinstance(final_state.executive_summary, str)
        assert len(final_state.executive_summary) > 0
        assert final_state.final_report is None

