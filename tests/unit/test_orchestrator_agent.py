"""Unit tests for OrchestratorAgent — Module 7."""

from unittest.mock import MagicMock

from app.agents.orchestrator.agent import OrchestratorAgent
from app.models.compliance import ComplianceResult, ComplianceStatus
from app.models.evidence import EvidenceSnippet
from app.models.policy import PolicyRule, RuleCategory, RuleSeverity
from app.models.report import ReviewStatus
from app.models.risk import RiskCategory, RiskFinding, RiskSeverity
from app.models.state import AgentStatus, AgentStatusEnum, WorkflowRunStatus, WorkflowState


def _make_snippet(ev_id: str = "EV-001") -> EvidenceSnippet:
    return EvidenceSnippet(
        evidence_id=ev_id,
        document_id="DEAL-001",
        page_number=1,
        text="Sample evidence text.",
        char_start=0,
        char_end=20,
    )


def _make_rule(rule_id: str = "POLICY-001") -> PolicyRule:
    return PolicyRule(
        rule_id=rule_id,
        name="dscr_min",
        description="Checks minimum DSCR.",
        category=RuleCategory.FINANCIAL,
        operator=">=",
        threshold="1.25x",
        severity=RuleSeverity.HIGH,
    )


def _make_state(**kwargs) -> WorkflowState:
    defaults = {
        "run_id": "RUN-TEST1234",
        "document_id": "DEAL-001",
        "evidence_registry": {"EV-001": _make_snippet("EV-001")},
        "policy_rules": [_make_rule("POLICY-001")],
    }
    defaults.update(kwargs)
    return WorkflowState(**defaults)


class TestOrchestratorAgentStateInspection:
    """Test OrchestratorAgent.inspect_state()."""

    def test_inspect_state_summary(self):
        agent = OrchestratorAgent()
        state = _make_state()
        summary = agent.inspect_state(state)

        assert summary["run_id"] == "RUN-TEST1234"
        assert summary["run_status"] == WorkflowRunStatus.INITIALIZED.value
        assert "agents" in summary
        assert summary["pending_handoffs"] == 0
        assert summary["escalations"] == 0


class TestOrchestratorAgentDependencyChecking:
    """Test OrchestratorAgent.check_dependencies()."""

    def test_all_dependencies_met(self):
        agent = OrchestratorAgent()
        state = _make_state()
        issues = agent.check_dependencies(state)
        assert issues == []

    def test_empty_evidence_registry(self):
        agent = OrchestratorAgent()
        state = _make_state(evidence_registry={})
        issues = agent.check_dependencies(state)
        assert len(issues) == 1
        assert "Evidence registry is empty" in issues[0]

    def test_missing_policy_rules(self):
        agent = OrchestratorAgent()
        state = _make_state(policy_rules=[])
        issues = agent.check_dependencies(state)
        assert len(issues) == 1
        assert "No policy rules configured" in issues[0]


class TestOrchestratorAgentRetryPolicy:
    """Test OrchestratorAgent.should_retry()."""

    def test_should_retry_when_failed_under_limit(self):
        agent = OrchestratorAgent(max_retries=2)
        statuses = {"term_extraction": AgentStatus(agent_name="term_extraction").mark_failed("error")}
        state = _make_state(agent_statuses=statuses, retry_counts={"term_extraction": 1})

        assert agent.should_retry("term_extraction", state) is True

    def test_should_not_retry_when_limit_reached(self):
        agent = OrchestratorAgent(max_retries=2)
        statuses = {"term_extraction": AgentStatus(agent_name="term_extraction").mark_failed("error")}
        state = _make_state(agent_statuses=statuses, retry_counts={"term_extraction": 2})

        assert agent.should_retry("term_extraction", state) is False

    def test_should_not_retry_when_not_failed(self):
        agent = OrchestratorAgent(max_retries=2)
        statuses = {"term_extraction": AgentStatus(agent_name="term_extraction").mark_running()}
        state = _make_state(agent_statuses=statuses, retry_counts={"term_extraction": 0})

        assert agent.should_retry("term_extraction", state) is False


class TestOrchestratorAgentHandoffRouting:
    """Test OrchestratorAgent.route_handoffs()."""

    def test_route_handoffs_delegates_to_router(self):
        agent = OrchestratorAgent()
        state = _make_state()

        mock_router = MagicMock()
        mock_router.route.return_value = state
        agent.router = mock_router

        result = agent.route_handoffs(state)
        assert result == state
        mock_router.route.assert_called_once_with(state, llm_client=None, max_attempts=1)


class TestOrchestratorAgentEscalationCollection:
    """Test OrchestratorAgent.collect_escalations()."""

    def test_collect_escalations_aggregates_and_deduplicates(self):
        agent = OrchestratorAgent()

        finding = RiskFinding(
            risk_id="RISK-001",
            title="Unclear DSCR",
            severity=RiskSeverity.HIGH,
            category=RiskCategory.CREDIT,
            description="DSCR calculation ambiguous",
            recommended_action="Verify EBITDA",
            requires_human_review=True,
        )

        state = _make_state(
            escalations=[{"escalation_id": "esc-1", "reason": "Handoff loop limit reached"}],
            risk_findings=[finding],
        )

        escalations = agent.collect_escalations(state)
        assert len(escalations) == 2
        assert escalations[0]["escalation_id"] == "esc-1"
        assert escalations[1]["escalation_id"] == "risk-RISK-001"
        assert escalations[1]["severity"] == "high"


class TestOrchestratorAgentReportAssembly:
    """Test OrchestratorAgent.assemble_report()."""

    def test_assemble_report_completed(self):
        agent = OrchestratorAgent()
        state = _make_state(
            executive_summary="Clean deal summary.",
            risk_findings=[
                RiskFinding(
                    risk_id="RISK-001",
                    title="Minor gap",
                    severity=RiskSeverity.LOW,
                    category=RiskCategory.OPERATIONAL,
                    description="Minor notice",
                    recommended_action="Monitor annually",
                    requires_human_review=False,
                )
            ],
            compliance_results={
                "COMP-001": ComplianceResult(
                    compliance_id="COMP-001",
                    rule_id="POLICY-001",
                    rule_name="Checks minimum DSCR.",
                    status=ComplianceStatus.PASS,
                    rationale="Pass",
                    confidence=1.0,
                    agent_name="compliance_review",
                )
            },
        )

        report = agent.assemble_report(state)
        assert report.review_status == ReviewStatus.COMPLETED
        assert report.executive_summary == "Clean deal summary."
        assert report.compliance_summary.total_rules == 1
        assert report.compliance_summary.pass_count == 1
        assert report.compliance_summary.overall_status == "ALL_PASS"

    def test_assemble_report_completed_with_human_review(self):
        agent = OrchestratorAgent()
        finding = RiskFinding(
            risk_id="RISK-001",
            title="Escalated finding",
            severity=RiskSeverity.HIGH,
            category=RiskCategory.CREDIT,
            description="Needs officer sign-off",
            recommended_action="Obtain credit committee approval",
            requires_human_review=True,
        )
        state = _make_state(risk_findings=[finding])

        report = agent.assemble_report(state)
        assert report.review_status == ReviewStatus.COMPLETED_WITH_HUMAN_REVIEW
        assert "1 finding(s) require human review" in report.review_status_rationale

    def test_assemble_report_failed_on_state_errors(self):
        agent = OrchestratorAgent()
        state = _make_state(errors={"ingest_document": "PDF corrupt"})

        report = agent.assemble_report(state)
        assert report.review_status == ReviewStatus.FAILED
        assert "PDF corrupt" in report.review_status_rationale
