"""Unit tests for ComplianceReviewAgent handoff generation and single-rule review — Module 5."""

from __future__ import annotations

import pytest

from app.agents.compliance.compliance_review import ComplianceReviewAgent
from app.models.compliance import ComplianceResult, ComplianceStatus
from app.models.evidence import EvidenceSnippet
from app.models.handoff import AGENT_COMPLIANCE, AGENT_TERM_EXTRACTION, HandoffPriority, HandoffType
from app.models.policy import PolicyRule, RuleCategory, RuleSeverity
from app.models.terms import DealTerm, TermCategory, TermStatus


@pytest.fixture
def agent() -> ComplianceReviewAgent:
    return ComplianceReviewAgent()


@pytest.fixture
def sample_rules() -> list[PolicyRule]:
    return [
        PolicyRule(
            rule_id="POLICY-001",
            name="interest_rate",
            description="Interest rate must not exceed 10%",
            category=RuleCategory.FINANCIAL,
            operator="<=",
            threshold="10%",
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


class TestComplianceHandoffGeneration:
    def test_generate_handoffs_for_insufficient_evidence(
        self, agent: ComplianceReviewAgent, sample_rules: list[PolicyRule]
    ):
        results = [
            ComplianceResult(
                compliance_id="COMP-001",
                rule_id="POLICY-001",
                status=ComplianceStatus.INSUFFICIENT_EVIDENCE,
                rationale="No term named 'interest_rate' found.",
                confidence=0.0,
                agent_name="compliance",
            ),
            ComplianceResult(
                compliance_id="COMP-002",
                rule_id="POLICY-002",
                status=ComplianceStatus.PASS,
                rationale="Leverage ratio is within limits.",
                confidence=1.0,
                agent_name="compliance",
            ),
        ]

        handoffs = agent.generate_clarification_handoffs(results, sample_rules)

        assert len(handoffs) == 1
        h = handoffs[0]
        assert h.source_agent == AGENT_COMPLIANCE
        assert h.target_agent == AGENT_TERM_EXTRACTION
        assert h.handoff_type == HandoffType.CLARIFICATION_REQUIRED
        assert h.rule_ids == ["POLICY-001"]
        assert h.requested_term_names == ["interest_rate"]
        assert h.priority == HandoffPriority.HIGH

    def test_no_handoffs_when_all_rules_pass_or_fail(
        self, agent: ComplianceReviewAgent, sample_rules: list[PolicyRule]
    ):
        results = [
            ComplianceResult(
                compliance_id="COMP-001",
                rule_id="POLICY-001",
                status=ComplianceStatus.PASS,
                rationale="Pass",
                confidence=1.0,
                agent_name="compliance",
            ),
            ComplianceResult(
                compliance_id="COMP-002",
                rule_id="POLICY-002",
                status=ComplianceStatus.FAIL,
                rationale="Fail",
                confidence=1.0,
                agent_name="compliance",
            ),
        ]

        handoffs = agent.generate_clarification_handoffs(results, sample_rules)
        assert handoffs == []

    def test_review_rule_single_rule_evaluation(self, agent: ComplianceReviewAgent, sample_rules: list[PolicyRule]):
        rule = sample_rules[0]
        term = DealTerm(
            term_id="TERM-001",
            name="interest_rate",
            value="8.5%",
            category=TermCategory.RATE,
            confidence=0.9,
            evidence_ids=["EV-001"],
            status=TermStatus.EXTRACTED,
        )
        terms_dict = {"TERM-001": term}

        result = agent.review_rule(
            rule=rule,
            extracted_terms=terms_dict,
            evidence_registry={},
            comp_id="COMP-001",
        )

        assert result.compliance_id == "COMP-001"
        assert result.rule_id == "POLICY-001"
        assert result.status == ComplianceStatus.PASS
        assert result.term_ids == ["TERM-001"]

    def test_insufficient_evidence_without_matching_keywords_escalates_to_human(
        self, agent: ComplianceReviewAgent, sample_rules: list[PolicyRule]
    ):
        results = [
            ComplianceResult(
                compliance_id="COMP-001",
                rule_id="POLICY-001",
                status=ComplianceStatus.INSUFFICIENT_EVIDENCE,
                rationale="Missing term.",
                confidence=0.0,
                agent_name="compliance",
            ),
        ]
        # Evidence registry containing unrelated text only
        evidence_registry = {
            "EV-001": EvidenceSnippet(
                evidence_id="EV-001",
                document_id="DEAL-001",
                page_number=1,
                text="The borrower is a limited liability company.",
                char_start=0,
                char_end=43,
            )
        }

        handoffs = agent.generate_clarification_handoffs(
            results=results,
            policy_rules=sample_rules,
            evidence_registry=evidence_registry,
        )

        assert len(handoffs) == 1
        h = handoffs[0]
        assert h.target_agent == "human"
        assert h.handoff_type == HandoffType.ESCALATION_REQUIRED
