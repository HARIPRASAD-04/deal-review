"""Handoff Router — Module 5.

The HandoffRouter is the orchestrator coordinator that routes handoffs between
agents and manages the Compliance → Extraction feedback loop.

Responsibilities:
- Inspect WorkflowState.pending_handoffs for pending requests.
- Track attempt counts per rule_id using WorkflowState.clarification_attempts
  to prevent infinite loops.
- If loop limit reached (attempt count >= max_attempts):
    - Escalate to human review.
    - Mark handoff as RESOLVED with escalation details.
    - Emit LOOP_LIMIT_REACHED audit event.
- If under loop limit:
    - Dispatch to TermExtractionAgent.clarify() for targeted re-extraction.
    - If terms are found:
        - Update state.extracted_terms.
        - Dispatch to ComplianceReviewAgent.review_rule() for compliance recheck.
        - Update state.compliance_results with new result.
        - Emit CLARIFICATION_COMPLETED and COMPLIANCE_RECHECK_COMPLETED events.
    - If terms are not found:
        - Emit CLARIFICATION_FAILED event.
    - Mark handoff as RESOLVED in state.resolved_handoffs.
- Emit HANDOFF_ROUTING_STARTED and HANDOFF_ROUTING_COMPLETED audit events.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.agents.compliance.compliance_review import ComplianceReviewAgent
from app.agents.extraction.term_extraction import TermExtractionAgent
from app.llm.client import LLMClient
from app.models.audit import AuditEventType, make_audit_event
from app.models.handoff import Handoff, HandoffStatus
from app.models.state import WorkflowState

logger = logging.getLogger(__name__)


class HandoffRouter:
    """Coordinator that routes agent handoffs and manages feedback loops.

    The router coordinates agent execution without embedding extraction or
    compliance evaluation logic directly.
    """

    AGENT_NAME: str = "orchestrator_router"

    def route(
        self,
        state: WorkflowState,
        llm_client: Optional[LLMClient] = None,
        max_attempts: int = 1,
    ) -> WorkflowState:
        """Route pending handoffs in state and manage clarification loops.

        Args:
            state: The current WorkflowState.
            llm_client: Optional LLMClient to pass to TermExtractionAgent.
            max_attempts: Maximum clarification attempts per rule_id (default=1).

        Returns:
            Updated WorkflowState.
        """
        pending = list(state.pending_handoffs)
        if not pending:
            logger.debug("[%s] No pending handoffs to route.", self.AGENT_NAME)
            return state

        current_state = state.add_audit_event(
            make_audit_event(
                run_id=state.run_id,
                event_type=AuditEventType.HANDOFF_ROUTING_STARTED,
                message=f"Handoff routing started with {len(pending)} pending handoff(s).",
                agent_name=self.AGENT_NAME,
                metadata={"pending_count": len(pending)},
            )
        )

        remaining_pending: list[Handoff] = []
        resolved_handoffs: list[Handoff] = list(current_state.resolved_handoffs)
        extracted_terms = dict(current_state.extracted_terms)
        compliance_results = dict(current_state.compliance_results)
        clarification_attempts = dict(current_state.clarification_attempts)
        escalations = list(current_state.escalations)

        for handoff in pending:
            if handoff.target_agent == "term_extraction":
                (
                    current_state,
                    extracted_terms,
                    compliance_results,
                    clarification_attempts,
                    escalations,
                    resolved,
                ) = self._process_clarification_handoff(
                    handoff=handoff,
                    state=current_state,
                    extracted_terms=extracted_terms,
                    compliance_results=compliance_results,
                    clarification_attempts=clarification_attempts,
                    escalations=escalations,
                    llm_client=llm_client,
                    max_attempts=max_attempts,
                )
                resolved_handoffs.append(resolved)
            elif handoff.target_agent == "human" or handoff.handoff_type.value == "ESCALATION_REQUIRED":
                rule_id = handoff.rule_ids[0] if handoff.rule_ids else "UNKNOWN_RULE"
                escalations.append({
                    "handoff_id": handoff.handoff_id,
                    "rule_id": rule_id,
                    "requested_terms": handoff.requested_term_names,
                    "reason": handoff.reason,
                    "status": "ESCALATION_REQUIRED",
                })
                resolved_handoffs.append(
                    handoff.model_copy(
                        update={
                            "status": HandoffStatus.RESOLVED,
                            "resolution_notes": (
                                "Escalated directly to human review (no supporting evidence found in document)."
                            ),
                        }
                    )
                )
            else:
                remaining_pending.append(handoff)

        updated_state = current_state.model_copy(
            update={
                "pending_handoffs": remaining_pending,
                "resolved_handoffs": resolved_handoffs,
                "extracted_terms": extracted_terms,
                "compliance_results": compliance_results,
                "clarification_attempts": clarification_attempts,
                "escalations": escalations,
            }
        )

        final_state = updated_state.add_audit_event(
            make_audit_event(
                run_id=state.run_id,
                event_type=AuditEventType.HANDOFF_ROUTING_COMPLETED,
                message=(
                    f"Handoff routing completed. "
                    f"{len(resolved_handoffs) - len(state.resolved_handoffs)} handoff(s) resolved."
                ),
                agent_name=self.AGENT_NAME,
                metadata={"resolved_count": len(resolved_handoffs)},
            )
        )

        return final_state

    def _process_clarification_handoff(
        self,
        handoff: Handoff,
        state: WorkflowState,
        extracted_terms: dict,
        compliance_results: dict,
        clarification_attempts: dict,
        escalations: list,
        llm_client: Optional[LLMClient],
        max_attempts: int,
    ) -> tuple[WorkflowState, dict, dict, dict, list, Handoff]:
        """Process a single clarification handoff targeting term_extraction."""
        rule_id = handoff.rule_ids[0] if handoff.rule_ids else "UNKNOWN_RULE"
        attempts = clarification_attempts.get(rule_id, 0)

        # Loop limit check
        if attempts >= max_attempts:
            logger.warning(
                "[%s] Loop limit reached for rule %s (attempts=%d, max=%d). Escalating to human.",
                self.AGENT_NAME,
                rule_id,
                attempts,
                max_attempts,
            )
            state = state.add_audit_event(
                make_audit_event(
                    run_id=state.run_id,
                    event_type=AuditEventType.LOOP_LIMIT_REACHED,
                    message=(
                        f"Clarification attempt limit ({max_attempts}) reached for rule '{rule_id}'. "
                        f"Escalating to human review."
                    ),
                    agent_name=self.AGENT_NAME,
                    metadata={
                        "rule_id": rule_id,
                        "attempts": attempts,
                        "handoff_id": handoff.handoff_id,
                    },
                )
            )
            escalations.append({
                "handoff_id": handoff.handoff_id,
                "rule_id": rule_id,
                "requested_terms": handoff.requested_term_names,
                "reason": (
                    f"Clarification attempt limit ({max_attempts}) reached. "
                    f"Missing term could not be resolved automatically."
                ),
                "status": "ESCALATION_REQUIRED",
            })
            resolved_handoff = handoff.model_copy(
                update={
                    "status": HandoffStatus.RESOLVED,
                    "resolution_notes": (
                        f"Max clarification attempts ({max_attempts}) reached for rule {rule_id}. "
                        f"Escalated to human review."
                    ),
                }
            )
            return (
                state,
                extracted_terms,
                compliance_results,
                clarification_attempts,
                escalations,
                resolved_handoff,
            )

        # Increment clarification attempt counter
        clarification_attempts[rule_id] = attempts + 1

        state = state.add_audit_event(
            make_audit_event(
                run_id=state.run_id,
                event_type=AuditEventType.CLARIFICATION_STARTED,
                message=(
                    f"Targeted clarification started for rule '{rule_id}' "
                    f"(attempt {attempts + 1}/{max_attempts})."
                ),
                agent_name=self.AGENT_NAME,
                metadata={
                    "rule_id": rule_id,
                    "requested_term_names": handoff.requested_term_names,
                    "attempt": attempts + 1,
                },
            )
        )

        # Perform targeted extraction via TermExtractionAgent if llm_client is available
        new_terms = []
        if llm_client is not None:
            extraction_agent = TermExtractionAgent(llm_client=llm_client)
            start_counter = len(extracted_terms) + 1
            new_terms, _ = extraction_agent.clarify(
                evidence_registry=state.evidence_registry,
                requested_term_names=handoff.requested_term_names,
                reason=handoff.reason,
                start_counter=start_counter,
            )

        if new_terms:
            for term in new_terms:
                extracted_terms[term.term_id] = term

            state = state.add_audit_event(
                make_audit_event(
                    run_id=state.run_id,
                    event_type=AuditEventType.CLARIFICATION_COMPLETED,
                    message=f"Targeted clarification found {len(new_terms)} new term(s) for rule '{rule_id}'.",
                    agent_name=self.AGENT_NAME,
                    metadata={"new_terms": [t.name for t in new_terms], "rule_id": rule_id},
                )
            )

            # Perform immediate compliance recheck for the affected rule via ComplianceReviewAgent
            rule = next((r for r in state.policy_rules if r.rule_id == rule_id), None)
            if rule:
                state = state.add_audit_event(
                    make_audit_event(
                        run_id=state.run_id,
                        event_type=AuditEventType.COMPLIANCE_RECHECK_STARTED,
                        message=f"Re-checking compliance for rule '{rule_id}' after clarification.",
                        agent_name=self.AGENT_NAME,
                        metadata={"rule_id": rule_id},
                    )
                )

                comp_id = "COMP-999"
                for res in compliance_results.values():
                    if res.rule_id == rule_id:
                        comp_id = res.compliance_id
                        break

                compliance_agent = ComplianceReviewAgent()
                updated_result = compliance_agent.review_rule(
                    rule=rule,
                    extracted_terms=extracted_terms,
                    evidence_registry=state.evidence_registry,
                    comp_id=comp_id,
                )
                compliance_results[comp_id] = updated_result

                state = state.add_audit_event(
                    make_audit_event(
                        run_id=state.run_id,
                        event_type=AuditEventType.COMPLIANCE_RECHECK_COMPLETED,
                        message=f"Compliance re-check for rule '{rule_id}' completed → {updated_result.status.value}.",
                        agent_name=self.AGENT_NAME,
                        metadata={"rule_id": rule_id, "new_status": updated_result.status.value},
                    )
                )
            resolution_notes = (
                f"Clarification succeeded: extracted {len(new_terms)} term(s) and re-evaluated compliance."
            )
        else:
            state = state.add_audit_event(
                make_audit_event(
                    run_id=state.run_id,
                    event_type=AuditEventType.CLARIFICATION_FAILED,
                    message=f"Targeted clarification returned no terms for rule '{rule_id}'.",
                    agent_name=self.AGENT_NAME,
                    metadata={"rule_id": rule_id},
                )
            )
            resolution_notes = f"Clarification returned no new terms for rule '{rule_id}'."

        resolved_handoff = handoff.model_copy(
            update={
                "status": HandoffStatus.RESOLVED,
                "resolution_notes": resolution_notes,
            }
        )

        return (
            state,
            extracted_terms,
            compliance_results,
            clarification_attempts,
            escalations,
            resolved_handoff,
        )
