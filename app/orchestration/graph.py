"""Minimal LangGraph orchestration graph — Module 1 skeleton.

This module establishes the foundational LangGraph StateGraph that all future
agent nodes will plug into.  In Module 1 the graph is intentionally minimal:

    START → initialize_run → END

Future modules will add nodes for:

* term_extraction   (Module 2)
* compliance_review (Module 3)
* risk_analysis     (Module 4)
* handoff_router    (Module 5)
* retry_handler     (Module 5)
* human_escalation  (Module 5)
* report_assembly   (Module 6)

Conditional routing between those nodes will be determined by the
Orchestrator, which inspects the shared WorkflowState after each step.

WHY LANGGRAPH?
--------------
LangGraph allows us to:
* Define the workflow as an explicit directed graph with typed state.
* Add conditional edges so the Orchestrator can route dynamically.
* Benefit from built-in checkpointing / replay when needed.
* Integrate LLM-backed agent nodes cleanly alongside deterministic nodes.

The graph uses ``WorkflowState`` (a Pydantic model) as its shared state.
LangGraph requires that state updates are returned as *dicts* or *TypedDicts*
by each node, so each node returns only the fields it modified.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.models.audit import AuditEventType, make_audit_event
from app.models.handoff import (
    AGENT_ORCHESTRATOR,
    AGENT_TERM_EXTRACTION,
    AGENT_COMPLIANCE,
    AGENT_RISK_SUMMARY,
)
from app.models.state import AgentStatus, AgentStatusEnum, WorkflowRunStatus, WorkflowState

logger = logging.getLogger(__name__)

# ── Node names — kept as constants to avoid magic strings ─────────────────────

NODE_INITIALIZE = "initialize_run"
NODE_TERM_EXTRACTION = "term_extraction"      # Placeholder — Module 2
NODE_COMPLIANCE_REVIEW = "compliance_review"  # Placeholder — Module 3
NODE_RISK_ANALYSIS = "risk_analysis"          # Placeholder — Module 4
NODE_HANDOFF_ROUTER = "handoff_router"        # Placeholder — Module 5
NODE_HUMAN_ESCALATION = "human_escalation"    # Placeholder — Module 5
NODE_REPORT_ASSEMBLY = "report_assembly"      # Placeholder — Module 6


# ── Helper: default agent statuses ────────────────────────────────────────────

def _initial_agent_statuses() -> dict[str, AgentStatus]:
    """Build the initial per-agent status map with all agents IDLE."""
    return {
        name: AgentStatus(agent_name=name)
        for name in [
            AGENT_ORCHESTRATOR,
            AGENT_TERM_EXTRACTION,
            AGENT_COMPLIANCE,
            AGENT_RISK_SUMMARY,
        ]
    }


# ── Graph Node Implementations ────────────────────────────────────────────────

def initialize_run(state: WorkflowState) -> dict[str, Any]:
    """Initialize the pipeline run.

    Responsibilities:
    * Set run status to INITIALIZED (already set, but explicit here).
    * Initialize per-agent statuses.
    * Record the first audit event.

    Returns only the state fields that this node modifies, as required by
    LangGraph's node contract.
    """
    logger.info("[%s] Initializing pipeline run.", state.run_id)

    start_event = make_audit_event(
        run_id=state.run_id,
        event_type=AuditEventType.RUN_STARTED,
        agent_name=AGENT_ORCHESTRATOR,
        message=f"Pipeline run started for document '{state.document_id}'.",
        metadata={"document_id": state.document_id},
    )

    agent_statuses = _initial_agent_statuses()

    # Mark orchestrator as running.
    agent_statuses[AGENT_ORCHESTRATOR] = agent_statuses[
        AGENT_ORCHESTRATOR
    ].mark_running()

    logger.info(
        "[%s] Run initialized. Agent statuses: %s",
        state.run_id,
        {k: v.status.value for k, v in agent_statuses.items()},
    )

    return {
        "run_status": WorkflowRunStatus.INITIALIZED,
        "agent_statuses": agent_statuses,
        "audit_events": [*state.audit_events, start_event],
    }


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Build and return the LangGraph StateGraph.

    In Module 1 the graph contains only the ``initialize_run`` node.
    Future modules will register additional nodes and conditional edges.

    Returns:
        A compiled LangGraph StateGraph ready for invocation.
    """
    # LangGraph requires a schema for the state it passes between nodes.
    # We use WorkflowState (a Pydantic model) as the schema — LangGraph
    # handles serialisation/deserialisation automatically.
    graph = StateGraph(WorkflowState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node(NODE_INITIALIZE, initialize_run)

    # ── Future nodes (commented — to be registered in later modules) ──────────
    # graph.add_node(NODE_TERM_EXTRACTION,   term_extraction_node)
    # graph.add_node(NODE_COMPLIANCE_REVIEW, compliance_review_node)
    # graph.add_node(NODE_RISK_ANALYSIS,     risk_analysis_node)
    # graph.add_node(NODE_HANDOFF_ROUTER,    handoff_router_node)
    # graph.add_node(NODE_HUMAN_ESCALATION,  human_escalation_node)
    # graph.add_node(NODE_REPORT_ASSEMBLY,   report_assembly_node)

    # ── Edges — Module 1: linear START → initialize → END ────────────────────
    graph.add_edge(START, NODE_INITIALIZE)
    graph.add_edge(NODE_INITIALIZE, END)

    # ── Future conditional edges (to be added in Module 5) ───────────────────
    # graph.add_conditional_edges(
    #     NODE_INITIALIZE,
    #     route_after_init,
    #     {
    #         "extract": NODE_TERM_EXTRACTION,
    #         "end": END,
    #     },
    # )

    return graph.compile()


# ── Module-level compiled graph instance ─────────────────────────────────────

deal_review_graph = build_graph()
"""Pre-compiled LangGraph instance.  Import this and call ``.invoke()`` or
``.stream()`` to execute the pipeline."""
