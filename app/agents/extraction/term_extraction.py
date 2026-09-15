"""Term Extraction Agent — Module 3.

The Term Extraction Agent is the first LLM-backed agent in the pipeline.

Responsibilities:
* Serialize the evidence registry into a structured text block.
* Call the LLM for structured extraction.
* Validate every returned candidate term:
    - Evidence IDs must exist in the registry.  Invalid IDs → candidate
      rejected, diagnostic reason recorded.
    - Required fields must be present and well-formed (Pydantic enforces this).
* Assign deterministic TERM-NNN IDs sequentially.
* Detect and represent conflicting values (same term name, different values)
  by marking both terms as AMBIGUOUS rather than silently choosing one.
* Merge duplicate terms (same name, same value) into one DealTerm with all
  supporting evidence IDs combined.
* Return the accepted DealTerm list and a list of human-readable rejection
  reasons for diagnostic/audit purposes.

What this agent does NOT do:
* It does not access the PDF.
* It does not evaluate compliance against policy rules.
* It does not assess risk.
* It does not invent terms absent from the evidence.

Dependency direction:
    TermExtractionAgent
          ↓
    LLMClient (protocol — injected via __init__)
          ↓
    EvidenceRegistry (dict[str, EvidenceSnippet] from WorkflowState)
          ↓
    DealTerm[]
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from pydantic import ValidationError

from app.agents.extraction.prompts import SYSTEM_PROMPT, serialize_evidence
from app.llm.client import LLMClient
from app.llm.schemas import ExtractionOutput, ExtractedTermSchema
from app.models.evidence import EvidenceSnippet
from app.models.terms import DealTerm, TermCategory, TermStatus

logger = logging.getLogger(__name__)

_EV_ID_PATTERN: re.Pattern[str] = re.compile(r"^EV-\d{3,}$")


def _is_valid_ev_id(ev_id: str) -> bool:
    """Return True if ev_id matches the EV-NNN+ pattern."""
    return bool(_EV_ID_PATTERN.match(ev_id))


class TermExtractionAgent:
    """Extracts structured DealTerm objects from an evidence registry.

    Args:
        llm_client: An object satisfying the ``LLMClient`` protocol.

    Usage::

        agent = TermExtractionAgent(llm_client=FakeLLMClient(response=...))
        terms, rejections = agent.extract(state.evidence_registry)
    """

    AGENT_NAME: str = "term_extraction"

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    # ── Public interface ───────────────────────────────────────────────────────

    def extract(
        self,
        evidence_registry: dict[str, EvidenceSnippet],
    ) -> tuple[list[DealTerm], list[str]]:
        """Extract material deal terms from the evidence registry.

        Args:
            evidence_registry: The populated evidence registry from WorkflowState
                               (``dict[str, EvidenceSnippet]``, keyed by EV-NNN).

        Returns:
            A 2-tuple:
            * ``accepted_terms``: List of validated ``DealTerm`` objects, ordered
              by TERM-NNN assignment.
            * ``rejection_reasons``: Human-readable strings describing each
              candidate that was rejected and why.  Non-empty when the LLM
              returned terms with invalid evidence IDs or malformed fields.
        """
        # ── Guard: empty registry ──────────────────────────────────────────────
        if not evidence_registry:
            logger.info(
                "[%s] Evidence registry is empty. Returning zero terms.",
                self.AGENT_NAME,
            )
            return [], []

        # ── Serialize evidence for prompt ──────────────────────────────────────
        snippets = list(evidence_registry.values())
        user_prompt = serialize_evidence(snippets)
        logger.debug(
            "[%s] Serialized %d evidence snippet(s) for LLM prompt.",
            self.AGENT_NAME,
            len(snippets),
        )

        # ── Call LLM ──────────────────────────────────────────────────────────
        raw_output: ExtractionOutput = self._llm.get_structured_completion(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_schema=ExtractionOutput,
        )
        logger.info(
            "[%s] LLM returned %d candidate term(s).",
            self.AGENT_NAME,
            len(raw_output.terms),
        )

        # ── Validate, deduplicate, assign IDs ─────────────────────────────────
        accepted, rejections = self._process_candidates(
            candidates=raw_output.terms,
            evidence_registry=evidence_registry,
        )

        logger.info(
            "[%s] Accepted %d term(s); %d candidate(s) rejected.",
            self.AGENT_NAME,
            len(accepted),
            len(rejections),
        )
        return accepted, rejections

    def clarify(
        self,
        evidence_registry: dict[str, EvidenceSnippet],
        requested_term_names: list[str],
        reason: Optional[str] = None,
        start_counter: int = 1,
    ) -> tuple[list[DealTerm], list[str]]:
        """Perform targeted re-extraction for specified terms based on clarification feedback.

        Args:
            evidence_registry: Evidence snippets to search.
            requested_term_names: List of term names specifically requested (e.g. ['interest_rate']).
            reason: Optional explanation of why clarification was requested.
            start_counter: Starting integer index for TERM-NNN assignment.

        Returns:
            A tuple of (accepted_terms, rejection_reasons).
        """
        if not evidence_registry:
            logger.info("[%s.clarify] Evidence registry is empty.", self.AGENT_NAME)
            return [], []

        from app.agents.extraction.prompts import CLARIFICATION_SYSTEM_PROMPT

        snippets = list(evidence_registry.values())
        serialized_evidence = serialize_evidence(snippets)

        terms_str = ", ".join(requested_term_names) if requested_term_names else "unspecified terms"
        user_prompt = (
            f"TARGETED CLARIFICATION REQUEST\n"
            f"Requested Term Name(s): {terms_str}\n"
            f"Reason / Context: {reason or 'Missing or incomplete term requested by compliance review.'}\n\n"
            f"EVIDENCE SNIPPETS:\n"
            f"{serialized_evidence}"
        )

        raw_output: ExtractionOutput = self._llm.get_structured_completion(
            system_prompt=CLARIFICATION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_schema=ExtractionOutput,
        )

        accepted, rejections = self._process_candidates(
            candidates=raw_output.terms,
            evidence_registry=evidence_registry,
        )

        # Adjust term_ids to start from start_counter
        renamed: list[DealTerm] = []
        for i, term in enumerate(accepted):
            new_id = f"TERM-{start_counter + i:03d}"
            renamed.append(term.model_copy(update={"term_id": new_id}))

        logger.info(
            "[%s.clarify] Clarification accepted %d term(s) (starting at TERM-%03d).",
            self.AGENT_NAME,
            len(renamed),
            start_counter,
        )
        return renamed, rejections

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _process_candidates(
        self,
        candidates: list[ExtractedTermSchema],
        evidence_registry: dict[str, EvidenceSnippet],
    ) -> tuple[list[DealTerm], list[str]]:
        """Validate candidates against the registry, merge duplicates, flag conflicts."""
        rejections: list[str] = []

        # Step 1: Validate each candidate's evidence IDs.
        # Collect (schema, merged_evidence_ids) for valid candidates only.
        valid: list[tuple[ExtractedTermSchema, list[str]]] = []

        for candidate in candidates:
            invalid_ids = [
                eid for eid in candidate.evidence_ids
                if not _is_valid_ev_id(eid) or eid not in evidence_registry
            ]
            if invalid_ids:
                reason = (
                    f"Candidate term '{candidate.name}' (value='{candidate.value}') "
                    f"rejected: evidence ID(s) not found in registry: {invalid_ids}. "
                    f"This candidate is NOT added to extracted_terms."
                )
                rejections.append(reason)
                logger.warning("[%s] %s", self.AGENT_NAME, reason)
                continue
            valid.append((candidate, list(candidate.evidence_ids)))

        # Step 2: Merge duplicates and detect conflicts.
        # Key: normalized term name (lowercase stripped).
        # Value: list of (schema, evidence_ids) with that name.
        by_name: dict[str, list[tuple[ExtractedTermSchema, list[str]]]] = {}
        for schema, ev_ids in valid:
            key = schema.name.strip().lower()
            by_name.setdefault(key, []).append((schema, ev_ids))

        # Step 3: Build DealTerm list with sequential IDs.
        accepted: list[DealTerm] = []
        counter = 1

        for key, entries in by_name.items():
            if len(entries) == 1:
                # Simple case — single candidate for this name.
                schema, ev_ids = entries[0]
                term = self._build_term(
                    counter=counter,
                    schema=schema,
                    evidence_ids=ev_ids,
                    status=TermStatus.EXTRACTED,
                )
                accepted.append(term)
                counter += 1
            else:
                # Multiple candidates for the same name.
                unique_values = {e[0].value.strip() for e in entries}
                if len(unique_values) == 1:
                    # Same value — merge into one term with all evidence IDs.
                    merged_ids: list[str] = []
                    for _, ev_ids in entries:
                        for eid in ev_ids:
                            if eid not in merged_ids:
                                merged_ids.append(eid)
                    schema = entries[0][0]
                    term = self._build_term(
                        counter=counter,
                        schema=schema,
                        evidence_ids=merged_ids,
                        status=TermStatus.EXTRACTED,
                    )
                    accepted.append(term)
                    counter += 1
                    logger.debug(
                        "[%s] Merged %d duplicate entries for '%s' → %s.",
                        self.AGENT_NAME,
                        len(entries),
                        key,
                        term.term_id,
                    )
                else:
                    # Conflicting values — preserve both, mark AMBIGUOUS.
                    logger.info(
                        "[%s] Conflicting values detected for '%s': %s. "
                        "Both terms marked AMBIGUOUS.",
                        self.AGENT_NAME,
                        key,
                        sorted(unique_values),
                    )
                    for schema, ev_ids in entries:
                        term = self._build_term(
                            counter=counter,
                            schema=schema,
                            evidence_ids=ev_ids,
                            status=TermStatus.AMBIGUOUS,
                            notes=(
                                (schema.notes + " | " if schema.notes else "")
                                + f"Conflicting values detected for term '{schema.name}'. "
                                  "Both versions preserved for downstream review."
                            ),
                        )
                        accepted.append(term)
                        counter += 1

        return accepted, rejections

    @staticmethod
    def _build_term(
        counter: int,
        schema: ExtractedTermSchema,
        evidence_ids: list[str],
        status: TermStatus,
        notes: Optional[str] = None,
    ) -> DealTerm:
        """Construct a ``DealTerm`` from a validated schema candidate."""
        return DealTerm(
            term_id=f"TERM-{counter:03d}",
            name=schema.name,
            value=schema.value,
            normalized_value=schema.normalized_value,
            category=schema.category,
            confidence=schema.confidence,
            evidence_ids=evidence_ids,
            status=status,
            notes=notes if notes is not None else schema.notes,
        )
