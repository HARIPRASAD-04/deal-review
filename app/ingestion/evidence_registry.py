"""Evidence Registry — Module 2.

An in-memory, dict-backed registry that serves as the authoritative
structured representation of document evidence for a pipeline run.

This is NOT a vector store.  It is NOT a semantic retrieval system.
It is a simple provenance layer: Evidence ID → EvidenceSnippet.

Design decisions:
* The registry is a plain Python class wrapping a dict.  No database, no
  persistence, no external dependencies.
* The registry is mutable (unlike the frozen Pydantic models).  The
  ingestion pipeline builds it incrementally, then passes it to
  WorkflowState via ``.to_dict()``.
* Duplicate ``evidence_id`` values are rejected with a ``ValueError``.
* Invalid ``evidence_id`` format (not matching ``EV-\\d{3,}``) is rejected.
* The registry is iterable and supports ``len()``.

Conceptual layout::

    EvidenceRegistry

    EV-001 → page 1 → section "1. PARTIES" → "Borrower: ..."
    EV-002 → page 1 → section None         → "Principal Amount: ..."
    EV-003 → page 2 → section "4.2 — Rate" → "Interest rate: 8.5% p.a."
"""

from __future__ import annotations

import re
import logging
from typing import Iterator, Optional

from app.models.evidence import EvidenceSnippet

logger = logging.getLogger(__name__)

_EV_ID_PATTERN: re.Pattern[str] = re.compile(r"^EV-\d{3,}$")


def _validate_evidence_id(evidence_id: str) -> None:
    """Raise ValueError if evidence_id does not match EV-NNN+ format."""
    if not _EV_ID_PATTERN.match(evidence_id):
        raise ValueError(
            f"Invalid evidence_id format: '{evidence_id}'. "
            f"Expected pattern: EV-NNN (e.g. EV-001, EV-042)."
        )


class EvidenceRegistry:
    """In-memory evidence registry for a single pipeline run.

    Usage::

        registry = EvidenceRegistry()
        registry.add(snippet)
        snippet = registry.get("EV-001")
        all_snippets = registry.list_all()
        state_dict = registry.to_dict()  # pass to WorkflowState

    Thread safety: NOT thread-safe.  Intended for use within a single
    synchronous pipeline run.
    """

    def __init__(self) -> None:
        self._store: dict[str, EvidenceSnippet] = {}

    # ── Write operations ───────────────────────────────────────────────────────

    def add(self, snippet: EvidenceSnippet) -> None:
        """Add an EvidenceSnippet to the registry.

        Args:
            snippet: The evidence snippet to add.

        Raises:
            ValueError: If the evidence_id is already present in the registry,
                        or if the evidence_id does not match the required format.
        """
        _validate_evidence_id(snippet.evidence_id)
        if snippet.evidence_id in self._store:
            raise ValueError(
                f"Duplicate evidence_id: '{snippet.evidence_id}' is already "
                f"registered in this registry."
            )
        self._store[snippet.evidence_id] = snippet
        logger.debug("Registered %s (page %d).", snippet.evidence_id, snippet.page_number)

    # ── Read operations ────────────────────────────────────────────────────────

    def get(self, evidence_id: str) -> EvidenceSnippet:
        """Return the EvidenceSnippet for the given ID.

        Args:
            evidence_id: The EV-NNN identifier to look up.

        Raises:
            KeyError: If the evidence_id is not in the registry.
        """
        _validate_evidence_id(evidence_id)
        if evidence_id not in self._store:
            raise KeyError(
                f"Evidence ID '{evidence_id}' not found in registry. "
                f"Available: {sorted(self._store.keys())}"
            )
        return self._store[evidence_id]

    def get_or_none(self, evidence_id: str) -> Optional[EvidenceSnippet]:
        """Return the EvidenceSnippet or None if not found.

        Does not raise; useful for soft lookups.
        """
        try:
            _validate_evidence_id(evidence_id)
        except ValueError:
            return None
        return self._store.get(evidence_id)

    def has(self, evidence_id: str) -> bool:
        """Return True if the evidence_id is present in the registry."""
        return evidence_id in self._store

    def list_all(self) -> list[EvidenceSnippet]:
        """Return all evidence snippets in insertion order."""
        return list(self._store.values())

    def list_for_document(self, document_id: str) -> list[EvidenceSnippet]:
        """Return all evidence snippets belonging to a specific document."""
        return [s for s in self._store.values() if s.document_id == document_id]

    def to_dict(self) -> dict[str, EvidenceSnippet]:
        """Return a copy of the internal store suitable for WorkflowState.

        WorkflowState.evidence_registry expects ``dict[str, EvidenceSnippet]``.
        """
        return dict(self._store)

    # ── Container protocol ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._store)

    def __iter__(self) -> Iterator[str]:
        """Iterate over evidence IDs in insertion order."""
        return iter(self._store)

    def __contains__(self, evidence_id: object) -> bool:
        return evidence_id in self._store

    def __repr__(self) -> str:
        return f"EvidenceRegistry({len(self._store)} snippets)"
