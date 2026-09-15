"""Evidence segmenter — Module 2.

Responsibility: Convert a ``DocumentPage`` into a list of ``EvidenceSnippet``
objects using PyMuPDF block-level text as input.

Design decisions:
* One EvidenceSnippet per meaningful text block (paragraph-like unit).
* Evidence IDs are assigned by the **caller** (the ingestion pipeline), not
  by this module.  ``segment_page()`` accepts an ``evidence_id_start``
  counter and returns a list of snippets with sequential IDs.
* Section detection is a best-effort heuristic.  We never invent section info.
  If no section can be reliably identified, ``section`` and ``clause`` are
  both None.
* Text normalization is minimal: strip surrounding whitespace, collapse
  excessive internal blank lines, preserve all symbols/numbers/qualifiers.
* Blocks exceeding ``MAX_BLOCK_CHARS`` characters are split on sentence
  boundaries into smaller snippets so no single snippet is excessively large.
* Blank or whitespace-only blocks are silently skipped.
* A page with no usable text produces an empty list of snippets (not an error).

Known limitations:
* Section detection is heuristic and will miss non-standard heading formats.
* Very complex multi-column layouts may produce text in unexpected order.
* Table content may not segment cleanly into paragraph-like units.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from app.models.evidence import EvidenceSnippet, SourceType
from app.models.document import DocumentPage

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

# Maximum characters in a single evidence snippet before it is split.
MAX_BLOCK_CHARS: int = 1_500

# Minimum characters in a block before it is worth creating a snippet.
MIN_BLOCK_CHARS: int = 10

# ── Section detection patterns ─────────────────────────────────────────────────
# Listed from most specific to least specific.
_SECTION_PATTERNS: list[re.Pattern[str]] = [
    # "Section 4.2 — Interest Rate" or "Section 4.2: Interest Rate"
    re.compile(r"^(Section\s+\d+(?:\.\d+)*(?:\s*[:\-—–]\s*.+)?)", re.IGNORECASE),
    # "Clause 3 — Covenants"
    re.compile(r"^(Clause\s+\d+(?:\.\d+)*(?:\s*[:\-—–]\s*.+)?)", re.IGNORECASE),
    # "Schedule A" / "Annexure B"
    re.compile(r"^(Schedule\s+[A-Z](?:\s*[:\-—–]\s*.+)?)", re.IGNORECASE),
    re.compile(r"^(Annexure\s+[A-Z](?:\s*[:\-—–]\s*.+)?)", re.IGNORECASE),
    # "4.2 — Interest Rate" (numeric prefix with separator)
    re.compile(r"^(\d+(?:\.\d+)+\s*[:\-—–]\s*.+)"),
    # "4.2" alone on first line (standalone numeric heading)
    re.compile(r"^(\d+(?:\.\d+)+)\s*$"),
]


def _detect_section(text: str) -> Optional[str]:
    """Return the section identifier from the first line of text, or None.

    Tries each pattern in order; returns the first match stripped of excess
    whitespace.  Returns None when no pattern matches.
    """
    first_line = text.split("\n")[0].strip()
    for pattern in _SECTION_PATTERNS:
        match = pattern.match(first_line)
        if match:
            return match.group(1).strip()
    return None


def _normalize_text(text: str) -> str:
    """Apply minimal normalization preserving all financial/legal content.

    Operations performed:
    * Strip leading/trailing whitespace.
    * Collapse runs of 3+ blank lines into a double newline.
    * Collapse runs of 3+ spaces within a line into a single space
      (preserves intentional indentation of 1–2 spaces).

    Operations NOT performed:
    * No removal of special characters or symbols.
    * No removal of percentages, currency values, or numeric qualifiers.
    * No rewording or paraphrasing.
    """
    text = text.strip()
    # Collapse excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse runs of 3+ spaces (but not leading spaces — preserve indentation).
    text = re.sub(r"(?<=\S) {3,}", "  ", text)
    return text


def _split_large_block(text: str) -> list[str]:
    """Split ``text`` into chunks each ≤ MAX_BLOCK_CHARS characters.

    Splits preferentially on double-newline paragraph breaks, then on
    sentence-ending periods followed by whitespace.  Falls back to a hard
    character split if necessary.
    """
    if len(text) <= MAX_BLOCK_CHARS:
        return [text]

    # Try paragraph splits first.
    parts = re.split(r"\n\n+", text)
    chunks: list[str] = []
    current = ""
    for part in parts:
        candidate = (current + "\n\n" + part).strip() if current else part
        if len(candidate) <= MAX_BLOCK_CHARS:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # If even a single paragraph is too large, split on sentences.
            if len(part) > MAX_BLOCK_CHARS:
                chunks.extend(_split_on_sentences(part))
                current = ""
            else:
                current = part
    if current:
        chunks.append(current)
    return chunks if chunks else [text[:MAX_BLOCK_CHARS]]


def _split_on_sentences(text: str) -> list[str]:
    """Split text on sentence boundaries, targeting ≤ MAX_BLOCK_CHARS each."""
    # Split on period + whitespace or newline.
    sentences = re.split(r"(?<=\.)\s+", text)
    chunks: list[str] = []
    current = ""
    for sent in sentences:
        candidate = (current + " " + sent).strip() if current else sent
        if len(candidate) <= MAX_BLOCK_CHARS:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = sent
    if current:
        chunks.append(current)
    return chunks if chunks else [text[:MAX_BLOCK_CHARS]]


def segment_page(
    page: DocumentPage,
    evidence_id_start: int,
    source_type: SourceType = SourceType.DEAL_DOCUMENT,
) -> list[EvidenceSnippet]:
    """Segment a ``DocumentPage`` into ordered ``EvidenceSnippet`` objects.

    Evidence IDs start at ``evidence_id_start`` (inclusive) and increment by
    1 for each snippet produced.  The caller must track the running counter
    across pages.

    Args:
        page:              The page to segment.
        evidence_id_start: Starting integer for evidence ID generation (≥ 1).
        source_type:       Source type to assign to all snippets on this page.

    Returns:
        Ordered list of EvidenceSnippet objects.  Empty if the page has no
        usable text.
    """
    if not page.has_extractable_text or not page.raw_text.strip():
        logger.debug(
            "[%s] Page %d skipped — no extractable text.",
            page.document_id,
            page.page_number,
        )
        return []

    # Split raw_text on double-newline (block boundaries from loader).
    raw_blocks = re.split(r"\n\n+", page.raw_text)
    snippets: list[EvidenceSnippet] = []
    counter = evidence_id_start

    for raw_block in raw_blocks:
        normalized = _normalize_text(raw_block)
        if len(normalized) < MIN_BLOCK_CHARS:
            continue  # Skip empty / trivially short blocks.

        # Handle large blocks by splitting into sub-chunks.
        chunks = _split_large_block(normalized)

        for chunk in chunks:
            chunk = chunk.strip()
            if len(chunk) < MIN_BLOCK_CHARS:
                continue

            section = _detect_section(chunk)
            # clause is None when no section can be detected — per spec.
            clause: Optional[str] = section  # same value; kept distinct for clarity

            evidence_id = f"EV-{counter:03d}"
            snippets.append(
                EvidenceSnippet(
                    evidence_id=evidence_id,
                    document_id=page.document_id,
                    page_number=page.page_number,
                    section=section,
                    clause=clause,
                    text=chunk,
                    source_type=source_type,
                )
            )
            counter += 1

    logger.debug(
        "[%s] Page %d → %d evidence snippet(s).",
        page.document_id,
        page.page_number,
        len(snippets),
    )
    return snippets
