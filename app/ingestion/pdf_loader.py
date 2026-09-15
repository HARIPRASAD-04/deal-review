"""PDF loader — Module 2.

Responsibility: Open a PDF file using PyMuPDF (``fitz``) and convert it into
structured ``DocumentMetadata`` + a list of ``DocumentPage`` objects.

This module is PURELY deterministic — no LLM calls, no network IO.

Design decisions:
* We return an ``IngestionResult`` dataclass that carries either a successful
  result (metadata + pages) OR a structured error message.  We never raise an
  uncontrolled exception from the public ``load_pdf()`` function; the caller
  decides how to handle failures.
* Page numbers exposed to callers are always **1-indexed** (human-facing).
  PyMuPDF uses 0-based internal indexing; the conversion happens here.
* Block-level text is preserved per page — the Segmenter consumes it.
* We do NOT perform OCR.  Scanned / image-only pages will produce empty text,
  and ``DocumentPage.has_extractable_text`` will be False for those pages.

Known limitations (documented per spec):
* PyMuPDF does not perform OCR.
* Complex multi-column layouts may produce out-of-order text.
* Tables may not always extract with perfect structure.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pymupdf as fitz  # PyMuPDF (pymupdf package; fitz is the legacy alias)

from app.models.document import DocumentMetadata, DocumentPage

logger = logging.getLogger(__name__)

# Minimum printable characters on a page before we consider it meaningful.
_MIN_TEXT_CHARS = 10


@dataclass
class IngestionResult:
    """Result of a PDF ingestion attempt.

    Either ``metadata`` + ``pages`` are populated (success), or ``error`` is
    set (failure).  Both cannot be None at the same time on failure; on
    success, ``error`` is always None.

    Attributes:
        metadata: Populated on success; None on failure.
        pages:    Ordered list of DocumentPage objects; empty on failure.
        error:    Human-readable failure description; None on success.
    """

    metadata: Optional[DocumentMetadata] = None
    pages: list[DocumentPage] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """True when ingestion completed without a fatal error."""
        return self.error is None and self.metadata is not None


def load_pdf(pdf_path: str | Path, document_id: str) -> IngestionResult:
    """Load a PDF file and return structured page representations.

    This is the primary public function of the loader.  It is deterministic:
    the same unchanged PDF will produce the same ``DocumentPage`` sequence.

    Args:
        pdf_path:    Path to the PDF file.
        document_id: Canonical deal identifier, e.g. ``"DEAL-002"``.

    Returns:
        An ``IngestionResult``.  Check ``result.success`` before using
        ``result.metadata`` and ``result.pages``.
    """
    pdf_path = Path(pdf_path)

    # ── Missing file ───────────────────────────────────────────────────────────
    if not pdf_path.exists():
        msg = f"PDF file not found: {pdf_path}"
        logger.warning(msg)
        return IngestionResult(error=msg)

    if not pdf_path.is_file():
        msg = f"Path is not a file: {pdf_path}"
        logger.warning(msg)
        return IngestionResult(error=msg)

    # ── Open PDF ───────────────────────────────────────────────────────────────
    doc: Optional[fitz.Document] = None
    try:
        doc = fitz.open(str(pdf_path))
    except fitz.FileDataError as exc:
        msg = f"Corrupt or unreadable PDF '{pdf_path}': {exc}"
        logger.warning(msg)
        return IngestionResult(error=msg)
    except Exception as exc:  # noqa: BLE001
        msg = f"Unexpected error opening PDF '{pdf_path}': {exc}"
        logger.error(msg, exc_info=True)
        return IngestionResult(error=msg)

    try:
        page_count = len(doc)
        if page_count == 0:
            msg = f"PDF '{pdf_path}' contains zero pages."
            logger.warning(msg)
            return IngestionResult(error=msg)

        file_size = os.path.getsize(pdf_path)
        metadata = DocumentMetadata(
            document_id=document_id,
            filename=pdf_path.name,
            source_path=str(pdf_path.resolve()),
            file_type="pdf",
            page_count=page_count,
            file_size_bytes=file_size,
        )

        pages: list[DocumentPage] = []
        for zero_idx in range(page_count):
            page_number = zero_idx + 1  # 1-indexed for humans
            fitz_page = doc[zero_idx]
            raw_text, block_count = _extract_page_text(fitz_page)
            has_text = len(raw_text.strip()) >= _MIN_TEXT_CHARS
            pages.append(
                DocumentPage(
                    document_id=document_id,
                    page_number=page_number,
                    raw_text=raw_text,
                    block_count=block_count,
                    has_extractable_text=has_text,
                )
            )
            if not has_text:
                logger.debug(
                    "[%s] Page %d has insufficient extractable text (may be image-only).",
                    document_id,
                    page_number,
                )

        logger.info(
            "[%s] Loaded %d pages from '%s'.",
            document_id,
            page_count,
            pdf_path.name,
        )
        return IngestionResult(metadata=metadata, pages=pages)

    finally:
        if doc is not None:
            doc.close()


def _extract_page_text(fitz_page: fitz.Page) -> tuple[str, int]:
    """Extract text blocks from a single PyMuPDF page.

    Uses block-level extraction so the Segmenter can iterate over
    paragraph-like units.  We join blocks with double newlines to mark
    paragraph boundaries.

    Args:
        fitz_page: An open PyMuPDF Page object.

    Returns:
        A tuple of (full_page_text, block_count).
        ``block_count`` counts ALL blocks including image blocks.
    """
    # get_text("blocks") returns: (x0, y0, x1, y1, text, block_no, block_type)
    # block_type: 0 = text, 1 = image
    blocks = fitz_page.get_text("blocks", sort=True)  # sort=True → reading order
    text_blocks = [
        b[4].strip() for b in blocks if b[6] == 0 and b[4].strip()
    ]
    full_text = "\n\n".join(text_blocks)
    return full_text, len(blocks)
