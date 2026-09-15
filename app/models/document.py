"""Document models — Module 2.

These models represent the *structural* outcome of loading a PDF deal
document.  They capture provenance metadata and per-page text, but they do
NOT store the raw PDF bytes inside WorkflowState.

Design decisions:
* Both models are frozen (immutable) following the project convention.
* ``DocumentPage.raw_text`` stores the concatenated text extracted by
  PyMuPDF for that page.  The Segmenter then breaks it into EvidenceSnippet
  objects; the page object itself is not stored in WorkflowState.
* ``DocumentMetadata`` is stored in ``WorkflowState.document_metadata``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Provenance record for an ingested deal document.

    Stored in ``WorkflowState.document_metadata`` after successful ingestion.
    The actual PDF is NOT embedded — this is a lightweight descriptor only.

    Attributes:
        document_id:     Canonical deal identifier, e.g. ``DEAL-002``.
        filename:        Original filename, e.g. ``deal_002.pdf``.
        source_path:     Absolute or relative path to the source file.
        file_type:       File format, typically ``"pdf"``.
        page_count:      Total number of pages.
        file_size_bytes: File size in bytes.
        ingested_at:     UTC timestamp of when ingestion was completed.
    """

    document_id: str = Field(
        ...,
        min_length=1,
        description="Canonical deal identifier, e.g. DEAL-002.",
    )
    filename: str = Field(
        ...,
        min_length=1,
        description="Original filename of the source document.",
    )
    source_path: str = Field(
        ...,
        min_length=1,
        description="Absolute or relative path to the source PDF file.",
    )
    file_type: str = Field(
        default="pdf",
        min_length=1,
        description="File format (e.g. 'pdf').",
    )
    page_count: int = Field(
        ...,
        ge=1,
        description="Total number of pages in the document.",
    )
    file_size_bytes: int = Field(
        ...,
        ge=0,
        description="File size in bytes.",
    )
    ingested_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when ingestion completed.",
    )

    model_config = {"frozen": True}


class DocumentPage(BaseModel):
    """Representation of a single page extracted from a source PDF.

    This model is an intermediate artefact produced by the PDF loader.
    It is consumed by the Segmenter to produce EvidenceSnippet objects.
    It is NOT stored in WorkflowState — only the resulting EvidenceSnippets are.

    Attributes:
        document_id:           Canonical deal identifier.
        page_number:           1-indexed page number (human-facing).
        raw_text:              Full concatenated text for the page as returned
                               by PyMuPDF.  May be empty for image-only pages.
        block_count:           Number of text blocks detected by PyMuPDF on
                               this page (before filtering).
        has_extractable_text:  False when PyMuPDF returned no meaningful text
                               (e.g. scanned/image-only pages).
    """

    document_id: str = Field(
        ...,
        min_length=1,
        description="Canonical deal identifier.",
    )
    page_number: int = Field(
        ...,
        ge=1,
        description="1-indexed page number.",
    )
    raw_text: str = Field(
        default="",
        description="Full page text as extracted by PyMuPDF.",
    )
    block_count: int = Field(
        default=0,
        ge=0,
        description="Number of text blocks detected on this page.",
    )
    has_extractable_text: bool = Field(
        default=True,
        description=(
            "False when PyMuPDF returned no meaningful text "
            "(e.g. scanned/image-only pages)."
        ),
    )

    model_config = {"frozen": True}
