"""Unit tests for DocumentMetadata and DocumentPage models."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.models.document import DocumentMetadata, DocumentPage


class TestDocumentMetadata:
    def _valid(self, **kwargs) -> dict:
        return {
            "document_id": "DEAL-002",
            "filename": "deal_002.pdf",
            "source_path": "/data/samples/deal_002.pdf",
            "page_count": 3,
            "file_size_bytes": 12226,
            **kwargs,
        }

    def test_valid_creation(self):
        meta = DocumentMetadata(**self._valid())
        assert meta.document_id == "DEAL-002"
        assert meta.filename == "deal_002.pdf"
        assert meta.page_count == 3
        assert meta.file_size_bytes == 12226
        assert meta.file_type == "pdf"
        assert isinstance(meta.ingested_at, datetime)

    def test_ingested_at_defaults_to_utc_now(self):
        meta = DocumentMetadata(**self._valid())
        assert meta.ingested_at.tzinfo == timezone.utc

    def test_is_immutable(self):
        meta = DocumentMetadata(**self._valid())
        with pytest.raises(Exception):
            meta.document_id = "DEAL-999"  # type: ignore[misc]

    def test_custom_file_type(self):
        meta = DocumentMetadata(**self._valid(file_type="doc"))
        assert meta.file_type == "doc"

    def test_empty_document_id_raises(self):
        with pytest.raises(ValidationError):
            DocumentMetadata(**self._valid(document_id=""))

    def test_empty_filename_raises(self):
        with pytest.raises(ValidationError):
            DocumentMetadata(**self._valid(filename=""))

    def test_empty_source_path_raises(self):
        with pytest.raises(ValidationError):
            DocumentMetadata(**self._valid(source_path=""))

    def test_page_count_zero_raises(self):
        with pytest.raises(ValidationError):
            DocumentMetadata(**self._valid(page_count=0))

    def test_negative_file_size_raises(self):
        with pytest.raises(ValidationError):
            DocumentMetadata(**self._valid(file_size_bytes=-1))

    def test_zero_file_size_allowed(self):
        meta = DocumentMetadata(**self._valid(file_size_bytes=0))
        assert meta.file_size_bytes == 0

    def test_serialisation_round_trip(self):
        meta = DocumentMetadata(**self._valid())
        restored = DocumentMetadata.model_validate(meta.model_dump())
        assert restored.document_id == meta.document_id
        assert restored.page_count == meta.page_count


class TestDocumentPage:
    def _valid(self, **kwargs) -> dict:
        return {
            "document_id": "DEAL-002",
            "page_number": 1,
            "raw_text": "Interest rate: 8.5% per annum.",
            "block_count": 5,
            "has_extractable_text": True,
            **kwargs,
        }

    def test_valid_creation(self):
        page = DocumentPage(**self._valid())
        assert page.page_number == 1
        assert page.has_extractable_text is True
        assert "8.5%" in page.raw_text

    def test_is_immutable(self):
        page = DocumentPage(**self._valid())
        with pytest.raises(Exception):
            page.page_number = 99  # type: ignore[misc]

    def test_zero_page_number_raises(self):
        with pytest.raises(ValidationError):
            DocumentPage(**self._valid(page_number=0))

    def test_negative_block_count_raises(self):
        with pytest.raises(ValidationError):
            DocumentPage(**self._valid(block_count=-1))

    def test_empty_text_defaults(self):
        page = DocumentPage(document_id="DEAL-002", page_number=2)
        assert page.raw_text == ""
        assert page.has_extractable_text is True

    def test_image_only_page(self):
        page = DocumentPage(
            document_id="DEAL-002",
            page_number=3,
            raw_text="",
            has_extractable_text=False,
        )
        assert page.has_extractable_text is False

    def test_empty_document_id_raises(self):
        with pytest.raises(ValidationError):
            DocumentPage(**self._valid(document_id=""))
