"""Unit tests for app/llm/client.py."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from app.llm.client import FakeLLMClient, get_llm_client
from app.llm.schemas import ExtractionOutput, ExtractedTermSchema
from app.models.terms import TermCategory


# ── Helpers ────────────────────────────────────────────────────────────────────

class _SimpleSchema(BaseModel):
    value: str


def _make_simple_output() -> ExtractionOutput:
    return ExtractionOutput(terms=[
        ExtractedTermSchema(
            name="interest_rate",
            value="8.5% per annum",
            category=TermCategory.RATE,
            confidence=0.99,
            evidence_ids=["EV-001"],
        )
    ])


# ── FakeLLMClient ─────────────────────────────────────────────────────────────

class TestFakeLLMClient:
    def test_returns_configured_response(self):
        output = _make_simple_output()
        client = FakeLLMClient(response=output)
        result = client.get_structured_completion(
            system_prompt="sys",
            user_prompt="user",
            response_schema=ExtractionOutput,
        )
        assert result is output

    def test_raises_configured_error(self):
        error = RuntimeError("connection refused")
        client = FakeLLMClient(raise_error=error)
        with pytest.raises(RuntimeError, match="connection refused"):
            client.get_structured_completion("sys", "user", ExtractionOutput)

    def test_call_count_incremented(self):
        client = FakeLLMClient(response=_make_simple_output())
        assert client.call_count == 0
        client.get_structured_completion("sys", "user", ExtractionOutput)
        client.get_structured_completion("sys", "user", ExtractionOutput)
        assert client.call_count == 2

    def test_raises_when_no_response_configured(self):
        client = FakeLLMClient()  # no response, no error
        with pytest.raises(ValueError, match="no configured response"):
            client.get_structured_completion("sys", "user", ExtractionOutput)

    def test_error_mode_raises_before_incrementing_is_fine(self):
        """Confirm error is raised (not swallowed)."""
        client = FakeLLMClient(raise_error=TimeoutError("timed out"))
        with pytest.raises(TimeoutError):
            client.get_structured_completion("sys", "user", ExtractionOutput)


# ── get_llm_client factory ─────────────────────────────────────────────────────

class TestGetLLMClientFactory:
    def _settings(self, provider="", model="", key=""):
        class _S:
            model_provider = provider
            model_name = model
            api_key = key
        return _S()

    def test_empty_provider_raises(self):
        with pytest.raises(ValueError, match="MODEL_PROVIDER is not configured"):
            get_llm_client(self._settings(provider=""))

    def test_unsupported_provider_raises(self):
        with pytest.raises(ValueError, match="Unsupported MODEL_PROVIDER"):
            get_llm_client(self._settings(provider="openai"))

    def test_anthropic_not_supported_yet(self):
        with pytest.raises(ValueError, match="Unsupported MODEL_PROVIDER"):
            get_llm_client(self._settings(provider="anthropic"))

    def test_google_provider_without_sdk_raises_import_error(self):
        """GoogleLLMClient raises ImportError if langchain-google-genai not installed.
        In the test environment it IS installed, so this tests the factory routes
        to GoogleLLMClient correctly (i.e. doesn't raise ValueError)."""
        import importlib
        try:
            importlib.import_module("langchain_google_genai")
            sdk_available = True
        except ImportError:
            sdk_available = False

        if sdk_available:
            # Factory succeeds in creating a GoogleLLMClient.
            # It will raise ValueError if api_key is empty.
            with pytest.raises(ValueError, match="api_key"):
                get_llm_client(self._settings(provider="google", key="", model="gemini-2.0-flash"))
        else:
            # Factory routes to google, GoogleLLMClient raises ImportError.
            with pytest.raises(ImportError, match="langchain-google-genai"):
                get_llm_client(self._settings(provider="google", key="testkey", model="gemini-2.0-flash"))

    def test_google_provider_empty_model_name_raises(self):
        """GoogleLLMClient should raise if model_name is empty."""
        import importlib
        try:
            importlib.import_module("langchain_google_genai")
        except ImportError:
            pytest.skip("langchain-google-genai not installed")
        with pytest.raises(ValueError, match="model_name"):
            get_llm_client(self._settings(provider="google", key="somekey", model=""))
