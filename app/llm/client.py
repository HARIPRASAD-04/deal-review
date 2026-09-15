"""LLM client abstraction — Module 3.

Design principles:
* ``LLMClient`` is a structural protocol (duck-typed interface).  Any object
  implementing ``get_structured_completion`` satisfies it.
* ``GoogleLLMClient`` is the only real provider implemented in Module 3.
  Additional providers (OpenAI, Anthropic) can be added later by implementing
  the same protocol without changing the agent code.
* ``FakeLLMClient`` is the test double used in all automated tests.  It
  returns a pre-configured response and never makes network calls.
* ``get_llm_client(settings)`` is the factory used by production code.  It
  reads the configured provider from settings and constructs the right client.

Provider-specific dependencies (langchain-google-genai) are imported lazily
inside the concrete client so that the base package remains importable even
when the provider SDK is not installed.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Protocol, runtime_checkable

from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ── Protocol ──────────────────────────────────────────────────────────────────

@runtime_checkable
class LLMClient(Protocol):
    """Structural protocol for LLM clients.

    Any object implementing ``get_structured_completion`` satisfies this
    interface — no inheritance required.

    Args:
        system_prompt:   System-level instructions for the model.
        user_prompt:     User-level content (e.g. serialised evidence).
        response_schema: A Pydantic ``BaseModel`` subclass that the model
                         must return.  The client is responsible for
                         instructing the model to produce valid JSON matching
                         this schema.

    Returns:
        An instance of ``response_schema`` populated with the model's output.

    Raises:
        Exception: Any provider-level error (auth, rate-limit, timeout, etc.).
                   Callers should wrap calls in try/except and handle failures
                   explicitly.
    """

    def get_structured_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: type[BaseModel],
    ) -> BaseModel:  # pragma: no cover
        ...


# ── Google Gemini ─────────────────────────────────────────────────────────────

class GoogleLLMClient:
    """LLM client backed by Google Gemini via langchain-google-genai.

    Uses ``with_structured_output`` to request JSON matching a Pydantic schema.
    The provider SDK (langchain-google-genai) is imported lazily so that the
    module remains importable when the SDK is not installed (e.g. in CI without
    the llm-google extra).

    Args:
        api_key:    Google AI Studio API key.
        model_name: Gemini model name, e.g. ``"gemini-2.0-flash"``.
    """

    def __init__(self, api_key: str, model_name: str) -> None:
        if not api_key:
            raise ValueError(
                "GoogleLLMClient requires a non-empty api_key. "
                "Set API_KEY in your .env file."
            )
        if not model_name:
            raise ValueError(
                "GoogleLLMClient requires a non-empty model_name. "
                "Set MODEL_NAME in your .env file."
            )
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "langchain-google-genai is not installed. "
                "Run: pip install 'deal-review[llm-google]'"
            ) from exc

        self._model = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0,   # deterministic extraction
        )
        self._model_name = model_name
        logger.debug("GoogleLLMClient initialized with model '%s'.", model_name)

    def get_structured_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: type[BaseModel],
    ) -> BaseModel:
        """Call Gemini with structured output mode.

        Args:
            system_prompt:   Grounding + extraction instructions.
            user_prompt:     Serialised evidence snippets.
            response_schema: Pydantic model for the expected response shape.

        Returns:
            An instance of ``response_schema`` populated by the model.
        """
        from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore

        structured_model = self._model.with_structured_output(response_schema)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        result = structured_model.invoke(messages)
        logger.debug(
            "GoogleLLMClient received structured response from '%s'.",
            self._model_name,
        )
        return result


# ── Fake client (test double) ─────────────────────────────────────────────────

class FakeLLMClient:
    """Deterministic LLM client for use in automated tests.

    Returns a pre-configured ``response``, sequence of ``responses``, or
    raises a pre-configured ``raise_error``.  Never makes network calls.

    Args:
        response:    The ``BaseModel`` instance to return on every call.
        responses:   Optional sequence of ``BaseModel`` instances to return
                     on consecutive calls.
        raise_error: If set, ``get_structured_completion`` raises this instead
                     of returning a response.
    """

    def __init__(
        self,
        response: Optional[BaseModel] = None,
        responses: Optional[list[BaseModel]] = None,
        raise_error: Optional[Exception] = None,
    ) -> None:
        self._response = response
        self._responses = list(responses) if responses is not None else None
        self._raise_error = raise_error
        self._call_count = 0

    @property
    def call_count(self) -> int:
        """Number of times ``get_structured_completion`` was called."""
        return self._call_count

    def get_structured_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: type[BaseModel],
    ) -> BaseModel:
        self._call_count += 1
        if self._raise_error is not None:
            raise self._raise_error
        if self._responses is not None:
            idx = min(self._call_count - 1, len(self._responses) - 1)
            return self._responses[idx]
        if self._response is None:
            raise ValueError(
                "FakeLLMClient has no configured response and no error. "
                "Pass response= or raise_error= when constructing FakeLLMClient."
            )
        return self._response


# ── Factory ───────────────────────────────────────────────────────────────────

def get_llm_client(settings: Any) -> LLMClient:
    """Construct and return the configured LLM client.

    Reads ``settings.model_provider``, ``settings.api_key``, and
    ``settings.model_name`` to determine which concrete client to create.

    Currently only ``"google"`` is supported.  Additional providers can be
    added here without changing any agent code.

    Args:
        settings: An instance of ``app.config.settings.Settings`` (or any
                  object with ``model_provider``, ``api_key``, and
                  ``model_name`` attributes).

    Returns:
        A concrete ``LLMClient`` instance.

    Raises:
        ValueError: If ``model_provider`` is empty or not supported.
    """
    provider = (settings.model_provider or "").strip().lower()

    if not provider:
        raise ValueError(
            "MODEL_PROVIDER is not configured. "
            "Set MODEL_PROVIDER=google (or another supported provider) in your .env file."
        )

    if provider == "google":
        return GoogleLLMClient(
            api_key=settings.api_key,
            model_name=settings.model_name,
        )

    raise ValueError(
        f"Unsupported MODEL_PROVIDER: '{provider}'. "
        "Currently only 'google' is implemented in Module 3. "
        "To add a new provider, implement the LLMClient protocol in app/llm/client.py."
    )
