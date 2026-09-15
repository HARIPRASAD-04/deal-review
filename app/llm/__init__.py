"""LLM layer package.

Exports:
    LLMClient       — protocol (interface) for all LLM clients
    GoogleLLMClient — concrete implementation backed by Google Gemini
    FakeLLMClient   — deterministic test double; never makes network calls
    get_llm_client  — factory that reads settings and returns the right client
"""

from app.llm.client import FakeLLMClient, GoogleLLMClient, LLMClient, get_llm_client

__all__ = [
    "LLMClient",
    "GoogleLLMClient",
    "FakeLLMClient",
    "get_llm_client",
]
