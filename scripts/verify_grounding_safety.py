"""Quick standalone test script for Evidence Grounding Safety (Module 3).

Verifies that if an LLM returns a hallucinated evidence ID (e.g. EV-999),
the Term Extraction Agent rejects the term, logs an audit warning, and does NOT
add the candidate term to extracted_terms.

Usage:
    python scripts/verify_grounding_safety.py
"""

from __future__ import annotations

import pathlib
import sys

# Add project root to sys.path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from app.agents.extraction.term_extraction import TermExtractionAgent
from app.ingestion.evidence_registry import EvidenceRegistry
from app.llm.client import FakeLLMClient
from app.llm.schemas import ExtractedTermSchema, ExtractionOutput
from app.models.evidence import EvidenceSnippet
from app.models.terms import TermCategory


def main() -> None:
    print("\n" + "=" * 64)
    print("  TESTING GROUNDING SAFETY (Hallucinated EV ID Rejection)")
    print("=" * 64)

    # 1. Populate Evidence Registry with ONLY EV-001
    registry = EvidenceRegistry()
    registry.add(
        EvidenceSnippet(
            evidence_id="EV-001",
            document_id="DEAL-001",
            page_number=1,
            text="The loan interest rate shall be 8.5% per annum.",
            char_start=0,
            char_end=48,
        )
    )

    print(f"\n[1] Evidence Registry loaded with: {list(registry.to_dict().keys())}")

    # 2. Simulate LLM returning a candidate term referencing fake EV-999
    fake_output = ExtractionOutput(
        terms=[
            ExtractedTermSchema(
                name="interest_rate",
                value="10%",
                category=TermCategory.RATE,
                confidence=0.95,
                evidence_ids=["EV-999"],  # <-- Hallucinated ID!
            )
        ]
    )

    print("[2] Simulated LLM output candidate: 'interest_rate' referencing ['EV-999']")

    # 3. Execute extraction agent
    agent = TermExtractionAgent(FakeLLMClient(response=fake_output))
    accepted_terms, rejections = agent.extract(registry.to_dict())

    # 4. Display Results
    print("\n" + "-" * 64)
    print("  RESULT SUMMARY")
    print("-" * 64)
    print(f"  Accepted Terms Count : {len(accepted_terms)}")
    print(f"  Rejections Count     : {len(rejections)}")

    if rejections:
        print("\n  Rejection Diagnostic Details:")
        for r in rejections:
            print(f"    [REJECTED] {r}")

    print("-" * 64)
    if len(accepted_terms) == 0 and len(rejections) == 1:
        print("  [SUCCESS] Grounding Safety Guard operational! Fake EV-999 was rejected.\n")
    else:
        print("  [FAILURE] Term was not rejected as expected.\n")


if __name__ == "__main__":
    main()
