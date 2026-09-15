"""Manual demo — Module 3 Term Extraction with real Google Gemini API.

Run from the project root (requires .env with API credentials configured):

    python scripts/demo_term_extraction.py

The script runs the full pipeline:

    data/samples/deal_002.pdf
        → Document Ingestion (Module 2)
        → Evidence Registry
        → Term Extraction Agent (Google Gemini)
        → DealTerm objects
        → Traceability Report

Manually verify at least 5 extracted terms to confirm:
    DealTerm → evidence_ids → EvidenceSnippet → page_number → source text

Exit code 0 = success, 1 = configuration / extraction error.
"""

from __future__ import annotations

import pathlib
import sys
import textwrap
import time

# Ensure project root is importable when run as a script.
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from app.config.settings import settings
from app.ingestion.evidence_registry import EvidenceRegistry
from app.ingestion.pdf_loader import load_pdf
from app.ingestion.segmenter import segment_page
from app.llm.client import get_llm_client
from app.models.terms import TermStatus
from app.orchestration.graph import build_graph
from app.models.state import WorkflowState


FIXTURE_PDF = pathlib.Path(__file__).parent.parent / "data" / "samples" / "deal_002.pdf"
SEPARATOR = "-" * 72


def _section(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def main() -> int:  # noqa: C901
    print("\n" + "=" * 72)
    print("  DEAL REVIEW PIPELINE -- Module 3 Term Extraction Demo")
    print("  Provider: " + (settings.model_provider or "(not configured)"))
    print("  Model:    " + (settings.model_name or "(not configured)"))
    print("=" * 72)

    # ── Configuration check ────────────────────────────────────────────────────
    if not settings.model_provider or not settings.api_key:
        print("\n[ERROR] LLM provider is not configured.")
        print("  Set MODEL_PROVIDER, MODEL_NAME, and API_KEY in your .env file.")
        return 1

    try:
        llm_client = get_llm_client(settings)
    except (ValueError, ImportError) as exc:
        print(f"\n[ERROR] Cannot create LLM client: {exc}")
        return 1

    # ── Ingestion ──────────────────────────────────────────────────────────────
    _section("STEP 1 -- Document Ingestion")
    if not FIXTURE_PDF.exists():
        print(f"  [ERROR] Fixture not found: {FIXTURE_PDF}")
        return 1

    print(f"  Loading: {FIXTURE_PDF.name}")
    result = load_pdf(str(FIXTURE_PDF), "DEAL-002")
    if not result.success:
        print(f"  [ERROR] Ingestion failed: {result.error}")
        return 1

    registry = EvidenceRegistry()
    counter = 1
    for page in result.pages:
        snippets = segment_page(page, evidence_id_start=counter)
        for s in snippets:
            registry.add(s)
        counter += len(snippets)

    print(f"  Pages: {result.metadata.page_count}")
    print(f"  Evidence snippets: {len(registry)}")
    print(f"  Sample IDs: {sorted(registry)[:5]}")

    # ── Term Extraction ────────────────────────────────────────────────────────
    _section("STEP 2 -- Term Extraction (LLM Call)")
    print("  Calling LLM for structured extraction ...")
    t0 = time.perf_counter()

    state = WorkflowState(
        document_id="DEAL-002",
        source_pdf_path=str(FIXTURE_PDF),
    )
    graph = build_graph(llm_client=llm_client)

    try:
        raw_result = graph.invoke(state)
    except Exception as exc:  # noqa: BLE001
        print(f"  [ERROR] Pipeline failed: {exc}")
        return 1

    elapsed = time.perf_counter() - t0
    final = WorkflowState(**raw_result) if isinstance(raw_result, dict) else raw_result

    if "extract_terms" in final.errors:
        print(f"  [ERROR] Term extraction failed: {final.errors['extract_terms']}")
        return 1

    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"  Terms extracted: {len(final.extracted_terms)}")

    # ── Rejection diagnostics ──────────────────────────────────────────────────
    from app.models.audit import AuditEventType
    rejections = [
        e for e in final.audit_events
        if e.event_type == AuditEventType.GENERIC and "[WARNING]" in e.message
    ]
    if rejections:
        _section(f"WARNINGS -- {len(rejections)} Candidate(s) Rejected")
        for r in rejections:
            print(f"  [!]  {r.message}")

    # ── Traceability Report ────────────────────────────────────────────────────
    _section(f"STEP 3 -- Traceability Report ({len(final.extracted_terms)} terms)")

    if not final.extracted_terms:
        print("  No terms were extracted from this document.")
        return 0

    for term in sorted(final.extracted_terms.values(), key=lambda t: t.term_id):
        status_marker = "[!] AMBIGUOUS" if term.status == TermStatus.AMBIGUOUS else "[+]"
        conf_str = f"{term.confidence:.0%}"
        print(f"\n  {status_marker} {term.term_id}  |  {term.name}  |  {conf_str} confidence")
        print(f"      Category : {term.category.value}")
        print(f"      Value    : {term.value}")
        if term.normalized_value:
            print(f"      Normalzd : {term.normalized_value}")
        if term.notes:
            print(f"      Notes    : {textwrap.shorten(term.notes, width=80)}")

        print(f"      Evidence :")
        for ev_id in term.evidence_ids:
            snippet = final.evidence_registry.get(ev_id)
            if snippet:
                section_str = f"Section: {snippet.section}  |  " if snippet.section else ""
                print(f"        {ev_id}  |  Page {snippet.page_number}  |  {section_str}")
                preview = textwrap.shorten(snippet.text.replace("\n", " "), width=72)
                print(f"          \"{preview}\"")
            else:
                print(f"        {ev_id}  ← [MISSING IN REGISTRY — should not happen]")

    # ── Summary ────────────────────────────────────────────────────────────────
    _section("SUMMARY")
    ambiguous = [t for t in final.extracted_terms.values() if t.status == TermStatus.AMBIGUOUS]
    print(f"  Total terms extracted : {len(final.extracted_terms)}")
    print(f"  Ambiguous terms       : {len(ambiguous)}")
    print(f"  Candidate rejections  : {len(rejections)}")
    print(f"  LLM call time         : {elapsed:.1f}s")
    print(f"  Evidence registry     : {len(final.evidence_registry)} snippets")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
