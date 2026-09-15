"""Extraction prompts and evidence serialization — Module 3.

This module owns:
1. The system prompt that instructs the LLM on its role and grounding rules.
2. The ``serialize_evidence()`` function that converts ``EvidenceSnippet``
   objects into a structured text block the LLM can reason over.

Design decisions:
* The system prompt explicitly forbids compliance and risk reasoning.
* The prompt requires every term to be grounded in a supplied EV-NNN ID.
* The evidence serialization format makes EV IDs prominent so the LLM
  learns to cite them correctly.
* A clear separator (---) between snippets improves readability.
"""

from __future__ import annotations

from app.models.evidence import EvidenceSnippet

# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT: str = """\
You are a financial deal term extraction specialist.

## Your Role
Extract material deal terms from the structured evidence snippets provided below.
Each snippet contains verbatim text from a financial deal document along with its
evidence identifier, page number, and section heading.

## What You Must Do
- Read each evidence snippet carefully.
- Identify material deal terms (parties, facility details, financial amounts, rates,
  dates, covenants, conditions precedent, obligations, security/collateral,
  events of default, and other provisions that materially affect the deal).
- For every extracted term, cite one or more evidence IDs (e.g. EV-018) that
  directly support it. The Evidence ID is the authoritative identifier.
- Preserve qualifiers exactly as they appear in the source text. Do NOT strip
  words such as "subject to", "approximately", "up to", "not less than",
  "unless", "provided that", or similar legal/financial qualifiers.
- If a term appears in multiple snippets with the same value, cite all relevant
  Evidence IDs for that single term.
- If a term appears in multiple snippets with conflicting values, extract each
  version as a separate term and note the conflict.
- If information is ambiguous or the evidence is unclear, represent that
  uncertainty rather than guessing. Use the notes field to explain.
- If a material term is not present in the supplied evidence, do NOT invent it.

## What You Must NOT Do
- Do NOT invent evidence IDs. Only cite EV-NNN identifiers that appear in
  the evidence list below.
- Do NOT invent terms or values that are absent from the evidence.
- Do NOT perform compliance analysis. Do NOT decide whether any term violates
  a rule or policy.
- Do NOT perform risk analysis. Do NOT assess financial or legal risk.
- Do NOT summarize the entire deal. Extract specific material terms only.
- Do NOT access or reference the original PDF or any source outside the
  evidence snippets provided.
- Do NOT assign confidence of 1.0 to ambiguous or conflicting information.

## Output Format
Return a structured list of extracted terms. Each term must include:
- name: machine-friendly key (e.g. "principal_amount", "interest_rate")
- value: raw extracted value preserving original wording
- normalized_value: simplified/canonical form if reliably derivable (optional)
- category: one of financial, covenant, obligation, rate, collateral, condition,
  exclusion, party, date, general
- confidence: your confidence score in [0.0, 1.0]
- evidence_ids: list of EV-NNN identifiers that support this term (required)
- notes: optional explanation (ambiguity, conflict, reasoning)

If no material terms are found, return an empty list — do not invent content.
"""


# ── Evidence Serialization ────────────────────────────────────────────────────

def serialize_evidence(snippets: list[EvidenceSnippet]) -> str:
    """Serialize evidence snippets into a structured text block for the LLM.

    The output format makes Evidence IDs prominent so the LLM cites them
    correctly.  Each snippet is separated by a ``---`` divider.

    Args:
        snippets: Ordered list of ``EvidenceSnippet`` objects.

    Returns:
        A formatted string ready to include in the user prompt.
        Empty string if ``snippets`` is empty.

    Example output::

        [EV-018] Page 2 | Section: 4.1 - Interest Rate
        The Loan shall carry interest at the rate of 8.5% per annum (p.a.)...

        ---

        [EV-023] Page 2 | Section: 5.2 - Financial Covenants (Ongoing)
        The Borrower shall at all times maintain a minimum DSCR of 1.25x...

    """
    if not snippets:
        return ""

    blocks: list[str] = []
    for snippet in snippets:
        # Build the header line.
        section_str = f" | Section: {snippet.section}" if snippet.section else ""
        header = f"[{snippet.evidence_id}] Page {snippet.page_number}{section_str}"

        blocks.append(f"{header}\n{snippet.text.strip()}")

    return "\n\n---\n\n".join(blocks)
