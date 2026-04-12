# Privacy-Preserving CDSS — Implementation Plan

> **Scope:** Three implementation tracks in priority order.
> Track 1 fixes active correctness bugs. Track 2 adds EHR integration via CDS Hooks.
> Track 3 delivers a minimal demo frontend. Each track is independent and can be
> completed in sequence without breaking the pipeline.

---

## Track 1 — Active Bug Fixes

These must be completed before any integration work. They affect correctness of the
core pipeline and will produce misleading output in the CDS Hooks layer if left unfixed.

---

### Bug 1 — gnomAD Falsy-Zero AF Logic

**File:** `app/pipeline/sources/gnomad_client.py`

**Problem:**

```python
af = genome.get("af") or exome.get("af") or 0.0
```

When gnomAD returns `af = 0.0` (variant observed in the database but with zero
allele frequency — which is a valid and clinically meaningful result), Python
evaluates `0.0` as falsy. The expression falls through to the `or exome.get("af")`
branch, and if that is also `0.0` or absent, returns `0.0` from the literal
fallback. This means the variant is reported as "not found" in the gnomAD chunk
that reaches the LLM, which is factually incorrect. A variant present in gnomAD
with AF=0.0 is different from a variant absent from gnomAD entirely.

**Fix:**

```python
# Replace this:
af = genome.get("af") or exome.get("af") or 0.0

# With this:
genome_af = genome.get("af")
exome_af  = exome.get("af")

if genome_af is not None:
    af = genome_af
elif exome_af is not None:
    af = exome_af
else:
    af = 0.0
```

Also update the downstream chunk text to distinguish between the two cases:

```python
# After the fix, differentiate "found with AF=0" from "not found"
if variant:
    # Variant exists in gnomAD
    found_in_gnomad = True
    af_text = f"AF={af:.8f}" if af > 0 else "AF=0.0 (observed but frequency rounds to zero)"
else:
    found_in_gnomad = False
```

And in `nodes.py`, the "not found" branch should only fire when gnomAD returns
`None` for the variant object, not when AF happens to be zero:

```python
# Current code fires for both "not in gnomAD" and "AF=0.0":
if freq_data:
    ...
else:
    trusted_chunks.append(RetrievedChunk(
        text=f"gnomAD lookup for {rsid}: Variant not found...",
        ...
    ))

# Fix: add a found flag to the return dict from get_allele_frequency()
# Return {"found": False} when variant is absent, {"found": True, "af": 0.0} when present
```

**Return dict change in `gnomad_client.py`:**

```python
# When variant is not found:
return {"found": False, "rsid": rsid}

# When variant is found:
return {
    "found":            True,
    "rsid":             rsid,
    "gnomad_id":        gnomad_id,
    "allele_frequency": af,
    "is_common":        af >= 0.05,
    "genome_ac":        genome.get("ac"),
    "genome_an":        genome.get("an"),
}
```

**Update `nodes.py` to check `found` key:**

```python
freq_data = get_allele_frequency(rsid)
if freq_data and freq_data.get("found"):
    freq_text = (
        f"Population frequency for {rsid}: "
        f"AF={freq_data['allele_frequency']:.8f}. "
        f"{'Variant is common in population (AF >= 5%).' if freq_data['is_common'] else 'Variant is rare in population.'}"
    )
    trusted_chunks.append(RetrievedChunk(
        text=freq_text, source="gnomAD", reference=rsid
    ))
elif freq_data and not freq_data.get("found"):
    trusted_chunks.append(RetrievedChunk(
        text=f"gnomAD lookup for {rsid}: Variant not found in gnomAD r4. Absent from population frequency data.",
        source="gnomAD",
        reference=rsid
    ))
# If freq_data is None entirely, gnomAD call failed — do not add a chunk
```

---

### Bug 2 — BRCA2 Gene Filter False Negative

**File:** `app/pipeline/graph/nodes.py` — `evaluate_node()`

**Problem:**

The gene filter in the Evaluator checks whether the target gene appears in the
chunk text:

```python
mentions_gene = gene_lower and gene_lower in chunk.text.lower()
if is_nccn and gene_lower and not mentions_gene:
    continue
```

For a BRCA2 query, `gene_lower = "brca2"`. Many NCCN chunks mention `"BRCA1/2"`
or `"BRCA1/BRCA2"` as a combined reference (which is standard NCCN notation for
recommendations that apply to both genes). The substring `"brca2"` is not present
in `"brca1/2"`, so these chunks are incorrectly dropped. A clinician asking about
BRCA2 loses the majority of the relevant NCCN guidance.

**Fix:**

```python
def _gene_mentioned(gene_lower: str, text_lower: str) -> bool:
    """
    Check if the target gene is referenced in the chunk text.
    Handles combined notations like BRCA1/2, BRCA1/BRCA2, BRCA 1/2.
    """
    if gene_lower in text_lower:
        return True

    # Handle BRCA1/2 combined notation for BRCA2 queries
    if gene_lower == "brca2":
        return any(p in text_lower for p in ["brca1/2", "brca1/brca2", "brca 1/2"])
    if gene_lower == "brca1":
        return any(p in text_lower for p in ["brca1/2", "brca1/brca2", "brca 1/2"])

    return False
```

Replace the filter line in `evaluate_node()`:

```python
# Before:
mentions_gene = gene_lower and gene_lower in chunk.text.lower()

# After:
mentions_gene = gene_lower and _gene_mentioned(gene_lower, chunk.text.lower())
```

Place `_gene_mentioned()` at the top of `nodes.py` as a module-level helper.

---

### Bug 3 — Connection Pool Exhaustion

**File:** `app/db/pool.py`

**Problem:**

The `ThreadedConnectionPool` has `maxconn=10`. Under concurrent requests (which
the parallel `Send()` fan-out produces — DB_Retriever and PDF_Retriever both
need connections simultaneously), if 10 connections are already borrowed and a
new request comes in, `getconn()` raises `PoolError: connection pool exhausted`.
The current `db_conn()` context manager has no retry or timeout logic.

**Fix:**

Add a retry loop with backoff to `get_conn()`:

```python
import time

def get_conn(retries: int = 5, delay: float = 0.3):
    """Borrow a connection from the pool with retry on exhaustion."""
    pool = _get_pool()
    for attempt in range(retries):
        try:
            return pool.getconn()
        except pg_pool.PoolError:
            if attempt == retries - 1:
                raise
            log.warning(
                f"Connection pool exhausted — retrying in {delay}s "
                f"(attempt {attempt + 1}/{retries})"
            )
            time.sleep(delay)
            delay *= 1.5  # exponential backoff
```

Also ensure connections are always returned even on exception — the current
`db_conn()` context manager already does this via `finally`, so no change needed
there. But add a pool health check on startup:

```python
# In app/main.py, after app is created:
from app.db.pool import _get_pool

@app.on_event("startup")
def startup_event():
    _get_pool()  # Force pool initialisation at startup, not on first request
    log.info("Connection pool pre-warmed on startup")
```

---

### Bug 4 — Dead Code Cleanup

**Files:** Multiple

These are not functional bugs but leave misleading code paths in the repository.

**`app/pipeline/graph/nodes.py`** — `citation_node()` receives `trusted_chunks`
and `candidate_chunks` from state but only uses `verified_chunks` for citation
extraction. `trusted_chunks` and `candidate_chunks` reads are dead. Remove them:

```python
# Remove these two lines from citation_node():
trusted_chunks = state.get("trusted_chunks", [])    # unused
# candidate_chunks already not used
```

**`app/pipeline/graph/state.py`** — The `CDSSGraphState` comment says
`Annotated[list, operator.add]` is needed so both retrievers can append. This is
correct, but `trusted_chunks` and `candidate_chunks` should have a brief docstring
explaining the merge behaviour for future contributors.

**`docs/manifest.json`** — The `"category"` value for the Genetic-Familial PDF
is `"screening"` but `retrieve_pdf_node` passes `category_filter="screening_protocol"`
to `multi_query_search()`, which passes it to `vector_client.search_documents()`
as a SQL `WHERE category = 'screening_protocol'` clause. This will never match
any row. Fix the manifest:

```json
{
  "Genetic-Familial High-Risk Assessment Breast Ovarian PancreaticandProstate.pdf": {
    "source": "Genetic-Familial High-Risk Assessment",
    "category": "screening_protocol",
    "gene": null,
    "parser": "docling"
  }
}
```

Re-run `python app/db/vector/indexing.py` after this change to re-index with the
correct category value.

---

## Track 2 — CDS Hooks EHR Integration

CDS Hooks is the HL7 open standard for integrating clinical decision support into
EHR workflows. Both Epic and Cerner support it natively. This track adds a
CDS Hooks adapter layer in front of the existing LangGraph pipeline. The pipeline
itself is not modified.

---

### Architecture

```
Epic / Cerner Sandbox
        │
        │  POST /cds-services/genomic-cdss
        │  (FHIR R4 context payload)
        ▼
app/api/cds_hooks_router.py          ← new
        │
        │  _parse_fhir_observations() ← new (app/api/fhir_parser.py)
        │  builds query string
        ▼
Existing LangGraph pipeline          ← unchanged
(cdss_app.invoke)
        │
        ▼
app/api/cds_hooks_router.py
        │
        │  _build_cards()
        ▼
CDS Hooks JSON response
(cards displayed in EHR)
```

---

### New Files

**`app/api/fhir_parser.py`**

Responsible for extracting clinically relevant fields from the FHIR `Observation`
bundle that the EHR prefetches and sends with the hook payload.

```python
"""
FHIR R4 Observation parser for genomic variant data.

FHIR stores genetic results as Observation resources with components
identified by LOINC codes:
  48018-6  → Gene studied (valueCodeableConcept → HGNC display name)
  81252-9  → Discrete genetic variant (valueCodeableConcept → rsID or HGVS)
  69547-8  → Genomic ref allele
  62374-4  → Human reference sequence assembly version

The parser extracts gene symbol and variant ID, then constructs a query
string compatible with the existing decompose_query() routing logic.
"""

from __future__ import annotations

LOINC_GENE_STUDIED       = "48018-6"
LOINC_DISCRETE_VARIANT   = "81252-9"
LOINC_GENOMIC_REF_ALLELE = "69547-8"


def parse_genomic_observations(observations: dict) -> dict:
    """
    Parse a FHIR Observation bundle and return extracted genomic fields.

    Returns:
        {
            "gene":    str | None,   e.g. "BRCA1"
            "rs_id":   str | None,   e.g. "rs879254116"
            "hgvs":    str | None,   e.g. "NM_007294.4:c.5266dupC"
        }
    """
    entries = observations.get("entry", [])
    gene  = None
    rs_id = None
    hgvs  = None

    for entry in entries:
        resource = entry.get("resource", {})
        if resource.get("resourceType") != "Observation":
            continue

        for comp in resource.get("component", []):
            loinc = (
                comp.get("code", {})
                    .get("coding", [{}])[0]
                    .get("code", "")
            )

            if loinc == LOINC_GENE_STUDIED:
                gene = (
                    comp.get("valueCodeableConcept", {})
                        .get("coding", [{}])[0]
                        .get("display", "")
                    or
                    comp.get("valueCodeableConcept", {})
                        .get("text", "")
                )

            elif loinc == LOINC_DISCRETE_VARIANT:
                coding = comp.get("valueCodeableConcept", {}).get("coding", [{}])[0]
                code   = coding.get("code", "")
                if code.startswith("rs"):
                    rs_id = code
                elif code.startswith("NM_") or ":" in code:
                    hgvs = code

    return {"gene": gene or None, "rs_id": rs_id or None, "hgvs": hgvs or None}


def build_query_from_fhir(observations: dict) -> str | None:
    """
    Convert a FHIR Observation bundle into a query string for the CDSS pipeline.

    The query format matches what decompose_query() is already tuned to handle:
      "What is the clinical significance of {rsid} in {gene} and what cancer
       screening protocol should the patient follow according to NCCN guidelines?
       Also confirm the ClinGen expert panel validity for {gene}."
    """
    fields = parse_genomic_observations(observations)
    gene   = fields["gene"]
    rs_id  = fields["rs_id"]
    hgvs   = fields["hgvs"]

    if not gene and not rs_id and not hgvs:
        return None

    variant_ref = rs_id or hgvs or "this variant"
    gene_clause = f"in {gene}" if gene else ""
    clingen_clause = f"Also confirm the ClinGen expert panel validity for {gene}." if gene else ""

    return (
        f"What is the clinical significance of {variant_ref} {gene_clause} "
        f"and what cancer screening protocol should the patient follow "
        f"according to NCCN guidelines? {clingen_clause}"
    ).strip()
```

---

**`app/api/cds_hooks_router.py`**

```python
"""
CDS Hooks adapter layer.

Exposes two endpoints required by the CDS Hooks specification:
  GET  /cds-services                     → service discovery
  POST /cds-services/genomic-cdss        → hook handler

The hook handler:
  1. Parses the FHIR Observation bundle from the prefetch block
  2. Constructs a query string via fhir_parser.build_query_from_fhir()
  3. Invokes the existing LangGraph pipeline (cdss_app.invoke)
  4. Formats the pipeline output as CDS Hooks "cards"

No changes are made to the LangGraph pipeline.
"""

import asyncio
import logging
from fastapi import APIRouter
from app.api.fhir_parser import build_query_from_fhir
from app.pipeline.graph.workflow import cdss_app
from app.pipeline.generation.guardrails import SAFE_FAILURE_MESSAGE

log = logging.getLogger("cdss.cds_hooks")
cds_router = APIRouter()


# ── Service Discovery ─────────────────────────────────────────────────────────

@cds_router.get("/cds-services")
def cds_discovery():
    """
    CDS Hooks specification requires every service to expose this endpoint.
    The EHR calls it on startup to learn which hooks the service handles
    and what FHIR data to prefetch.
    """
    return {
        "services": [
            {
                "hook":        "patient-view",
                "id":          "genomic-cdss",
                "title":       "Genomic Variant Clinical Decision Support",
                "description": (
                    "Interprets hereditary cancer variants (BRCA1/2 and related genes) "
                    "against ClinVar, gnomAD, ClinGen, and NCCN guidelines. "
                    "Returns variant classification, ClinGen expert panel status, "
                    "and applicable cancer screening protocols."
                ),
                "prefetch": {
                    "patient": "Patient/{{context.patientId}}",
                    "observations": (
                        "Observation"
                        "?patient={{context.patientId}}"
                        "&category=laboratory"
                        "&code=http://loinc.org|81247-9"  # Master HL7 genetic panel
                    )
                }
            }
        ]
    }


# ── Hook Handler ──────────────────────────────────────────────────────────────

@cds_router.post("/cds-services/genomic-cdss")
async def handle_genomic_hook(payload: dict):
    """
    Main CDS Hook handler. Called by the EHR when a patient chart is opened
    and the patient has genetic laboratory observations.
    """
    prefetch     = payload.get("prefetch", {})
    observations = prefetch.get("observations", {})
    hook_instance = payload.get("hookInstance", "unknown")

    log.info(f"[CDS Hook] Received hook instance: {hook_instance}")

    # ── Step 1: Parse FHIR observations → query string ────────────────────────
    query = build_query_from_fhir(observations)

    if not query:
        log.info("[CDS Hook] No genomic observations found in prefetch — returning empty cards")
        return {"cards": []}

    log.info(f"[CDS Hook] Built query: {query[:100]}...")

    # ── Step 2: Invoke LangGraph pipeline ─────────────────────────────────────
    try:
        initial_state = {"query": query}
        final_state = await asyncio.to_thread(cdss_app.invoke, initial_state)
    except Exception as e:
        log.error(f"[CDS Hook] Pipeline error: {e}")
        return {
            "cards": [_error_card(str(e))]
        }

    # ── Step 3: Format as CDS Hooks cards ─────────────────────────────────────
    cards = _build_cards(final_state)
    log.info(f"[CDS Hook] Returning {len(cards)} card(s), confidence={final_state.get('confidence')}")
    return {"cards": cards}


# ── Card Builders ─────────────────────────────────────────────────────────────

_INDICATOR_MAP = {
    "high":   "warning",   # High confidence pathogenic result → flag prominently
    "medium": "info",
    "low":    "info",
}


def _build_cards(final_state: dict) -> list[dict]:
    answer     = final_state.get("final_answer", "")
    confidence = final_state.get("confidence", "low")
    citations  = final_state.get("citations", [])

    # Safe failure — pipeline could not produce a reliable answer
    if not answer or answer == SAFE_FAILURE_MESSAGE:
        return [{
            "summary":   "Insufficient genomic data for clinical recommendation",
            "detail":    (
                "The system could not retrieve sufficient verified clinical data "
                "to generate a recommendation. Please consult primary databases "
                "or a qualified clinical geneticist."
            ),
            "indicator": "info",
            "source":    {"label": "Privacy-Preserving CDSS"},
        }]

    indicator = _INDICATOR_MAP.get(confidence, "info")

    cards = []

    # Card 1 — Main clinical summary with full markdown answer
    cards.append({
        "summary":   "Hereditary cancer variant interpretation available",
        "detail":    answer,
        "indicator": indicator,
        "source": {
            "label": "Privacy-Preserving CDSS",
            "url":   "http://localhost:5656/docs",
        },
        "links": [
            {
                "label": "Open full analysis in CDSS",
                "url":   "http://localhost:5656/docs#/default/handle_query_query_post",
                "type":  "absolute",
            }
        ],
    })

    # Card 2 — Citation summary (only if citations exist)
    if citations:
        citation_lines = "\n".join(
            f"- **{c.source}**: {c.reference}"
            for c in citations
        )
        cards.append({
            "summary":   f"Data sources used ({len(citations)} citations)",
            "detail":    citation_lines,
            "indicator": "info",
            "source":    {"label": "Privacy-Preserving CDSS"},
        })

    return cards


def _error_card(error_message: str) -> dict:
    return {
        "summary":   "CDSS pipeline error — manual review required",
        "detail":    (
            f"The genomic decision support pipeline encountered an error. "
            f"Please review the variant manually.\n\nError: {error_message}"
        ),
        "indicator": "warning",
        "source":    {"label": "Privacy-Preserving CDSS"},
    }
```

---

### Register in `app/main.py`

Add two lines:

```python
from app.api.cds_hooks_router import cds_router   # add this

app.include_router(router)
app.include_router(cds_router)                     # add this
```

---

### Testing Without a Real EHR

**Option 1 — curl (immediate, no accounts needed):**

```bash
curl -X POST http://localhost:5656/cds-services/genomic-cdss \
  -H "Content-Type: application/json" \
  -d '{
    "hook": "patient-view",
    "hookInstance": "test-brca1-001",
    "context": {"patientId": "synthetic-001"},
    "prefetch": {
      "observations": {
        "resourceType": "Bundle",
        "entry": [
          {
            "resource": {
              "resourceType": "Observation",
              "component": [
                {
                  "code": {"coding": [{"code": "48018-6"}]},
                  "valueCodeableConcept": {
                    "coding": [{"display": "BRCA1"}]
                  }
                },
                {
                  "code": {"coding": [{"code": "81252-9"}]},
                  "valueCodeableConcept": {
                    "coding": [{"code": "rs879254116"}]
                  }
                }
              ]
            }
          }
        ]
      }
    }
  }'
```

Expected response: a JSON object with a `cards` array containing your pipeline's
full clinical summary rendered as a CDS Hooks warning card.

**Option 2 — SMART on FHIR sandbox (free, no approval):**

1. Go to `https://launch.smarthealthit.org`
2. Select a synthetic patient with genetic observations
3. Set your CDS Hooks service URL to `http://localhost:5656`
4. Launch — the sandbox fires real hook payloads at your service

**Option 3 — Epic sandbox (free developer account):**

1. Register at `https://fhir.epic.com`
2. Create an app pointing to `http://localhost:5656/cds-services`
3. Use their built-in CDS Hooks tester with synthetic patient data

**Option 4 — Cerner Ignite sandbox:**

1. Register at `https://developers.cerner.com`
2. Same process as Epic — point to your local service URL

---

### Deployment Note for Sandbox Testing

Epic and Cerner sandboxes require your service to be publicly reachable (not
`localhost`). Use `ngrok` to expose your local FastAPI instance during testing:

```bash
ngrok http 5656
# Copy the generated https://xxxxx.ngrok.io URL
# Use that as your CDS Hooks service URL in the EHR sandbox
```

---

## Track 3 — Minimal Demo Frontend

A single self-contained HTML file. No build tools, no framework, no dependencies
beyond a CDN-loaded markdown renderer. The file lives at
`frontend/templates/index.html` (already a placeholder in the repo) and is served
by FastAPI as a static file.

---

### Register Static Files in `app/main.py`

```python
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "frontend" / "static"),
    name="static"
)

@app.get("/ui", response_class=HTMLResponse)
def serve_ui():
    html_path = BASE_DIR / "frontend" / "templates" / "index.html"
    return HTMLResponse(content=html_path.read_text())
```

Access the UI at `http://localhost:5656/ui`.

---

### `frontend/templates/index.html`

The UI uses the dark brutalist aesthetic already established in the project:
black backgrounds, Space Mono monospace font, neon yellow-green (`#c8f135`)
accents, exposed grid lines, industrial feel.

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Privacy-Preserving CDSS</title>

  <!-- Fonts -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Bebas+Neue&display=swap" rel="stylesheet">

  <!-- Markdown renderer -->
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>

  <style>
    :root {
      --bg:        #000000;
      --surface:   #0a0a0a;
      --border:    #1a1a1a;
      --accent:    #c8f135;
      --accent-dim:#7a9120;
      --text:      #e0e0e0;
      --muted:     #555555;
      --error:     #ff4444;
      --high:      #c8f135;
      --medium:    #f0a500;
      --low:       #888888;
      --mono:      'Space Mono', monospace;
      --display:   'Bebas Neue', sans-serif;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: var(--mono);
      font-size: 13px;
      line-height: 1.6;
      min-height: 100vh;
    }

    /* ── Header ── */
    header {
      border-bottom: 1px solid var(--accent);
      padding: 20px 32px;
      display: flex;
      align-items: baseline;
      gap: 16px;
    }

    header h1 {
      font-family: var(--display);
      font-size: 28px;
      color: var(--accent);
      letter-spacing: 2px;
    }

    header span {
      color: var(--muted);
      font-size: 11px;
    }

    /* ── Layout ── */
    .layout {
      display: grid;
      grid-template-columns: 420px 1fr;
      height: calc(100vh - 65px);
    }

    /* ── Left panel — query ── */
    .panel-query {
      border-right: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      padding: 24px;
      gap: 16px;
    }

    .panel-label {
      font-size: 10px;
      color: var(--accent);
      letter-spacing: 3px;
      text-transform: uppercase;
      border-bottom: 1px solid var(--border);
      padding-bottom: 8px;
    }

    textarea {
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--text);
      font-family: var(--mono);
      font-size: 12px;
      line-height: 1.7;
      padding: 14px;
      resize: none;
      flex: 1;
      outline: none;
      transition: border-color 0.2s;
    }

    textarea:focus { border-color: var(--accent); }

    textarea::placeholder { color: var(--muted); }

    button#submit {
      background: var(--accent);
      border: none;
      color: #000;
      cursor: pointer;
      font-family: var(--display);
      font-size: 18px;
      letter-spacing: 2px;
      padding: 12px;
      transition: background 0.15s;
      text-align: center;
    }

    button#submit:hover   { background: #d4ff40; }
    button#submit:active  { background: var(--accent-dim); }
    button#submit:disabled {
      background: var(--border);
      color: var(--muted);
      cursor: not-allowed;
    }

    /* ── Pipeline status ── */
    .pipeline {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 6px;
    }

    .node {
      border: 1px solid var(--border);
      padding: 6px 8px;
      font-size: 10px;
      color: var(--muted);
      display: flex;
      align-items: center;
      gap: 6px;
      transition: border-color 0.3s, color 0.3s;
    }

    .node.active  { border-color: var(--accent); color: var(--accent); }
    .node.done    { border-color: var(--accent-dim); color: var(--accent-dim); }
    .node.error   { border-color: var(--error); color: var(--error); }

    .node-dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: currentColor;
      flex-shrink: 0;
    }

    /* ── Right panel — output ── */
    .panel-output {
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    .output-header {
      padding: 16px 24px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      flex-shrink: 0;
    }

    .confidence-badge {
      font-size: 10px;
      letter-spacing: 2px;
      padding: 3px 10px;
      border: 1px solid currentColor;
      display: none;
    }

    .confidence-badge.high   { color: var(--high); }
    .confidence-badge.medium { color: var(--medium); }
    .confidence-badge.low    { color: var(--low); }

    .output-body {
      flex: 1;
      overflow-y: auto;
      padding: 24px;
    }

    /* Scrollbar */
    .output-body::-webkit-scrollbar       { width: 4px; }
    .output-body::-webkit-scrollbar-track { background: var(--bg); }
    .output-body::-webkit-scrollbar-thumb { background: var(--border); }

    /* Empty state */
    .empty-state {
      height: 100%;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 12px;
      color: var(--muted);
    }

    .empty-state .glyph {
      font-family: var(--display);
      font-size: 64px;
      color: var(--border);
      letter-spacing: 4px;
    }

    .empty-state p { font-size: 11px; letter-spacing: 1px; }

    /* Loading */
    .loading {
      display: none;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      height: 100%;
      gap: 16px;
      color: var(--accent);
    }

    .spinner {
      width: 32px;
      height: 32px;
      border: 2px solid var(--border);
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }

    @keyframes spin { to { transform: rotate(360deg); } }

    .loading p { font-size: 11px; letter-spacing: 2px; color: var(--muted); }

    /* Markdown render */
    .markdown-output { display: none; }

    .markdown-output strong { color: var(--accent); }

    .markdown-output h2 {
      font-family: var(--display);
      font-size: 16px;
      letter-spacing: 2px;
      color: var(--accent);
      margin: 24px 0 10px;
      padding-bottom: 6px;
      border-bottom: 1px solid var(--border);
    }

    .markdown-output p   { margin-bottom: 10px; }
    .markdown-output ul  { margin: 8px 0 12px 20px; }
    .markdown-output li  { margin-bottom: 6px; color: var(--text); }

    .markdown-output code {
      background: var(--surface);
      border: 1px solid var(--border);
      padding: 1px 5px;
      font-family: var(--mono);
      font-size: 11px;
    }

    /* Citations panel */
    .citations {
      display: none;
      border-top: 1px solid var(--border);
      padding: 16px 24px;
      flex-shrink: 0;
    }

    .citations-label {
      font-size: 10px;
      color: var(--accent);
      letter-spacing: 3px;
      margin-bottom: 10px;
    }

    .citation-list {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }

    .citation-item {
      font-size: 11px;
      color: var(--muted);
      border-left: 2px solid var(--border);
      padding-left: 10px;
    }

    .citation-item strong { color: var(--accent-dim); }

    /* Error state */
    .error-msg {
      display: none;
      color: var(--error);
      border: 1px solid var(--error);
      padding: 14px;
      font-size: 12px;
    }
  </style>
</head>

<body>

<header>
  <h1>🧬 PRIVACY-PRESERVING CDSS</h1>
  <span>Hybrid RAG · Local LLM · Zero Hallucination Tolerance</span>
</header>

<div class="layout">

  <!-- ── Left: Query Panel ── -->
  <aside class="panel-query">
    <div class="panel-label">Clinical Query</div>

    <textarea
      id="query-input"
      rows="10"
      placeholder="Enter a clinical genomic query...

Example:
What is the clinical significance of rs879254116 in BRCA1 and what cancer screening protocol should the patient follow according to NCCN guidelines? Also confirm the ClinGen expert panel validity for BRCA1."
    ></textarea>

    <button id="submit">SUBMIT QUERY</button>

    <!-- Pipeline status nodes -->
    <div class="panel-label" style="margin-top:8px">Pipeline Status</div>
    <div class="pipeline">
      <div class="node" id="node-decomposer">
        <div class="node-dot"></div>DECOMPOSER
      </div>
      <div class="node" id="node-db">
        <div class="node-dot"></div>DB RETRIEVER
      </div>
      <div class="node" id="node-pdf">
        <div class="node-dot"></div>PDF RETRIEVER
      </div>
      <div class="node" id="node-evaluator">
        <div class="node-dot"></div>EVALUATOR
      </div>
      <div class="node" id="node-generator">
        <div class="node-dot"></div>GENERATOR
      </div>
      <div class="node" id="node-enforcer">
        <div class="node-dot"></div>ENFORCER
      </div>
    </div>
  </aside>

  <!-- ── Right: Output Panel ── -->
  <main class="panel-output">

    <div class="output-header">
      <div class="panel-label" style="border:none;padding:0">Clinical Output</div>
      <span class="confidence-badge" id="confidence-badge"></span>
    </div>

    <div class="output-body" id="output-body">

      <!-- Empty state -->
      <div class="empty-state" id="empty-state">
        <div class="glyph">CDSS</div>
        <p>Submit a query to begin</p>
      </div>

      <!-- Loading -->
      <div class="loading" id="loading">
        <div class="spinner"></div>
        <p>RUNNING LANGGRAPH PIPELINE</p>
      </div>

      <!-- Error -->
      <div class="error-msg" id="error-msg"></div>

      <!-- Markdown output -->
      <div class="markdown-output" id="markdown-output"></div>

    </div>

    <!-- Citations footer -->
    <div class="citations" id="citations-panel">
      <div class="citations-label">DATA SOURCES</div>
      <div class="citation-list" id="citation-list"></div>
    </div>

  </main>
</div>

<script>
  const API_BASE = "http://localhost:5656";

  const nodes = ["decomposer","db","pdf","evaluator","generator","enforcer"];
  const nodeEls = {};
  nodes.forEach(n => { nodeEls[n] = document.getElementById(`node-${n}`); });

  function resetNodes() {
    nodes.forEach(n => {
      nodeEls[n].className = "node";
    });
  }

  function setNodeState(name, state) {
    if (nodeEls[name]) nodeEls[name].className = `node ${state}`;
  }

  // Simulate pipeline stage progression during the request
  // (pipeline runs synchronously server-side, so we animate client-side)
  let stageTimer = null;

  function animatePipeline() {
    const stages = [
      { node: "decomposer", delay: 200 },
      { node: "db",         delay: 800 },
      { node: "pdf",        delay: 900 },
      { node: "evaluator",  delay: 3500 },
      { node: "generator",  delay: 5000 },
      { node: "enforcer",   delay: 8000 },
    ];

    // Mark each node active then done in sequence
    stages.forEach(({ node, delay }, i) => {
      setTimeout(() => setNodeState(node, "active"), delay);
      const nextDelay = stages[i + 1]?.delay ?? delay + 1200;
      setTimeout(() => setNodeState(node, "done"), nextDelay - 100);
    });
  }

  function stopAnimation() {
    clearTimeout(stageTimer);
  }

  function showLoading() {
    document.getElementById("empty-state").style.display   = "none";
    document.getElementById("error-msg").style.display     = "none";
    document.getElementById("markdown-output").style.display = "none";
    document.getElementById("citations-panel").style.display = "none";
    document.getElementById("confidence-badge").style.display = "none";
    document.getElementById("loading").style.display       = "flex";
  }

  function showError(msg) {
    document.getElementById("loading").style.display = "none";
    const el = document.getElementById("error-msg");
    el.textContent = `ERROR: ${msg}`;
    el.style.display = "block";
    nodes.forEach(n => setNodeState(n, "error"));
  }

  function showResult(data) {
    document.getElementById("loading").style.display = "none";

    // Confidence badge
    const badge = document.getElementById("confidence-badge");
    badge.textContent = `${data.confidence.toUpperCase()} CONFIDENCE`;
    badge.className = `confidence-badge ${data.confidence}`;
    badge.style.display = "block";

    // Markdown answer
    const mdEl = document.getElementById("markdown-output");
    mdEl.innerHTML = marked.parse(data.answer || "No answer returned.");
    mdEl.style.display = "block";

    // Citations
    if (data.citations && data.citations.length > 0) {
      const list = document.getElementById("citation-list");
      list.innerHTML = data.citations.map((c, i) =>
        `<div class="citation-item">
          <strong>[${i + 1}] ${c.source}</strong> — ${c.reference}
        </div>`
      ).join("");
      document.getElementById("citations-panel").style.display = "block";
    }

    // Mark all pipeline nodes done
    nodes.forEach(n => setNodeState(n, "done"));
  }

  document.getElementById("submit").addEventListener("click", async () => {
    const query = document.getElementById("query-input").value.trim();
    if (!query) return;

    const btn = document.getElementById("submit");
    btn.disabled = true;
    btn.textContent = "RUNNING...";

    resetNodes();
    showLoading();
    animatePipeline();

    try {
      const res = await fetch(`${API_BASE}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query })
      });

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      }

      const data = await res.json();
      showResult(data);

    } catch (err) {
      showError(err.message);
    } finally {
      btn.disabled = false;
      btn.textContent = "SUBMIT QUERY";
    }
  });

  // Allow Ctrl+Enter to submit
  document.getElementById("query-input").addEventListener("keydown", e => {
    if (e.ctrlKey && e.key === "Enter") {
      document.getElementById("submit").click();
    }
  });
</script>

</body>
</html>
```

Access at `http://localhost:5656/ui` once the static file route is registered.

---

## Implementation Order

Complete these in sequence. Each track builds on the previous.

| Priority | Task | File(s) Changed | Effort |
|---|---|---|---|
| 1 | Fix gnomAD falsy-zero AF | `gnomad_client.py`, `nodes.py` | 30 min |
| 2 | Fix BRCA2 gene filter | `nodes.py` | 20 min |
| 3 | Fix manifest category mismatch | `docs/manifest.json` + re-index | 10 min + index time |
| 4 | Fix connection pool exhaustion | `pool.py`, `main.py` | 20 min |
| 5 | Remove dead code | `nodes.py` | 10 min |
| 6 | Add `fhir_parser.py` | new file | 45 min |
| 7 | Add `cds_hooks_router.py` | new file | 45 min |
| 8 | Register CDS router in `main.py` | `main.py` | 5 min |
| 9 | Test CDS Hooks with curl | — | 20 min |
| 10 | Add static file serving to `main.py` | `main.py` | 10 min |
| 11 | Write `frontend/templates/index.html` | new file | done (above) |

Total estimated effort: **~4 hours** for all three tracks.

---

## Disclaimer

This system is a clinical decision support tool only. It does not replace
professional medical judgment. All outputs must be reviewed by a qualified
clinician before use in patient care. CDS Hooks integration with live EHR
systems requires additional security review, authentication, and clinical
validation before deployment.
