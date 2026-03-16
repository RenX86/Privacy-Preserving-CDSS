import json
import ollama
from pydantic import BaseModel, Field
from app.config import settings
from app.pipeline.sources.vector_client import search_documents

# ─── STEP 0: JSON SCHEMA FOR QUERY EXPANSION ──────────────────────────────────
# This Pydantic model defines the EXACT JSON structure we expect from the LLM.
# When passed to ollama.chat(format=...), Ollama uses grammar-constrained decoding:
# the model physically CANNOT output anything except valid JSON matching this schema.
# This eliminates preamble text like "Here are three queries..." at the source.

class ExpandedQueries(BaseModel):
    queries: list[str] = Field(
        description="Focused clinical search queries. No preamble. No numbering. No explanation. Only the raw query strings."
    )
    # ↑ 'queries' is the key name in the JSON output
    # ↑ list[str] means the value must be a JSON array of strings
    # ↑ Field(description=...) adds the description to the JSON schema —
    #   the LLM reads this description and understands the expected format

# Patterns that indicate LLM preamble text rather than an actual query
_PREAMBLE_PATTERNS = (
    "here are", "the following", "below are", "sure, here",
    "certainly", "of course", "these queries", "search queries:"
)

def _is_preamble(line: str) -> bool:
    """Detect LLM explanation text that leaked into the query list."""
    low = line.lower().strip()
    # Numbered list items: "1.", "2.", "- ", "* "
    if low and low[0].isdigit() and "." in low[:3]:
        return True
    if low.startswith(("- ", "* ", "• ")):
        return False  # These are valid bullet queries — keep them but strip the prefix
    return any(low.startswith(p) for p in _PREAMBLE_PATTERNS)

def _clean_variants(raw: str, n: int) -> list[str]:
    """Parse LLM output into clean query strings, stripping preamble and bullets."""
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    cleaned = []
    for line in lines:
        if _is_preamble(line):
            print(f"  [MultiQuery] Skipping preamble: {line[:60]}")
            continue
        # Strip leading bullet characters
        for prefix in ("- ", "* ", "• "):
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
                break
        if line:
            cleaned.append(line)
        if len(cleaned) >= n:
            break
    return cleaned


def expand_queries(original_query: str, n: int = 3, query_type: str = "general") -> list[str]:
    """
    Generate targeted clinical sub-queries to retrieve specific guideline sections.
    Focuses on extracting ACMG criteria codes and classification rules.
    """
    if query_type == "screening_retrieval":
        prompt = f"""You are a clinical genetics search assistant.
Generate {n} short, specific search queries to retrieve cancer screening and surveillance sections.
Focus on: BRCA carrier management, surveillance schedules (mammography, MRI ages), risk-reducing surgery (RRSO, mastectomy timing), prophylactic interventions, hereditary cancer prevention.
DO NOT generate ACMG criteria codes or treatment chemotherapy protocols.
Return ONLY the queries, one per line, no numbering, no explanation.

Original query: {original_query}

Example good queries:
BRCA1 carrier annual breast MRI mammography starting age
risk-reducing salpingo-oophorectomy RRSO age recommendation BRCA
hereditary breast ovarian cancer surveillance screening protocol"""

    elif query_type == "protocol_retrieval":
        prompt = f"""You are a clinical oncology search assistant.
Generate {n} short, specific search queries to retrieve cancer treatment protocol sections.
Focus on: chemotherapy regimens, radiation, surgery, targeted therapy, immunotherapy, staging, disease management.
DO NOT generate ACMG criteria codes or screening intervals.
Return ONLY the queries, one per line, no numbering, no explanation.

Original query: {original_query}

Example good queries:
breast cancer HER2 positive treatment chemotherapy regimen
stage III breast cancer neoadjuvant surgery radiation protocol
triple negative breast cancer immunotherapy targeted therapy"""

    else:
        prompt = f"""You are a clinical genetics search assistant.
Generate {n} short, specific search queries to retrieve ACMG/clinical guideline sections.
Focus on: criteria codes (PVS1, PS1-4, PM1-6, PP1-5, BA1, BS1-4, BP1-7), classification rules, evidence weights, pathogenicity thresholds.
DO NOT rephrase the full question. Generate short targeted keyword queries.
Return ONLY the queries, one per line, no numbering, no explanation.

Original query: {original_query}

Example good queries:
ACMG PVS1 null variant loss of function criteria
pathogenic classification criteria strong evidence PS1 PS2
benign variant population frequency BA1 BS1 criteria"""

    try:
        # ── JSON Schema constrained generation ────────────────────────────────
        # format=ExpandedQueries.model_json_schema() produces this JSON schema:
        # {"properties": {"queries": {"type": "array", "items": {"type": "string"}, ...}}}
        # Ollama uses this schema as a grammar constraint — the model MUST output
        # valid JSON matching this schema. Preamble text is literally impossible.
        response = ollama.chat(
            model=settings.LOCAL_LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    # System role gives the model its identity before reading the prompt.
                    # This improves instruction-following significantly for smaller models.
                    "content": "You are a clinical search assistant. Output ONLY a JSON object with a 'queries' key containing a list of search query strings. No explanation."
                },
                {"role": "user", "content": prompt}
            ],
            format=ExpandedQueries.model_json_schema()
            # ↑ This is the grammar constraint. Without this, the model outputs free text.
            # With this, every token MUST be valid JSON matching the ExpandedQueries schema.
        )

        # ── Parse the guaranteed-JSON response ────────────────────────────────
        raw = response.message.content
        # raw is a JSON string like: '{"queries": ["ACMG PVS1 criteria", "PS1 evidence", "PM2 absent"]}'  
        
        data = json.loads(raw)
        # ↑ json.loads() converts the JSON string into a Python dict
        # result: {"queries": ["ACMG PVS1 criteria", ...]}
        
        variants = data.get("queries", [])[:n]
        # ↑ .get("queries", []) safely extracts the list (falls back to [] if key missing)
        # ↑ [:n] limits to exactly n items regardless of how many the LLM generated

        # ── Backup: strip any preamble that somehow slipped through ───────────
        # This should never trigger with JSON schema output, but is a safety net
        variants = _clean_variants("\n".join(variants), n) if variants else []

        all_queries = [original_query] + variants
        print(f"  [MultiQuery] Generated {len(variants)} targeted queries (JSON schema):")
        for q in variants:
            print(f"    • {q[:80]}")
        return all_queries

    except json.JSONDecodeError as e:
        # If JSON parsing fails (e.g. model produced malformed JSON), fall back to
        # the plain-text preamble-stripping approach on the raw content
        print(f"  [MultiQuery] JSON parse failed: {e} — falling back to text parsing")
        raw = response.message.content.strip()
        variants = _clean_variants(raw, n)
        return [original_query] + variants

    except Exception as e:
        print(f"  [MultiQuery] LLM expansion failed: {e} -- using original only")
        return [original_query]



# ─── STEP 2: SEARCH + DEDUPLICATE ─────────────────────────────────────────────

def multi_query_search(
    query: str,
    top_k: int = 5,
    category_filter: str = None,
    n_variants: int = 3
) -> list[dict]:
    """
    1. Expand the query into N variants
    2. Run vector search for each variant
    3. Deduplicate results by chunk text (same chunk found multiple times = keep once)
    4. Return the full unique pool — let the reranker pick the best ones
    """
    # Route to dedicated prompt per category
    if category_filter == "screening_protocol":
        expand_type = "screening_retrieval"
    elif category_filter == "protocol":
        expand_type = "protocol_retrieval"
    else:
        expand_type = "rule_retrieval"
    queries = expand_queries(query, n=n_variants, query_type=expand_type)

    seen_texts = set()
    unique_results = []

    for q in queries:
        results = search_documents(q, top_k=top_k, category_filter=category_filter)
        for r in results:
            # Deduplicate by chunk_text (small chunks - more variety than parent_text)
            # parent_text deduplication was too aggressive: all 20 results had same 2 parents
            key = (r.get("chunk_text", "") or r.get("parent_text", ""))[:160]
            if key not in seen_texts:
                seen_texts.add(key)
                unique_results.append(r)

    print(f"  [MultiQuery] {len(queries)} queries × up to {top_k} results = {len(unique_results)} unique chunks")
    return unique_results
