import re
from dataclasses import dataclass

@dataclass
class SubQuery:
    text: str 
    target: str
    query_type: str


VARIANT_ID_PATTERNS = re.compile(
    r'\brs\d+\b'
    r'|NM_\d+\.\d+'
    r'|NP_\d+\.\d+',
    re.IGNORECASE
)

# Protocol = active cancer TREATMENT (chemotherapy, surgery, radiation, staging)
# These keywords should be SPECIFIC to treatment — not generic clinical words
PROTOCOL_KEYWORDS = [
    "chemotherapy", "radiotherapy", "radiation therapy", "targeted therapy",
    "immunotherapy", "biological therapy", "neoadjuvant", "adjuvant",
    "systemic therapy", "hormone therapy", "endocrine therapy",
    "tumor staging", "cancer staging", "metastatic", "stage iv", "stage iii",
    "oncology treatment", "treatment regimen", "treatment protocol"
]

# Screening = carrier surveillance, prophylaxis, hereditary risk management
SCREENING_KEYWORDS = [
    "nccn", "screening", "surveillance", "high-risk", "prevention",
    "early detection", "mammography", "mri", "prophylaxis", "prophylactic",
    "risk-reducing", "rrso", "carrier", "hereditary", "germline"
]

CLINGEN_KEYWORDS = ["gene validity", "clingen", "gene-specific",
                     "expert panel", "expert panel recommendation",
                     "actionability curation", "dosage sensitivity"]


def decompose_query(query: str, gene: str = None) -> list[SubQuery]:
    """
    Break the user's multi-topic query into FOCUSED single-topic sub-queries.

    KEY DESIGN RULE:
        The sub-query 'text' is used as the SEED for vector search and reranking.
        It must contain ONLY the terms relevant to that sub-query's topic.

        ❌ BAD:  text = full user query (mixes multiple topics → diluted embedding)
        ✅ GOOD: text = focused topic keywords (matches target document embeddings)

    WHY THIS MATTERS:
        When the reranker sees focused topic text vs a relevant document chunk,
        it scores HIGH (0.1–0.6).  When it sees the full 40-word mixed query,
        the same chunk scores much lower → gets dropped by CRAG.
    """

    sub_queries = []
    query_lower = query.lower()
    gene_tag = gene or ""

    # Extract variant IDs (rsXXX) — used in both postgres and focused text
    variant_matches = VARIANT_ID_PATTERNS.findall(query)

    # ── Postgres: structured DB lookup (already focused) ─────────────────────
    for match in variant_matches:
        sub_queries.append(SubQuery(
            text=f"Get clinical significance for {match}",
            target="postgres",
            query_type="data_extraction"
        ))

    # ── Screening: focused on surveillance schedules & prophylaxis ────────────
    has_screening = any(keyword in query_lower for keyword in SCREENING_KEYWORDS)

    if any(keyword in query_lower for keyword in PROTOCOL_KEYWORDS):
        if not has_screening:
            focused = (
                f"cancer treatment protocol chemotherapy surgery radiation "
                f"staging systemic therapy {gene_tag}"
            ).strip()
            print(f"  [Decomposer] Protocol sub-query: {focused[:80]}")
            sub_queries.append(SubQuery(
                text=focused,
                target="vector_db",
                query_type="protocol_retrieval"
            ))
        else:
            print("  [Decomposer] Protocol keywords found but screening also detected — skipping protocol_retrieval")

    if has_screening:
        # Mentions mammography, MRI, risk-reducing surgery — exactly what
        # NCCN Genetic/Familial High-Risk Assessment sections contain.
        focused = (
            f"NCCN cancer screening surveillance mammography MRI "
            f"risk-reducing prophylactic salpingo-oophorectomy "
            f"{gene_tag} carrier management"
        ).strip()
        print(f"  [Decomposer] Screening sub-query: {focused[:80]}")
        sub_queries.append(SubQuery(
            text=focused,
            target="vector_db",
            query_type="screening_retrieval"
        ))

    # ── ClinGen: API call, text is just for logging ──────────────────────────
    if any(keyword in query_lower for keyword in CLINGEN_KEYWORDS):
        sub_queries.append(SubQuery(
            text=query,     # ClinGen is an API call, not vector search
            target="clingen",
            query_type="clingen_lookup"
        ))

    # ── Fallback: generic vector search with no category filter ──────────────
    if not sub_queries:
        sub_queries.append(SubQuery(
            text=query,
            target="vector_db",
            query_type="general"
        ))

    return sub_queries