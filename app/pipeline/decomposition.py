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

ACMG_KEYWORDS = ["acmg", "pathogenic", "likely_pathogenic", "vus",
                 "uncertain_significance", "benign", "likely_benign", "variant classification", "pvs1",
                 "ps1", "pm2", "pp2", "bp2", "bs2", "bp4", "bs4", "criteria"]

# Protocol = active cancer TREATMENT (chemotherapy, surgery, radiation, staging)
# These keywords should be SPECIFIC to treatment — not generic clinical words
PROTOCOL_KEYWORDS = [
    "chemotherapy", "radiotherapy", "radiation therapy", "targeted therapy",
    "immunotherapy", "biological therapy", "neoadjuvant", "adjuvant",
    "systemic therapy", "hormone therapy", "endocrine therapy",
    "tumor staging", "cancer staging", "metastatic", "stage iv", "stage iii",
    "oncology treatment", "treatment regimen", "treatment protocol"
]
# Removed: 'cancer', 'tumor', 'stage', 'treatment', 'management', 'surgery', 'oncology'
# These are too generic — they appear in screening/ACMG queries and cause false triggers

# Screening = carrier surveillance, prophylaxis, hereditary risk management
SCREENING_KEYWORDS = [
    "nccn", "screening", "surveillance", "high-risk", "prevention",
    "early detection", "mammography", "mri", "prophylaxis", "prophylactic",
    "risk-reducing", "rrso", "carrier", "hereditary", "germline"
]

CLINGEN_KEYWORDS = ["gene validity", "clingen", "gene-specific",
                     "expert panel", "expert panel recommendation",
                     "actionability curation", "dosage sensitivity"]


def decompose_query(query: str) -> list[SubQuery]:

    sub_queries = []
    query_lower = query.lower()

    variant_matches = VARIANT_ID_PATTERNS.findall(query)
    for match in variant_matches:
        sub_queries.append(SubQuery(
            text =f"Get clinical significance for {match}",
            target="postgres",
            query_type="data_extraction"
        ))

    if any(keyword in query_lower for keyword in ACMG_KEYWORDS):
        sub_queries.append(SubQuery(
            text=query,           # raw query → multi-query expander generates targeted criteria queries
            target="vector_db",
            query_type="rule_retrieval"
        ))

    has_screening = any(keyword in query_lower for keyword in SCREENING_KEYWORDS)

    if any(keyword in query_lower for keyword in PROTOCOL_KEYWORDS):
        # Only fire protocol retrieval if we are NOT already doing a screening search
        # A BRCA1 "screening protocol" query should go to screening_retrieval, NOT here
        # Treatment protocol is for active cancer patients, not germline carriers
        if not has_screening:
            sub_queries.append(SubQuery(
                text=query,
                target="vector_db",
                query_type="protocol_retrieval"
            ))
        else:
            print("  [Decomposer] Protocol keywords found but screening also detected — skipping protocol_retrieval to avoid noise")

    if has_screening:
        sub_queries.append(SubQuery(
            text=query,
            target="vector_db",
            query_type="screening_retrieval"
        ))

    if any(keyword in query_lower for keyword in CLINGEN_KEYWORDS):
        sub_queries.append(SubQuery(
            text=query,
            target="clingen",
            query_type="rule_retrieval"
        ))

    if not sub_queries:
        sub_queries.append(SubQuery(
            text=query,
            target="vector_db",
            query_type="rule_retrieval"
        ))

    return sub_queries