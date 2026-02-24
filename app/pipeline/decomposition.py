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

NCCN_KEYWORDS = ["nccn", "protocol", "screening", "surveillance", "chemotherapy",
                 "radiotherapy", "surgery", "targeted therapy", "biological therapy", "immunotherapy", 
                 "treatment guideline", "oncology", "cancer", "tumor", "mutation", "mutation testing"]

CLINGEN_KEYWORDS = ["clinvar", "clinvar variant", "clinvar variant id", "clinvar variant name",
                     "clinvar variant description", "clinvar variant classification",
                     "clinvar variant classification criteria", "gene validity", "clingen", "gene-specific",
                     "expert panel", "expert panel recommendation", "expert panel recommendation criteria"]


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
            text=f"Get ACMG guidelines for {query}",
            target="vector_db",
            query_type="rule_retrieval"
        ))

    if any(keyword in query_lower for keyword in NCCN_KEYWORDS):
        sub_queries.append(SubQuery(
            text=query,
            target="vector_db",
            query_type="protocol_retrieval"
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