import re
import httpx

# ClinGen uses search.clinicalgenome.org/api/genes (the old /api/gene-validity endpoint is 404)
CLINGEN_GENES_URL = "https://search.clinicalgenome.org/api/genes"

def get_gene_validity(gene_symbol: str) -> list[dict]:
    """
    Query ClinGen for gene-level validity and curation status.
    Returns a list of dicts (one per matching gene) with fields:
      - symbol, hgnc_id, has_validity, has_actionability, has_dosage,
        has_variant, date_last_curated, location
    """
    try:
        response = httpx.get(
            CLINGEN_GENES_URL,
            params={"page": 1, "pageSize": 5, "search": gene_symbol},
            headers={"Accept": "application/json"},
            timeout=10.0
        )
        response.raise_for_status()
        data = response.json()
        rows = data.get("rows", [])

        # Filter to exact symbol match only (e.g. BRCA1, not BRCA1P1)
        exact = [r for r in rows if r.get("symbol", "").upper() == gene_symbol.upper()]
        results = exact if exact else rows

        print(f"  [ClinGen] {len(results)} record(s) for {gene_symbol}")
        for r in results:
            print(f"    symbol={r.get('symbol')} | validity={r.get('has_validity')} "
                  f"| actionability={r.get('has_actionability')} "
                  f"| curated={r.get('date_last_curated')}")

        return results

    except httpx.TimeoutException:
        print(f"  [ClinGen] Timeout for {gene_symbol}")
        return []
    except httpx.HTTPStatusError as e:
        print(f"  [ClinGen] HTTP {e.response.status_code} for {gene_symbol}")
        return []
    except Exception as e:
        print(f"  [ClinGen] Unexpected error for {gene_symbol}: {e}")
        return []


# Common acronyms that match the gene regex but are NOT genes.
# Extend this list if false positives appear.
_NON_GENE_ACRONYMS = {
    "ACMG", "NCCN", "VUS", "LOF", "DNA", "RNA", "PCR", "SNP", "CNV",
    "RRSO", "MRI", "CT", "PET", "NGS", "WGS", "WES", "FISH", "IHC",
    "HER", "ER", "PR", "HR", "OS", "PFS", "FDA", "EMA", "US", "UK",
}

def extract_gene_from_query(query_text: str) -> str | None:
    """
    Extract a gene symbol from the query using a regex pattern.
    Matches tokens that look like gene symbols (1 uppercase letter + 1-9 uppercase
    letters/digits) and filters out common non-gene acronyms.

    This replaces the previous hardcoded 28-gene allowlist, which silently
    returned None for any gene not in the list (CDH1, MUTYH, BMPR1A, etc.).
    """
    gene_pattern = re.compile(r'\b[A-Z][A-Z0-9]{1,9}\b')
    matches = gene_pattern.findall(query_text)
    for match in matches:
        if match not in _NON_GENE_ACRONYMS:
            return match
    return None