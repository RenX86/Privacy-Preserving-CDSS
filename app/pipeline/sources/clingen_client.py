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


def extract_gene_from_query(query_text: str) -> str | None:
    """
    Extract a known cancer/genetic gene symbol from the query text.
    Checks against a curated list of commonly queried genes.
    """
    known_genes = {
        "BRCA1", "BRCA2", "TP53", "EGFR", "ALK", "ROS1", "MET", "ERBB2",
        "PDGFRA", "KIT", "BRAF", "MSH2", "MLH1", "MSH6", "PMS2",
        "APC", "RB1", "VHL", "KRAS", "PTEN", "CDKN2A", "STK11",
        "ATM", "CHEK2", "PALB2", "RAD51C", "RAD51D", "NBN",
    }

    gene_pattern = re.compile(r'\b[A-Z][A-Z0-9]{1,9}\b')
    matches = gene_pattern.findall(query_text)
    for match in matches:
        if match in known_genes:
            return match
    return None