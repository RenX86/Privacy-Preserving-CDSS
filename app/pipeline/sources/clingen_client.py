import httpx
from app.config import settings

def get_gene_validity(get_symbol: str) -> list[dict]:
    url = f"{settings.CLINGEN_API_URL}/gene-validity"
    params = {
        "search": gene_symbol,
        "limit": 10
    }

    try:
        
        response = httpx.get(url, params=params, timeout=10.0)
        response.raise_for_status()
        data = response.json()

        return data.get("results", [])
    
    except httpx.TimeoutExeption:

        print(f"[Clingen] HTTP error for {gene_symbol}")
        return []

    except httpx.HTTPStatusError as e:
        print(f"[Clingen] HTTP error for {gene_symbol}: {e.response.status_code}")
        return []

    except Exception as e:
        print(f"[Clingen] Unexpected error for {gene_symbol}: {e}")
        return []

def extract_gene_from_query(query_text: str) -> str | None:

    import re 

    gene_pattern = re.compile(r'\b[A-Z][A-Z0-9]{1,9}\b')

    known_genes = {
        "BRCA1", "BRCA2", "TP53", "EGFR", "ALK", "ROS1", "MET", "HER2", "PDGFRA", "KIT",
        "BRAF", "KIT", "PDGFRA", "EGFR", "ALK", "ROS1", "MET", "HER2", "PDGFRA", "KIT",
        "MSH6", "PMS2", "APC", "RB1", "VHL", "EGFR", "KRAS"
    }

    matches = gene_pattern.findall(query_text)

    for match in matches:
        if match in known_genes:
            return match
    
    return None