import httpx

GENOMAD_API_URL = "https://gnomad.broadinstitute.org/api"

def get_allele_frequency(rsid: str) -> dict | None:

    query = """
    query VariantFrequency($variantId: String!) {
        variant(variantId: $variantId, dataset: gnomad_r4) {
            variant_id
            rsids
            genome {
                af
                ac
                an
            }
            exome {
                af
                ac
                an
            }
        }
    }
    """
    try:
        response = httpx.post(
            GENOMAD_API_URL,
            json={"query": query, "variables": {"variantId": rsid}},
            timeout=10.0
        )
        response.raise_for_status()
        data = response.json()

        variant = data.get("data", {}).get("variant")
        if not variant:
            return None

        genome = variant.get("genome") or {}
        exome  = variant.get("exome") or {}
        af     = genome.get("af") or exome.get("af") or 0.0

        return {
            "rsid":             rsid,
            "allele_frequency": af,
            "is_common":        af >= 0.05,
            "ba1_applicable":   af >= 0.05,
        }
    except Exception as e:
        print(f"[gnomAD] Error fetching {rsid}: {e}")
        return None