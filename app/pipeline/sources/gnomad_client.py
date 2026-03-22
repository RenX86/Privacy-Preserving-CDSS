import httpx
import psycopg2.extras
from app.config import settings
from app.db.pool import db_conn

GNOMAD_API_URL = "https://gnomad.broadinstitute.org/api"

def _get_variant_position(rsid: str) -> dict | None:
    """
    Look up chromosome, position, ref, alt from our local ClinVar PostgreSQL table.
    We stored this during ClinVar ingestion (GRCh38 coordinates).
    Returns None if not found.
    """
    sql = """
        SELECT chromosome, position, ref_allele, alt_allele
        FROM variants
        WHERE rsid = %s
        LIMIT 1;
    """
    try:
        with db_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, (rsid,))
                row = cur.fetchone()
        return dict(row) if row else None
    except Exception as e:
        print(f"  [gnomAD] DB lookup failed for {rsid}: {e}")
        return None


def get_allele_frequency(rsid: str) -> dict | None:
    """
    Get allele frequency from gnomAD v4 for a given rsID.

    gnomAD's GraphQL API requires positional variant IDs (chrom-pos-ref-alt),
    not rsIDs. We resolve the position from our local ClinVar PostgreSQL table
    first, then query gnomAD with the correct ID format.
    """
    # Step 1: resolve rsID to chromosomal position from local ClinVar DB
    pos_data = _get_variant_position(rsid)
    if not pos_data:
        print(f"  [gnomAD] No position data in local DB for {rsid} — skipping")
        return None

    chrom = pos_data["chromosome"].replace("chr", "")  # gnomAD uses "17" not "chr17"
    pos   = pos_data["position"]
    ref   = pos_data["ref_allele"]
    alt   = pos_data["alt_allele"]

    gnomad_id = f"{chrom}-{pos}-{ref}-{alt}"
    print(f"  [gnomAD] Querying variant {gnomad_id} (from {rsid})")

    # Step 2: query gnomAD with parameterized GraphQL variables (not string interpolation)
    query = """
    query($id: String!) {
      variant(variantId: $id, dataset: gnomad_r4) {
        variant_id
        rsids
        genome { af ac an }
        exome  { af ac an }
      }
    }
    """

    try:
        response = httpx.post(
            GNOMAD_API_URL,
            json={"query": query, "variables": {"id": gnomad_id}},
            timeout=12.0
        )
        response.raise_for_status()
        data = response.json()

        if "errors" in data:
            print(f"  [gnomAD] API error for {gnomad_id}: {data['errors'][0].get('message')}")
            return None

        variant = data.get("data", {}).get("variant")
        if not variant:
            print(f"  [gnomAD] Variant {gnomad_id} not found in gnomAD r4")
            return None

        genome = variant.get("genome") or {}
        exome  = variant.get("exome")  or {}
        af = genome.get("af") or exome.get("af") or 0.0

        print(f"  [gnomAD] AF={af:.6f} for {gnomad_id}")

        return {
            "rsid":             rsid,
            "gnomad_id":        gnomad_id,
            "allele_frequency": af,
            "is_common":        af >= 0.05,
            "ba1_applicable":   af >= 0.05,  # ACMG BA1: >5% AF = stand-alone benign
            "genome_ac":        genome.get("ac"),
            "genome_an":        genome.get("an"),
        }

    except httpx.TimeoutException:
        print(f"  [gnomAD] Timeout querying {gnomad_id}")
        return None
    except Exception as e:
        print(f"  [gnomAD] Error for {gnomad_id}: {e}")
        return None