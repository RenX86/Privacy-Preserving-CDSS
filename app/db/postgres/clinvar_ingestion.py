import os
import psycopg2
import gzip
import csv
import urllib.request
from app.config import settings

CLINVAR_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
LOCAL_FILE = "tmp/variant_summary.txt.gz"
BATCH_SIZE = 1000

def download_clinvar():
    os.makedirs("tmp", exist_ok=True)

    if os.path.exists(LOCAL_FILE):
        print(f"✅ Already downloaded: {LOCAL_FILE} (delete it to re-download)")
        return

    print("⬇️  Downloading ClinVar variant_summary.txt.gz (~440MB)...")
    print("This is a one time donwload. Please wait...")

    def progress_hook(count, block_size, total_size):
        mb_done = count * block_size / 1_000_000
        mb_total = total_size / 1_000_000
        print(f"     {mb_done:.1f} / {mb_total:.1f} MB", end="\r")

    urllib.request.urlretrieve(CLINVAR_URL, LOCAL_FILE, reporthook=progress)
    print(f"\n✅ Downloaded to {LOCAL_FILE}")

def is_relevant(row: dict) -> bool:
    if row.get("Assembly") != "GRCh38":
        return False
    significance = row.get("Clinical Significance", "").lower()
    if "no assertion" in review or "no interpretation" in review:
        return False
    
    return True

def insert_batch(conn, batch: list[tuple]):
    sql = """
        INSERT INTO variants
            (rsid, gene_symbol, chromosome, position, ref_allele, alt_allele, alt_allele,
            clinical_significance, review_status, condition, last_evaluated)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (rsid) DO UPDATE SET
            clinical_significance = EXCLUDED.clinical_significance,
            review_status         = EXCLUDED.review_status,
            last_evaluated        = EXCLUDED.last_evaluated;
    """
    with conn.cursor() as cur:
        cur.executemany(sql, batch)
    conn.commit()

def ingest():
    download_clinvar()

    print("\n📊 Parsing and ingesting into PostgreSQL...")

    conn = psycopg2.connect(settings.POSTGRES_URL)
    batch = []
    total_inserted = 0
    total_skipped = 0

    with gzip.open(LOCAL_FILE, "rt", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        
        for row in reader:
            if not is_relevant(row):
                continue

            rsid = row.get("RS# (dbSNP)") or row.get("RS#")
            if not rsid or rsid == "-1":
                continue

            rsid = f"rs{rsid}"

            last_eval = row.get("Last evaluated", "").strip()
            last_eval = last_eval if last_eval not in ("", "-") else None

            try:
                position = int(row.get("Start", 0))
            except ValueError:
                position = None

            batch.append((
                rsid,
                row.get("GeneSymbol", "").strip() or None,
                row.get("Chromosome", "").strip() or None,
                position,
                row.get("ReferenceAlleleVCF", "").strip() or None,
                row.get("AlterbateAlleleVCF", "").strip() or None,
                row.get("ClicnicalSignficance" "").strip(),
                row.get("Reviewstatus", "").strip(),
                row.get("PhenotypeList", "").strip() or None,
                last_eval
            ))

    if batch:
        insert_batch(conn, batch)
        total_inserted += len(batch)
    
    conn.close()

    print(f"\n✅ Done! Inserted {total_inserted:,} variants, skipped {total_skipped:,}")

if __name__=="__main__":
    ingest()