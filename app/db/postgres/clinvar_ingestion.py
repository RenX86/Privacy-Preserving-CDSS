import os
import psycopg2
import gzip
import csv
import urllib.request
import hashlib
from app.config import settings

CLINVAR_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
LOCAL_FILE = "tmp/variant_summary.txt.gz"
CLINVAR_MD5_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz.md5"
LOCAL_MD5_FILE  = "tmp/variant_summary.txt.gz.md5"
BATCH_SIZE = 1000

def  compute_md5(filepath: str) -> str:
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()

def verify_checksum() -> bool:
    
    print("🔍 Downloading official MD5 checksum from NCBI...")
    urllib.request.urlretrieve(CLINVAR_MD5_URL, LOCAL_MD5_FILE)

    with open(LOCAL_MD5_FILE, "r") as f:
        expected_md5 = f.read().strip().split()[0]
    
    print (f"   Expected MD5 : {expected_md5}")
    print("   Computing MD5 of downloaded file...")
    actual_md5 = compute_md5(LOCAL_FILE)
    print(f"   Actual MD5   : {actual_md5}")

    return actual_md5 == expected_md5

def download_clinvar():
    os.makedirs("tmp", exist_ok=True)

    if os.path.exists(LOCAL_FILE):
        print(f"✅ File already exists: {LOCAL_FILE}")
        print("   Verifying checksum anyway...")
        if verify_checksum():
            print("✅ Checksum valid — using cached file")
            return
        else:
            print("   ❌ Checksum FAILED — file is corrupted! Re-downloading...")
            os.remove(LOCAL_FILE)
    print("⬇️  Downloading ClinVar variant_summary.txt.gz (~250MB)...")

    def progress(count, block_size, total_size):
        mb_done = count * block_size / 1_000_000
        mb_total = total_size / 1_000_000
        print(f"   {mb_done:.1f} / {mb_total:.1f} MB", end="\r")
    
    urllib.request.urlretrieve(CLINVAR_URL, LOCAL_FILE, reporthook=progress)
    print(f"\n✅ Downloaded to {LOCAL_FILE}")

    print("\n🔍 Verifying download integrity...")
    if verify_checksum():
        print("✅ Checksum verified — file is intact!")
    else:
        os.remove(LOCAL_FILE)
        raise RuntimeError(
            "❌ MD5 checksum FAILED after download.\n"
            "The file was corrupted during transfer. Please run the script again."
        )

def is_relevant(row: dict) -> bool:
    if row.get("Assembly") != "GRCh38":
        return False
    significance = row.get("ClinicalSignificance", "").lower()
    if not any(s in significance for s in [
        "pathogenic", "likely pathogenic", "benign", "likely benign"
    ]):
        return False

    review = row.get("ReviewStatus", "").lower()
    if "no assertion" in review or "no interpretation" in review:
        return False
    
    return True

def insert_batch(conn, batch: list[tuple]):
    sql = """
        INSERT INTO variants
            (rsid, gene_symbol, chromosome, position, ref_allele, alt_allele,
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

    conn            = psycopg2.connect(settings.POSTGRES_URL)
    batch           = []
    total_inserted  = 0
    total_skipped   = 0

    with gzip.open(LOCAL_FILE, "rt", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        
        for row in reader:
            if not is_relevant(row):
                total_skipped += 1
                continue

            rsid = row.get("RS# (dbSNP)") or row.get("RS#")
            if not rsid or rsid == "-1":
                continue

            rsid = f"rs{rsid}"

            last_eval = row.get("LastEvaluated", "").strip()
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
                row.get("AlternateAlleleVCF", "").strip() or None,
                row.get("ClinicalSignificance", "").strip(),
                row.get("ReviewStatus", "").strip(),
                row.get("PhenotypeList", "").strip() or None,
                last_eval
            ))

            if len(batch) >= BATCH_SIZE:
                insert_batch(conn, batch)
                total_inserted += len(batch)
                batch = []
                print(f"   Inserted {total_inserted:,} variants so far...", end="\r")
            
    if batch:
        insert_batch(conn, batch)
        total_inserted += len(batch)
    
    conn.close()

    print(f"\n✅ Done! Inserted {total_inserted:,} variants, skipped {total_skipped:,}")

if __name__=="__main__":
    ingest()