import os
import re
import psycopg2
import json
import fitz   # PyMuPDF — installed as 'pymupdf', imported as 'fitz'
from app.models.embeddings import embed_text
from app.config import settings

DOCS_FOLDER = "docs"


# ─── STEP 1: READ THE FILE ────────────────────────────────────────────────────

def read_file(filepath: str) -> str:

    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".pdf":
        doc = fitz.open(filepath)
        pages = []
        for page in doc:
            page_text = page.get_text().strip()
            if page_text:
                pages.append(page_text)
        doc.close()
        # Join pages with double newlines as a natural boundary marker
        return "\n\n".join(pages)

    elif ext == ".txt":
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .pdf or .txt")


# ─── STEP 2: SEMANTIC SPLITTER ────────────────────────────────────────────────

# Matches common clinical document section headers:
# - Numbered: "1. Introduction", "1.1 Background", "Table 1", "Figure 2"
# - ALL CAPS lines: "INTRODUCTION", "METHODS"
# - Title-case lines followed by colon: "Criteria:"
# - Markdown: "## Section"
SECTION_HEADER = re.compile(
    r'(?=^(?:'
    r'\d+\.\d*\s+[A-Z]'              # 1. Title or 1.1 Title
    r'|\d+\.\s+[A-Z]'                # 1. Title
    r'|[A-Z]{3,}(?:\s+[A-Z]{3,})*\s*$'  # ALL CAPS HEADER
    r'|#{1,3}\s'                     # Markdown ## Header
    r'|Table\s+\d+'                  # Table 1, Table 2
    r'|Figure\s+\d+'                 # Figure 1
    r'|Appendix'                     # Appendix sections
    r'))',
    re.MULTILINE
)

MIN_SECTION_CHARS = 200   # ignore tiny fragments like title pages

def semantic_split(text: str, target_sections: int = 30) -> list[str]:
    """
    Split document text into meaningful parent sections.
    Falls back progressively to ensure we always get multiple sections.
    """
    # Attempt 1: regex-based section headers
    sections = SECTION_HEADER.split(text)
    sections = [s.strip() for s in sections if len(s.strip()) >= MIN_SECTION_CHARS]

    if len(sections) >= 5:
        print(f"   Regex split -> {len(sections)} sections")
        return sections

    # Attempt 2: split on double-newlines (paragraph boundaries = page boundaries from read_file)
    sections = [p.strip() for p in text.split("\n\n") if len(p.strip()) >= MIN_SECTION_CHARS]

    if len(sections) >= 5:
        print(f"   Paragraph split -> {len(sections)} sections")
        return sections

    # Attempt 3: fixed-size chunking — always works regardless of PDF format
    print("   [WARN] Falling back to fixed-size chunking")
    chunk_size = max(500, len(text) // target_sections)
    sections = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size].strip()
        if len(chunk) >= MIN_SECTION_CHARS:
            sections.append(chunk)
    print(f"   Fixed-size split -> {len(sections)} sections")
    return sections


# ─── STEP 3: SMALL-TO-BIG CHUNKING ───────────────────────────────────────────

def make_child_chunks(parent_text: str, max_chars: int = 200) -> list[str]:
    
    sentences = re.split(r'(?<=[.!?])\s+', parent_text)
    children  = []
    current   = ""

    for sentence in sentences:
        if len(current) + len(sentence) <= max_chars:
            current += " " + sentence
        else:
            if current.strip():
                children.append(current.strip())
            current = sentence

    if current.strip():
        children.append(current.strip())

    return [c for c in children if len(c) > 40]


# ─── STEP 4: DATABASE INSERT ──────────────────────────────────────────────────

def insert_chunk(conn, source, category, gene, child_text, parent_text, embedding):
    sql = """
        INSERT INTO medical_documents
            (source, category, gene, chunk_text, parent_text, embedding)
        VALUES (%s, %s, %s, %s, %s, %s::vector);
    """
    with conn.cursor() as cur:
        cur.execute(sql, (source, category, gene, child_text, parent_text, embedding))
    conn.commit()


# Patterns that identify preamble/metadata sections - these should NOT be indexed
NOISE_PATTERNS = [
    "Conflicts of Interest",
    "Current address",
    "Disclaimer",
    "Acknowledgments",
    "Correspondence to",
    "No commercial conflict",
    "The following workgroup",
    "ARUP Institute",
    "Elaine Lyon",
    "Department of Pathology, ARUP",
    "These ACMG Standards and Guidelines were developed",
    "Adherence to these standards",
    "Copyright",
    "All rights reserved",
]

def is_noise_section(text: str) -> bool:
    """Return True if this section is preamble/metadata -- should NOT be indexed."""
    sample = text[:400]   # only check the opening to avoid false positives
    return any(pattern.lower() in sample.lower() for pattern in NOISE_PATTERNS)


# --- STEP 5: INDEX ONE DOCUMENT ---

def index_document(filepath, source, category, gene=None):
    print(f"\n--- Reading: {os.path.basename(filepath)}")

    text            = read_file(filepath)
    parent_sections = semantic_split(text)

    print(f"   Semantic split -> {len(parent_sections)} sections")

    conn           = psycopg2.connect(settings.POSTGRES_URL)
    total_children = 0

    try:
        skipped = 0
        for parent in parent_sections:
            if is_noise_section(parent):
                skipped += 1
                continue
            children = make_child_chunks(parent)
            for child in children:
                embedding = embed_text(child)
                insert_chunk(conn, source, category, gene, child, parent, embedding)
                total_children += 1
    finally:
        conn.close()

    print(f"   [OK] {total_children} child chunks indexed | {skipped} noise sections skipped")

def discover_documents(docs_folder: str) -> list[dict]:

    manifest_path = os.path.join(docs_folder, "manifest.json")

    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    else:
        manifest = {}
        print("[WARN] No manifest.json found -- using filenames as source names")

    documents = []
    category_map = {
        "guidelines": "guideline",
        "protocols": "protocol",
    }

    for subfolder, category in category_map.items():
        folder_path = os.path.join(docs_folder, subfolder)
        if not os.path.exists(folder_path):
            continue
        for filename in sorted(os.listdir(folder_path)):
            if not filename.endswith((".pdf", ".txt")):
                continue
            meta = manifest.get(filename, {})
            documents.append({
                "filepath": os.path.join(folder_path, filename),
                "source": meta.get("source", filename.rsplit(".",1)[0]),
                "category": category,
                "gene": meta.get("gene", None)
            })

    return documents

if __name__=="__main__":
    print("*** Starting semantic document indexing...")

    documents = discover_documents(DOCS_FOLDER)

    if not documents:
        print("[WARN] No documents found. Add PDFs to docs/guidelines/ or docs/protocols/")
    else:
        print(f" Found {len(documents)} documents:")
        for doc in documents:
            index_document(**doc)

    print("\n[OK] Indexing complete!")
