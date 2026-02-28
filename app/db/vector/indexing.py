import os
import re
import psycopg2
import json
import fitz   # PyMuPDF — reads PDFs
from sentence_transformers import SentenceTransformer
from app.config import settings

DOCS_FOLDER = "docs"
_model      = SentenceTransformer(settings.EMBEDDING_MODEL)


# ─── STEP 1: READ THE FILE ────────────────────────────────────────────────────

def read_file(filepath: str) -> str:
    
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".pdf":
        doc = fitz.open(filepath)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text

    elif ext == ".txt":
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .pdf or .txt")


# ─── STEP 2: SEMANTIC SPLITTER ────────────────────────────────────────────────

SECTION_HEADER = re.compile(
    r'(?=^(?:[A-Z]{2,5}\d?\s[-–]|#{1,3}\s|\d+\.\s[A-Z]))',
    re.MULTILINE
)

def semantic_split(text: str) -> list[str]:
    
    sections = SECTION_HEADER.split(text)
    sections = [s.strip() for s in sections if len(s.strip()) > 80]

    if len(sections) <= 1:
        sections = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 80]

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


# ─── STEP 5: INDEX ONE DOCUMENT ───────────────────────────────────────────────

def index_document(filepath, source, category, gene=None):
    print(f"\n📄 Reading: {os.path.basename(filepath)}")

    text            = read_file(filepath)
    parent_sections = semantic_split(text)

    print(f"   Semantic split → {len(parent_sections)} sections")

    conn           = psycopg2.connect(settings.POSTGRES_URL)
    total_children = 0

    try:
        for parent in parent_sections:
            children = make_child_chunks(parent)
            for child in children:
                embedding = _model.encode(child).tolist()
                insert_chunk(conn, source, category, gene, child, parent, embedding)
                total_children += 1
    finally:
        conn.close()

    print(f"   ✅ {total_children} child chunks indexed")

def discover_documents(docs_folder: str) -> list[dict]:

    manifest_path = os.path.join(docs_folder, "manifest.json")

    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    else:
        manifest = []
        print("⚠️  No manifest.json found — using filenames as source names")

    documents = []
    category_map = {
        "guidelines": "guideline",
        "protocols": "protocol",
    }

    for subfolder, category in category_map.items():
        subfolder_path = os.path.join(docs_folder, subfolder)
        if not os.path.exists(folder_path):
            continue
        for filename in sorted(os.listdir(folder_path)):
            if not filename.endswith((".pdf", ".txt")):
                continue
            meta = manifest.get(filename, {})
            documents.append({
                "filepath": os.path.join(folder_path, filename),
                "source": meta.get("source", filename.resplit(".",1)[0]),
                "category": category,
                "gene": meta.get("gene", None)
            })

    return documents

if __name__=="__main__":
    print("🚀 Starting semantic document indexing...")

    documents = discover_documents(DOCS_FOLDER)

    if not documents:
        print("⚠️  No documents found. Add PDFs to docs/guidelines/ or docs/protocols/")
    else:
        print(f" Found {len(documents)} documents:")
        for doc in documents:
            index_document(**doc)

    print("\n✅ Indexing complete!")


