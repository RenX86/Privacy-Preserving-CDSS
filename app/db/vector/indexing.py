import os
import re
import psycopg2
import json
import pymupdf4llm
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from app.models.embeddings import embed_text
from app.config import settings

DOCS_FOLDER = "docs"

# ─── GLOBAL INIT ──────────────────────────────────────────────────────────────
print("Initializing Docling layout models (this may take a moment)...")
_pipeline_options = PdfPipelineOptions(
    do_table_structure=True,
    do_ocr=False,
)
doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=_pipeline_options)
    }
)
print("   [OK] Docling ready.")


# ─── BATCH CONVERTER FOR NCCN ─────────────────────────────────────────────────

def convert_pdf_in_batches(filepath: str, batch_size: int = 10) -> str:
    import fitz
    import tempfile

    src_doc = fitz.open(filepath)
    total_pages = len(src_doc)
    all_markdown = []

    for start in range(0, total_pages, batch_size):
        end = min(start + batch_size, total_pages)
        print(f"   Processing pages {start+1}-{end} of {total_pages}...")

        tmp_doc = fitz.open()
        tmp_doc.insert_pdf(src_doc, from_page=start, to_page=end - 1)

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            tmp_doc.save(tmp_path)
            tmp_doc.close()
            result = doc_converter.convert(tmp_path)
            batch_md = result.document.export_to_markdown()
            all_markdown.append(batch_md)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    src_doc.close()
    return "\n\n".join(all_markdown)


# ─── STEP 1: READ & SCRUB ─────────────────────────────────────────────────────

def read_and_scrub_file(filepath: str, parser: str) -> str | list:
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".pdf":
        if parser == "docling":
            print(f"   [Router] Docling (table-aware) parser...")
            md_text = convert_pdf_in_batches(filepath, batch_size=10)

            # ── NCCN SCRUBBING ────────────────────────────────────────────────
            md_text = re.sub(
                r'PLEASE NOTE that use of this NCCN Content.*?artificial intelligence model or tool\.?\s*',
                '', md_text, flags=re.IGNORECASE | re.DOTALL
            )
            md_text = re.sub(
                r'Printed by.*?All Rights Reserved\.?\s*',
                '', md_text, flags=re.IGNORECASE | re.DOTALL
            )
            md_text = re.sub(
                r'Version \d+\.\d+.*?permission of NCCN\.?\s*',
                '', md_text, flags=re.IGNORECASE | re.DOTALL
            )
            md_text = re.sub(r'(?im)^NCCN Guidelines Index.*?$', '', md_text)
            md_text = re.sub(r'(?im)^Table of Contents\s*$', '', md_text)
            md_text = re.sub(r'(?im)^Discussion\s*$', '', md_text)
            md_text = re.sub(r'(?im)^UPDATES\s*$', '', md_text)
            return md_text

        else:
            print(f"   [Router] PyMuPDF4LLM (page-chunk) parser...")
            # page_chunks=True returns one dict per page — avoids issues with
            # journal PDFs that have minimal font-size-based headers
            page_chunks = pymupdf4llm.to_markdown(filepath, page_chunks=True)

            # ── Journal PDF scrubbing (per page) ─────────────────────────────
            scrubbed = []
            for page in page_chunks:
                text = page["text"]
                text = re.sub(r'(?i)Genet Med\..{0,100}?doi:.*?\n', '', text)
                text = re.sub(r'Author [mM]anuscript\n?', '', text)
                text = re.sub(r'HHS Public Access\n?', '', text)
                text = re.sub(
                    r'(?i)Users may view, print, copy.{0,200}?editorial_policies\n?',
                    '', text, flags=re.DOTALL
                )
                text = text.strip()
                if text:
                    scrubbed.append({
                        "text": text,
                        "metadata": {"page": page.get("metadata", {}).get("page", "?")}
                    })
            # Return list of page dicts directly — no further splitting needed
            return scrubbed

    elif ext == ".txt":
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .pdf or .txt")


# ─── STEP 2: SPLITTING ────────────────────────────────────────────────────────

def semantic_markdown_split(md_text: str) -> list:
    headers_to_split_on = [
        ("#",   "Header_1"),
        ("##",  "Header_2"),
        ("###", "Header_3"),
    ]
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )
    return splitter.split_text(md_text)


def table_aware_split(md_text: str) -> list:
    header_chunks = semantic_markdown_split(md_text)

    result = []
    for chunk in header_chunks:
        content = chunk.page_content.strip()
        if not content:
            continue

        if re.search(r'^\|.+\|', content, re.MULTILINE):
            sub_parts = re.split(r'(\n(?:\|.+\|\n)+)', content)
            for part in sub_parts:
                part = part.strip()
                if part:
                    result.append({"text": part, "metadata": chunk.metadata})
        else:
            result.append({"text": content, "metadata": chunk.metadata})

    return result


# ─── STEP 3: NOISE FILTERING ──────────────────────────────────────────────────

def is_bibliography_section(text: str) -> bool:
    return bool(re.match(r'^\s*\d{1,3}\.\s+[A-Z][a-z]+\s+[A-Z]{1,2}[,\s]', text))


# ─── STEP 4: DATABASE INSERTION ───────────────────────────────────────────────

def insert_chunk(conn, source, category, gene, chunk_text, metadata, parent_context, embedding):
    sql = """
        INSERT INTO medical_documents
            (source, category, gene, chunk_text, metadata, parent_text, embedding)
        VALUES (%s, %s, %s, %s, %s, %s, %s::vector);
    """
    with conn.cursor() as cur:
        cur.execute(sql, (source, category, gene, chunk_text, json.dumps(metadata), parent_context, embedding))
    conn.commit()


# ─── STEP 5: MAIN INDEXING LOGIC ──────────────────────────────────────────────

def index_document(filepath, source, category, gene=None, parser="pymupdf"):
    print(f"\n--- Reading: {os.path.basename(filepath)}")

    result = read_and_scrub_file(filepath, parser)

    # Build sections depending on what read_and_scrub_file returned:
    # - docling  → returns str  → table_aware_split
    # - pymupdf  → returns list → already page-chunked, use directly
    # - txt      → returns str  → semantic_markdown_split
    if parser == "docling":
        sections = table_aware_split(result)
        print(f"   Table-aware split -> {len(sections)} sections")
    elif isinstance(result, list):
        # pymupdf page_chunks path
        sections = result
        print(f"   Page-chunk split -> {len(sections)} sections")
    else:
        # txt fallback
        raw_sections = semantic_markdown_split(result)
        sections = [{"text": s.page_content, "metadata": s.metadata} for s in raw_sections]
        print(f"   Semantic split -> {len(sections)} sections")

    child_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    conn = psycopg2.connect(settings.POSTGRES_URL)
    total_children = 0
    skipped = 0

    try:
        for section in sections:
            content = section["text"].strip()
            metadata = section["metadata"]

            if not content or is_bibliography_section(content):
                skipped += 1
                continue

            metadata_str = " | ".join([f"{k}: {v}" for k, v in metadata.items()])
            parent_context = f"[{metadata_str}]\n{content}" if metadata_str else content

            children = child_splitter.split_text(content)

            for child in children:
                enriched_child = f"[{metadata_str}]\n{child}" if metadata_str else child
                embedding = embed_text(enriched_child)
                insert_chunk(conn, source, category, gene, enriched_child, metadata, parent_context, embedding)
                total_children += 1

    finally:
        conn.close()

    print(f"   [OK] {total_children} child chunks indexed | {skipped} sections skipped")


# ─── STEP 6: DOCUMENT DISCOVERY ───────────────────────────────────────────────

def discover_documents(docs_folder: str) -> list[dict]:
    manifest_path = os.path.join(docs_folder, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    else:
        manifest = {}
        print("[WARN] No manifest.json found -- defaulting all files to pymupdf parser")

    documents = []
    category_map = {"guidelines": "guideline", "protocols": "protocol", "screening": "screening_protocol"}

    for subfolder, category in category_map.items():
        folder_path = os.path.join(docs_folder, subfolder)
        if not os.path.exists(folder_path):
            continue
        for filename in sorted(os.listdir(folder_path)):
            if not filename.endswith((".pdf", ".txt")):
                continue
            meta = manifest.get(filename, {})
            parser = meta.get("parser") or ("docling" if category == "protocol" else "pymupdf")

            documents.append({
                "filepath": os.path.join(folder_path, filename),
                "source":   meta.get("source", filename.rsplit(".", 1)[0]),
                "category": category,
                "gene":     meta.get("gene", None),
                "parser":   parser,
            })
    return documents


if __name__ == "__main__":
    print("*** Starting Hybrid Markdown Document Indexing...")
    documents = discover_documents(DOCS_FOLDER)

    if not documents:
        print("[WARN] No documents found. Add PDFs to docs/guidelines/ or docs/protocols/")
    else:
        print(f"Found {len(documents)} documents:")
        for doc in documents:
            index_document(**doc)

    print("\n[OK] Indexing complete!")