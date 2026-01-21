"""
Ingest Documents Script

PURPOSE:
Process PDF documents from data/raw/, create embeddings, and store in database.
This is the main script to populate your knowledge base!

USAGE:
    # Ingest all PDFs in data/raw/
    python scripts/ingest_documents.py
    
    # Ingest a specific file
    python scripts/ingest_documents.py --file acmg_guidelines.pdf
    
    # Re-ingest (delete existing and re-process)
    python scripts/ingest_documents.py --reingest
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.settings import settings
from src.utils.logger import logger
from src.processing.pdf_extractor import extract_text_from_pdf
from src.processing.text_cleaner import clean_text
from src.processing.chunker import chunk_text
from src.embedding.generate_embeddings import embed_and_store
from src.database.db_operations import get_document_count, get_unique_sources, delete_by_source
from src.database.db_connection import test_connection, close_connection


def ingest_pdf(pdf_path: Path, reingest: bool = False) -> int:
    """
    Process a single PDF file and store in database.
    
    Args:
        pdf_path: Path to the PDF file
        reingest: If True, delete existing chunks first
        
    Returns:
        Number of chunks ingested
    """
    filename = pdf_path.name
    
    # Check if already ingested
    existing_sources = get_unique_sources()
    if filename in existing_sources:
        if reingest:
            logger.info(f"Re-ingesting {filename} - deleting existing chunks")
            delete_by_source(filename)
        else:
            logger.warning(f"Skipping {filename} - already ingested. Use --reingest to update.")
            return 0
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Processing: {filename}")
    logger.info(f"{'='*50}")
    
    # Step 1: Extract text from PDF
    logger.info("Step 1: Extracting text...")
    raw_text = extract_text_from_pdf(pdf_path)
    logger.info(f"  Extracted {len(raw_text):,} characters")
    
    # Step 2: Clean the text
    logger.info("Step 2: Cleaning text...")
    cleaned_text = clean_text(raw_text)
    logger.info(f"  Cleaned to {len(cleaned_text):,} characters")
    
    # Step 3: Chunk the text
    logger.info("Step 3: Chunking text...")
    chunks = chunk_text(cleaned_text)
    logger.info(f"  Created {len(chunks)} chunks")
    
    # Step 4: Generate embeddings and store
    logger.info("Step 4: Generating embeddings and storing...")
    count = embed_and_store(chunks, source_file=filename)
    
    logger.info(f"✓ Ingested {count} chunks from {filename}")
    return count


def ingest_all_pdfs(reingest: bool = False) -> dict:
    """
    Process all PDFs in the data/raw/ directory.
    
    Returns:
        Dict with ingestion statistics
    """
    raw_path = settings.DATA_RAW_PATH
    
    if not raw_path.exists():
        logger.error(f"Data directory not found: {raw_path}")
        raw_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {raw_path}")
        return {"total": 0, "ingested": 0, "skipped": 0}
    
    pdf_files = list(raw_path.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {raw_path}")
        logger.info("Add your clinical guideline PDFs to this folder.")
        return {"total": 0, "ingested": 0, "skipped": 0}
    
    logger.info(f"Found {len(pdf_files)} PDF file(s) to process")
    
    stats = {"total": len(pdf_files), "ingested": 0, "skipped": 0, "chunks": 0}
    
    for pdf_path in pdf_files:
        try:
            count = ingest_pdf(pdf_path, reingest=reingest)
            if count > 0:
                stats["ingested"] += 1
                stats["chunks"] += count
            else:
                stats["skipped"] += 1
        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}")
            stats["skipped"] += 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Ingest PDF documents into the knowledge base")
    parser.add_argument("--file", type=str, help="Specific PDF file to ingest (filename only)")
    parser.add_argument("--reingest", action="store_true", help="Re-ingest existing documents")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  CDSS Document Ingestion")
    print("=" * 60)
    print()
    
    # Test database connection first
    logger.info("Checking database connection...")
    if not test_connection():
        logger.error("Database connection failed! Make sure Docker is running.")
        return 1
    
    # Show current state
    current_count = get_document_count()
    current_sources = get_unique_sources()
    logger.info(f"Current database: {current_count} chunks from {len(current_sources)} sources")
    
    # Process documents
    if args.file:
        # Process specific file
        pdf_path = settings.DATA_RAW_PATH / args.file
        if not pdf_path.exists():
            logger.error(f"File not found: {pdf_path}")
            return 1
        
        count = ingest_pdf(pdf_path, reingest=args.reingest)
        stats = {"total": 1, "ingested": 1 if count > 0 else 0, "skipped": 0 if count > 0 else 1, "chunks": count}
    else:
        # Process all PDFs
        stats = ingest_all_pdfs(reingest=args.reingest)
    
    # Summary
    print("\n" + "=" * 60)
    print("  Ingestion Complete!")
    print("=" * 60)
    print(f"  Files processed: {stats['total']}")
    print(f"  Successfully ingested: {stats['ingested']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Total chunks created: {stats.get('chunks', 0)}")
    print()
    
    # Final count
    final_count = get_document_count()
    final_sources = get_unique_sources()
    print(f"  Database now contains: {final_count} chunks from {len(final_sources)} sources")
    
    close_connection()
    return 0


if __name__ == "__main__":
    sys.exit(main())
