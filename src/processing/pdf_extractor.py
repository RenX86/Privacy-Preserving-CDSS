"""
PDF Extractor Module

PURPOSE:
Extract text content from PDF files (clinical guidelines, research papers).
Handles multi-page PDFs and preserves structure where possible.

USAGE:
    from src.processing.pdf_extractor import extract_text_from_pdf
    
    text = extract_text_from_pdf("data/raw/acmg_guidelines.pdf")
    print(text[:500])  # First 500 characters
"""

from pathlib import Path
from typing import Optional
import pypdf

from src.utils.logger import logger


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """
    Extract all text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as a single string
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    if not pdf_path.suffix.lower() == ".pdf":
        logger.warning(f"File may not be a PDF: {pdf_path}")
    
    logger.info(f"Extracting text from: {pdf_path.name}")
    
    try:
        text_parts = []
        
        with open(pdf_path, "rb") as file:
            reader = pypdf.PdfReader(file)
            total_pages = len(reader.pages)
            
            logger.info(f"Processing {total_pages} pages...")
            
            for page_num, page in enumerate(reader.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
                
                # Log progress for large documents
                if page_num % 50 == 0:
                    logger.debug(f"Processed {page_num}/{total_pages} pages")
        
        full_text = "\n\n".join(text_parts)
        logger.info(f"✓ Extracted {len(full_text):,} characters from {total_pages} pages")
        
        return full_text
        
    except Exception as e:
        logger.error(f"Failed to extract PDF: {e}")
        raise


def extract_text_with_metadata(pdf_path: str | Path) -> dict:
    """
    Extract text along with PDF metadata.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dict with 'text', 'metadata', 'page_count'
    """
    pdf_path = Path(pdf_path)
    
    with open(pdf_path, "rb") as file:
        reader = pypdf.PdfReader(file)
        
        # Extract metadata
        metadata = {}
        if reader.metadata:
            metadata = {
                "title": reader.metadata.get("/Title", ""),
                "author": reader.metadata.get("/Author", ""),
                "subject": reader.metadata.get("/Subject", ""),
                "creator": reader.metadata.get("/Creator", ""),
            }
        
        # Extract text
        text_parts = [page.extract_text() or "" for page in reader.pages]
        
        return {
            "text": "\n\n".join(text_parts),
            "metadata": metadata,
            "page_count": len(reader.pages),
            "source_file": pdf_path.name
        }


def get_pdf_info(pdf_path: str | Path) -> dict:
    """
    Get basic info about a PDF without extracting all text.
    Useful for previewing before processing.
    """
    pdf_path = Path(pdf_path)
    
    with open(pdf_path, "rb") as file:
        reader = pypdf.PdfReader(file)
        
        # Get first page preview
        first_page_text = ""
        if reader.pages:
            first_page_text = reader.pages[0].extract_text() or ""
            first_page_text = first_page_text[:500] + "..." if len(first_page_text) > 500 else first_page_text
        
        return {
            "filename": pdf_path.name,
            "page_count": len(reader.pages),
            "has_metadata": reader.metadata is not None,
            "first_page_preview": first_page_text
        }


# Quick test when running this file directly
if __name__ == "__main__":
    import sys
    from src.config.settings import settings
    
    print("=== PDF Extractor Test ===\n")
    
    # Check for PDFs in raw data folder
    raw_path = settings.DATA_RAW_PATH
    print(f"Looking for PDFs in: {raw_path}")
    
    if not raw_path.exists():
        print(f"Creating directory: {raw_path}")
        raw_path.mkdir(parents=True, exist_ok=True)
    
    pdf_files = list(raw_path.glob("*.pdf"))
    
    if not pdf_files:
        print("\nNo PDF files found. Add PDFs to data/raw/ folder.")
        print("Example: data/raw/acmg_guidelines.pdf")
    else:
        print(f"\nFound {len(pdf_files)} PDF(s):")
        for pdf in pdf_files:
            info = get_pdf_info(pdf)
            print(f"\n  📄 {info['filename']}")
            print(f"     Pages: {info['page_count']}")
            print(f"     Preview: {info['first_page_preview'][:100]}...")
