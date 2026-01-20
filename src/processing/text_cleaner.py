"""
Text Cleaner Module

PURPOSE:
Clean and normalize text extracted from PDFs.
Removes noise, fixes formatting issues, standardizes whitespace.

WHY THIS MATTERS:
- PDFs often have weird line breaks, headers/footers, page numbers
- Clean text = better embeddings = better search results
- Garbage in, garbage out!

USAGE:
    from src.processing.text_cleaner import clean_text
    
    raw_text = "Page 1\\n\\nGuideline   Text\\n\\n\\n\\nPage 2"
    clean = clean_text(raw_text)
"""

import re
from typing import Optional

from src.utils.logger import logger


def clean_text(text: str, aggressive: bool = False) -> str:
    """
    Clean and normalize text from PDF extraction.
    
    Args:
        text: Raw text from PDF
        aggressive: If True, apply more aggressive cleaning (may remove useful content)
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    original_length = len(text)
    
    # Step 1: Normalize unicode characters
    text = normalize_unicode(text)
    
    # Step 2: Fix common PDF extraction issues
    text = fix_pdf_artifacts(text)
    
    # Step 3: Normalize whitespace
    text = normalize_whitespace(text)
    
    # Step 4: Remove page numbers and headers/footers (optional)
    if aggressive:
        text = remove_page_markers(text)
    
    # Step 5: Final cleanup
    text = text.strip()
    
    cleaned_length = len(text)
    reduction = ((original_length - cleaned_length) / original_length * 100) if original_length > 0 else 0
    
    logger.debug(f"Cleaned text: {original_length:,} → {cleaned_length:,} chars ({reduction:.1f}% reduction)")
    
    return text


def normalize_unicode(text: str) -> str:
    """Replace common unicode characters with ASCII equivalents."""
    replacements = {
        '\u2018': "'",   # Left single quote
        '\u2019': "'",   # Right single quote
        '\u201c': '"',   # Left double quote
        '\u201d': '"',   # Right double quote
        '\u2013': '-',   # En dash
        '\u2014': '-',   # Em dash
        '\u2026': '...', # Ellipsis
        '\u00a0': ' ',   # Non-breaking space
        '\u00b7': '-',   # Middle dot
        '\uf0b7': '-',   # Bullet point (private use)
        '\uf0a7': '-',   # Another bullet
    }
    
    for unicode_char, replacement in replacements.items():
        text = text.replace(unicode_char, replacement)
    
    return text


def fix_pdf_artifacts(text: str) -> str:
    """Fix common issues from PDF text extraction."""
    
    # Fix hyphenated words split across lines
    # "guide-\nline" → "guideline"
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    
    # Fix words split across lines without hyphen (common in justified text)
    # This is tricky - only do for obvious cases
    text = re.sub(r'(\w{3,})\n(\w{3,})', r'\1 \2', text)
    
    # Remove form feed characters
    text = text.replace('\f', '\n\n')
    
    # Fix multiple spaces
    text = re.sub(r' +', ' ', text)
    
    return text


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace while preserving paragraph structure."""
    
    # Replace multiple newlines with double newline (paragraph break)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove trailing whitespace from lines
    text = re.sub(r' +\n', '\n', text)
    
    # Remove leading whitespace from lines (except indentation)
    text = re.sub(r'\n +', '\n', text)
    
    return text


def remove_page_markers(text: str) -> str:
    """Remove page numbers and common headers/footers."""
    
    # Remove standalone page numbers
    text = re.sub(r'\n\d{1,4}\n', '\n', text)
    
    # Remove "Page X of Y" patterns
    text = re.sub(r'[Pp]age\s+\d+\s*(of\s+\d+)?', '', text)
    
    # Remove common footer patterns (be careful with these)
    # text = re.sub(r'©.*?\d{4}.*?\n', '', text)  # Copyright lines
    
    return text


def extract_sections(text: str) -> list[dict]:
    """
    Attempt to identify and extract document sections.
    
    Looks for numbered sections like "1. Introduction" or "Section 2:"
    
    Returns:
        List of dicts with 'title' and 'content'
    """
    # Pattern for section headers
    section_pattern = re.compile(
        r'^(?:(?:\d+\.?\s*)+|[A-Z]+\.?\s*|Section\s+\d+:?\s*)([A-Z][^\n]+)',
        re.MULTILINE
    )
    
    sections = []
    matches = list(section_pattern.finditer(text))
    
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        
        sections.append({
            'title': match.group(1).strip(),
            'content': text[start:end].strip(),
            'start_pos': match.start()
        })
    
    return sections


# Quick test when running this file directly
if __name__ == "__main__":
    print("=== Text Cleaner Test ===\n")
    
    # Test with sample messy text
    sample = """
    Page 1
    
    Clinical Guide-
    lines for Variant
    
    Interpretation
    
    
    
    This document provides    guidance for
    the    interpretation of genetic variants.
    
    Page 2
    
    1. Introduction
    
    The American College of Medical Genetics...
    """
    
    print("Original text:")
    print("-" * 40)
    print(sample)
    
    print("\nCleaned text:")
    print("-" * 40)
    cleaned = clean_text(sample, aggressive=True)
    print(cleaned)
    
    print("\n✓ Text cleaner module ready!")
