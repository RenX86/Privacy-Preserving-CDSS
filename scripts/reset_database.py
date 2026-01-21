"""
Reset Database Script

PURPOSE:
Clear and recreate the documents table with new vector dimensions.
Run this when changing embedding models with different dimensions.

USAGE:
    python scripts/reset_database.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.db_connection import get_db_cursor, close_connection, test_connection
from src.config.settings import settings
from src.utils.logger import logger


def reset_database():
    """Drop and recreate the documents table with new dimensions."""
    
    print("=" * 60)
    print("  Database Reset for New Embedding Dimensions")
    print("=" * 60)
    print()
    
    # Confirm
    print(f"⚠️  WARNING: This will DELETE all existing embeddings!")
    print(f"   New dimension: {settings.EMBEDDING_DIMENSION}")
    print()
    
    confirm = input("Type 'YES' to confirm: ")
    if confirm != "YES":
        print("Aborted.")
        return False
    
    print()
    
    # Test connection
    if not test_connection():
        print("❌ Database connection failed!")
        return False
    
    try:
        with get_db_cursor() as cursor:
            # Drop existing table and indexes
            logger.info("Dropping existing table...")
            cursor.execute("DROP TABLE IF EXISTS documents CASCADE")
            
            # Create new table with updated dimensions
            logger.info(f"Creating new table with vector({settings.EMBEDDING_DIMENSION})...")
            cursor.execute(f"""
                CREATE TABLE documents (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding vector({settings.EMBEDDING_DIMENSION}),
                    source_file VARCHAR(255),
                    chunk_index INTEGER,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create index for similarity search
            # Note: pgvector limits indexes to 2000 dimensions
            if settings.EMBEDDING_DIMENSION <= 2000:
                logger.info("Creating IVFFlat index...")
                cursor.execute("""
                    CREATE INDEX documents_embedding_idx 
                    ON documents 
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                """)
            else:
                logger.warning(f"⚠ Skipping vector index (dimension {settings.EMBEDDING_DIMENSION} > 2000 limit)")
                logger.warning("  Similarity searches will use brute-force (slower but works)")
            
            # Create source index
            cursor.execute("CREATE INDEX documents_source_idx ON documents(source_file)")
            
        print()
        print("✅ Database reset complete!")
        print(f"   New vector dimension: {settings.EMBEDDING_DIMENSION}")
        print()
        print("Next step: Run 'python scripts/ingest_documents.py' to re-ingest documents")
        
        return True
        
    except Exception as e:
        logger.error(f"Reset failed: {e}")
        return False
    finally:
        close_connection()


if __name__ == "__main__":
    success = reset_database()
    sys.exit(0 if success else 1)
