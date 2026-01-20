"""
Setup Database Script

PURPOSE:
Tests database connection and verifies the schema is ready.
Run this after starting Docker to make sure everything works.

USAGE:
    python scripts/setup_database.py
"""

import sys
from pathlib import Path

# Add project root to path so we can import src modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.db_connection import test_connection, check_pgvector_extension, get_db_cursor, close_connection
from src.database.db_operations import get_document_count
from src.utils.logger import logger


def main():
    print("=" * 50)
    print("  CDSS Database Setup Verification")
    print("=" * 50)
    print()
    
    # Step 1: Test basic connection
    print("1. Testing database connection...")
    if not test_connection():
        print("\n❌ FAILED: Cannot connect to database!")
        print("   Make sure Docker is running:")
        print("   docker-compose up -d")
        return False
    print("   ✓ Connection successful\n")
    
    # Step 2: Check pgvector extension
    print("2. Checking pgvector extension...")
    if not check_pgvector_extension():
        print("\n❌ FAILED: pgvector extension not found!")
        print("   Check that init.sql ran correctly")
        return False
    print("   ✓ pgvector enabled\n")
    
    # Step 3: Check documents table
    print("3. Checking documents table...")
    try:
        with get_db_cursor() as cursor:
            cursor.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'documents'
                ORDER BY ordinal_position
            """)
            columns = cursor.fetchall()
            
            if not columns:
                print("   ❌ documents table not found!")
                return False
                
            print("   ✓ documents table exists")
            print("   Columns:")
            for col_name, col_type in columns:
                print(f"     - {col_name}: {col_type}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    print()
    
    # Step 4: Check document count
    print("4. Checking existing data...")
    count = get_document_count()
    print(f"   Documents in database: {count}")
    print()
    
    # Cleanup
    close_connection()
    
    print("=" * 50)
    print("  ✓ All checks passed! Database is ready.")
    print("=" * 50)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
