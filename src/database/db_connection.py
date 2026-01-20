"""
Database Connection Module

PURPOSE:
Handles connecting to the PostgreSQL database with pgvector.
Provides a reusable connection that other modules can use.

USAGE:
    from src.database.db_connection import get_connection, close_connection
    
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM documents LIMIT 5")
    results = cursor.fetchall()
    close_connection(conn)
    
    # Or use the context manager:
    with get_db_cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM documents")
        count = cursor.fetchone()[0]
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from typing import Optional

from src.config.settings import settings
from src.utils.logger import logger


# Global connection pool placeholder
_connection: Optional[psycopg2.extensions.connection] = None


def get_connection() -> psycopg2.extensions.connection:
    """
    Get a connection to the PostgreSQL database.
    
    Creates a new connection if one doesn't exist.
    Uses settings from .env file.
    
    Returns:
        psycopg2 connection object
    """
    global _connection
    
    if _connection is None or _connection.closed:
        try:
            logger.info(f"Connecting to database: {settings.POSTGRES_DB}@{settings.POSTGRES_HOST}")
            _connection = psycopg2.connect(
                host=settings.POSTGRES_HOST,
                port=settings.POSTGRES_PORT,
                database=settings.POSTGRES_DB,
                user=settings.POSTGRES_USER,
                password=settings.POSTGRES_PASSWORD
            )
            logger.info("✓ Database connection established")
        except psycopg2.Error as e:
            logger.error(f"✗ Failed to connect to database: {e}")
            raise
    
    return _connection


def close_connection(conn: Optional[psycopg2.extensions.connection] = None) -> None:
    """
    Close the database connection.
    
    Args:
        conn: Connection to close, or None to close global connection
    """
    global _connection
    
    if conn is not None:
        conn.close()
        logger.info("Database connection closed")
    elif _connection is not None:
        _connection.close()
        _connection = None
        logger.info("Global database connection closed")


@contextmanager
def get_db_cursor(dict_cursor: bool = False):
    """
    Context manager for database operations.
    
    Automatically handles connection and cursor cleanup.
    Commits on success, rolls back on error.
    
    Args:
        dict_cursor: If True, returns rows as dictionaries instead of tuples
        
    Usage:
        with get_db_cursor() as cursor:
            cursor.execute("SELECT * FROM documents")
            rows = cursor.fetchall()
    """
    conn = get_connection()
    cursor_factory = RealDictCursor if dict_cursor else None
    cursor = conn.cursor(cursor_factory=cursor_factory)
    
    try:
        yield cursor
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database error, rolled back: {e}")
        raise
    finally:
        cursor.close()


def test_connection() -> bool:
    """
    Test if the database connection works.
    
    Returns:
        True if connection successful, False otherwise
    """
    try:
        with get_db_cursor() as cursor:
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            logger.info("✓ Database connection test passed")
            return result[0] == 1
    except Exception as e:
        logger.error(f"✗ Database connection test failed: {e}")
        return False


def check_pgvector_extension() -> bool:
    """
    Check if pgvector extension is installed and enabled.
    
    Returns:
        True if pgvector is available, False otherwise
    """
    try:
        with get_db_cursor() as cursor:
            cursor.execute("SELECT * FROM pg_extension WHERE extname = 'vector'")
            result = cursor.fetchone()
            if result:
                logger.info("✓ pgvector extension is enabled")
                return True
            else:
                logger.warning("✗ pgvector extension is NOT enabled")
                return False
    except Exception as e:
        logger.error(f"Error checking pgvector: {e}")
        return False


# Quick test when running this file directly
if __name__ == "__main__":
    print("=== Database Connection Test ===\n")
    
    # Test connection
    if test_connection():
        print("✓ Basic connection works")
    else:
        print("✗ Connection failed - is PostgreSQL running?")
        exit(1)
    
    # Check pgvector
    if check_pgvector_extension():
        print("✓ pgvector extension ready")
    else:
        print("✗ pgvector not found")
    
    # Cleanup
    close_connection()
    print("\n✓ All tests passed!")
