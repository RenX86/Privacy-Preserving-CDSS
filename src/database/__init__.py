"""Database module - exports connection and operations"""
from .db_connection import get_connection, close_connection, get_db_cursor, test_connection
from .db_operations import insert_document, insert_documents_batch, search_similar

__all__ = [
    "get_connection",
    "close_connection", 
    "get_db_cursor",
    "test_connection",
    "insert_document",
    "insert_documents_batch",
    "search_similar"
]
