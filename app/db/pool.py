"""
Shared PostgreSQL connection pool.
Import db_conn() context manager instead of calling psycopg2.connect() directly.
"""
import logging
import time
from contextlib import contextmanager
from psycopg2 import pool as pg_pool
from app.config import settings

log = logging.getLogger("cdss.pool")

_pool: pg_pool.ThreadedConnectionPool | None = None


def _get_pool() -> pg_pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        _pool = pg_pool.ThreadedConnectionPool(
            minconn=2,
            maxconn=10,
            dsn=settings.POSTGRES_URL
        )
        log.info("PostgreSQL connection pool initialised (min=2, max=10)")
    return _pool


def get_conn(retries: int = 5, delay: float = 0.3):
    """Borrow a connection from the pool with retry on exhaustion."""
    pool = _get_pool()
    for attempt in range(retries):
        try:
            return pool.getconn()
        except pg_pool.PoolError:
            if attempt == retries - 1:
                raise
            log.warning(
                f"Connection pool exhausted — retrying in {delay}s "
                f"(attempt {attempt + 1}/{retries})"
            )
            time.sleep(delay)
            delay *= 1.5  # exponential backoff



def release_conn(conn):
    """Return a connection to the pool."""
    _get_pool().putconn(conn)


@contextmanager
def db_conn():
    """Context manager — borrows a connection and always returns it."""
    conn = get_conn()
    try:
        yield conn
    finally:
        release_conn(conn)
