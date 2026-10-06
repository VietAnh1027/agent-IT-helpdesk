"""Thread-safe psycopg2 pool and transaction-scoped connections."""

from contextlib import contextmanager
from threading import Lock
from typing import Iterator

from psycopg2.extensions import connection as PgConnection
from psycopg2.pool import ThreadedConnectionPool

from .config import Settings, get_settings


_pool: ThreadedConnectionPool | None = None
_pool_lock = Lock()


def create_pool(settings: Settings | None = None) -> ThreadedConnectionPool:
    """Create a pool without opening one merely by importing this module."""
    settings = settings or get_settings()
    return ThreadedConnectionPool(
        settings.db_pool_min,
        settings.db_pool_max,
        **settings.db_connect_kwargs,
    )


def get_pool() -> ThreadedConnectionPool:
    """Return the process-wide pool, creating it on first use."""
    global _pool
    with _pool_lock:
        if _pool is None or _pool.closed:
            _pool = create_pool()
        return _pool


@contextmanager
def get_connection() -> Iterator[PgConnection]:
    """Commit on success, roll back on failure, then return the connection."""
    pool = get_pool()
    conn = pool.getconn()
    try:
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
    finally:
        pool.putconn(conn)


def close_pool() -> None:
    """Close every pooled connection after request processing has stopped."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.closeall()
            _pool = None
