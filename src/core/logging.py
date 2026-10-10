"""Standard logging with a ticket ID attached to each processing context."""

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

from .config import get_settings


_ticket_id: ContextVar[str | None] = ContextVar("ticket_id", default=None)


class TicketContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.ticket_id = _ticket_id.get() or "-"
        return True


@contextmanager
def ticket_context(ticket_id: str) -> Iterator[None]:
    """Tag logs emitted while handling a ticket; restore earlier context on exit."""
    token = _ticket_id.set(str(ticket_id))
    try:
        yield
    finally:
        _ticket_id.reset(token)


def configure_logging(level: str | int | None = None) -> None:
    """Install one ticket-aware handler on the root logger."""
    if level is None:
        level = get_settings().log_level
    if isinstance(level, str):
        level = level.upper()
        if level not in logging.getLevelNamesMapping():
            raise ValueError(f"Unknown log level: {level}")

    root = logging.getLogger()
    root.setLevel(level)
    if any(getattr(handler, "_ticket_context_handler", False) for handler in root.handlers):
        return

    handler = logging.StreamHandler()
    handler._ticket_context_handler = True
    handler.addFilter(TicketContextFilter())
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s ticket_id=%(ticket_id)s %(message)s"
        )
    )
    root.addHandler(handler)
