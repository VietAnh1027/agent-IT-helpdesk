"""Shared configuration, database access, and logging for the application."""
from .config import get_settings

__all__ = ["get_settings"]