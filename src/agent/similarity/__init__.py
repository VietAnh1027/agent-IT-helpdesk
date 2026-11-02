"""Similarity Subsystem package: Phát hiện trùng lặp, spam và sự cố diện rộng."""

from src.agent.similarity.models import IncidentRecord, SimilarityMatchResult, TicketRecord
from src.agent.similarity.repository import InMemoryTicketRepository, PostgresTicketRepository, TicketRepository
from src.agent.similarity.ticket_matcher import (
    broadcast_incident_resolution,
    check_or_link_mass_incident,
    check_recurring_resolved,
    check_repeat_unresolved,
    check_spam_flood,
    match_ticket_state,
)

__all__ = [
    "TicketRecord",
    "IncidentRecord",
    "SimilarityMatchResult",
    "TicketRepository",
    "InMemoryTicketRepository",
    "PostgresTicketRepository",
    "check_spam_flood",
    "check_repeat_unresolved",
    "check_or_link_mass_incident",
    "check_recurring_resolved",
    "broadcast_incident_resolution",
    "match_ticket_state",
]

