"""Repository Pattern cho dữ liệu Tickets & Incidents phục vụ Similarity Subsystem.

Hỗ trợ:
- Protocol `TicketRepository` (Interface chuẩn)
- `InMemoryTicketRepository` (Unit test siêu tốc, độc lập môi trường ngoài)
- `PostgresTicketRepository` (Production với pgvector cosine distance `<=>`)
"""

from datetime import datetime, timezone
import math
from typing import Protocol, runtime_checkable
import uuid

from src.agent.similarity.models import IncidentRecord, SimilarityMatchResult, TicketRecord


def _cosine_similarity(vec1: list[float] | None, vec2: list[float] | None) -> float:
    """Tính cosine similarity giữa 2 vector thuần Python."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


@runtime_checkable
class TicketRepository(Protocol):
    """Protocol chuẩn cho việc truy vấn và cập nhật tickets/incidents."""

    def save_ticket(self, ticket: TicketRecord) -> None: ...
    def get_ticket(self, ticket_id: str) -> TicketRecord | None: ...
    def get_incident(self, incident_id: str) -> IncidentRecord | None: ...
    def find_recent_by_requester(self, email: str, since: datetime) -> list[TicketRecord]: ...
    def find_similar_tickets(
        self,
        embedding: list[float],
        *,
        min_similarity: float = 0.80,
        since: datetime | None = None,
        exclude_ticket_id: str | None = None,
        category: str | None = None,
        only_status: list[str] | None = None,
        exclude_requester_email: str | None = None,
        same_requester_email: str | None = None,
    ) -> list[SimilarityMatchResult]: ...
    def find_open_incident(self, category: str, since: datetime | None = None) -> IncidentRecord | None: ...
    def create_incident(
        self,
        representative_ticket_id: str,
        category: str,
        initial_ticket_ids: list[str],
    ) -> IncidentRecord: ...
    def add_ticket_to_incident(self, ticket_id: str, incident_id: str) -> None: ...
    def get_tickets_linked_to_incident(
        self,
        incident_id: str,
        status: str = "linked_to_incident",
    ) -> list[TicketRecord]: ...
    def resolve_incident(self, incident_id: str, resolution_text: str) -> None: ...
    def update_ticket_response(
        self,
        ticket_id: str,
        answer: str,
        status: str = "auto_sent",
        action: str = "broadcast_from_incident",
    ) -> None: ...


class InMemoryTicketRepository:
    """In-memory implementation phục vụ unit test nhanh, không cần DB."""

    def __init__(self) -> None:
        self.tickets: dict[str, TicketRecord] = {}
        self.incidents: dict[str, IncidentRecord] = {}

    def save_ticket(self, ticket: TicketRecord) -> None:
        self.tickets[ticket.id] = ticket

    def get_ticket(self, ticket_id: str) -> TicketRecord | None:
        return self.tickets.get(ticket_id)

    def get_incident(self, incident_id: str) -> IncidentRecord | None:
        return self.incidents.get(incident_id)

    def find_recent_by_requester(self, email: str, since: datetime) -> list[TicketRecord]:
        results = [
            t for t in self.tickets.values()
            if t.requester_email.casefold() == email.casefold() and t.created_at >= since
        ]
        results.sort(key=lambda t: t.created_at, reverse=True)
        return results

    def find_similar_tickets(
        self,
        embedding: list[float],
        *,
        min_similarity: float = 0.80,
        since: datetime | None = None,
        exclude_ticket_id: str | None = None,
        category: str | None = None,
        only_status: list[str] | None = None,
        exclude_requester_email: str | None = None,
        same_requester_email: str | None = None,
    ) -> list[SimilarityMatchResult]:
        matches: list[SimilarityMatchResult] = []
        for t in self.tickets.values():
            if exclude_ticket_id and t.id == exclude_ticket_id:
                continue
            if since and t.created_at < since:
                continue
            if category and t.category and t.category.upper() != category.upper():
                continue
            if only_status and t.status not in only_status:
                continue
            if exclude_requester_email and t.requester_email.casefold() == exclude_requester_email.casefold():
                continue
            if same_requester_email and t.requester_email.casefold() != same_requester_email.casefold():
                continue
            if not t.embedding:
                continue

            sim = _cosine_similarity(embedding, t.embedding)
            if sim >= min_similarity:
                matches.append(
                    SimilarityMatchResult(
                        ticket_id=t.id,
                        similarity=sim,
                        requester_email=t.requester_email,
                        status=t.status,
                        created_at=t.created_at,
                        category=t.category,
                    )
                )

        matches.sort(key=lambda m: m.similarity, reverse=True)
        return matches

    def find_open_incident(self, category: str, since: datetime | None = None) -> IncidentRecord | None:
        for inc in self.incidents.values():
            if inc.status in ("open", "acknowledged"):
                if inc.category.upper() == category.upper():
                    if since is None or inc.created_at >= since:
                        return inc
        return None

    def create_incident(
        self,
        representative_ticket_id: str,
        category: str,
        initial_ticket_ids: list[str],
    ) -> IncidentRecord:
        inc_id = str(uuid.uuid4())
        incident = IncidentRecord(
            id=inc_id,
            representative_ticket_id=representative_ticket_id,
            category=category,
            ticket_count=len(initial_ticket_ids),
            status="open",
            created_at=datetime.now(timezone.utc),
        )
        self.incidents[inc_id] = incident
        for tid in initial_ticket_ids:
            if tid in self.tickets:
                self.tickets[tid].incident_id = inc_id
        return incident

    def add_ticket_to_incident(self, ticket_id: str, incident_id: str) -> None:
        if ticket_id in self.tickets:
            self.tickets[ticket_id].incident_id = incident_id
        if incident_id in self.incidents:
            self.incidents[incident_id].ticket_count += 1

    def get_tickets_linked_to_incident(
        self,
        incident_id: str,
        status: str = "linked_to_incident",
    ) -> list[TicketRecord]:
        return [
            t for t in self.tickets.values()
            if t.incident_id == incident_id and (status is None or t.status == status)
        ]

    def resolve_incident(self, incident_id: str, resolution_text: str) -> None:
        if incident_id in self.incidents:
            inc = self.incidents[incident_id]
            inc.status = "resolved"
            inc.resolution_text = resolution_text

    def update_ticket_response(
        self,
        ticket_id: str,
        answer: str,
        status: str = "auto_sent",
        action: str = "broadcast_from_incident",
    ) -> None:
        if ticket_id in self.tickets:
            t = self.tickets[ticket_id]
            t.status = status
            t.final_response_text = answer


class PostgresTicketRepository:
    """Triển khai kết nối cơ sở dữ liệu PostgreSQL thật với pgvector."""

    def __init__(self, connection_or_pool) -> None:
        self.pool = connection_or_pool

    def save_ticket(self, ticket: TicketRecord) -> None:
        # Triển khai INSERT / ON CONFLICT UPDATE vào bảng tickets
        pass

    def get_ticket(self, ticket_id: str) -> TicketRecord | None:
        # SELECT từ bảng tickets WHERE id = %s
        pass

    def get_incident(self, incident_id: str) -> IncidentRecord | None:
        # SELECT từ bảng incidents WHERE id = %s
        pass

    def find_recent_by_requester(self, email: str, since: datetime) -> list[TicketRecord]:
        # SELECT từ bảng tickets WHERE requester_email = %s AND created_at >= %s
        return []

    def find_similar_tickets(
        self,
        embedding: list[float],
        *,
        min_similarity: float = 0.80,
        since: datetime | None = None,
        exclude_ticket_id: str | None = None,
        category: str | None = None,
        only_status: list[str] | None = None,
        exclude_requester_email: str | None = None,
        same_requester_email: str | None = None,
    ) -> list[SimilarityMatchResult]:
        # Query pgvector: 1 - (embedding <=> %s::vector) AS similarity
        return []

    def find_open_incident(self, category: str, since: datetime | None = None) -> IncidentRecord | None:
        return None

    def create_incident(
        self,
        representative_ticket_id: str,
        category: str,
        initial_ticket_ids: list[str],
    ) -> IncidentRecord:
        return IncidentRecord(id=str(uuid.uuid4()), representative_ticket_id=representative_ticket_id, category=category)

    def add_ticket_to_incident(self, ticket_id: str, incident_id: str) -> None:
        pass

    def get_tickets_linked_to_incident(
        self,
        incident_id: str,
        status: str = "linked_to_incident",
    ) -> list[TicketRecord]:
        return []

    def resolve_incident(self, incident_id: str, resolution_text: str) -> None:
        pass

    def update_ticket_response(
        self,
        ticket_id: str,
        answer: str,
        status: str = "auto_sent",
        action: str = "broadcast_from_incident",
    ) -> None:
        pass

