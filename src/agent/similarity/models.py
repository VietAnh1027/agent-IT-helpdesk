"""Data models cho Subsystem Tương đồng & Sự cố diện rộng (Similarity, Duplicate & Incident).

Đặc tả dựa trên:
- documents/02_ARCHITECTURE.md (Mục 3)
- documents/03_DATA_MODEL.md (Mục 1 & 8)
- implementation_plan/12_SIMILARITY_AND_INCIDENT_DETECTION_PLAN.md
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class TicketRecord:
    """Biểu diễn một bản ghi ticket trong kho dữ liệu phục vụ so khớp."""

    id: str
    requester_email: str
    subject: str = ""
    body: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    category: str | None = None
    status: str = "new"
    embedding: list[float] | None = None
    incident_id: str | None = None
    recurring_issue: bool = False
    final_response_text: str | None = None
    sources: list[dict] | None = None


@dataclass
class IncidentRecord:
    """Biểu diễn một sự cố diện rộng gộp nhiều ticket trùng lặp."""

    id: str
    representative_ticket_id: str
    category: str
    ticket_count: int = 1
    status: str = "open"  # 'open' | 'acknowledged' | 'resolved'
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolution_text: str | None = None


@dataclass
class SimilarityMatchResult:
    """Kết quả tìm kiếm tương đồng vector giữa các ticket."""

    ticket_id: str
    similarity: float
    requester_email: str
    status: str
    created_at: datetime
    category: str | None = None

