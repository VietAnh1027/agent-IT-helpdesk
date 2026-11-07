"""Core TicketMatcher: Xử lý 4 nghiệp vụ so khớp tương đồng, phát hiện trùng lặp & sự cố diện rộng.

Đặc tả dựa trên:
- documents/02_ARCHITECTURE.md (Mục 3, 12)
- documents/03_DATA_MODEL.md (Mục 1, 6, 8)
- implementation_plan/12_SIMILARITY_AND_INCIDENT_DETECTION_PLAN.md
"""

from datetime import datetime, timedelta, timezone

from src.agent.similarity.models import IncidentRecord, TicketRecord
from src.agent.similarity.repository import TicketRepository
from src.agent.state import AgentState
from src.core.config import get_settings


def check_spam_flood(
    requester_email: str,
    repo: TicketRepository,
    current_time: datetime | None = None,
    window_minutes: int | None = None,
) -> bool:
    """Kiểm tra kịch bản 1: Spam do bực bội (1 người gửi dồn dập trong 5 phút)."""
    settings = get_settings()
    now = current_time or datetime.now(timezone.utc)
    window = window_minutes if window_minutes is not None else settings.similarity_spam_window_minutes
    since = now - timedelta(minutes=window)

    recent_tickets = repo.find_recent_by_requester(requester_email, since)
    return len(recent_tickets) > 0


def check_repeat_unresolved(
    requester_email: str,
    embedding: list[float],
    repo: TicketRepository,
    current_time: datetime | None = None,
    window_days: int | None = None,
    min_similarity: float | None = None,
    current_ticket_id: str | None = None,
) -> tuple[bool, str | None]:
    """Kiểm tra kịch bản 2: Repeat-unresolved (1 người hỏi lại cùng vấn đề trong 7 ngày)."""
    settings = get_settings()
    now = current_time or datetime.now(timezone.utc)
    window = window_days if window_days is not None else settings.similarity_repeat_window_days
    since = now - timedelta(days=window)
    threshold = min_similarity if min_similarity is not None else settings.similarity_repeat_threshold

    matches = repo.find_similar_tickets(
        embedding=embedding,
        min_similarity=threshold,
        since=since,
        same_requester_email=requester_email,
        exclude_ticket_id=current_ticket_id,
    )

    if matches:
        return True, matches[0].ticket_id
    return False, None


def check_or_link_mass_incident(
    ticket_id: str,
    requester_email: str,
    category: str,
    embedding: list[float],
    repo: TicketRepository,
    current_time: datetime | None = None,
    window_minutes: int | None = None,
    threshold: int | None = None,
    min_similarity: float | None = None,
) -> tuple[str | None, bool]:
    """Kiểm tra kịch bản 3: Mass Incident (Sự cố diện rộng 2 giai đoạn: candidate và confirmed).

    Trả về (incident_id, is_linked_to_existing).
    - Nếu incident đã open từ trước: gộp vé vào incident, is_linked_to_existing = True.
    - Nếu đạt mốc vé thứ N (kích hoạt tạo incident mới): tạo incident, gán hồi tố, is_linked_to_existing = False.
    - Nếu chưa đủ ngưỡng: (None, False).
    """
    settings = get_settings()
    now = current_time or datetime.now(timezone.utc)
    window = window_minutes if window_minutes is not None else settings.similarity_incident_window_minutes
    since = now - timedelta(minutes=window)
    target_thresh = threshold if threshold is not None else settings.similarity_incident_min_tickets
    min_sim = min_similarity if min_similarity is not None else settings.similarity_incident_threshold

    # Bước A: Kiểm tra xem đã có incident nào đang mở trong cùng category không
    open_incident = repo.find_open_incident(category=category, since=since)
    if open_incident:
        repo.add_ticket_to_incident(ticket_id, open_incident.id)
        return open_incident.id, True

    # Bước B: Giai đoạn candidate - tìm các ticket tương đồng từ người khác trong cửa sổ window_minutes
    candidates = repo.find_similar_tickets(
        embedding=embedding,
        min_similarity=min_sim,
        since=since,
        category=category,
        exclude_ticket_id=ticket_id,
        exclude_requester_email=requester_email,
    )

    # Đếm số người dùng khác biệt gửi ticket tương đồng
    distinct_users: dict[str, str] = {}  # email -> ticket_id
    for c in candidates:
        email_key = c.requester_email.casefold()
        if email_key not in distinct_users:
            distinct_users[email_key] = c.ticket_id

    total_unique_count = len(distinct_users) + 1  # Cộng thêm ticket hiện tại

    # Nếu chạm hoặc vượt ngưỡng N vé (mặc định 5)
    if total_unique_count >= target_thresh:
        # Lấy danh sách ticket records để xác định representative (vé đầu tiên theo thời gian)
        matched_ticket_ids = list(distinct_users.values())
        candidate_tickets: list[TicketRecord] = []
        for tid in matched_ticket_ids:
            t = repo.get_ticket(tid)
            if t:
                candidate_tickets.append(t)

        candidate_tickets.sort(key=lambda x: x.created_at)
        representative_id = candidate_tickets[0].id if candidate_tickets else ticket_id

        all_ids = matched_ticket_ids + [ticket_id]
        new_incident = repo.create_incident(
            representative_ticket_id=representative_id,
            category=category,
            initial_ticket_ids=all_ids,
        )
        return new_incident.id, False

    return None, False


def check_recurring_resolved(
    embedding: list[float],
    repo: TicketRepository,
    min_similarity: float | None = None,
    current_ticket_id: str | None = None,
) -> bool:
    """Kiểm tra kịch bản 4: Recurring Resolved (Lỗi cũ tái phát sau khi đã resolved/closed)."""
    settings = get_settings()
    threshold = min_similarity if min_similarity is not None else settings.similarity_recurring_threshold

    matches = repo.find_similar_tickets(
        embedding=embedding,
        min_similarity=threshold,
        only_status=["closed", "resolved"],
        exclude_ticket_id=current_ticket_id,
    )
    return len(matches) > 0


def broadcast_incident_resolution(
    representative_ticket_id: str,
    final_text: str,
    repo: TicketRepository,
) -> int:
    """Cơ chế Event-Driven Broadcast: Khi Ticket #1 được IT duyệt, tự động cập nhật mọi vé đang chờ."""
    # Tìm incident mà representative_ticket_id làm đại diện
    incident: IncidentRecord | None = None
    rep_ticket = repo.get_ticket(representative_ticket_id)
    if rep_ticket and rep_ticket.incident_id:
        incident = repo.get_incident(rep_ticket.incident_id)

    if not incident:
        # Tra cứu từ bảng incidents
        for inc in getattr(repo, "incidents", {}).values():
            if inc.representative_ticket_id == representative_ticket_id:
                incident = inc
                break

    if not incident:
        return 0

    # Cập nhật incident
    repo.resolve_incident(incident.id, resolution_text=final_text)

    # Lấy toàn bộ ticket đang xếp hàng chờ linked_to_incident
    queued_tickets = repo.get_tickets_linked_to_incident(incident.id, status="linked_to_incident")
    for t in queued_tickets:
        repo.update_ticket_response(
            ticket_id=t.id,
            answer=final_text,
            status="auto_sent",
            action="broadcast_from_incident",
        )

    return len(queued_tickets)


def match_ticket_state(
    state: AgentState,
    embedding: list[float],
    repo: TicketRepository,
    current_time: datetime | None = None,
) -> AgentState:
    """Quy trình tích hợp so khớp tương đồng và cập nhật trực tiếp vào AgentState."""
    category = (state.classification or {}).get("category", "")

    # 1. Kiểm tra spam dồn dập
    if state.requester_email:
        state.is_spam_duplicate = check_spam_flood(
            requester_email=state.requester_email,
            repo=repo,
            current_time=current_time,
        )

    # 2. Kiểm tra repeat-unresolved
    if state.requester_email:
        is_repeat, _ = check_repeat_unresolved(
            requester_email=state.requester_email,
            embedding=embedding,
            repo=repo,
            current_time=current_time,
            current_ticket_id=state.ticket_id,
        )
        if is_repeat:
            state.repeat_ticket_suspected = True

    # 3. Kiểm tra recurring resolved
    state.recurring_issue = check_recurring_resolved(
        embedding=embedding,
        repo=repo,
        current_ticket_id=state.ticket_id,
    )

    # 4. Kiểm tra sự cố diện rộng (Mass incident)
    if category:
        inc_id, is_linked = check_or_link_mass_incident(
            ticket_id=state.ticket_id,
            requester_email=state.requester_email,
            category=category,
            embedding=embedding,
            repo=repo,
            current_time=current_time,
        )
        if inc_id:
            state.incident_id = inc_id
            if is_linked:
                # Vé thứ 6 trở đi: kiểm tra câu trả lời của Ticket #1 đại diện
                incident = repo.get_incident(inc_id)
                rep_ticket = repo.get_ticket(incident.representative_ticket_id) if incident else None

                if rep_ticket and rep_ticket.status in ("auto_sent", "approved") and rep_ticket.final_response_text:
                    # Fast-path: tái sử dụng câu trả lời chuẩn đã duyệt của Ticket #1
                    state.routing_decision = "auto"
                    state.routing_reason = "incident_fast_reply_from_representative"
                    state.generation = {
                        "answer": rep_ticket.final_response_text,
                        "sources": rep_ticket.sources or [],
                    }
                    state.generation_complete = True
                else:
                    # Safe-path: đưa vào hàng chờ duyệt và gửi thông báo tạm thời
                    state.routing_decision = "escalate"
                    state.routing_reason = "linked_to_incident"
                    state.generation = {
                        "answer": "Sự cố này đang được đội ngũ kỹ thuật xử lý chung trên toàn hệ thống. "
                                  "Phản hồi chính thức sẽ được gửi tới bạn ngay khi hoàn tất.",
                        "sources": [],
                    }
                    state.generation_complete = True

    return state

