"""Unit tests cho Subsystem Tương đồng (Similarity & Duplicate / Incident Detection).

Đặc tả dựa trên:
- documents/02_ARCHITECTURE.md (Mục 3, 12)
- documents/03_DATA_MODEL.md (Mục 1, 6, 8)
- implementation_plan/12_SIMILARITY_AND_INCIDENT_DETECTION_PLAN.md
"""

from datetime import datetime, timedelta, timezone
import unittest

from src.agent.similarity.models import IncidentRecord, TicketRecord
from src.agent.similarity.repository import InMemoryTicketRepository
from src.agent.similarity.ticket_matcher import (
    broadcast_incident_resolution,
    check_or_link_mass_incident,
    check_recurring_resolved,
    check_repeat_unresolved,
    check_spam_flood,
    match_ticket_state,
)
from src.agent.state import AgentState


def make_embedding(seed_val: float, dim: int = 768) -> list[float]:
    """Helper tạo vector embedding giả lập đơn vị cho test."""
    vec = [0.0] * dim
    vec[0] = seed_val
    vec[1] = (1.0 - seed_val**2) ** 0.5 if seed_val <= 1.0 else 0.0
    return vec


class TestSpamFloodDetection(unittest.TestCase):
    """Kiểm tra kịch bản 1: Spam do bực bội (cùng email gửi dồn dập trong 5 phút)."""

    def setUp(self):
        self.repo = InMemoryTicketRepository()
        self.now = datetime.now(timezone.utc)

    def test_spam_flood_detected_within_window(self):
        # Ticket trước gửi cách đây 2 phút (< 5 phút)
        self.repo.save_ticket(
            TicketRecord(
                id="T_OLD",
                requester_email="user@company.com",
                subject="VPN lỗi",
                body="Không vào được VPN",
                created_at=self.now - timedelta(minutes=2),
                category="VPN",
                status="new",
            )
        )
        is_spam = check_spam_flood(
            requester_email="user@company.com",
            repo=self.repo,
            current_time=self.now,
            window_minutes=5,
        )
        self.assertTrue(is_spam, "Phải phát hiện spam khi cùng email gửi tiếp trong 5 phút.")

    def test_spam_flood_ignored_outside_window(self):
        # Ticket trước gửi cách đây 10 phút (> 5 phút)
        self.repo.save_ticket(
            TicketRecord(
                id="T_OLD",
                requester_email="user@company.com",
                subject="VPN lỗi",
                body="Không vào được VPN",
                created_at=self.now - timedelta(minutes=10),
                category="VPN",
                status="new",
            )
        )
        is_spam = check_spam_flood(
            requester_email="user@company.com",
            repo=self.repo,
            current_time=self.now,
            window_minutes=5,
        )
        self.assertFalse(is_spam, "Không được coi là spam nếu ticket cũ cách đây hơn 5 phút.")

    def test_spam_flood_different_email_ignored(self):
        # Ticket khác email gửi cách đây 1 phút
        self.repo.save_ticket(
            TicketRecord(
                id="T_OTHER",
                requester_email="other@company.com",
                subject="Lỗi máy in",
                body="Kẹt giấy",
                created_at=self.now - timedelta(minutes=1),
                category="PRINTER",
                status="new",
            )
        )
        is_spam = check_spam_flood(
            requester_email="user@company.com",
            repo=self.repo,
            current_time=self.now,
            window_minutes=5,
        )
        self.assertFalse(is_spam, "Khác email thì không được tính là spam dồn dập của user này.")


class TestRepeatUnresolvedDetection(unittest.TestCase):
    """Kiểm tra kịch bản 2: Repeat-unresolved (1 người hỏi lại cùng vấn đề trong 7 ngày)."""

    def setUp(self):
        self.repo = InMemoryTicketRepository()
        self.now = datetime.now(timezone.utc)
        self.vec_vpn = make_embedding(0.99)
        self.vec_printer = make_embedding(0.10)

    def test_repeat_unresolved_same_email_high_similarity(self):
        # Ticket cũ đã trả lời 2 ngày trước
        self.repo.save_ticket(
            TicketRecord(
                id="T_REP_1",
                requester_email="user1@company.com",
                subject="VPN lỗi kết nối",
                body="Cisco AnyConnect báo timeout",
                created_at=self.now - timedelta(days=2),
                category="VPN",
                status="auto_sent",
                embedding=self.vec_vpn,
            )
        )
        is_repeat, old_id = check_repeat_unresolved(
            requester_email="user1@company.com",
            embedding=self.vec_vpn,
            repo=self.repo,
            current_time=self.now,
            window_days=7,
            min_similarity=0.85,
        )
        self.assertTrue(is_repeat)
        self.assertEqual(old_id, "T_REP_1")

    def test_repeat_unresolved_ignored_if_different_email(self):
        # Cùng nội dung VPN nhưng từ người khác gửi -> Không tính là repeat cá nhân
        self.repo.save_ticket(
            TicketRecord(
                id="T_REP_2",
                requester_email="colleague@company.com",
                subject="VPN lỗi kết nối",
                body="Cisco AnyConnect báo timeout",
                created_at=self.now - timedelta(days=1),
                category="VPN",
                status="auto_sent",
                embedding=self.vec_vpn,
            )
        )
        is_repeat, old_id = check_repeat_unresolved(
            requester_email="user1@company.com",
            embedding=self.vec_vpn,
            repo=self.repo,
            current_time=self.now,
            window_days=7,
            min_similarity=0.85,
        )
        self.assertFalse(is_repeat)
        self.assertIsNone(old_id)

    def test_repeat_unresolved_ignored_if_low_similarity(self):
        # Cùng người gửi nhưng khác hẳn nội dung (hỏi máy in thay vì VPN)
        self.repo.save_ticket(
            TicketRecord(
                id="T_REP_3",
                requester_email="user1@company.com",
                subject="Hết mực máy in tầng 3",
                body="Nhờ IT thay mực",
                created_at=self.now - timedelta(days=1),
                category="PRINTER",
                status="auto_sent",
                embedding=self.vec_printer,
            )
        )
        is_repeat, old_id = check_repeat_unresolved(
            requester_email="user1@company.com",
            embedding=self.vec_vpn,
            repo=self.repo,
            current_time=self.now,
            window_days=7,
            min_similarity=0.85,
        )
        self.assertFalse(is_repeat)
        self.assertIsNone(old_id)


class TestMassIncidentDetection(unittest.TestCase):
    """Kiểm tra kịch bản 3: Mass Incident (Sự cố diện rộng 2 giai đoạn: candidate và confirmed)."""

    def setUp(self):
        self.repo = InMemoryTicketRepository()
        self.now = datetime.now(timezone.utc)
        self.vec_outage = make_embedding(0.95)

    def test_candidate_phase_under_5_tickets_does_not_create_incident(self):
        # Thêm 3 ticket từ 3 người khác nhau trong 10 phút qua (chưa đủ ngưỡng N=5)
        for i in range(1, 4):
            self.repo.save_ticket(
                TicketRecord(
                    id=f"T_CAND_{i}",
                    requester_email=f"user{i}@company.com",
                    subject="Sập mạng toàn văn phòng",
                    body="Không ai vào được mạng nội bộ",
                    created_at=self.now - timedelta(minutes=15 - i),
                    category="VPN",
                    status="pending_review",
                    embedding=self.vec_outage,
                )
            )

        # Ticket thứ 4 xuất hiện
        incident_id, is_linked = check_or_link_mass_incident(
            ticket_id="T_CAND_4",
            requester_email="user4@company.com",
            category="VPN",
            embedding=self.vec_outage,
            repo=self.repo,
            current_time=self.now,
            window_minutes=30,
            threshold=5,
            min_similarity=0.82,
        )

        self.assertIsNone(incident_id, "Khi chưa đủ 5 vé thì không được tạo incident.")
        self.assertFalse(is_linked)
        self.assertEqual(len(self.repo.incidents), 0)

    def test_triggers_incident_at_5th_ticket_and_retroactive_link(self):
        # Đã có sẵn 4 ticket tương đồng từ 4 người khác nhau trong 20 phút qua
        for i in range(1, 5):
            self.repo.save_ticket(
                TicketRecord(
                    id=f"T_INC_{i}",
                    requester_email=f"user{i}@company.com",
                    subject="Mất mạng",
                    body="Mất mạng văn phòng",
                    created_at=self.now - timedelta(minutes=25 - i * 3),
                    category="VPN",
                    status="pending_review",
                    embedding=self.vec_outage,
                )
            )

        # Ticket thứ 5 đến từ user thứ 5 (được lưu vào repo lúc intake)
        self.repo.save_ticket(
            TicketRecord(
                id="T_INC_5",
                requester_email="user5@company.com",
                subject="Mất mạng",
                body="Mất mạng văn phòng",
                created_at=self.now,
                category="VPN",
                status="new",
                embedding=self.vec_outage,
            )
        )

        incident_id, is_linked = check_or_link_mass_incident(
            ticket_id="T_INC_5",
            requester_email="user5@company.com",
            category="VPN",
            embedding=self.vec_outage,
            repo=self.repo,
            current_time=self.now,
            window_minutes=30,
            threshold=5,
            min_similarity=0.82,
        )

        self.assertIsNotNone(incident_id, "Chạm ngưỡng 5 vé phải tạo incident ngay lập tức.")
        self.assertFalse(is_linked, "Ticket thứ 5 hoàn tất việc kích hoạt, không bị coi là linked skip.")

        # Kiểm tra bảng incidents
        incident = self.repo.get_incident(incident_id)
        self.assertIsNotNone(incident)
        self.assertEqual(incident.representative_ticket_id, "T_INC_1", "Representative phải là vé đầu tiên.")
        self.assertEqual(incident.ticket_count, 5)

        # Kiểm tra 5 ticket đầu đều đã được gán incident_id hồi tố
        for i in range(1, 6):
            t = self.repo.get_ticket(f"T_INC_{i}")
            self.assertEqual(t.incident_id, incident_id, f"Ticket T_INC_{i} phải được gắn incident_id.")

    def test_ticket_6_uses_fast_reply_if_representative_already_resolved(self):
        # Giả lập incident đã tồn tại với Ticket #1 là đại diện và ĐÃ ĐƯỢC DUYỆT (auto_sent / approved)
        self.repo.save_ticket(
            TicketRecord(
                id="T_REP",
                requester_email="user1@company.com",
                subject="Mất kết nối VPN",
                body="Cổng VPN gateway lỗi",
                created_at=self.now - timedelta(minutes=20),
                category="VPN",
                status="auto_sent",
                embedding=self.vec_outage,
                final_response_text="Sự cố VPN đã được IT xử lý thành công, hãy kết nối lại.",
            )
        )
        incident = self.repo.create_incident(
            representative_ticket_id="T_REP",
            category="VPN",
            initial_ticket_ids=["T_REP"],
        )

        # Ticket thứ 6 đến
        state = AgentState(
            ticket_id="T_6",
            requester_email="user6@company.com",
            subject="VPN bị mất kết nối",
            body="Tôi không vào được VPN",
        )
        state.classification = {"category": "VPN"}

        matched_state = match_ticket_state(
            state=state,
            embedding=self.vec_outage,
            repo=self.repo,
            current_time=self.now,
        )

        self.assertEqual(matched_state.incident_id, incident.id)
        # Vì representative đã có final_response_text, ticket 6 nhận ngay câu trả lời đó
        self.assertEqual(matched_state.routing_decision, "auto")
        self.assertEqual(matched_state.routing_reason, "incident_fast_reply_from_representative")
        self.assertIn("Sự cố VPN đã được IT xử lý thành công", matched_state.generation["answer"])

    def test_ticket_6_queued_if_representative_is_still_pending(self):
        # Giả lập incident đã tồn tại nhưng Ticket #1 vẫn đang pending_review (chưa có câu trả lời duyệt)
        self.repo.save_ticket(
            TicketRecord(
                id="T_REP_PENDING",
                requester_email="user1@company.com",
                subject="Mất kết nối VPN",
                body="Cổng VPN gateway lỗi",
                created_at=self.now - timedelta(minutes=20),
                category="VPN",
                status="pending_review",
                embedding=self.vec_outage,
                final_response_text=None,
            )
        )
        incident = self.repo.create_incident(
            representative_ticket_id="T_REP_PENDING",
            category="VPN",
            initial_ticket_ids=["T_REP_PENDING"],
        )

        # Ticket thứ 6 đến
        state = AgentState(
            ticket_id="T_6_PENDING",
            requester_email="user6@company.com",
            subject="VPN bị mất kết nối",
            body="Tôi không vào được VPN",
        )
        state.classification = {"category": "VPN"}

        matched_state = match_ticket_state(
            state=state,
            embedding=self.vec_outage,
            repo=self.repo,
            current_time=self.now,
        )

        self.assertEqual(matched_state.incident_id, incident.id)
        self.assertEqual(matched_state.routing_decision, "escalate")
        self.assertEqual(matched_state.routing_reason, "linked_to_incident")
        self.assertIn("Sự cố này đang được đội ngũ kỹ thuật xử lý chung", matched_state.generation["answer"])


class TestBroadcastIncidentResolution(unittest.TestCase):
    """Kiểm tra cơ chế Event-Driven Broadcast khi Ticket #1 được IT duyệt."""

    def setUp(self):
        self.repo = InMemoryTicketRepository()
        self.now = datetime.now(timezone.utc)

    def test_broadcast_updates_all_queued_tickets(self):
        # Tạo Representative Ticket
        self.repo.save_ticket(
            TicketRecord(
                id="T_REP_MAIN",
                requester_email="lead@company.com",
                subject="Đứt cáp",
                body="Mất mạng",
                created_at=self.now - timedelta(minutes=30),
                category="VPN",
                status="pending_review",
            )
        )
        incident = self.repo.create_incident(
            representative_ticket_id="T_REP_MAIN",
            category="VPN",
            initial_ticket_ids=["T_REP_MAIN"],
        )

        # Thêm 3 ticket đang xếp hàng chờ (linked_to_incident)
        for i in range(1, 4):
            self.repo.save_ticket(
                TicketRecord(
                    id=f"T_QUEUED_{i}",
                    requester_email=f"client{i}@company.com",
                    subject="Mất mạng",
                    body="Tôi cũng bị mất mạng",
                    created_at=self.now - timedelta(minutes=10 - i),
                    category="VPN",
                    status="linked_to_incident",
                    incident_id=incident.id,
                )
            )

        # IT Agent bấm Approve cho Ticket đại diện với nội dung chính thức
        solution_text = "IT đã khởi động lại gateway VPN, mọi người có thể kết nối bình thường."
        updated_count = broadcast_incident_resolution(
            representative_ticket_id="T_REP_MAIN",
            final_text=solution_text,
            repo=self.repo,
        )

        self.assertEqual(updated_count, 3, "Phải cập nhật đúng 3 vé đang chờ.")

        # Kiểm tra từng vé đã được đổi status sang auto_sent và nhận đúng solution_text
        for i in range(1, 4):
            t = self.repo.get_ticket(f"T_QUEUED_{i}")
            self.assertEqual(t.status, "auto_sent")
            self.assertEqual(t.final_response_text, solution_text)

        # Kiểm tra trạng thái Incident chuyển sang resolved
        inc = self.repo.get_incident(incident.id)
        self.assertEqual(inc.status, "resolved")
        self.assertEqual(inc.resolution_text, solution_text)


class TestRecurringResolvedDetection(unittest.TestCase):
    """Kiểm tra kịch bản 4: Recurring Resolved (Lỗi cũ tái phát sau khi đã đóng)."""

    def setUp(self):
        self.repo = InMemoryTicketRepository()
        self.now = datetime.now(timezone.utc)
        self.vec_printer = make_embedding(0.92)

    def test_recurring_resolved_detected_for_past_closed_ticket(self):
        # Ticket đã đóng từ 2 tháng trước
        self.repo.save_ticket(
            TicketRecord(
                id="T_CLOSED_OLD",
                requester_email="any_user@company.com",
                subject="Kẹt giấy máy in HP LaserJet",
                body="Máy in báo kẹt giấy khay 2 liên tục",
                created_at=self.now - timedelta(days=60),
                category="PRINTER",
                status="closed",
                embedding=self.vec_printer,
            )
        )

        is_recurring = check_recurring_resolved(
            embedding=self.vec_printer,
            repo=self.repo,
            min_similarity=0.88,
        )
        self.assertTrue(is_recurring, "Phải phát hiện lỗi tái phát khi trùng với ticket đã closed trong quá khứ.")

    def test_recurring_resolved_ignored_if_similarity_below_threshold(self):
        vec_other = make_embedding(0.20)
        self.repo.save_ticket(
            TicketRecord(
                id="T_CLOSED_OLD_2",
                requester_email="any_user@company.com",
                subject="Kẹt giấy máy in HP LaserJet",
                body="Máy in báo kẹt giấy khay 2 liên tục",
                created_at=self.now - timedelta(days=60),
                category="PRINTER",
                status="closed",
                embedding=self.vec_printer,
            )
        )

        is_recurring = check_recurring_resolved(
            embedding=vec_other,
            repo=self.repo,
            min_similarity=0.88,
        )
        self.assertFalse(is_recurring)


if __name__ == "__main__":
    unittest.main()

