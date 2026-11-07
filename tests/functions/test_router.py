"""Unit tests cho AgentState và Routing Rules theo 08_AGENT_STATE.md và 04_CONFIDENCE_EVAL_POLICY.md."""

import unittest
from unittest.mock import patch

from src.agent.state import AgentState
from src.router import (
    CONFIDENCE_AUTO_THRESHOLD,
    CONFIDENCE_CLARIFY_THRESHOLD,
    CONFIDENCE_DRAFT_THRESHOLD,
    DEFAULT_CANARY_RATE,
    route_ticket,
)


def make_confidence(score=0.90, status="ok", errors=None):
    """Helper tạo dict confidence chuẩn theo src/confidence.py."""
    return {
        "c_retrieval": score,
        "c_consensus": 1.0,
        "c_groundedness": score,
        "c_rag": score,
        "status": status,
        "errors": errors or [],
    }


def make_parent(parent_id="p1"):
    return {
        "parent_chunk_id": parent_id,
        "article_title": "VPN Guide",
        "text": "Content",
        "relevance_score": 0.9,
    }


class AgentStateTests(unittest.TestCase):
    """Kiểm tra hợp đồng của dataclass AgentState theo 08_AGENT_STATE.md."""

    def test_state_initialization_defaults(self):
        state = AgentState(ticket_id="TC_001", subject="Lỗi VPN", body="Không kết nối được VPN")
        self.assertEqual(state.ticket_id, "TC_001")
        self.assertEqual(state.subject, "Lỗi VPN")
        self.assertEqual(state.body, "Không kết nối được VPN")
        self.assertIsNone(state.classification)
        self.assertEqual(state.missing_fields, [])
        self.assertEqual(state.clarification_rounds, 0)
        self.assertIsNone(state.clarification_question)
        self.assertEqual(state.retrieved_parents, [])
        self.assertIsNone(state.generation)
        self.assertFalse(state.generation_complete)
        self.assertIsNone(state.groundedness)
        self.assertIsNone(state.groundedness_score)
        self.assertIsNone(state.confidence)
        self.assertIsNone(state.routing_decision)
        self.assertIsNone(state.routing_reason)
        self.assertEqual(state.errors, [])
        self.assertIsNone(state.automation_stage)
        self.assertIsNone(state.canary_selected)
        self.assertFalse(state.is_reopened)
        self.assertFalse(state.repeat_ticket_suspected)
        self.assertEqual(state.requester_email, "")
        self.assertIsNone(state.incident_id)
        self.assertFalse(state.is_spam_duplicate)
        self.assertFalse(state.recurring_issue)
        self.assertIsNone(state.diagnostic_data)

    def test_state_list_isolation_between_instances(self):
        """Đảm bảo default_factory tạo list độc lập, không dùng chung bộ nhớ giữa 2 ticket."""
        s1 = AgentState(ticket_id="T1", subject="S1", body="B1")
        s2 = AgentState(ticket_id="T2", subject="S2", body="B2")
        s1.missing_fields.append("os_version")
        s1.errors.append({"stage": "generation", "message": "error"})
        s1.retrieved_parents.append({"id": 1})

        self.assertEqual(s1.missing_fields, ["os_version"])
        self.assertEqual(s2.missing_fields, [])
        self.assertEqual(len(s1.errors), 1)
        self.assertEqual(s2.errors, [])
        self.assertEqual(len(s1.retrieved_parents), 1)
        self.assertEqual(s2.retrieved_parents, [])


class RouterEscalationTriggersTests(unittest.TestCase):
    """Kiểm tra các trigger escalate bắt buộc (Mục 4 — 04_CONFIDENCE_EVAL_POLICY.md)."""

    def setUp(self):
        self.base_state = AgentState(
            ticket_id="TC_TEST",
            subject="Cần hỗ trợ",
            body="Mô tả ticket",
            retrieved_parents=[make_parent()],
            generation={"answer": "Hướng dẫn cụ thể.", "sources": [{"parent_chunk_id": "p1", "title": "VPN Guide"}]},
            generation_complete=True,
            confidence=make_confidence(score=0.90, status="ok"),
            automation_stage="auto",
        )

    def test_escalate_when_out_of_scope(self):
        self.base_state.classification = {"category": "out_of_scope", "reason": "Hỏi mua pizza"}
        routed = route_ticket(self.base_state)
        self.assertEqual(routed.routing_decision, "escalate")
        self.assertEqual(routed.routing_reason, "out_of_scope")

    def test_escalate_when_out_of_scope_uppercase(self):
        self.base_state.classification = {"category": "OUT_OF_SCOPE"}
        routed = route_ticket(self.base_state)
        self.assertEqual(routed.routing_decision, "escalate")
        self.assertEqual(routed.routing_reason, "out_of_scope")

    def test_escalate_when_multi_issue_suspected(self):
        self.base_state.classification = {"category": "VPN", "multi_issue_suspected": True}
        routed = route_ticket(self.base_state)
        self.assertEqual(routed.routing_decision, "escalate")
        self.assertEqual(routed.routing_reason, "multi_issue_suspected")

    def test_escalate_when_prompt_injection_suspected(self):
        self.base_state.classification = {"category": "VPN", "prompt_injection_suspected": True}
        routed = route_ticket(self.base_state)
        self.assertEqual(routed.routing_decision, "escalate")
        self.assertEqual(routed.routing_reason, "prompt_injection_suspected")

    def test_escalate_when_sentiment_angry(self):
        self.base_state.classification = {"category": "VPN", "sentiment_flag": "angry"}
        routed = route_ticket(self.base_state)
        self.assertEqual(routed.routing_decision, "escalate")
        self.assertEqual(routed.routing_reason, "sentiment_angry")

    def test_escalate_when_security_concern_flag(self):
        self.base_state.classification = {"category": "PASSWORD", "security_concern": True}
        routed = route_ticket(self.base_state)
        self.assertEqual(routed.routing_decision, "escalate")
        self.assertEqual(routed.routing_reason, "security_concern")

    def test_escalate_when_security_keyword_in_body(self):
        self.base_state.body = "Tài khoản của tôi bị khóa do nghi ngờ có kẻ xâm nhập trái phép."
        self.base_state.classification = {"category": "PASSWORD"}
        routed = route_ticket(self.base_state)
        self.assertEqual(routed.routing_decision, "escalate")
        self.assertEqual(routed.routing_reason, "security_concern")

    def test_escalate_when_reopened_ticket(self):
        self.base_state.is_reopened = True
        routed = route_ticket(self.base_state)
        self.assertEqual(routed.routing_decision, "escalate")
        self.assertEqual(routed.routing_reason, "reopened_ticket")

    def test_escalate_when_repeat_ticket(self):
        self.base_state.repeat_ticket_suspected = True
        routed = route_ticket(self.base_state)
        self.assertEqual(routed.routing_decision, "escalate")
        self.assertEqual(routed.routing_reason, "repeat_ticket")

    def test_escalate_when_retrieval_empty(self):
        self.base_state.retrieved_parents = []
        routed = route_ticket(self.base_state)
        self.assertEqual(routed.routing_decision, "escalate")
        self.assertEqual(routed.routing_reason, "retrieval_unavailable")


class RouterThresholdRoutingTests(unittest.TestCase):
    """Kiểm tra các dải ngưỡng confidence và phân luồng (Mục 3 — 04_CONFIDENCE_EVAL_POLICY.md)."""

    def setUp(self):
        self.base_state = AgentState(
            ticket_id="TC_002",
            subject="Hỏi cấu hình",
            body="Cách đổi mật khẩu",
            classification={"category": "PASSWORD"},
            retrieved_parents=[make_parent()],
            generation={"answer": "Vào portal để đổi.", "sources": [{"parent_chunk_id": "p1", "title": "VPN Guide"}]},
            generation_complete=True,
            automation_stage="draft",
        )

    def test_escalate_when_confidence_unavailable(self):
        self.base_state.confidence = {"c_rag": None, "status": "unavailable", "errors": ["Retrieval fail"]}
        routed = route_ticket(self.base_state)
        self.assertEqual(routed.routing_decision, "escalate")
        self.assertEqual(routed.routing_reason, "confidence_unavailable")

    def test_escalate_when_confidence_under_0_4(self):
        self.base_state.confidence = make_confidence(score=0.35, status="ok")
        routed = route_ticket(self.base_state)
        self.assertEqual(routed.routing_decision, "escalate")
        self.assertEqual(routed.routing_reason, "low_confidence")

    def test_clarify_when_between_0_4_and_0_6_and_has_missing_fields(self):
        self.base_state.confidence = make_confidence(score=0.50, status="ok")
        self.base_state.missing_fields = ["os_type", "error_code"]
        self.base_state.clarification_rounds = 0
        routed = route_ticket(self.base_state)
        self.assertEqual(routed.routing_decision, "clarify")
        self.assertEqual(routed.routing_reason, "needs_clarification")

    def test_escalate_when_between_0_4_and_0_6_but_no_missing_fields(self):
        """Chính sách: nếu không xác định được thiếu thông tin cụ thể nào, dù trong dải 0.4-0.6 vẫn escalate."""
        self.base_state.confidence = make_confidence(score=0.50, status="ok")
        self.base_state.missing_fields = []
        self.base_state.clarification_rounds = 0
        routed = route_ticket(self.base_state)
        self.assertEqual(routed.routing_decision, "escalate")
        self.assertEqual(routed.routing_reason, "missing_fields_unidentified")

    def test_escalate_when_between_0_4_and_0_6_and_clarification_rounds_exceeded(self):
        """Chính sách: nếu clarification_rounds >= 1 mà vẫn không đạt >= 0.6 thì escalate."""
        self.base_state.confidence = make_confidence(score=0.50, status="ok")
        self.base_state.missing_fields = ["os_type"]
        self.base_state.clarification_rounds = 1
        routed = route_ticket(self.base_state)
        self.assertEqual(routed.routing_decision, "escalate")
        self.assertEqual(routed.routing_reason, "clarification_rounds_exceeded")

    def test_draft_when_medium_confidence(self):
        """Dải 0.6 <= C_RAG < 0.85 -> Tạo draft chờ người duyệt."""
        self.base_state.confidence = make_confidence(score=0.75, status="ok")
        routed = route_ticket(self.base_state)
        self.assertEqual(routed.routing_decision, "draft")
        self.assertEqual(routed.routing_reason, "medium_confidence")

    def test_draft_boundary_0_6(self):
        self.base_state.confidence = make_confidence(score=0.60, status="ok")
        routed = route_ticket(self.base_state)
        self.assertEqual(routed.routing_decision, "draft")
        self.assertEqual(routed.routing_reason, "medium_confidence")

    def test_draft_boundary_0_849(self):
        self.base_state.confidence = make_confidence(score=0.849, status="ok")
        routed = route_ticket(self.base_state)
        self.assertEqual(routed.routing_decision, "draft")
        self.assertEqual(routed.routing_reason, "medium_confidence")


class RouterAutoAndCanaryTests(unittest.TestCase):
    """Kiểm tra điều kiện Auto-reply (>=0.85) và Canary sampling (05_CATEGORY_GRADUATION.md)."""

    def setUp(self):
        self.base_state = AgentState(
            ticket_id="TC_AUTO",
            subject="Hướng dẫn VPN",
            body="Cách kết nối VPN văn phòng",
            classification={"category": "VPN"},
            retrieved_parents=[make_parent()],
            generation={"answer": "Mở FortiClient và nhập vpn.company.com", "sources": [{"parent_chunk_id": "p1", "title": "VPN Guide"}]},
            generation_complete=True,
            confidence=make_confidence(score=0.92, status="ok"),
            automation_stage="auto",
        )

    def test_auto_reply_success(self):
        """Đạt điểm >= 0.85, status ok, stage auto, không canary -> auto."""
        self.base_state.canary_selected = False
        routed = route_ticket(self.base_state)
        self.assertEqual(routed.routing_decision, "auto")
        self.assertEqual(routed.routing_reason, "high_confidence_auto")

    def test_draft_when_category_not_in_auto_stage(self):
        """Đạt điểm >= 0.85 nhưng category đang ở draft mode -> draft."""
        self.base_state.automation_stage = "draft"
        routed = route_ticket(self.base_state)
        self.assertEqual(routed.routing_decision, "draft")
        self.assertEqual(routed.routing_reason, "category_not_in_auto_stage")

    def test_draft_when_confidence_status_is_degraded(self):
        """Đạt điểm >= 0.85 nhưng confidence bị degraded (ví dụ claim mâu thuẫn) -> draft."""
        self.base_state.confidence = make_confidence(score=0.90, status="degraded", errors=["Claim mâu thuẫn"])
        routed = route_ticket(self.base_state)
        self.assertEqual(routed.routing_decision, "draft")
        self.assertEqual(routed.routing_reason, "confidence_degraded")

    def test_draft_when_generation_incomplete(self):
        """Đạt điểm >= 0.85 nhưng câu trả lời bị cắt ngang -> draft."""
        self.base_state.generation_complete = False
        routed = route_ticket(self.base_state)
        self.assertEqual(routed.routing_decision, "draft")
        self.assertEqual(routed.routing_reason, "generation_incomplete")

    def test_draft_when_canary_selected_pre_marked(self):
        """Category ở Auto nhưng ticket được đánh dấu trúng canary sampling -> draft."""
        self.base_state.canary_selected = True
        routed = route_ticket(self.base_state)
        self.assertEqual(routed.routing_decision, "draft")
        self.assertEqual(routed.routing_reason, "canary_sample")

    def test_canary_sampling_auto_selects(self):
        """Khi canary_selected là None và canary_rate = 1.0 -> chắc chắn trúng canary."""
        self.base_state.canary_selected = None
        routed = route_ticket(self.base_state, canary_rate=1.0)
        self.assertTrue(routed.canary_selected)
        self.assertEqual(routed.routing_decision, "draft")
        self.assertEqual(routed.routing_reason, "canary_sample")

    def test_canary_sampling_zero_rate(self):
        """Khi canary_rate = 0.0 -> không bao giờ trúng canary."""
        self.base_state.canary_selected = None
        routed = route_ticket(self.base_state, canary_rate=0.0)
        self.assertFalse(routed.canary_selected)
        self.assertEqual(routed.routing_decision, "auto")
        self.assertEqual(routed.routing_reason, "high_confidence_auto")


if __name__ == "__main__":
    unittest.main()

