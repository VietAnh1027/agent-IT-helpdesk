"""Routing policy: Quyết định luồng xử lý ticket (auto, draft, clarify, escalate).

Đặc tả quy tắc dựa trên:
- documents/04_CONFIDENCE_EVAL_POLICY.md (Mục 3: Ngưỡng xử lý, Mục 4: Escalation Triggers)
- documents/05_CATEGORY_GRADUATION.md (Mục 5: Canary sampling)
- documents/08_AGENT_STATE.md (Mục 4: Routing và lỗi)
"""

import random

from src.agent.guardrails.input_guardrails import check_fast_security_concern
from src.agent.state import AgentState

CONFIDENCE_AUTO_THRESHOLD = 0.85
CONFIDENCE_DRAFT_THRESHOLD = 0.60
CONFIDENCE_CLARIFY_THRESHOLD = 0.40
DEFAULT_CANARY_RATE = 0.10


def _check_escalation_triggers(state: AgentState) -> str | None:
    """Kiểm tra các trigger escalate bắt buộc theo Mục 4 của Policy.

    Trả về routing_reason nếu vi phạm, ngược lại None.
    """
    cls = state.classification or {}

    # 1. Classifier gắn out_of_scope
    cat = str(cls.get("category", "")).casefold()
    if cat in ("out_of_scope", "out-of-scope"):
        return "out_of_scope"

    # 2. Nghi ngờ đa vấn đề
    if cls.get("multi_issue_suspected") is True:
        return "multi_issue_suspected"

    # 3. Nghi ngờ prompt injection
    if cls.get("prompt_injection_suspected") is True:
        return "prompt_injection_suspected"

    # 4. Khách hàng giận dữ
    if str(cls.get("sentiment_flag", "")).casefold() == "angry":
        return "sentiment_angry"

    # 5. Từ khóa bảo mật / nghi ngờ hack / khóa tài khoản
    if cls.get("security_concern") is True:
        return "security_concern"

    if check_fast_security_concern(state.subject, state.body):
        return "security_concern"

    # 6. Ticket reopened (đã auto_sent/closed nhưng có comment mới)
    if state.is_reopened:
        return "reopened_ticket"

    # 7. Ticket lặp lại nhiều lần
    if state.repeat_ticket_suspected:
        return "repeat_ticket"

    # 8. Retrieval không tìm được chunk hợp lệ
    if not state.retrieved_parents:
        return "retrieval_unavailable"

    return None


def route_ticket(
    state: AgentState,
    *,
    canary_rate: float = DEFAULT_CANARY_RATE,
    random_seed: int | None = None,
) -> AgentState:
    """Xác định routing_decision và routing_reason cho AgentState.

    Quyết định gồm 4 hành động:
    - 'escalate': Chuyển ngay cho IT nhân viên xử lý.
    - 'clarify': Gửi câu hỏi làm rõ (chỉ khi có missing_fields cụ thể và rounds = 0).
    - 'draft': Tạo draft câu trả lời, lưu pending_review chờ duyệt.
    - 'auto': Gửi câu trả lời trực tiếp (khi confidence >= 0.85, status ok, stage auto, không canary).
    """
    # 1. Kiểm tra Early Escalation Triggers
    trigger_reason = _check_escalation_triggers(state)
    if trigger_reason is not None:
        state.routing_decision = "escalate"
        state.routing_reason = trigger_reason
        return state

    # 2. Kiểm tra Confidence Record & Status
    if state.confidence is None:
        state.routing_decision = "escalate"
        state.routing_reason = "confidence_unavailable"
        return state

    c_rag = state.confidence.get("c_rag")
    status = state.confidence.get("status")

    if c_rag is None or status == "unavailable":
        state.routing_decision = "escalate"
        state.routing_reason = "confidence_unavailable"
        return state

    # 3. Phân luồng theo dải điểm số
    # Dải < 0.4: Escalate
    if c_rag < CONFIDENCE_CLARIFY_THRESHOLD:
        state.routing_decision = "escalate"
        state.routing_reason = "low_confidence"
        return state

    # Dải 0.4 <= C_RAG < 0.6: Clarify (nếu đủ điều kiện) hoặc Escalate
    if c_rag < CONFIDENCE_DRAFT_THRESHOLD:
        if state.clarification_rounds >= 1:
            state.routing_decision = "escalate"
            state.routing_reason = "clarification_rounds_exceeded"
            return state

        if state.missing_fields:
            state.routing_decision = "clarify"
            state.routing_reason = "needs_clarification"
            return state

        # Không xác định được thiếu trường thông tin cụ thể nào
        state.routing_decision = "escalate"
        state.routing_reason = "missing_fields_unidentified"
        return state

    # Dải 0.6 <= C_RAG < 0.85: Tạo Draft
    if c_rag < CONFIDENCE_AUTO_THRESHOLD:
        state.routing_decision = "draft"
        state.routing_reason = "medium_confidence"
        return state

    # Dải >= 0.85: Xét điều kiện Auto-reply
    # Kiểm tra tính toàn vẹn câu trả lời
    answer = (state.generation or {}).get("answer", "")
    if not state.generation_complete or not answer.strip():
        state.routing_decision = "draft"
        state.routing_reason = "generation_incomplete"
        return state

    # Không được auto khi confidence bị degraded (ví dụ mâu thuẫn evidence hoặc judge lỗi)
    if status == "degraded":
        state.routing_decision = "draft"
        state.routing_reason = "confidence_degraded"
        return state

    # Kiểm tra category đã graduate lên stage 'auto' chưa (nguồn duy nhất: automation_stage)
    if str(state.automation_stage).casefold() != "auto":
        state.routing_decision = "draft"
        state.routing_reason = "category_not_in_auto_stage"
        return state

    # Cơ chế Canary Sampling (lấy mẫu 1 phần nhỏ đi qua Draft để giám sát chất lượng)
    if state.canary_selected is None:
        if canary_rate > 0:
            rng = random.Random(random_seed) if random_seed is not None else random
            state.canary_selected = rng.random() < canary_rate
        else:
            state.canary_selected = False

    if state.canary_selected:
        state.routing_decision = "draft"
        state.routing_reason = "canary_sample"
        return state

    # Đủ tất cả điều kiện: Tự động gửi
    state.routing_decision = "auto"
    state.routing_reason = "high_confidence_auto"
    return state

