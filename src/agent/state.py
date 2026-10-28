"""AgentState: đối tượng trạng thái dùng chung trong suốt pipeline xử lý 1 ticket.

Đặc tả thiết kế dựa trên documents/08_AGENT_STATE.md.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class AgentState:
    """Gom dữ liệu của một lượt xử lý ticket:

    Đầu vào, kết quả phân loại, evidence KB, câu trả lời, đánh giá groundedness,
    điểm confidence, quyết định routing và các lỗi phát sinh.
    """

    ticket_id: str
    subject: str = ""
    body: str = ""

    # 1. Phân loại & Guardrail sớm
    classification: dict | None = None
    missing_fields: list[str] = field(default_factory=list)
    clarification_rounds: int = 0
    clarification_question: str | None = None

    # 2. Retrieval, Generation & Đánh giá
    retrieved_parents: list[dict] = field(default_factory=list)
    generation: dict | None = None
    generation_complete: bool = False
    groundedness: dict | None = None
    groundedness_score: float | None = None
    confidence: dict | None = None

    # 3. Quyết định Routing & Lỗi
    routing_decision: Literal["draft", "auto", "clarify", "escalate"] | None = None
    routing_reason: str | None = None
    errors: list[dict] = field(default_factory=list)

    # 4. Trạng thái Category Automation & Vận hành (theo 05_CATEGORY_GRADUATION.md)
    automation_stage: str | None = None  # "shadow" | "draft" | "auto"
    auto_enabled: bool = False
    canary_selected: bool | None = None
    is_reopened: bool = False
    repeat_ticket_suspected: bool = False

    def add_error(self, stage: str, message: str) -> None:
        """Ghi nhận lỗi vào state mà không xóa câu trả lời đã sinh."""
        self.errors.append({"stage": stage, "message": message})

