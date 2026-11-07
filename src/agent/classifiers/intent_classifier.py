"""Intent Classifier: phân loại danh mục, phát hiện guardrails và trích xuất missing_fields.

Đặc tả dựa trên:
- documents/02_ARCHITECTURE.md (Mục 4)
- documents/04_CONFIDENCE_EVAL_POLICY.md (Mục 3, 4)
- implementation_plan/11_CLASSIFIER_AND_GUARDRAILS_IMPLEMENTATION_PLAN.md
"""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

from src.agent.classifiers.sentiment_checker import check_fast_angry_sentiment
from src.agent.guardrails.input_guardrails import (
    check_fast_security_concern,
    sanitize_missing_fields,
)
from src.agent.llm_client import generate, render_prompt
from src.agent.state import AgentState
from src.core.config import get_settings


class ClassificationResult(BaseModel):
    model_config = ConfigDict(strict=False, extra="ignore")

    category: Literal["VPN", "PASSWORD", "SOFTWARE", "PRINTER", "MAIL", "OUT_OF_SCOPE"]
    multi_issue_suspected: bool = False
    prompt_injection_suspected: bool = False
    sentiment_flag: Literal["neutral", "frustrated", "angry"] = "neutral"
    missing_fields: list[str] = Field(default_factory=list)
    security_concern: bool = False
    reason: str = ""




def classify_ticket(
    subject: str,
    body: str,
    *,
    use_cloud_model: bool | None = None,
) -> ClassificationResult:
    """Phân loại ticket qua 2 lớp: Fast-path Rules và LLM Classifier."""
    # 1. Lớp Fast-path: Kiểm tra an ninh / xâm nhập tài khoản
    is_angry = check_fast_angry_sentiment(f"{subject} {body}")
    if check_fast_security_concern(subject, body):
        return ClassificationResult(
            category="OUT_OF_SCOPE",
            multi_issue_suspected=False,
            prompt_injection_suspected=False,
            sentiment_flag="angry" if is_angry else "neutral",
            missing_fields=[],
            security_concern=True,
            reason="security_concern_keyword",
        )

    # 2. Lớp LLM Classifier
    settings = get_settings()
    if use_cloud_model is None:
        use_cloud_model = bool(
            getattr(settings, "openrouter_api_key", None)
            and settings.openrouter_api_key.get_secret_value().strip()
        )

    model = settings.openrouter_model if use_cloud_model else settings.ollama_model
    system_prompt = render_prompt("classify_system.jinja2")
    task_prompt = render_prompt(
        "classify_task.jinja2",
        ticket_subject=subject,
        ticket_body=body,
    )

    response = generate(
        task_prompt,
        model_name=model,
        system_prompt=system_prompt,
        response_schema=ClassificationResult.model_json_schema(),
        use_cloud_model=use_cloud_model,
        max_tokens=512,
    )

    parsed = ClassificationResult.model_validate_json(response)

    # Nếu fast-path phát hiện sự tức giận cực độ, ưu tiên gán angry
    if is_angry:
        parsed.sentiment_flag = "angry"

    # Lọc missing_fields theo Whitelist 2C
    parsed.missing_fields = sanitize_missing_fields(parsed.category, parsed.missing_fields)
    return parsed


def classify_state(
    state: AgentState,
    *,
    use_cloud_model: bool | None = None,
) -> AgentState:
    """Phân loại ticket và cập nhật kết quả trực tiếp vào AgentState."""
    try:
        result = classify_ticket(state.subject, state.body, use_cloud_model=use_cloud_model)
        state.classification = result.model_dump()
        state.missing_fields = list(result.missing_fields)
    except Exception as exc:
        state.add_error("classification", str(exc))
        # Fallback an toàn: đưa về OUT_OF_SCOPE để router chuyển sang Escalate
        state.classification = {
            "category": "OUT_OF_SCOPE",
            "multi_issue_suspected": False,
            "prompt_injection_suspected": False,
            "sentiment_flag": "neutral",
            "missing_fields": [],
            "security_concern": False,
            "reason": f"Classification fallback due to error: {exc}",
        }
        state.missing_fields = []

    return state

