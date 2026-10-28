"""Input Guardrails: Phòng thủ đầu vào trước khi suy luận hoặc gọi LLM.

Đặc tả dựa trên:
- documents/09_GUARDRAILS.md
- documents/02_ARCHITECTURE.md (Mục 7)
- documents/04_CONFIDENCE_EVAL_POLICY.md (Mục 4)
- documents/08_AGENT_STATE.md (Mục 2)
"""

import re
from typing import Sequence

# Whitelist các trường thông tin kỹ thuật được phép hỏi theo từng Category (Cơ chế 2C)
CATEGORY_WHITELIST_FIELDS: dict[str, set[str]] = {
    "VPN": {"os_name", "error_code"},
    "PRINTER": {"printer_name", "error_code"},
    "SOFTWARE": {"software_name", "error_code"},
    "MAIL": {"client_type", "error_code"},
    "PASSWORD": set(),
    "OUT_OF_SCOPE": set(),
}

# Các trường nhạy cảm tuyệt đối không được phép hỏi người dùng
FORBIDDEN_SENSITIVE_FIELDS: set[str] = {
    "password",
    "pass",
    "pwd",
    "mat_khau",
    "otp",
    "otp_code",
    "secret",
    "token",
    "key",
    "credential",
}

# Danh sách từ khóa an ninh thông tin / tấn công / xâm nhập hệ thống
SECURITY_KEYWORDS: Sequence[str] = (
    "hack",
    "bị hack",
    "hacked",
    "ransomware",
    "tống tiền",
    "mã độc",
    "bị tấn công",
    "tấn công mạng",
    "xâm nhập",
    "chiếm quyền",
    "mất quyền kiểm soát",
    "khóa tài khoản",
    "locked account",
    "unauthorized access",
    "lộ lọt dữ liệu",
    "lộ dữ liệu",
    "data leak",
    "phát tán virus",
    "ddos",
    "nghi ngờ",
)

# Mẫu Regex phát hiện Prompt Injection & Jailbreak cơ bản
PROMPT_INJECTION_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"ignore\s+(?:(?:all|previous|above)\s+)*instructions", re.IGNORECASE),
    re.compile(r"bỏ\s+qua\s+(?:(?:mọi|tất\s+cả|các|những|trên|trước)\s+)*hướng\s+dẫn", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(?:a|an)?", re.IGNORECASE),
    re.compile(r"hãy\s+đóng\s+vai\s+", re.IGNORECASE),
    re.compile(r"(?:tiết\s+lộ|reveal|in\s+ra)\s+(?:toàn\s+bộ\s+)?(?:system\s+prompt|prompt)", re.IGNORECASE),
    re.compile(r"dan\s+mode|jailbreak", re.IGNORECASE),
)


def check_fast_security_concern(subject: str = "", body: str = "") -> bool:
    """Quét nhanh từ khóa bảo mật / sự cố an ninh / nghi ngờ xâm nhập (< 1ms)."""
    content = f"{subject} {body}".casefold()
    return any(keyword in content for keyword in SECURITY_KEYWORDS)


def check_prompt_injection(text: str) -> bool:
    """Kiểm tra heuristic phát hiện mẫu câu prompt injection / jailbreak."""
    content = text.strip()
    if not content:
        return False
    return any(pattern.search(content) is not None for pattern in PROMPT_INJECTION_PATTERNS)


def sanitize_missing_fields(category: str, raw_fields: list[str]) -> list[str]:
    """Lọc danh sách missing_fields theo đúng Whitelist của category và loại bỏ trường nhạy cảm."""
    cat_upper = category.upper()
    allowed = CATEGORY_WHITELIST_FIELDS.get(cat_upper, set())
    cleaned: list[str] = []
    for f in raw_fields:
        f_clean = f.strip().lower()
        if not f_clean:
            continue
        # Chặn các từ khóa nhạy cảm
        if any(sens in f_clean for sens in FORBIDDEN_SENSITIVE_FIELDS):
            continue
        # Chỉ giữ lại trường nằm trong Whitelist
        if f_clean in allowed:
            cleaned.append(f_clean)
    return cleaned