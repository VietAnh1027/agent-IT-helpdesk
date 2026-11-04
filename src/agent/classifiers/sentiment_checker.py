"""Sentiment Checker: nhận diện cảm xúc người dùng (neutral, frustrated, angry).

Hỗ trợ kiểm tra nhanh qua regex (Fast-path).
"""

import re

# Từ khóa biểu hiện sự tức giận dữ dội, xúc phạm hoặc đe dọa
ANGRY_KEYWORDS = (
    "lừa đảo",
    "vô trách nhiệm",
    "gọi quản lý",
    "quản lý ngay",
    "báo công an",
    "kiện",
    "làm ăn kiểu gì",
    "khốn nạn",
    "đồ tồi",
    "bọn mày",
    "chết tiệt",
    "điên mất",
    "scam",
    "sue",
    "lawyer",
    "worst service",
    "completely unacceptable",
)

ANGRY_REGEX = re.compile(
    r"\b(" + "|".join(re.escape(kw) for kw in ANGRY_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


def check_fast_angry_sentiment(text: str) -> bool:
    """Quét nhanh regex các dấu hiệu tức giận cực độ, đe dọa hoặc chửi bới.

    Dùng trong Fast-path để phát hiện nhanh các ca khẩn cấp mà không cần tốn token LLM.
    """
    if not text or not text.strip():
        return False

    normalized = " ".join(text.casefold().split())
    if ANGRY_REGEX.search(normalized):
        return True

    # Kiểm tra trường hợp gõ toàn bộ bằng chữ in hoa kèm nhiều dấu chấm than (hò hét)
    words = [w for w in text.split() if w.isalpha()]
    if len(words) >= 5 and sum(1 for w in words if w.isupper()) / len(words) >= 0.8:
        if "!" in text or "?" in text:
            return True

    return False