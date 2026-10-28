"""Script kiểm tra kết nối và tình trạng OpenRouter API Key qua gói OpenAI.

Chạy: python test_openrouter_api.py
"""

import json
import time
from openai import OpenAI
from src.core.config import get_settings
from src.agent.llm_client import generate


def mask_key(key: str) -> str:
    """Ẩn phần giữa của API key để bảo mật."""
    if not key or len(key) < 8:
        return "***"
    return f"{key[:8]}...{key[-4:]}"


def test_direct_openrouter():
    print("=" * 60)
    print("BƯỚC 1: KIỂM TRA OPENROUTER API KEY TRỰC TIẾP QUA GÓI OPENAI")
    print("=" * 60)

    settings = get_settings()
    key_obj = settings.openrouter_api_key
    if not key_obj or not key_obj.get_secret_value().strip():
        print("[LỖI] OPENROUTER_API_KEY chưa được cấu hình hoặc bị để trống trong file .env!")
        print("  -> Vui lòng mở file .env và điền:")
        print("     OPENROUTER_API_KEY = sk-or-v1-xxxx...")
        print("  -> Lấy key miễn phí hoặc trả phí tại: https://openrouter.ai/settings/keys")
        return False

    raw_key = key_obj.get_secret_value().strip()
    model = settings.openrouter_model
    base_url = settings.openrouter_base_url
    print(f"- Base URL: {base_url}")
    print(f"- Model kiểm tra: {model}")
    print(f"- API Key phát hiện được: {mask_key(raw_key)}")
    print(f"- Đang khởi tạo OpenAI client và gửi request kiểm tra...")

    start_time = time.time()
    try:
        client = OpenAI(
            base_url=base_url,
            api_key=raw_key,
            timeout=settings.llm_timeout_seconds,
            default_headers={
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "Agent IT Helpdesk Test",
            },
        )

        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Xin chào! Bạn là mô hình AI nào? Hãy trả lời trong 1 câu ngắn gọn."}
            ],
            max_tokens=60,
            temperature=0.2,
        )

        latency = time.time() - start_time
        print(f"\n[THÀNH CÔNG] Kết nối OpenRouter thành công sau {latency:.2f}s!")
        choice = completion.choices[0]
        content = choice.message.content
        print(f"- Phản hồi từ model:\n  \"{content}\"")
        print(f"- Finish reason: {choice.finish_reason}")
        if completion.usage:
            print(f"- Prompt tokens: {completion.usage.prompt_tokens}")
            print(f"- Completion tokens: {completion.usage.completion_tokens}")
            print(f"- Total tokens: {completion.usage.total_tokens}")
        return True

    except Exception as exc:
        latency = time.time() - start_time
        print(f"\n[THẤT BẠI] Gọi OpenRouter bị lỗi sau {latency:.2f}s!")
        print(f"Chi tiết lỗi: {type(exc).__name__}: {exc}")
        if "401" in str(exc):
            print("=> Gợi ý: API Key không hợp lệ hoặc đã bị revoke. Hãy kiểm tra lại key tại openrouter.ai.")
        elif "429" in str(exc):
            print("=> Gợi ý: Rate limit hoặc tài khoản hết credit/hết quota.")
        elif "404" in str(exc):
            print(f"=> Gợi ý: Model '{model}' không tìm thấy trên OpenRouter.")
        return False


def test_structured_output():
    print("\n" + "=" * 60)
    print("BƯỚC 2: KIỂM TRA TÍNH NĂNG JSON SCHEMA (GROUNDEDNESS JUDGE)")
    print("=" * 60)

    settings = get_settings()
    key_obj = settings.openrouter_api_key
    if not key_obj or not key_obj.get_secret_value().strip():
        print("[BỎ QUA] Thiếu API key.")
        return False

    raw_key = key_obj.get_secret_value().strip()
    model = settings.openrouter_model
    base_url = settings.openrouter_base_url

    schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok", "error"]},
            "tested_model": {"type": "string"},
            "message": {"type": "string"},
        },
        "required": ["status", "tested_model", "message"],
        "additionalProperties": False,
    }

    print(f"- Đang gửi request với JSON schema định dạng...")
    try:
        client = OpenAI(
            base_url=base_url,
            api_key=raw_key,
            timeout=settings.llm_timeout_seconds,
        )
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Hãy sinh JSON theo đúng schema, message: 'Kiểm tra schema thành công'"}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "test_schema",
                    "schema": schema,
                },
            },
            max_tokens=1000,
            temperature=0,
        )
        content = completion.choices[0].message.content
        print(f"[THÀNH CÔNG] JSON output nhận được:")
        print(f"  {content}")
        parsed = json.loads(content)
        print(f"-> Parsed JSON hợp lệ: {parsed}")
        return True
    except Exception as exc:
        print(f"[CẢNH BÁO] Không hỗ trợ strict json_schema hoặc lỗi: {exc}")
        return False


def test_via_llm_client():
    print("\n" + "=" * 60)
    print("BƯỚC 3: KIỂM TRA HÀM generate(..., use_cloud_model=True)")
    print("=" * 60)

    try:
        print("- Đang gọi generate('Chào bạn!', use_cloud_model=True) ...")
        resp = generate("Chào bạn! Hãy trả lời ngắn gọn trong 1 câu.", use_cloud_model=True)
        print("[THÀNH CÔNG] Kết quả qua llm_client:")
        print(f"  \"{resp}\"")
        return True
    except Exception as exc:
        print(f"[THẤT BẠI] Lỗi khi gọi generate: {type(exc).__name__}: {exc}")
        return False


if __name__ == "__main__":
    print("\n>>> BẮT ĐẦU KIỂM TRA OPENROUTER API & OPENAI SDK <<<\n")
    step1 = test_direct_openrouter()
    if step1:
        step2 = test_structured_output()
        step3 = test_via_llm_client()
        print("\n" + "=" * 60)
        print("KẾT LUẬN: OpenRouter hoạt động tốt! Sẵn sàng chạy eval_confidence_scorer.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("KẾT LUẬN: Chưa thể kết nối OpenRouter. Vui lòng kiểm tra lại OPENROUTER_API_KEY trong .env.")
        print("=" * 60)
