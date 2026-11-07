"""Interactive Console CLI cho IT Helpdesk AI Agent.

Cho phép trải nghiệm và kiểm thử trực tiếp toàn bộ luồng pipeline LangGraph:
- Similarity & Incident Detection
- Intent Classification & Sentiment Analysis
- Diagnostic Tool Calling (Read-only)
- Hybrid Retrieval & Evidence Context
- LLM Answer Generation
- Confidence Scorer & Groundedness
- Router Policy (Auto, Draft, Clarify, Escalate)

Chạy:
    python cli.py
"""

import json
import sys
import time

from src.agent.similarity.repository import InMemoryTicketRepository
from src.agent.state import AgentState
from src.agent.tools.diagnostic_adapter import MockDiagnosticBackendAdapter
from src.pipeline import process_ticket

# Mã màu ANSI hiển thị trên Terminal
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_banner():
    banner = f"""
{CYAN}╔══════════════════════════════════════════════════════════════════════════════════════╗
║               {BOLD}🤖 IT HELPDESK AI AGENT — LIVE INTERACTIVE CONSOLE{RESET}{CYAN}                    ║
║         Pipeline điều phối đa tầng xây dựng bằng LangGraph & Phân luồng thông minh   ║
╚══════════════════════════════════════════════════════════════════════════════════════╝{RESET}
"""
    print(banner)


def setup_demo_environment():
    """Khởi tạo môi trường dữ liệu giả lập cho Demo CLI."""
    repo = InMemoryTicketRepository()
    adapter = MockDiagnosticBackendAdapter()

    # Seed dữ liệu chẩn đoán kỹ thuật có kịch bản
    adapter.seed_vpn_gateway("default", status="maintenance", last_incident_at="2026-11-04T08:00:00Z")
    adapter.seed_printer("PRINTER_01", status="online", queue_length=0)
    adapter.seed_printer("PRINTER_02", status="jammed", queue_length=5, last_error="Kẹt giấy khay 2 (Paper jam)")
    adapter.seed_license("Figma", total_seats=10, used_seats=10)  # Hết license
    adapter.seed_license("Photoshop", total_seats=5, used_seats=2)  # Còn 3 license

    return repo, adapter


def display_pipeline_telemetry(state: AgentState):
    """In chi tiết từng chặng xử lý của AgentState ra màn hình."""
    print(f"\n{BOLD}{'='*30} TIẾN TRÌNH XỬ LÝ (TELEMETRY) {'='*30}{RESET}\n")

    # 1. Similarity Check
    print(f"{CYAN}[Chặng 1: Phân Tích Tương Đồng & Sự Cố]{RESET}")
    print(f"  • Spam dồn dập (Flood) : {'Có (Bị phát hiện)' if state.is_spam_duplicate else 'Không'}")
    print(f"  • Hỏi lặp lại (Repeat) : {'Có (Chưa dứt điểm)' if state.repeat_ticket_suspected else 'Không'}")
    print(f"  • Sự cố diện rộng (Inc) : {state.incident_id or 'Không (Ticket đơn lẻ)'}")
    print(f"  • Lỗi tái phát (Recur)  : {'Có (Từng resolve trong quá khứ)' if state.recurring_issue else 'Không'}")

    # 2. Classifier & Guardrail
    cls = state.classification or {}
    print(f"\n{CYAN}[Chặng 2: Phân Loại Ý Định & An Toàn]{RESET}")
    cat = cls.get("category", "UNKNOWN")
    sentiment = cls.get("sentiment_flag", "neutral")
    sec_concern = cls.get("security_concern", False)
    injection = cls.get("prompt_injection_suspected", False)

    print(f"  • Category nhận diện    : {BOLD}{cat}{RESET}")
    print(f"  • Cảm xúc (Sentiment)   : {RED if sentiment == 'angry' else GREEN}{sentiment.upper()}{RESET}")
    print(f"  • Dấu hiệu Hack/Bảo mật : {RED + 'CẢNH BÁO BẢO MẬT' + RESET if sec_concern else 'An toàn'}")
    print(f"  • Prompt Injection      : {RED + 'PHÁT HIỆN INJECTION' + RESET if injection else 'Không'}")
    if cls.get("missing_fields"):
        print(f"  • Thông tin còn thiếu   : {YELLOW}{cls.get('missing_fields')}{RESET}")

    # 3. Diagnostic Tool
    print(f"\n{CYAN}[Chặng 3: Công Cụ Chẩn Đoán Kỹ Thuật (Read-Only)]{RESET}")
    if state.diagnostic_data:
        print(f"  • Kết quả tra cứu Tool  : {YELLOW}{json.dumps(state.diagnostic_data, ensure_ascii=False)}{RESET}")
    else:
        print("  • Không kích hoạt Tool chẩn đoán (Category không yêu cầu tra cứu).")

    # 4. Retrieval KB
    print(f"\n{CYAN}[Chặng 4: Truy Vấn Tri Thức (Hybrid Retrieval)]{RESET}")
    if state.retrieved_parents:
        print(f"  • Số đoạn tri thức tìm được : {len(state.retrieved_parents)} đoạn")
        for i, p in enumerate(state.retrieved_parents[:3], 1):
            title = p.get("article_title", "Tài liệu")
            score = p.get("relevance_score", 0.0)
            print(f"    [{i}] {title} (Điểm phù hợp: {score:.3f})")
    else:
        print("  • Không có tài liệu KB (bị ngắt sớm hoặc không tìm thấy).")

    # 5. Answer Generation
    gen = state.generation or {}
    print(f"\n{CYAN}[Chặng 5: Sinh Câu Trả Lời (LLM Answer Generation)]{RESET}")
    answer = gen.get("answer", "")
    if answer:
        print(f"  • Hoàn tất sinh văn bản : {'Có' if state.generation_complete else 'Bị cắt ngắn/lỗi'}")
    else:
        print("  • Không sinh văn bản (đã ngắt luồng sớm để bảo vệ an toàn).")

    # 6. Confidence & Groundedness
    conf = state.confidence or {}
    print(f"\n{CYAN}[Chặng 6: Đánh Giá Độ Tin Cậy & Tính Xác Thực]{RESET}")
    c_rag = conf.get("c_rag")
    status = conf.get("status", "unknown")
    if c_rag is not None:
        print(f"  • Điểm tổng hợp C_RAG   : {BOLD}{c_rag:.3f}{RESET} (Trạng thái: {status})")
        print(f"    - Retrieval Score     : {conf.get('c_retrieval', 0.0):.3f}")
        print(f"    - Consensus Score     : {conf.get('c_consensus', 0.0):.3f}")
        if conf.get("c_groundedness") is not None:
            print(f"    - Groundedness Score  : {conf.get('c_groundedness', 0.0):.3f}")
    else:
        print(f"  • Điểm C_RAG            : Không áp dụng ({conf.get('errors') or 'Bị bỏ qua'})")

    # 7. Router Decision
    decision = state.routing_decision or "CHƯA XÁC ĐỊNH"
    reason = state.routing_reason or "Không rõ"

    color_map = {
        "auto": GREEN,
        "draft": YELLOW,
        "clarify": BLUE,
        "escalate": RED,
    }
    color = color_map.get(decision.lower(), RESET)

    print(f"\n{BOLD}{'='*25} QUYẾT ĐỊNH PHÂN LUỒNG CUỐI CÙNG {'='*25}{RESET}")
    print(f"  🎯 HÀNH ĐỘNG ROUTER : {color}{BOLD}[ {decision.upper()} ]{RESET}")
    print(f"  📌 LÝ DO QUYẾT ĐỊNH : {BOLD}{reason}{RESET}")
    print(f"{BOLD}{'='*78}{RESET}\n")

    if answer:
        print(f"{GREEN}{BOLD}💬 NỘI DUNG PHẢN HỒI GỬI TỚI NGƯỜI DÙNG:{RESET}")
        print(f"----------------------------------------------------------------------")
        print(answer)
        print(f"----------------------------------------------------------------------\n")


def run_ticket_flow(subject: str, body: str, email: str, repo, adapter):
    """Thực thi luồng ticket qua pipeline và in kết quả."""
    print(f"\n{YELLOW}Đang khởi chạy luồng xử lý qua LangGraph...{RESET}")
    start_time = time.time()

    state = process_ticket(
        subject=subject,
        body=body,
        requester_email=email,
        repo=repo,
        adapter=adapter,
    )

    elapsed = time.time() - start_time
    display_pipeline_telemetry(state)
    print(f"⏱️  Tổng thời gian xử lý: {elapsed:.2f} giây\n")


def main():
    print_banner()
    repo, adapter = setup_demo_environment()

    SCENARIOS = {
        "1": (
            "Lỗi VPN văn phòng",
            "Tôi không thể kết nối vào cổng VPN default được, ứng dụng cứ báo connecting rồi timeout hoài.",
            "developer@company.com",
            "Demo: Chẩn đoán Tool VPN (Gateway đang bảo trì -> Đưa vào ngữ cảnh giải thích)",
        ),
        "2": (
            "Máy in tầng 4 kẹt giấy",
            "Máy in PRINTER_02 ở tầng 4 bị kẹt giấy liên tục không thể in hợp đồng cho sếp được.",
            "accountant@company.com",
            "Demo: Chẩn đoán Tool Máy In (PRINTER_02 bị kẹt giấy khay 2)",
        ),
        "3": (
            "Xin cấp bản quyền phần mềm",
            "Tôi mới vào dự án, nhờ IT cấp bản quyền phần mềm Figma để làm việc với team UI/UX.",
            "designer@company.com",
            "Demo: Chẩn đoán Tool License (Figma đã hết license 10/10 -> Hướng dẫn mua thêm)",
        ),
        "4": (
            "CẢNH BÁO BẢO MẬT",
            "Tài khoản của tôi hình như bị hack rồi, có ai đó đang xâm nhập và chiếm quyền điều khiển máy tính!",
            "victim@company.com",
            "Demo: Fast-path Security (Chặn đứng tức thì trong < 1ms -> Escalate khẩn cấp)",
        ),
        "5": (
            "Bọn làm ăn vô trách nhiệm",
            "BỌN MÀY LÀM ĂN KIỂU GÌ ĐẤY HẢ, GỌI QUẢN LÝ RA ĐÂY TAO NÓI CHUYỆN, ĐỒ TỒI!",
            "angry_user@company.com",
            "Demo: Sentiment Angry (Phát hiện xúc phạm/giận dữ -> Escalate ngay lập tức)",
        ),
        "6": (
            "Hỏi mua đồ ăn",
            "Căng tin công ty trưa nay có bán cơm tấm sườn trứng không nhỉ mọi người?",
            "hungry_employee@company.com",
            "Demo: Out of scope (Hỏi ngoài phạm vi IT -> Escalate an toàn)",
        ),
    }

    while True:
        print(f"{BOLD}CHỌN KỊCH BẢN ĐỂ CHẠY THỬ NGHIỆM:{RESET}")
        print("  [C]  ✍️  Nhập Ticket tùy ý (Custom Subject, Body, Email)")
        for k, v in SCENARIOS.items():
            print(f"  [{k}]  📋 {v[0]} — {CYAN}{v[3]}{RESET}")
        print("  [0]  🚪 Thoát chương trình")

        choice = input(f"\n{BOLD}Nhập lựa chọn của bạn: {RESET}").strip()

        if choice == "0":
            print("\nTạm biệt! Chúc bạn một ngày làm việc hiệu quả.\n")
            break

        if choice.upper() == "C":
            print("\n--- NHẬP THÔNG TIN TICKET ---")
            subj = input("1. Tiêu đề (Subject): ").strip()
            if not subj:
                print(f"{RED}Tiêu đề không được để trống!{RESET}\n")
                continue
            body = input("2. Nội dung (Body): ").strip()
            email = input("3. Email người gửi [mặc định: user@company.com]: ").strip() or "user@company.com"

            run_ticket_flow(subj, body, email, repo, adapter)

        elif choice in SCENARIOS:
            subj, body, email, _ = SCENARIOS[choice]
            print(f"\n{BOLD}ĐÃ CHỌN KỊCH BẢN [{choice}]: {subj}{RESET}")
            print(f"• Người gửi: {email}")
            print(f"• Nội dung : {body}")
            run_ticket_flow(subj, body, email, repo, adapter)
        else:
            print(f"{RED}Lựa chọn không hợp lệ, vui lòng chọn lại!{RESET}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nĐã dừng chương trình.\n")
        sys.exit(0)

