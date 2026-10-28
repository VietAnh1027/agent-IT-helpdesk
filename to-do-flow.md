## Giai đoạn 1 — Nền tảng (không có AI, chỉ hạ tầng)

1. **Project scaffolding**: repo structure, Docker Compose (Postgres+pgvector, Redis nếu dùng queue), `.env` config, dependency management (poetry/pip-tools).
2. **Database schema**: viết migration cho toàn bộ bảng trong Data Model Doc (`tickets`, `kb_articles`, `kb_chunks`, `ai_responses`, `review_actions`, `eval_set`, `incidents`, `webhook_events`, `category_automation_state`, `category_metric_snapshots`). Làm hết 1 lần ở đây để tránh phải sửa schema lắt nhắt sau này.

## Giai đoạn 2 — Dữ liệu (trước khi có dòng logic AI nào)

3. **Viết KB tay** (ít nhất phần lõi, như đã bàn — đừng để AI sinh 100%).
4. **Sinh ticket synthetic** + **eval set** (100-150 mẫu có golden_answer) — làm trước khi code AI vì đây chính là "bài kiểm tra" bạn sẽ dùng liên tục ở các bước sau.

## Giai đoạn 3 — Lõi AI, xây theo thứ tự có thể đo được ngay từng bước

5. **Embedding + chunking + retrieval**: ingest KB vào `kb_chunks`, build retrieval query. Chạy thử eval set, đo **Retrieval Recall@5** ngay — đây là bước đầu tiên có số liệu thật, làm sớm để biết chunking strategy có ổn không trước khi tốn công build phần sau.
6. **LLM answer generation** (Ollama, model nhỏ) — nối vào sau retrieval, sinh answer thuần; backend đính kèm sources từ đúng parent context đã cấp cho LLM.
7. **Eval harness**: script chạy toàn bộ eval set qua pipeline, xuất ra `answer accuracy`, `hallucination rate`. Làm bước này **sớm**, không để cuối — vì từ đây trở đi mọi thay đổi (prompt, model, threshold) đều nên so sánh trước/sau bằng harness này.
8. **Confidence scorer + routing rules** (draft/clarify/escalate) — logic thuần, dễ unit test độc lập.
9. **Duplicate/incident detection, sentiment check, multi-issue check** — các nhánh phụ đã thiết kế, thêm dần, mỗi cái có thể unit test riêng vì đều là logic dựa trên rule + similarity, không phụ thuộc UI.

## Giai đoạn 4 — Tích hợp & giao diện (làm sau cùng, vì phụ thuộc vào phần lõi đã chạy đúng)

10. **Zendesk integration**: webhook nhận vào, API gửi comment ra. Có thể mock trước (giả lập webhook bằng script gửi request thủ công) để test pipeline không cần chờ set up Zendesk trial, rồi nối thật sau.
11. **Category graduation job** — chạy được vì đã có đủ dữ liệu review_actions từ các bước test trước đó.
12. **IT Agent Dashboard** (duyệt draft, xem incident, category automation status, metrics) — cố tình để cuối, vì đây là phần "vỏ" hiển thị dữ liệu đã có sẵn từ backend, không tạo ra logic mới. Build sớm dễ bị cám dỗ code UI đẹp trong khi AI phía sau còn tệ.
13. **Deploy** (Docker Compose → VPS) theo hướng hybrid đã chốt.

---

## Chi tiết kế hoạch các bước triển khai tiếp theo (Sau khi hoàn thành Confidence Scorer)

Hiện tại hệ thống đã hoàn thành:
- **Giai đoạn 1 & 2:** DB migration, Ingestion parent-child KB, Golden dataset (150 mẫu).
- **Lõi AI (bước 5, 6, 7 & nửa đầu bước 8):** Hybrid retrieval (Dense + BM25 + RRF), Generator + Backend sources, Eval harnesses, và Confidence Scorer (`src/confidence.py` hỗ trợ cả full/lite mode).

Dưới đây là kế hoạch chi tiết các bước triển khai tiếp theo theo đúng thứ tự logic core-first:

### Bước 8.2: Hoàn thiện Routing Rules & Khung AgentState
1. **Định nghĩa `AgentState` (`src/agent/state.py`)**:
   - Xây dựng dataclass `AgentState` theo thiết kế tại `08_AGENT_STATE.md`.
   - Làm đối tượng dữ liệu duy nhất truyền qua từng node trong pipeline: chứa ticket info, classification, evidence, answer, groundedness, confidence, routing decision và errors.
2. **Triển khai Router logic (`src/router.py`)**:
   - Viết hàm `route_ticket(state: AgentState)` theo chính sách ngưỡng tại `04_CONFIDENCE_EVAL_POLICY.md`:
     - **Auto-reply:** $C_{RAG} \ge 0.85$, status `ok`, không dính guardrail, category ở stage `auto` (kèm cơ chế lấy mẫu canary draft).
     - **Draft (chờ duyệt):** $0.6 \le C_{RAG} < 0.85$.
     - **Clarify (hỏi làm rõ):** $0.4 \le C_{RAG} < 0.6$ **và** thiếu thông tin cụ thể theo category **và** `clarification_rounds == 0`.
     - **Escalate (chuyển người):** $C_{RAG} < 0.4$, hoặc đã hết lượt clarify, hoặc dính trigger escalate bắt buộc.
   - Viết unit test độc lập cho `src/router.py`.

### Bước 9.1: Xây dựng các bộ Phân loại sớm (Classifier & Guardrail Triggers)
1. **Intent Classifier (`src/agent/classifiers/intent_classifier.py`)**:
   - Phân loại ticket vào 1 trong 5 categories hoặc `out_of_scope`.
   - Phát hiện các trigger escalate sớm: `multi_issue_suspected = true`, từ khóa bảo mật nghi ngờ hack/khóa tài khoản.
   - Trích xuất `missing_fields` phục vụ câu hỏi clarify (dùng prompt `classify_system.jinja2`).
2. **Sentiment Checker (`src/agent/classifiers/sentiment_checker.py`)**:
   - Nhận diện cảm xúc người dùng: nếu `sentiment_flag = "angry"` $\rightarrow$ escalate ngay lập tức với priority cao, bỏ qua toàn bộ RAG phía sau.

### Bước 9.2: Subsystem Tương đồng (Similarity & Duplicate / Incident Detection)
- Xây dựng module dùng chung `src/agent/similarity/ticket_matcher.py` (theo `02_ARCHITECTURE.md` mục 3), dùng pgvector để thực hiện 4 nghiệp vụ:
  1. **Spam do bực bội:** Cùng email gửi dồn dập trong vài phút $\rightarrow$ gộp thread.
  2. **Repeat-unresolved:** Cùng email, tương đồng cao với ticket trong 7 ngày đã trả lời $\rightarrow$ escalate kèm link ticket cũ.
  3. **Mass Incident (Sự cố diện rộng 2 giai đoạn):**
     - Đếm ngầm trong cửa sổ 30 phút. Nếu $\ge 5$ ticket khác email có độ tương đồng cao $\rightarrow$ tự động tạo bản ghi `incidents`.
     - Từ ticket thứ 6 trở đi $\rightarrow$ gắn `status = linked_to_incident`, bỏ qua generation, gửi phản hồi "đã ghi nhận sự cố".
  4. **Recurring Resolved:** Tương đồng với ticket đã đóng trước đó $\rightarrow$ gắn cờ `recurring_issue = true` để theo dõi chất lượng KB.

### Bước 9.3: Tích hợp Tool-Use IT (`07_TOOLS.md`)
1. **Triển khai 5 Tool IT**:
   - **Tier 0 (Read-only):** `check_vpn_gateway_status`, `check_license_availability`, `check_printer_status`.
   - **Tier 1 (State-changing):** `trigger_password_reset_email`, `trigger_mailbox_resync`.
2. **Cơ chế Guardrail an toàn (`src/agent/tools/executor.py`)**:
   - Ép email nhận reset password phải lấy trực tiếp từ `tickets.requester_email` trong DB, cấm LLM tự sinh email ngoài.

### Bước 9.4: Sâu chuỗi Orchestrator Pipeline hoàn chỉnh (`src/pipeline.py`)
- Kết nối tất cả các thành phần thành luồng xử lý hoàn chỉnh theo `02_ARCHITECTURE.md`:
  $$\text{Ticket Input} \rightarrow \text{Duplicate/Incident Check} \rightarrow \text{Classify \& Sentiment} \rightarrow \text{Retrieval} \rightarrow \text{Tool} \rightarrow \text{Generate} \rightarrow \text{Confidence} \rightarrow \text{Router}$$
- Chạy benchmark đo lường toàn diện trên `golden_dataset.json` và `out_of_scope.json`.

### Giai đoạn 4: Tích hợp & Vận hành Production
1. **Background Job - Category Graduation (`src/jobs/graduation_job.py`)**:
   - Quét định kỳ bảng `review_actions` (7 ngày gần nhất) để tính approval rate, tự động promote/demote category (`shadow` $\rightarrow$ `draft` $\rightarrow$ `auto`) theo `05_CATEGORY_GRADUATION.md`.
2. **Incident Broadcast Job (`src/jobs/incident_broadcast_job.py`)**:
   - Tự động broadcast câu trả lời khi ticket đại diện của incident được duyệt.
3. **Zendesk Webhook & Client API (`src/api/` & `src/integrations/`)**:
   - Endpoint FastAPI nhận webhook idempotent (`webhook_event_id`).
   - Gọi API Zendesk gửi internal note (Draft) hoặc public reply (Auto/Clarify).
4. **IT Agent Dashboard (`web/`)**:
   - Giao diện duyệt/sửa draft, quản lý incident và xem metrics tự động hóa.
5. **Đóng gói Docker Compose & Deploy VPS**.
