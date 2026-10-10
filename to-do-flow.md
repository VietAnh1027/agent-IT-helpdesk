Đúng, đến bước code — nhưng có một điểm khác biệt quan trọng so với "coding thứ tự chuẩn" của một CRUD app thông thường: dự án này là **AI-heavy**, nên thứ tự nên đi theo hướng **eval-driven**, không phải **UI-first**. Nghĩa là: xây được cái đo lường được chất lượng AI càng sớm càng tốt, trước khi xây UI đẹp — vì UI đẹp mà AI phía sau chưa đo được gì thì bạn sẽ không biết mình đang tiến hay lùi.

Thứ tự đề xuất, theo 4 giai đoạn lớn:

## Giai đoạn 1 — Nền tảng (không có AI, chỉ hạ tầng)

1. **Project scaffolding**: repo structure, Docker Compose (Postgres+pgvector, Redis nếu dùng queue), `.env` config, dependency management (poetry/pip-tools).
2. **Database schema**: viết migration cho toàn bộ bảng trong Data Model Doc (`tickets`, `kb_articles`, `kb_chunks`, `ai_responses`, `review_actions`, `eval_set`, `incidents`, `webhook_events`, `category_automation_state`, `category_metric_snapshots`). Làm hết 1 lần ở đây để tránh phải sửa schema lắt nhắt sau này.

## Giai đoạn 2 — Dữ liệu (trước khi có dòng logic AI nào)

3. **Viết KB tay** (ít nhất phần lõi, như đã bàn — đừng để AI sinh 100%).
4. **Sinh ticket synthetic** + **eval set** (100-150 mẫu có golden_answer) — làm trước khi code AI vì đây chính là "bài kiểm tra" bạn sẽ dùng liên tục ở các bước sau.

## Giai đoạn 3 — Lõi AI, xây theo thứ tự có thể đo được ngay từng bước

5. **Embedding + chunking + retrieval**: ingest KB vào `kb_chunks`, build retrieval query. Chạy thử eval set, đo **Retrieval Recall@5** ngay — đây là bước đầu tiên có số liệu thật, làm sớm để biết chunking strategy có ổn không trước khi tốn công build phần sau.
6. **LLM answer generation** (Ollama, model nhỏ) — nối vào sau retrieval, sinh câu trả lời kèm trích dẫn.
7. **Eval harness**: script chạy toàn bộ eval set qua pipeline, xuất ra `answer accuracy`, `hallucination rate`. Làm bước này **sớm**, không để cuối — vì từ đây trở đi mọi thay đổi (prompt, model, threshold) đều nên so sánh trước/sau bằng harness này.
8. **Confidence scorer + routing rules** (draft/clarify/escalate) — logic thuần, dễ unit test độc lập.
9. **Duplicate/incident detection, sentiment check, multi-issue check** — các nhánh phụ đã thiết kế, thêm dần, mỗi cái có thể unit test riêng vì đều là logic dựa trên rule + similarity, không phụ thuộc UI.

## Giai đoạn 4 — Tích hợp & giao diện (làm sau cùng, vì phụ thuộc vào phần lõi đã chạy đúng)

10. **Zendesk integration**: webhook nhận vào, API gửi comment ra. Có thể mock trước (giả lập webhook bằng script gửi request thủ công) để test pipeline không cần chờ set up Zendesk trial, rồi nối thật sau.
11. **Category graduation job** — chạy được vì đã có đủ dữ liệu review_actions từ các bước test trước đó.
12. **IT Agent Dashboard** (duyệt draft, xem incident, category automation status, metrics) — cố tình để cuối, vì đây là phần "vỏ" hiển thị dữ liệu đã có sẵn từ backend, không tạo ra logic mới. Build sớm dễ bị cám dỗ code UI đẹp trong khi AI phía sau còn tệ.
13. **Deploy** (Docker Compose → VPS) theo hướng hybrid đã chốt.

## Vì sao thứ tự này khác "chuẩn CRUD app" thông thường

Nhiều tutorial sẽ khuyên "làm UI trước để có cái nhìn thấy sớm, tạo động lực". Với dự án AI, lời khuyên đó dễ phản tác dụng: bạn dễ tưởng hệ thống "chạy được" chỉ vì UI đẹp, trong khi retrieval/answer generation phía dưới chưa được đo đạc gì. **Ưu tiên có eval harness chạy được sớm (bước 7) quan trọng hơn có dashboard đẹp sớm** — đây cũng chính là điểm bạn nên nhấn khi kể trong phỏng vấn về cách bạn tiếp cận dự án AI khác với web app thông thường.
