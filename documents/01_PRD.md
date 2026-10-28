# PRD — IT Support Ticket AI Agent

## 1. Bối cảnh & vấn đề

Phòng IT Support ở hầu hết công ty nhận một lượng lớn ticket lặp đi lặp lại (VPN, reset password, cấp quyền phần mềm, lỗi máy in, lỗi email) mà câu trả lời đã tồn tại sẵn trong knowledge base (KB). Nhân viên IT tốn thời gian trả lời thủ công những câu hỏi này thay vì tập trung vào việc phức tạp hơn.

Dự án này xây một AI agent đọc ticket, tra cứu KB, và **soạn câu trả lời (draft)** cho nhân viên IT duyệt, tiến tới **tự động trả lời** cho một số loại ticket đã được chứng minh an toàn và chính xác.

> Đây là dự án cá nhân (portfolio), không có ticket/KB thật từ công ty. Dữ liệu được tổng hợp bán thực tế (synthetic, có ghi rõ nguồn) để mô phỏng môi trường production.

## 2. Mục tiêu

**Mục tiêu sản phẩm:**
- Tự động xử lý được phần lớn ticket thuộc nhóm "có thể trả lời bằng KB", giảm workload cho IT support.
- Có cơ chế an toàn: không bao giờ tự gửi câu trả lời sai một cách tự tin.

**Mục tiêu portfolio:**
- Thể hiện khả năng thiết kế hệ thống end-to-end (không chỉ demo RAG).
- Có số liệu đo đạc cụ thể (retrieval accuracy, hallucination rate, resolution rate) để nói trong phỏng vấn.
- Thể hiện tư duy rủi ro/production: human-in-the-loop, escalation, logging, eval.

## 3. Đối tượng sử dụng (giả lập)

| Vai trò | Mô tả | Tương tác với hệ thống |
|---|---|---|
| End user | Nhân viên công ty (giả lập) gửi ticket | Gửi ticket qua form/API |
| IT Agent (người duyệt) | Nhân viên IT support | Duyệt/sửa draft trên dashboard |
| Hệ thống | AI Agent | Phân loại, tra cứu, sinh câu trả lời, chấm điểm tin cậy |

## 4. Phạm vi (Scope)

### Trong phạm vi (v1)
5 nhóm ticket:
1. Kết nối VPN (cài đặt, lỗi kết nối, quên cấu hình)
2. Reset/quên mật khẩu (tự phục vụ, không phải tài khoản bị khóa do bảo mật)
3. Xin cấp quyền / cài đặt phần mềm
4. Lỗi máy in (không nhận diện, kẹt giấy, driver)
5. Lỗi đồng bộ email/Outlook

### Ngoài phạm vi (v1 — ghi rõ để tránh scope creep)
- Ticket liên quan bảo mật (nghi ngờ xâm nhập, khóa tài khoản do vi phạm chính sách)
- Ticket cần thao tác hệ thống thật (cấp quyền thật, reset AD thật) — v1 chỉ dừng ở tư vấn/hướng dẫn
- Ticket đa ngôn ngữ (chỉ tiếng Việt hoặc chỉ tiếng Anh, chọn 1)
- Tích hợp với ticketing tool thật của bên thứ ba (Zendesk/Jira) — thiết kế tương thích nhưng không bắt buộc tích hợp thật ở v1

## 5. User Stories

- Là end user, tôi gửi ticket mô tả vấn đề IT, và nhận được câu trả lời nhanh nếu thuộc loại đã hỗ trợ.
- Là IT Agent, tôi thấy draft câu trả lời AI đề xuất kèm danh sách tài liệu KB đã cung cấp cho AI (backend đính kèm; không khẳng định từng câu đã được nguồn hỗ trợ), có thể duyệt nguyên văn hoặc sửa trước khi gửi.
- Là IT Agent, tôi thấy ticket bị AI từ chối xử lý (do không đủ tin cậy hoặc ngoài phạm vi) được gắn cờ rõ ràng kèm lý do.
- Là chủ hệ thống, tôi xem được dashboard số liệu: tỷ lệ tự động hóa, tỷ lệ duyệt nguyên văn, tỷ lệ escalate.

## 6. Success Metrics

| Metric | Mục tiêu v1 | Ý nghĩa |
|---|---|---|
| Retrieval Recall@5 | ≥ 85% | KB đúng có nằm trong top 5 kết quả tra cứu không |
| Draft approval rate (duyệt nguyên văn, không sửa) | ≥ 60% | Chất lượng câu trả lời AI |
| Hallucination rate (trả lời sai nhưng tự tin) | < 3% | Chỉ số an toàn quan trọng nhất |
| Auto-resolution rate (sau khi bật auto-reply) | ≥ 30% tổng ticket trong scope | Hiệu quả giảm tải |
| Escalation accuracy | ≥ 90% ticket nên escalate được escalate đúng | An toàn |

## 7. Roadmap theo giai đoạn

1. **Shadow mode** — AI trả lời nội bộ, không hiển thị, dùng để đo baseline so với "ground truth" tự viết.
2. **Draft mode** — AI đề xuất, "IT Agent" (chính bạn đóng vai) duyệt/sửa. Thu thập dữ liệu chênh lệch draft vs. final.
3. **Auto-reply có giới hạn** — Chỉ bật cho category đạt ngưỡng chất lượng ổn định qua Draft mode. Việc chuyển sang Auto (và tự rút lui nếu chất lượng giảm) không làm thủ công mà theo cơ chế tự động dựa trên số liệu — xem chi tiết ở `05_CATEGORY_GRADUATION.md`.

## 8. Ràng buộc kỹ thuật đã biết

- Local LLM chạy trên GPU cá nhân (RTX 3050, 4GB VRAM) — giới hạn kích thước model có thể chạy mượt.
- Muốn có link demo sống trên VPS/cloud nhỏ — cần tính toán vì VPS phổ thông thường không có GPU (xem Architecture Doc, mục "Quyết định mở").
- Vector DB: pgvector (Postgres).

## 9. Rủi ro & giả định

- **Giả định**: dữ liệu ticket/KB synthetic đủ đa dạng để mô phỏng thực tế — cần kiểm tra bằng cách tự đọc lại, không chỉ generate rồi dùng luôn.
- **Rủi ro**: model nhỏ (do giới hạn VRAM/CPU) có thể chất lượng thấp hơn model lớn — cần document rõ trade-off này trong Architecture Doc để không bị hỏi "sao không dùng GPT-4" mà không trả lời được.
- **Rủi ro**: nếu không tự viết KB tay mà để AI sinh toàn bộ, dự án mất giá trị chứng minh hiểu domain — cần tự viết ít nhất một phần KB.
