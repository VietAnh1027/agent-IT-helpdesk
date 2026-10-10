# Confidence & Evaluation Policy — IT Support Ticket AI Agent

## 1. Mục đích

Doc này định nghĩa: (1) cách tính điểm tin cậy, (2) ngưỡng để quyết định draft / auto-reply / escalate, (3) cách đo chất lượng hệ thống bằng eval set, (4) các trigger escalate bắt buộc.

## 2. Công thức tính Confidence Score (đề xuất khởi điểm)

```
confidence = w1 * retrieval_similarity_top1
           + w2 * retrieval_score_gap (top1 - top2, càng lớn càng chắc chắn đúng nguồn)
           + w3 * llm_self_reported_confidence (prompt LLM tự chấm 0-1)
```

Gợi ý trọng số khởi điểm: w1 = 0.5, w2 = 0.2, w3 = 0.3 — **đây chỉ là điểm khởi đầu, cần calibrate lại bằng eval set thật**, đừng để cố định mãi. Việc đo lại và điều chỉnh trọng số chính là phần thể hiện tư duy engineering trong CV.

## 3. Ngưỡng xử lý (threshold routing)

| Confidence                                                  | Điều kiện thêm                                                                                                                                             | Hành động                                                                                                                                                                                                                                                                                                                                                                 |
| ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ≥ 0.85                                                     | —                                                                                                                                                             | Đủ điều kiện Auto-reply*nếu* category đang ở stage `auto` (xem cơ chế tự động promote/demote category ở `05_CATEGORY_GRADUATION.md`). Kể cả khi đủ điều kiện, 1 phần nhỏ ticket vẫn được lấy mẫu ngẫu nhiên (canary) đi qua Draft mode thay vì auto-gửi, để duy trì tín hiệu giám sát chất lượng — chi tiết ở cùng doc |
| 0.6 – 0.85                                                 | —                                                                                                                                                             | Tạo Draft, chờ người duyệt                                                                                                                                                                                                                                                                                                                                              |
| 0.4 – 0.6                                                  | AI xác định thiếu thông tin cụ thể (thiếu trường bắt buộc theo category, VD: hệ điều hành, mã lỗi)**và** `clarification_rounds = 0` | Sinh 1 câu hỏi làm rõ, gửi public reply, chuyển`status = awaiting_clarification`, tăng `clarification_rounds` lên 1                                                                                                                                                                                                                                              |
| < 0.4, hoặc (0.4–0.6 nhưng`clarification_rounds ≥ 1`) | —                                                                                                                                                             | Escalate, không sinh câu trả lời cho end-user, chỉ ghi log nội bộ                                                                                                                                                                                                                                                                                                     |

> Câu hỏi làm rõ chỉ được sinh khi AI xác định rõ **thiếu trường thông tin cụ thể nào** (không phải "confidence thấp chung chung"). Nếu không xác định được thiếu gì cụ thể, dù confidence ở dải 0.4–0.6 vẫn escalate thẳng — tránh AI hỏi những câu vô nghĩa kiểu "bạn có thể nói rõ hơn không?".

> Các con số trên là điểm khởi đầu để code. Sau khi chạy Shadow mode và Draft mode, dùng dữ liệu approval rate thật để hiệu chỉnh lại ngưỡng — nên vẽ biểu đồ precision/recall theo từng mức ngưỡng để chọn điểm tối ưu, đây là phần rất đáng đưa vào README.

## 4. Escalation Triggers bắt buộc (bỏ qua confidence score)

Các trường hợp sau **luôn escalate ngay lập tức**, không qua LLM answer generation:

- Ticket bị Classifier gắn `out_of_scope`
- Ticket chứa từ khóa liên quan bảo mật/tài khoản bị khóa do nghi vấn (danh sách từ khóa tự định nghĩa, ví dụ: "hack", "xâm nhập", "nghi ngờ", "khóa tài khoản")
- Retrieval không tìm được KB chunk nào vượt ngưỡng similarity tối thiểu (không có nguồn để trích dẫn)
- Ticket lặp lại nhiều lần (cùng requester, cùng category, trong khoảng thời gian ngắn) — dấu hiệu câu trả lời trước chưa giải quyết được vấn đề
- Nghi ngờ prompt injection (nội dung ticket chứa chỉ thị nhắm vào hệ thống AI, vd "ignore previous instructions")
- Ticket đã dùng hết lượt clarification (`clarification_rounds ≥ 1`) mà vẫn không đủ thông tin để đạt confidence ≥ 0.6
- Ticket đã ở status `auto_sent` hoặc `closed`, nhưng có comment mới từ end-user (reopened) — escalate ngay, không chạy lại AI pipeline
- `sentiment_flag = angry` — escalate ngay bất kể confidence, gắn priority cao
- `multi_issue_suspected = true` — không cố trả lời gộp nửa vời, route thẳng cho người

## 5. Eval Set

- Kích thước khởi điểm: 100–150 ticket mẫu (trải đều 5 category + một số out-of-scope + một số adversarial).
- Cấu trúc: xem bảng `eval_set` trong Data Model Doc.
- Phân bố độ khó:
  - Easy (60%): câu hỏi rõ ràng, khớp trực tiếp 1 bài KB
  - Medium (25%): câu hỏi mơ hồ, cần suy luận nhẹ hoặc kết hợp nhiều đoạn KB
  - Hard/Adversarial (15%): ticket viết sai chính tả nhiều, ticket cố tình đánh lừa hệ thống, ticket ngoài phạm vi được ngụy trang giống trong phạm vi

## 6. Các Metrics cần đo và cách tính

| Metric                        | Công thức / cách đo                                                                                                                                                                   |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Retrieval Recall@5            | % ticket mà KB article đúng nằm trong top-5 kết quả retrieval                                                                                                                       |
| Answer Accuracy               | So sánh draft answer với golden_answer, có thể chấm tay hoặc dùng LLM-as-judge (ghi rõ phương pháp nào được dùng, không giấu)                                           |
| Hallucination Rate            | % câu trả lời có confidence cao nhưng nội dung sai/không khớp golden_answer —**chỉ số quan trọng nhất, phải báo cáo riêng**                                        |
| Draft Approval Rate           | Từ bảng`review_actions`: % approved_as_is trên tổng draft                                                                                                                           |
| Escalation Precision/Recall   | Trong các case đáng lẽ phải escalate (theo eval set gán sẵn), hệ thống có escalate đúng không                                                                                |
| Latency                       | Thời gian trung bình từ nhận ticket đến có draft                                                                                                                                   |
| Clarification Resolution Rate | Trong số ticket được hỏi làm rõ 1 lần, bao nhiêu % sau đó đạt confidence ≥ 0.6 (tức là câu hỏi làm rõ thực sự hữu ích, không chỉ làm chậm ticket)             |
| Incident Grouping Rate        | % ticket được gộp đúng vào 1 incident thay vì xử lý riêng lẻ trong lúc sự cố diện rộng — đo mức tiết kiệm compute/thời gian duyệt                                 |
| Reopened Rate                 | % ticket đã`auto_sent`/`closed` nhưng bị end-user reopen — chỉ số chất lượng thật của auto-reply, quan trọng hơn cả approval rate vì đo hậu quả thật sau khi gửi |
| Recurring Issue Rate          | % ticket gắn`recurring_issue = true` theo category — chỉ ra category nào có vấn đề gốc chưa fix triệt để                                                                   |

## 7. Cách dùng eval set trong quá trình phát triển

1. Chạy toàn bộ eval set qua hệ thống trước khi thay đổi gì (baseline).
2. Mỗi khi đổi: embedding model, chunking strategy, prompt, threshold — chạy lại eval set, so sánh số liệu trước/sau.
3. Lưu lại lịch sử các lần chạy (có thể chỉ cần 1 file CSV/JSON theo version) để vẽ biểu đồ tiến bộ theo thời gian — đây là bằng chứng thuyết phục nhất trong phỏng vấn, hơn hẳn chỉ nói "tôi đã cải thiện hệ thống".

## 8. Guardrail test bắt buộc trước khi coi v1 "hoàn thành"

- [ ] Test với ít nhất 10 ticket adversarial (prompt injection, câu hỏi ngoài scope ngụy trang)
- [ ] Test với ticket cố tình chứa thông tin nhạy cảm giả lập, xác nhận hệ thống escalate đúng
- [ ] Xác nhận không có trường hợp nào confidence cao mà câu trả lời sai trong eval set (nếu có, phải hạ threshold hoặc cải thiện retrieval trước khi tiếp tục)
