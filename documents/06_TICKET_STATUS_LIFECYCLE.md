# Ticket Status Lifecycle — IT Support Ticket AI Agent

Tổng hợp toàn bộ giá trị của `tickets.status`, theo đúng thứ tự thường xảy ra trong vòng đời 1 ticket. Tham chiếu chi tiết logic ở `02_ARCHITECTURE.md` và `03_DATA_MODEL.md`.

## Bảng tổng hợp

| Status | Được gán ở giai đoạn nào (bước nào trong pipeline) | Hành vi/hệ quả ngay sau khi gán |
|---|---|---|
| `new` | Ngay khi webhook từ Zendesk/Jira được nhận và lưu vào `tickets`, trước khi chạy bất kỳ bước xử lý nào | Ticket chờ vào hàng đợi xử lý (nếu dùng queue bất đồng bộ) hoặc được xử lý ngay (nếu xử lý đồng bộ trong request webhook) |
| `linked_to_incident` | Bước "Duplicate & Incident Detection" — **chỉ khi** ticket khớp với 1 incident đã ở trạng thái **confirmed** (đã có ≥5 ticket từ trước, không phải lần đầu chạm ngưỡng) | Pipeline dừng lại **ngay tại đây**, không chạy Classifier/Retrieval/Generation. Gửi placeholder "đã ghi nhận, đang xử lý" qua Zendesk/Jira API. Chờ `representative_ticket_id` của incident có kết quả cuối cùng để broadcast |
| `classified` | Sau bước Intent Classifier chạy xong **và** LLM Answer Generation + Confidence Scorer chạy xong luôn trong cùng 1 lượt (không có sự kiện bên ngoài nào chen giữa, nên gộp chung, không tách `drafted` riêng như bản nháp trước) | Ngay lập tức rẽ tiếp: trong scope + confidence cao → `pending_review`/`auto_sent`; trung bình + thiếu info → `awaiting_clarification`; ngoài scope/thấp/đa vấn đề/giận dữ → `escalated` |
| `pending_review` | Ngay sau khi có kết quả generate, khi category đang ở stage `draft` (xem `05_CATEGORY_GRADUATION.md`), hoặc category đang `auto` nhưng ticket này rơi vào canary sample | Draft hiển thị trên Dashboard IT Agent, chờ hành động duyệt/sửa/từ chối. Đây là điểm chính con người can thiệp |
| `awaiting_clarification` | Confidence Scorer chấm ở mức trung bình (0.4–0.6) **và** AI xác định được cụ thể trường thông tin còn thiếu **và** `clarification_rounds = 0` | Hệ thống gửi 1 câu hỏi làm rõ (public reply) qua Zendesk/Jira, tăng `clarification_rounds` lên 1. Ticket "đóng băng" chờ end-user reply — khi webhook mới tới, hệ thống tra row mới nhất cùng `(external_ticket_id, provider)`, thấy status này thì tạo row mới và chạy lại pipeline với **toàn bộ lịch sử** các row cùng `external_ticket_id` để có đủ context (thay cho `parent_thread_id` đã bỏ) |
| `approved` | IT Agent bấm duyệt (nguyên văn hoặc có sửa) trên Dashboard, ghi vào `review_actions` với action `approved_as_is`/`approved_with_edit` | Hệ thống gọi Zendesk/Jira API gửi `final_text` cho end-user. Nếu ticket này là `representative_ticket_id` của 1 incident → trigger broadcast sang các ticket `linked_to_incident` cùng nhóm |
| `auto_sent` | Confidence ≥0.85 **và** category đang ở stage `auto` **và** ticket không rơi vào canary sample — gửi thẳng, không qua người duyệt | Hệ thống gọi Zendesk/Jira API gửi câu trả lời ngay. Nếu sau đó end-user reply tiếp, webhook mới sẽ tra thấy row này có status `auto_sent` → escalate ngay theo rule reopened (ghi vào 1 row mới, không sửa row này) |
| `escalated` | Nhiều điểm trong pipeline có thể dẫn tới trạng thái này: Classifier trả về ngoài scope; `multi_issue_suspected = true`; `sentiment_flag = angry`; Confidence Scorer chấm dưới 0.4 (hoặc 0.4–0.6 nhưng đã hết lượt clarification); ticket reopened sau khi đã `auto_sent`/`closed`; hoặc IT Agent chủ động chọn `rejected`/`escalated` khi duyệt draft | Không gửi câu trả lời tự động cho end-user. Ticket được đưa vào hàng chờ xử lý thủ công hoàn toàn bởi IT Agent, ngoài phạm vi AI. Nếu ticket là representative của 1 incident → toàn bộ `linked_to_incident` cùng nhóm cũng escalate theo, không broadcast |
| `closed` | Sau khi IT Agent (hoặc chính end-user qua Zendesk) xác nhận vấn đề đã giải quyết xong — **là một hành động thủ công**, hệ thống AI không tự đóng ticket dù đã `approved`/`auto_sent` | Kết thúc vòng đời xử lý AI cho ticket này. Vẫn tiếp tục được dùng làm dữ liệu tham chiếu cho các cơ chế similarity sau này (repeat-unresolved, recurring-resolved) |

## Sơ đồ: sau khi AI generate xong, đi đâu tiếp — phụ thuộc stage category + canary

```mermaid
flowchart TD
    A["AI vừa generate xong<br/>(confidence >= 0.6)"] --> B{Category đang ở<br/>stage nào?}
    B -->|shadow hoặc draft| PR1["status = pending_review<br/>Luôn qua IT Agent duyệt"]
    B -->|auto| C{Roll ngẫu nhiên:<br/>rơi vào canary sample?<br/>(mặc định 10%)}
    C -->|Có, 10%| PR2["status = pending_review<br/>Vẫn qua duyệt dù category đã 'auto'<br/>-> nguồn dữ liệu giám sát chất lượng"]
    C -->|Không, 90%| AS["status = auto_sent<br/>Gửi thẳng, không ai duyệt"]

    PR1 --> APR[IT Agent duyệt/sửa/từ chối]
    PR2 --> APR
    APR -->|approved_as_is / approved_with_edit| SENT[Gửi cho end-user qua Zendesk/Jira]
    APR -->|rejected / escalated| ESC[status = escalated]

    style AS fill:#d4f4dd
    style PR1 fill:#fff4d4
    style PR2 fill:#fff4d4
    style ESC fill:#f8d4d4
```

**Đọc sơ đồ này cùng với cơ chế ở `05_CATEGORY_GRADUATION.md`**: chính nhóm `pending_review` màu vàng ở nhánh canary (PR2) là dữ liệu nuôi lại `approval_rate` cho category đã lên `auto` — nếu approval rate của riêng nhóm này tụt, job đánh giá định kỳ sẽ tự động đẩy category quay về `stage = draft`, khiến toàn bộ ticket sau đó lại đi theo nhánh PR1 (100% qua duyệt) cho tới khi đạt lại ngưỡng.

## Vài quy tắc xuyên suốt đáng nhớ

- **Không có status nào do AI tự gán mà bỏ qua khả năng escalate** — mọi nhánh "không chắc chắn" đều dẫn về `escalated`, không có trạng thái "AI đoán đại rồi thôi".
- **`closed` luôn là hành động của con người**, không phải hệ quả tự động của `approved`/`auto_sent` — tách biệt "đã gửi câu trả lời" và "vấn đề đã thật sự được giải quyết" là 2 việc khác nhau, đúng với bài học rút ra từ case "reopened" đã bàn trước đó.
- **`linked_to_incident` là trạng thái "chờ", không phải trạng thái cuối** — luôn kết thúc bằng cách chuyển sang `auto_sent`/`closed` (nếu broadcast thành công) hoặc `escalated` (nếu representative bị từ chối).
- Một `external_ticket_id` **có thể có nhiều row `tickets`** theo thời gian — mỗi lần cần AI đánh giá lại (VD sau `awaiting_clarification`) tạo 1 row mới, không sửa `status` của row cũ. Muốn xem lại toàn bộ lịch sử xử lý của 1 ticket Zendesk/Jira thật, query mọi row cùng `(external_ticket_id, provider)`, sắp theo `created_at`.
