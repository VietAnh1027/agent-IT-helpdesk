# Data Model Document — IT Support Ticket AI Agent

## 1. Bảng `tickets`

| Cột | Kiểu | Mô tả |
|---|---|---|
| id | UUID (PK) | |
| subject | text | Tiêu đề ticket |
| body | text | Nội dung mô tả vấn đề |
| requester_email | text | Email giả lập của người gửi |
| created_at | timestamp | |
| category | text (nullable) | Kết quả phân loại: vpn / password / software_access / printer / email_sync / out_of_scope |
| status | text | new / classified / pending_review / approved / auto_sent / awaiting_clarification / linked_to_incident / escalated / closed — **đã bỏ `drafted`**: không có sự kiện bên ngoài nào chen giữa lúc AI generate xong và lúc chuyển tiếp, nên gộp thẳng `classified` → `pending_review`/`auto_sent`, xem `06_TICKET_STATUS_LIFECYCLE.md` |
| confidence_score | float (nullable) | Điểm tin cậy cuối cùng (0–1) |
| clarification_rounds | int, default 0 | Số lần hệ thống đã hỏi làm rõ cho ticket này. Giới hạn cứng = 1 ở v1 — vượt quá thì bắt buộc escalate, không hỏi thêm |
| external_ticket_id | text | ID ticket thật bên Zendesk/Jira. **Nhiều row trong `tickets` có thể cùng `external_ticket_id`** — mỗi lần webhook mang một sự kiện cần AI xử lý (ticket mới, hoặc reply cần đánh giá lại) sẽ tạo 1 row mới, thay vì update row cũ. Đây là cách nhóm toàn bộ lịch sử xử lý của 1 ticket thật, thay thế hoàn toàn cho `parent_thread_id` (đã bỏ) |
| provider | text | zendesk / jira / ... — đi kèm `external_ticket_id` vì ID có thể trùng giữa các provider khác nhau |
| embedding | vector(N) | Embedding của `subject + body`, dùng để so sánh ticket-với-ticket (không phải ticket-với-KB). Nền tảng cho mọi loại similarity check bên dưới |
| incident_id | UUID (nullable, FK → incidents.id) | Gắn vào 1 incident nếu ticket này được xác định là 1 trong nhiều ticket trùng lặp của cùng sự cố diện rộng |
| recurring_issue | boolean, default false | true nếu ticket này khớp similarity cao với 1 ticket **đã resolved từ lâu** (không phải repeat gần đây) — tín hiệu lỗi hay tái diễn, không ảnh hưởng cách trả lời, chỉ phục vụ metric |
| sentiment_flag | text (nullable) | neutral / frustrated / angry — chấm bởi bước phân loại sentiment nhẹ, dùng để escalate ưu tiên |
| multi_issue_suspected | boolean, default false | true nếu Classifier phát hiện tín hiệu thuộc ≥2 category cùng lúc vượt ngưỡng |

## 2. Bảng `kb_articles`

| Cột | Kiểu | Mô tả |
|---|---|---|
| id | UUID (PK) | |
| title | text | |
| category | text | Khớp với category của ticket |
| content | text | Nội dung đầy đủ bài viết |
| content_hash | text | SHA-256 của `content` — tính **sau khi đã tách 3 dòng metadata đầu PDF ra khỏi nội dung**. Ingest pipeline so sánh hash mới vs cũ trước khi chunk+embed lại — **hash không đổi thì bỏ qua, giữ nguyên chunk/embedding cũ**, tránh tốn compute cho nội dung không đổi (quan trọng vì chạy local GPU) |
| date_added | timestamp | Thời điểm record được thêm vào hệ thống — set 1 lần lúc insert, **không đổi** dù nội dung được cập nhật sau này |
| date_created | date | Ngày tài liệu **thật sự được soạn**, khác `date_added`. Đọc từ dòng metadata `date_created: DD/MM/YYYY` ở đầu file PDF nguồn — không tự sinh, không suy ra từ hệ thống |
| last_updated | timestamp | Cập nhật mỗi lần `content` đổi (tức mỗi lần `content_hash` đổi). Dùng mô phỏng "KB có thể lỗi thời" cho eval |
| source | text | vd "tự viết" / "synthetic" — ghi rõ để minh bạch |
| is_active | boolean, default true | Đọc từ dòng metadata `is_activate: true/false` ở đầu file PDF. Bài viết ngừng dùng (lỗi thời, đã thay thế bởi bài khác) thì set false thay vì xóa — giữ lịch sử, loại khỏi retrieval qua filter, không qua xóa cứng |

> **Metadata giờ đọc trực tiếp từ PDF, không hardcode trong ingest script**: 3 dòng đầu mỗi file PDF nguồn phải theo đúng thứ tự `category` / `is_activate` / `date_created` (định dạng `key: value`, date theo `DD/MM/YYYY`). Ingest script parse 3 dòng này thành metadata, **loại bỏ khỏi nội dung** trước khi hash/chunk/embed — 3 dòng metadata không bao giờ xuất hiện trong `content` hay bất kỳ `kb_chunks` nào. Vì `category` không còn cố định theo tên file, ingest script quét toàn bộ `*.pdf` trong thư mục nguồn thay vì dùng danh sách file cố định.
>
> `category` và `is_active` được **denormalize xuống bảng chunk** (mục 3), vì đây là 2 cột thực sự dùng để filter cứng lúc query similarity — tránh phải JOIN ngược `kb_articles` mỗi lần search. Khi metadata đổi nhưng nội dung thân bài không đổi (hash khớp), ingest script vẫn cập nhật `category`/`is_active`/`date_created` ở cả `kb_articles` lẫn `kb_chunks`, chỉ bỏ qua bước re-chunk/re-embed tốn kém.

## 3. Bảng `kb_chunks` (gộp parent-child, self-referencing)

Thay vì 2 bảng riêng, gộp thành 1 bảng — parent và child là **cùng 1 loại thực thể** (đoạn nội dung của 1 bài viết), chỉ khác vai trò, nên dùng pattern self-referencing (adjacency list) thay vì tách bảng.

| Cột | Kiểu | Mô tả |
|---|---|---|
| id | UUID (PK) | |
| kb_article_id | UUID (FK → kb_articles, ON DELETE CASCADE) | |
| parent_chunk_id | UUID (nullable, FK → kb_chunks.id, ON DELETE CASCADE) | **NULL** nếu row này là parent. **Có giá trị** (trỏ tới 1 row khác trong cùng bảng) nếu là child |
| chunk_index | int | Vị trí trong bài viết gốc (nếu là parent) hoặc trong parent chunk (nếu là child) |
| content | text | Nội dung đoạn — với parent là đoạn lớn (context cho LLM), với child là đoạn nhỏ (dùng để search) |
| embedding | vector(N), nullable | **NULL bắt buộc** với parent (tiết kiệm chi phí, không cần embed đoạn lớn). **NOT NULL bắt buộc** với child |
| category | text | Denormalize từ `kb_articles.category` — cho phép filter cứng trước similarity search mà không cần JOIN |
| is_active | boolean | Denormalize từ `kb_articles.is_active` — cùng lý do, loại bỏ chunk của bài đã inactive ngay ở tầng filter |

**Ràng buộc toàn vẹn (CHECK constraint), bắt buộc để tránh dữ liệu sai lệch vai trò:**
```sql
CHECK (
    (parent_chunk_id IS NULL AND embedding IS NULL)      -- là parent
    OR
    (parent_chunk_id IS NOT NULL AND embedding IS NOT NULL)  -- là child
)
```

**Index**: ANN index (`ivfflat`/`hnsw`) chỉ nên là **partial index** (`WHERE parent_chunk_id IS NOT NULL`) — chỉ index các row child có embedding, tránh lãng phí và tránh vô tình match phải parent row (vốn không có embedding nên thực ra không match được, nhưng partial index vẫn nên làm để tối ưu kích thước index).

> **Cách dùng trong retrieval (hybrid search)**: query similarity trên `kb_chunks.embedding` **kèm điều kiện lọc cứng** `WHERE parent_chunk_id IS NOT NULL AND category = :ticket_category AND is_active = true` trước khi rank theo cosine similarity — đây là điểm khác so với thiết kế trước (chỉ similarity thuần), giúp tránh nhầm giữa 2 bài viết nội dung gần giống nhau nhưng khác category/đã lỗi thời. Sau khi tìm được child chunk khớp nhất, lấy `parent_chunk_id` để `SELECT content FROM kb_chunks WHERE id = :parent_chunk_id` lấy đoạn lớn làm context đưa cho LLM — không đưa thẳng child (thường quá ngắn) vào prompt. Trích dẫn nguồn cho end-user/IT Agent xem thì hiển thị theo `kb_article_id` gốc, không cần lộ chi tiết parent/child.

## 5. Bảng `ai_responses`

| Cột | Kiểu | Mô tả |
|---|---|---|
| id | UUID (PK) | |
| ticket_id | UUID (FK) | |
| draft_text | text | Câu trả lời AI sinh ra |
| cited_kb_chunk_ids | UUID[] | Trỏ tới `kb_chunks.id` của các **child row** (chunk thực sự match qua similarity search) — bắt buộc có, dùng để trace lại parent content + article gốc khi cần hiển thị/audit |
| retrieval_top_k_scores | jsonb | Lưu lại điểm similarity của top-k để phân tích sau |
| confidence_score | float | |
| model_used | text | Tên/version model tại thời điểm sinh |
| latency_ms | int | Đo hiệu năng |

## 6. Bảng `review_actions`

| Cột | Kiểu | Mô tả |
|---|---|---|
| id | UUID (PK) | |
| ai_response_id | UUID (FK) | |
| action | text | approved_as_is / approved_with_edit / rejected / escalated / **broadcast_from_incident** |
| final_text | text (nullable) | Nội dung cuối cùng nếu có sửa |
| reviewer_note | text (nullable) | |
| reviewed_at | timestamp | |

> Bảng này chính là nguồn dữ liệu để tính **draft approval rate** — metric quan trọng nhất trong PRD.

## 7. Bảng `eval_set`

| Cột | Kiểu | Mô tả |
|---|---|---|
| id | UUID (PK) | |
| ticket_text | text | Ticket mẫu (không nhất thiết trong bảng tickets chính) |
| expected_category | text | |
| expected_kb_article_id | UUID (FK, nullable) | |
| golden_answer | text | Câu trả lời chuẩn tự viết |
| difficulty | text | easy / medium / hard / adversarial (test prompt injection, ticket mơ hồ) |

## 8. Bảng `incidents`

| Cột | Kiểu | Mô tả |
|---|---|---|
| id | UUID (PK) | |
| representative_ticket_id | UUID (FK → tickets.id) | Ticket đầu tiên trong cụm, dùng làm nội dung để generate câu trả lời chung 1 lần |
| category | text | |
| ticket_count | int | Đếm số ticket đã gộp vào, cập nhật realtime |
| status | text | open / acknowledged / resolved |
| created_at | timestamp | |
| confirmed_ticket_count | int | Số ticket đã đạt ngưỡng tại thời điểm confirm (thường = N, khởi điểm 5) — cố định, không đổi, để phân biệt với `ticket_count` (tổng số, tăng tiếp sau đó) |

> **Cơ chế 2 giai đoạn — candidate rồi mới confirmed, tránh gắn nhầm incident chỉ vì 2 ticket tình cờ giống nhau:**
>
> - **Candidate (âm thầm)**: mọi ticket luôn đi qua pipeline xử lý bình thường (classify → retrieve → generate → draft/auto/escalate) — không có ngoại lệ. Song song, hệ thống chỉ **đếm** số ticket similarity cao, khác requester, trong cửa sổ 30 phút, KHÔNG tạo bản ghi `incidents`, KHÔNG đổi `status` của ticket nào.
> - **Confirmed (khi đạt ngưỡng N=5)**: khi ticket thứ N trong cụm xuất hiện, tạo bản ghi `incidents` thật, `representative_ticket_id` = ticket **đầu tiên** trong cụm. Gắn `incident_id` cho cả 5 ticket đã có (chỉ để thống kê/audit — **không huỷ hay ghi đè** draft/response chúng đã có).
> - **Từ ticket thứ N+1 trở đi**: mới áp dụng hành vi đặc biệt — skip pipeline generate, `status = linked_to_incident`, gửi placeholder "đã ghi nhận, đang xử lý", chờ broadcast.
>
> **Broadcast sau khi representative có kết quả cuối cùng:**
> - Representative đi qua routing bình thường theo category (Draft mode chờ duyệt, hoặc Auto mode tự gửi — nếu rơi vào canary sample thì vẫn phải qua duyệt như draft).
> - Khi representative được `approved_as_is` / `approved_with_edit` / `auto_sent` → trigger gửi `final_text` đó tới toàn bộ ticket đang `linked_to_incident` trong cùng incident, ghi 1 `ai_responses` + `review_actions` (action = `broadcast_from_incident`) cho từng ticket để giữ audit trail, chuyển `status = auto_sent` hoặc `closed`.
> - Nếu representative bị `rejected` / `escalated` → **không broadcast gì cả**, toàn bộ ticket `linked_to_incident` chuyển sang escalate thủ công — tránh nhân lỗi lên N lần nếu draft gốc sai.

## 9. Bảng `webhook_events`

| Cột | Kiểu | Mô tả |
|---|---|---|
| id | UUID (PK) | |
| provider | text | zendesk / jira / ... |
| external_event_id | text | ID event do Zendesk/Jira cung cấp |
| received_at | timestamp | |
| processed | boolean | |

> Unique constraint trên `(provider, external_event_id)`. Nếu webhook đến trùng ID đã xử lý → bỏ qua ngay, không chạy lại pipeline. Xử lý idempotency, tránh tạo draft/incident trùng lặp khi Zendesk/Jira retry webhook.

## 10. Quan hệ tổng quát

```
tickets 1---1 ai_responses 1---1 review_actions
tickets N---1 incidents (nhiều ticket có thể thuộc cùng 1 incident)
tickets N---1 "external ticket" (nhiều row tickets cùng external_ticket_id + provider
                                  = nhiều lần xử lý AI cho cùng 1 ticket Zendesk/Jira thật)
kb_articles 1---N kb_chunks (self-referencing qua parent_chunk_id: 1 parent row 1---N child row)
ai_responses N---N kb_chunks (qua cited_kb_chunk_ids, chỉ trỏ tới child row)
```

## 11. Ghi chú thiết kế

- `kb_chunks` gộp parent-child vào 1 bảng self-referencing (adjacency list) thay vì 2 bảng riêng — đơn giản hóa schema, dễ query "lấy parent + toàn bộ child của nó" bằng 1 self-join, không cần 2 lượt JOIN qua bảng khác. Đánh đổi: cần CHECK constraint để giữ đúng bất biến (parent luôn `embedding IS NULL`, child luôn `parent_chunk_id NOT NULL`), và partial index để ANN search không quét nhầm parent row.
- Muốn thử nghiệm nhiều chiến lược chunking khác nhau (kích thước parent, kích thước child, cách chia) mà không phá dữ liệu gốc: có thể thêm cột `chunking_strategy_version` vào `kb_chunks`, giữ song song nhiều bộ chunk cho cùng 1 `kb_article_id` để so sánh.
- **Xóa `kb_articles` tự động dọn sạch toàn bộ chunk liên quan** — `kb_chunks.kb_article_id` có `ON DELETE CASCADE`, và bản thân `parent_chunk_id` (self-FK) cũng `ON DELETE CASCADE` — xóa 1 parent row tự động xóa hết child của nó, không cần logic dọn dẹp thủ công ở tầng ứng dụng, tránh mồ côi dữ liệu (orphan rows).
- `retrieval_top_k_scores` lưu dạng jsonb thay vì chỉ lưu 1 số, để sau này phân tích được "khi nào retrieval yếu" chứ không chỉ biết yếu hay không.
- Cân nhắc thêm bảng `audit_log` riêng nếu muốn thể hiện rõ khả năng viết hệ thống có compliance/traceability — không bắt buộc ở v1.
- **Một cơ chế similarity dùng chung, nhiều mục đích**: cột `tickets.embedding` + so sánh cosine similarity là hạ tầng duy nhất phục vụ cả 4 việc: (1) phát hiện repeat-unresolved cùng người, (2) gộp incident diện rộng nhiều người, (3) gắn tag recurring_issue với ticket cũ đã resolved, (4) rate-limit/gộp thread khi 1 người spam liên tục. Không cần 4 hệ thống riêng — chỉ cần 1 hàm similarity search với tham số cửa sổ thời gian/số người khác nhau theo từng mục đích. Xem chi tiết luồng ở Architecture Doc, mục "Duplicate & Incident Detection".
- **Webhook mới thuộc loại gì — quyết định bằng rule dựa trên status, không cần AI đoán ý định**: khi 1 webhook "ticket updated/comment added" tới, tra row `tickets` mới nhất (theo `created_at`) có cùng `(external_ticket_id, provider)`, nhìn `status` của nó để biết xử lý ra sao:
  - `awaiting_clarification` → đây là câu trả lời làm rõ → chạy lại pipeline với full lịch sử (query mọi row cùng `external_ticket_id`, sắp theo `created_at`)
  - `auto_sent` hoặc `closed` → đây là reopened → escalate ngay, không chạy AI
  - `pending_review` / `escalated` / `linked_to_incident` → người dùng bổ sung info trong lúc đã có người xử lý → không cần AI can thiệp, người phụ trách tự thấy comment mới trên Zendesk/dashboard
  
  Không cần field trạng thái ý định riêng — toàn bộ logic suy ra từ `status` sẵn có, tránh trùng lặp nguồn sự thật.

