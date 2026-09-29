# Category Graduation Mechanism — IT Support Ticket AI Agent

## 1. Mục đích

Doc này định nghĩa cơ chế **tự động tăng/giảm mức độ tự động hóa cho từng category ticket**, dựa trên số liệu chất lượng thực đo, thay vì con người bật/tắt `auto_enabled` bằng tay. Đây là phần trả lời cho câu hỏi "hệ thống có quá phụ thuộc con người không" — câu trả lời là: phụ thuộc con người nhiều **lúc đầu, có chủ đích**, và **tự giảm dần khi đủ bằng chứng tin cậy**, không phải giữ nguyên mãi.

Nguyên tắc cốt lõi, xuyên suốt cơ chế: **promote chậm và khó, demote nhanh và dễ** (asymmetric trust). Lý do: chi phí một lần tự động trả lời sai trong domain IT support cao hơn chi phí giữ một ticket ở chế độ draft thêm vài ngày.

## 2. State machine cho mỗi category

```
SHADOW ──(đủ mẫu + đo baseline ổn định)──> DRAFT ──(đạt ngưỡng promote, giữ ổn định X ngày)──> AUTO
   ▲                                          ▲                                                  │
   └──────────────────────────────────────────┴───────────── demote khi vi phạm ngưỡng ──────────┘
```

- Category **mới thêm vào hệ thống** (VD sau này mở rộng scope) luôn bắt đầu ở `SHADOW`.
- Việc chuyển `SHADOW → DRAFT` vẫn có thể làm thủ công (ít rủi ro, vì Draft mode luôn có người duyệt trước khi gửi).
- Việc chuyển `DRAFT → AUTO` và `AUTO → DRAFT` là **tự động, chạy định kỳ**, đây là phần chính của doc này.

## 3. Data Model bổ sung (thêm vào Data Model Doc khi code)

### Bảng `category_automation_state`

| Cột | Kiểu | Mô tả |
|---|---|---|
| category | text (PK) | |
| stage | text | shadow / draft / auto |
| auto_enabled | boolean | Cờ thực tế dùng trong routing logic (mục 6, Confidence Policy) |
| stage_entered_at | timestamp | Thời điểm vào stage hiện tại — dùng để tính "giữ ổn định X ngày" |
| last_evaluated_at | timestamp | Lần chạy job đánh giá gần nhất |
| last_demotion_reason | text (nullable) | Ghi lại lý do bị demote gần nhất, phục vụ debug/báo cáo |
| manual_override | boolean, default false | true nếu con người khóa cứng trạng thái (VD tắt hẳn auto cho category nhạy cảm dù đạt ngưỡng) — job đánh giá bỏ qua category có cờ này |

### Bảng `category_metric_snapshots`

| Cột | Kiểu | Mô tả |
|---|---|---|
| id | UUID (PK) | |
| category | text | |
| snapshot_at | timestamp | |
| sample_size | int | Số draft/ticket dùng để tính trong cửa sổ này |
| approval_rate | float (nullable) | Chỉ có nghĩa khi stage = draft (xem mục 5 về vấn đề mất tín hiệu ở stage auto) |
| hallucination_rate | float | |
| reopened_rate | float | |
| stage_at_snapshot | text | |

> Bảng này chính là nguồn để **vẽ biểu đồ "độ tin cậy của category theo thời gian"** — đây là artifact trực quan rất mạnh cho CV/phỏng vấn, cho thấy quá trình hệ thống "học cách tin tưởng chính nó" một cách có dữ liệu.

## 4. Công thức & lịch đánh giá

Chạy 1 job định kỳ (VD mỗi ngày, hoặc mỗi khi đủ N ticket mới — chọn theo tốc độ tích lũy dữ liệu thật):

```
Với mỗi category có auto_enabled = false, manual_override = false:
    window = 50 draft gần nhất (rolling), tối thiểu cần đủ 50 mới xét — 
             nếu chưa đủ, bỏ qua, chưa đủ bằng chứng để quyết định

    approval_rate = approved_as_is / total trong window
    hallucination_rate = số case được đánh dấu sai-nhưng-tự-tin / total
    reopened_rate = số ticket reopened / total đã gửi trong window

    Lưu snapshot vào category_metric_snapshots

    NẾU approval_rate >= 0.90
       VÀ hallucination_rate == 0
       VÀ reopened_rate <= 0.05
       VÀ (now - stage_entered_at) >= 7 ngày   # giữ ổn định đủ lâu, tránh may mắn ngắn hạn
    THÌ:
       promote sang AUTO, set auto_enabled = true, stage_entered_at = now
```

```
Với mỗi category có auto_enabled = true:
    # Ở stage AUTO không còn "approval_rate" theo nghĩa cũ vì không ai duyệt nữa (xem mục 5)
    window = 50 ticket auto_sent gần nhất
    reopened_rate = số reopened / total
    hallucination_rate = ước lượng qua canary sampling (mục 5)

    NẾU reopened_rate > 0.08
       HOẶC hallucination_rate > 0        # bất kỳ hallucination nào cũng demote ngay — không chờ tích lũy
       HOẶC có ≥1 báo cáo thủ công nghiêm trọng từ IT Agent (nút "flag as wrong" trên dashboard)
    THÌ:
       demote về DRAFT ngay lập tức, set auto_enabled = false,
       ghi last_demotion_reason, stage_entered_at = now
```

**Bất đối xứng có chủ đích**: điều kiện promote cần *nhiều tín hiệu cộng dồn trong ≥7 ngày*; điều kiện demote chỉ cần *một tín hiệu xấu* là đủ, không cần chờ tích lũy. Đây là phần nên nhấn mạnh khi giải thích thiết kế — thể hiện tư duy ưu tiên an toàn hơn tối ưu automation rate.

## 5. Vấn đề quan trọng: mất tín hiệu chất lượng sau khi graduate

Khi category còn ở Draft mode, `approval_rate` là tín hiệu chất lượng rất tốt vì **mọi response đều được người xem qua**. Sau khi chuyển sang Auto, không còn ai duyệt nữa → **mất nguồn tín hiệu chính**, chỉ còn lại tín hiệu gián tiếp (reopened_rate, complaint thủ công) — vốn chậm và ít nhạy hơn.

**Giải pháp: Canary Sampling.** Ngay cả khi category đã ở Auto, **vẫn giữ lại một tỷ lệ nhỏ ticket (khởi điểm 10%), chọn ngẫu nhiên**, không tự gửi mà đưa qua Draft mode như bình thường để người duyệt. Phần 90% còn lại mới auto-gửi thật.

- 10% này chính là nguồn dữ liệu để tiếp tục tính `approval_rate` xấp xỉ, dùng làm tín hiệu sớm để phát hiện suy giảm chất lượng, thay vì phải chờ đến khi có complaint/reopened thật.
- Tỷ lệ canary có thể giảm dần theo thời gian nếu category chứng minh ổn định lâu dài (VD sau 30 ngày auto ổn định, giảm còn 5%) — nhưng **không bao giờ về 0%**, để luôn có đường giám sát sống.

Cập nhật lại công thức demote ở mục 4: `hallucination_rate` ở stage AUTO thực chất được tính **từ chính 10% canary sample này**, không phải suy đoán mơ hồ.

## 6. Chiến lược triển khai trong code

**Cấu trúc job:**
- 1 scheduled job riêng (`evaluate_category_automation`), chạy độc lập với luồng xử lý ticket real-time — không chặn pipeline chính.
- Đọc dữ liệu từ `review_actions`, `ai_responses`, `tickets` (để lấy reopened) trong cửa sổ rolling.
- Ghi kết quả vào `category_metric_snapshots`, cập nhật `category_automation_state` nếu đủ điều kiện chuyển stage.
- Mọi lần promote/demote **bắt buộc ghi log rõ ràng** (lý do, số liệu tại thời điểm quyết định) — không chỉ đổi cờ âm thầm, vì đây là quyết định ảnh hưởng trực tiếp tới việc AI có tự gửi cho end-user hay không.

**Nơi routing logic đọc cờ này:**
- Bước 9 trong Architecture Doc ("Routing") không còn hard-code danh sách category whitelist nữa, mà **query `category_automation_state.auto_enabled`** tại thời điểm xử lý từng ticket.
- Thêm bước canary: nếu `auto_enabled = true`, roll random theo `canary_rate` hiện tại của category để quyết định ticket này đi Auto thật hay đi qua Draft mode (dù confidence cao).

**Idempotency & an toàn khi job lỗi:**
- Job chỉ đọc + ghi snapshot + update state, không gửi gì ra ngoài — an toàn để retry nếu lỗi giữa chừng.
- Nếu job không chạy được (lỗi, downtime) trong nhiều ngày, category **giữ nguyên trạng thái cũ**, không tự ý đổi — thiên về an toàn (fail-safe, không fail-open).

## 7. Cách sử dụng (cho người vận hành / chính bạn khi demo)

- Dashboard nên có 1 màn hình riêng "Category Automation Status": mỗi category hiển thị stage hiện tại, số liệu snapshot gần nhất, biểu đồ theo thời gian (dùng `category_metric_snapshots`).
- Khi 1 category bị demote tự động, dashboard cần cảnh báo rõ ràng (không chỉ đổi số âm thầm) — vì đây là sự kiện đáng chú ý: "category X vừa bị hạ khỏi Auto mode do reopened_rate tăng lên 12%".
- Nút `manual_override` cho phép bạn tự khóa 1 category ở Draft mode vĩnh viễn nếu muốn (VD: dù password-reset đạt mọi ngưỡng, bạn vẫn có thể quyết định không bao giờ để nó tự động vì tính nhạy cảm) — đây là điểm hay để nói trong phỏng vấn: **hệ thống đề xuất, con người vẫn giữ quyền phủ quyết cuối cùng**.

## 8. Runbook khi bị demote

1. Xem `last_demotion_reason` + snapshot gần nhất để biết nguyên nhân cụ thể.
2. Kiểm tra các ticket reopened/canary-rejected gần đây trong khoảng thời gian đó — có phải do KB lỗi thời, hay do 1 thay đổi quy trình IT chưa cập nhật vào KB.
3. Sửa nguồn gốc (thường là cập nhật KB), sau đó category quay lại chu trình đánh giá bình thường (không cần thao tác gì thêm — job tự đánh giá lại theo lịch).
4. **Không** khuyến khích thao tác "ép promote lại ngay" thủ công trừ khi có lý do rất rõ ràng — vì làm vậy phá vỡ chính nguyên tắc "promote chậm, cần bằng chứng" của toàn bộ cơ chế.
