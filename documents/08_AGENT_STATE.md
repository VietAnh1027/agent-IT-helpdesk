# Thiết kế AgentState

Ngày đối chiếu: 08/10/2026. Trạng thái: **đề xuất thiết kế, chưa triển khai class trong Python**.

## 1. Dự án đã có AgentState chưa?

Chưa có định nghĩa `AgentState` trong các file Markdown hoặc code nguồn được kiểm tra.

- `02_ARCHITECTURE.md` và `workflow.md` mô tả thứ tự các bước và nhánh xử lý, chưa quy định đối tượng dữ liệu chung giữa các bước.
- `03_DATA_MODEL.md` mô tả dữ liệu lưu trong DB; `06_TICKET_STATUS_LIFECYCLE.md` mô tả vòng đời ticket. Đây không phải state của một lượt chạy agent.
- `05_CATEGORY_GRADUATION.md` có state của mức tự động hóa theo category; `07_TOOLS.md` có state độ tin cậy của tool. Chúng có vòng đời riêng.
- `src/pipeline.py` hiện truyền dữ liệu qua biến cục bộ. `eval/harness/eval_generation.py` có dictionary `result` gần với một state, nhưng chứa thêm đáp án chuẩn và kết quả chấm offline.
- `src/confidence.py` và `src/router.py` hiện chưa có implementation. Không coi các trường confidence/routing dưới đây là tính năng đã chạy được.

## 2. AgentState dùng để làm gì?

`AgentState` gom dữ liệu của **một lượt xử lý một ticket**: đầu vào, kết quả phân loại, evidence, câu trả lời, đánh giá groundedness, quyết định routing và lỗi.

Mỗi bước đọc những trường cần thiết và ghi kết quả vào state đó. Không cần truyền một danh sách tham số dài giữa các bước, cũng không phải dùng biến global để giữ ticket đang xử lý.

```text
Ticket → Classify → Retrieve → Generate → Build answer → Groundedness → Confidence → Route
            └── chặn sớm ──────────────────────────────────────────────────────→ Route
```

State không tự gọi model, không thay thế DB và không tự tạo lịch sử hội thoại. Mỗi request tạo state mới. Client LLM có thể được tái sử dụng độc lập với state.

## 3. Kiểu dữ liệu đề xuất

Dùng `dataclass` của Python cho pipeline tuần tự hiện tại. Chưa cần thêm LangGraph hoặc một framework quản lý state. Đoạn dưới là thiết kế minh họa; vị trí dự kiến khi triển khai là `src/agent/state.py`.

```python
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class AgentState:
    ticket_id: str
    subject: str
    body: str

    classification: dict | None = None
    missing_fields: list[str] = field(default_factory=list)
    clarification_rounds: int = 0
    clarification_question: str | None = None

    retrieved_parents: list[dict] = field(default_factory=list)
    generation: dict | None = None
    generation_complete: bool = False
    groundedness: dict | None = None
    groundedness_score: float | None = None
    confidence_score: float | None = None

    routing_decision: Literal["draft", "auto", "clarify", "escalate"] | None = None
    routing_reason: str | None = None
    errors: list[dict] = field(default_factory=list)
```

Quy ước:

- `None`: chưa có kết quả hoặc không tính được; không thay bằng điểm `0` vì điểm 0 là một kết quả đánh giá.
- Danh sách rỗng: chưa có phần tử. Với retrieval, chỉ được hiểu là "không tìm thấy" sau khi hàm retrieval chạy thành công; lỗi retrieval phải có trong `errors`.
- `default_factory=list` tạo danh sách riêng cho mỗi instance, tránh ticket sau dùng nhầm dữ liệu ticket trước.
- `dataclass` và type hint không tự validate dữ liệu. Webhook, kết quả classify/judge và dữ liệu DB vẫn cần được kiểm tra ở điểm tiếp nhận bằng cơ chế validation hiện có.
- Các `dict` bên dưới có cấu trúc xác định. Chỉ tách thêm type riêng khi triển khai thực sự cần; không dựng nhiều class phụ ngay trong bước thiết kế.

## 4. Ý nghĩa và nơi cập nhật từng biến

### Đầu vào và phân loại

| Biến | Ý nghĩa | Nơi ghi |
|---|---|---|
| `ticket_id` | ID của lần xử lý ticket trong hệ thống; khi eval dùng ID testcase. Không đồng nhất với ID ticket bên Zendesk/Jira. | Intake hoặc eval adapter |
| `subject` | Tiêu đề ticket. Với dataset chỉ có `question`, có thể dùng tiêu đề rỗng và đưa câu hỏi vào `body`. | Intake |
| `body` | Nội dung ticket cần giải quyết trong lượt này, không chứa đáp án chuẩn. | Intake |
| `classification` | Kết quả đã validate gồm `category`, `multi_issue_suspected`, `prompt_injection_suspected`, `sentiment_flag`, `reason`, khớp prompt classify hiện tại. `None` nghĩa là chưa phân loại. | Classifier |
| `missing_fields` | Thông tin cụ thể còn thiếu, ví dụ loại tài khoản hoặc mã lỗi. Chỉ thêm khi thực sự cần để xử lý; không yêu cầu đủ mọi thông tin cho mọi ticket. Không chứa mật khẩu/token. | Bước kiểm tra thông tin |
| `clarification_rounds` | Số lượt đã hỏi làm rõ của cùng ticket bên ngoài; nạp từ DB và kế thừa khi có webhook trả lời, không reset về 0 lúc tiếp tục ticket. Giới hạn thiết kế là 1. | Intake; cập nhật bền vững khi gửi câu hỏi thành công |
| `clarification_question` | Một câu hỏi được tạo từ `missing_fields` khi router chọn `clarify`; không ghi đè câu trả lời đã sinh. | Bước clarify |

Category runtime nên dùng tên trong prompt hiện tại: `VPN`, `PASSWORD`, `SOFTWARE`, `PRINTER`, `MAIL`, `OUT_OF_SCOPE`. Tài liệu DB cũ dùng tên như `email_sync`, `software_access`; cần ánh xạ rõ ở lớp tích hợp nếu DB còn dùng bộ tên cũ, không trộn hai bộ tên trong state.

### Retrieval, generation và đánh giá

| Biến | Ý nghĩa | Nơi ghi |
|---|---|---|
| `retrieved_parents` | Danh sách parent do `retrieve_top_k_parents()` trả về, giữ nguyên thứ tự xếp hạng và các trường `parent_chunk_id`, `parent_index`, `text`, `article_title`, `similarity`. Giữ metadata để phân tích score và truy vết. | Retriever |
| `generation` | `None` khi chưa nhận được output; khi có output là `{"answer": str, "citations": list[str]}`. Text do `generate()` tạo, citations do `build_answer()` tách ra. | Generation và hậu xử lý |
| `generation_complete` | `True` khi đã nhận đủ text; `False` nếu chưa sinh hoặc output bị cắt. Có text không đồng nghĩa generation đã hoàn tất hay nội dung đúng. | Wrapper generation |
| `groundedness` | Kết quả judge đã validate: `verdict`, `reason`, `problematic_claims`, `citation_support`, theo schema eval hiện tại. Chỉ dựa trên ticket, answer và evidence thực sự đưa vào generation. | Groundedness judge |
| `groundedness_score` | Điểm số 0–1 dùng cho confidence khi đã có cách chấm/chuyển đổi được kiểm chứng. Hiện judge trả verdict, chưa trả điểm này; mặc định `None`. | Bộ tính score, chưa triển khai |
| `confidence_score` | Điểm tổng hợp 0–1 từ retrieval và groundedness, dùng cho routing sau các rule bắt buộc. Không phải xác suất câu trả lời đúng đã được hiệu chuẩn. | Confidence scorer, chưa triển khai |

Không lưu thêm một bản evidence lớn trong `kb_contexts` của state. Khi render prompt, tạo dữ liệu tương thích với code hiện tại:

```python
kb_contexts = [
    {"source": parent["article_title"], "content": parent["text"]}
    for parent in state.retrieved_parents
]
```

Task template hiện gọi danh sách này là `supplied_knowledge`. Groundedness phải dùng đúng tập evidence đó, không retrieve lại một tập khác khi chấm.

Lưu ý tên trường: `source` trong prompt là nhãn trích dẫn lấy từ `article_title`; nó không phải cột `kb_articles.source` (nguồn gốc biên soạn như synthetic/tự viết) trong tài liệu DB.

`citations` là các nhãn mô hình đã viết, có thể sai. Khi có lỗi citation, giữ nguyên answer và nhãn để debug, ghi lỗi vào `errors`; không coi danh sách này là các nguồn đã xác minh.

### Confidence: tránh lưu hai bản của cùng dữ liệu

Hai đầu vào retrieval được tính từ `retrieved_parents`, chưa cần thành hai field riêng:

- `retrieval_similarity_top1`: similarity của parent đầu tiên sau khi dedupe theo parent.
- `retrieval_score_gap`: top1 trừ top2 của hai parent khác nhau. Đây là quy ước đề xuất cần giữ nhất quán khi hiệu chỉnh ngưỡng; không trộn với gap giữa hai child cùng parent.
- Không có kết quả: chặn generation theo policy. Chỉ có một parent: gap chưa xác định (`None`), không tự coi là gap lớn hay độ chắc chắn cao.
- Cosine similarity và gap không mặc nhiên là xác suất. Scorer phải định nghĩa chuẩn hóa, xử lý thiếu dữ liệu và hiệu chỉnh trọng số trước khi kết hợp với groundedness.

`04_CONFIDENCE_EVAL_POLICY.md` và một số mô tả cũ vẫn dùng `llm_self_reported_confidence`. Thiết kế này theo hướng đã thống nhất sau đó: thay thành phần đó bằng groundedness; không đưa `self_reported_confidence` hay `needs_human_review` trở lại output của generator.

Chưa chốt ánh xạ verdict sang điểm số. Không tự gán `supported = 1` rồi coi đó là độ tin cậy tuyệt đối. Khi thiếu score hoặc judge lỗi, không đủ điều kiện auto-reply.

### Routing và lỗi

| Biến | Ý nghĩa | Nơi ghi |
|---|---|---|
| `routing_decision` | Hành động backend chọn: tạo draft, tự gửi, hỏi rõ hoặc chuyển người. `None` nghĩa là chưa quyết định. | Router |
| `routing_reason` | Lý do cụ thể, ví dụ `prompt_injection`, `low_confidence`, `invalid_citation`, `canary_sample`. Hữu ích cho review/debug. | Router |
| `errors` | Danh sách `{"stage": str, "message": str}` theo định dạng eval đang dùng. Ghi thêm lỗi mới, không xóa câu trả lời đã có. Không đưa secret vào message. | Bước gặp lỗi |

State không có `needs_human_review`: routing do backend quyết định từ rule, score và chính sách category; không hỏi generator tự quyết.

Quyết định `auto` không có nghĩa đã gửi thành công. Chỉ cập nhật `tickets.status = auto_sent` khi tích hợp gửi phản hồi thành công. Tương tự, chỉ ghi nhận một lượt clarification đã gửi khi thao tác gửi được xác nhận, có idempotency.

## 5. Cập nhật state theo từng bước

1. **Intake:** tạo state mới với ID, subject, body. Khi có clarification reply, lấy số vòng và lịch sử đúng ticket từ DB.
2. **Classify/guardrail:** ghi `classification`; chặn sớm nếu ngoài scope, nghi injection, đa vấn đề, angry hoặc security trigger. Không cần generate để biết phải escalate.
3. **Retrieve:** gọi hàm retrieval dùng chung, ghi `retrieved_parents`. Không có evidence đủ ngưỡng thì route theo policy, không bịa câu trả lời.
4. **Generate:** render prompt từ ticket và parent contexts; ghi answer ngay khi có output. Output bị cắt vẫn được lưu nhưng `generation_complete = False`.
5. **Build answer:** bổ sung citations. Sai nhãn không xóa text; ghi lỗi. Text đầy đủ nhưng citation sai vẫn có thể được judge trong eval.
6. **Groundedness/confidence:** chỉ chấm text hoàn chỉnh. Groundedness không thấy golden answer. Giữ `None` nếu chưa đánh giá được; không tự đổi lỗi thành score cao.
7. **Route:** rule bắt buộc có ưu tiên trên confidence. Citation chưa xác minh, text bị cắt hoặc judge lỗi không đủ điều kiện auto; có thể giữ draft cho IT hoặc escalate theo chính sách.
8. **Persist/send:** chuyển các trường cần thiết sang DB và kết quả trả về. Không serialize toàn bộ state rồi gửi tất cả cho LLM hay người dùng.

Ngưỡng trong tài liệu policy được hiểu không chồng lấn: thấp `< 0.4`; clarification `0.4 <= score < 0.6` và đủ điều kiện hỏi; draft `0.6 <= score < 0.85`; xét auto `>= 0.85` rồi kiểm tra category/canary/guardrail. Đây vẫn là ngưỡng khởi điểm cần hiệu chỉnh.

## 6. Phần mở rộng khi tích hợp đầy đủ sản phẩm

Không thêm các trường sau vào class lõi ngay khi mới chạy RAG/eval. Chỉ thêm lúc module tương ứng được triển khai:

| Nhóm | Trường dự kiến và tác dụng |
|---|---|
| Webhook/DB | `provider`, `external_ticket_id`: định danh thread bên ngoài; `external_event_id`: khóa idempotency cùng provider; `requester_email`: danh tính đã xác thực lấy từ DB/provider, không trích từ nội dung LLM. |
| Clarification history | `history`: các thông điệp có thứ tự của đúng `(provider, external_ticket_id)`, được nạp chủ động khi tiếp tục ticket. Không mang lịch sử ticket khác vào state. |
| Similarity/incident | `incident_id`: liên kết incident; `recurring_issue`: nhãn thống kê; kết quả dispatcher phân biệt repeat/reopened/incident-confirmed. Candidate chưa confirmed không được tự dừng generation. |
| Category automation | `automation_stage`, `auto_enabled`, `canary_selected`: snapshot policy backend dùng khi routing. Ghi quyết định canary một lần để retry không chọn lại tùy tiện. Kiểm tra quyền auto trước khi gửi. |
| Tools | `tool_results`: kết quả tool có tên, trạng thái và dữ liệu; `tool_proposals`: đề xuất chờ duyệt Tier 1. Kết quả này tách khỏi nguồn KB. Tool executor vẫn kiểm tra danh tính, whitelist, trust và ghi log riêng. |
| Trace retrieval | `kb_article_id`, ID child match: mở rộng kết quả retriever nếu cần lưu `cited_kb_chunk_ids` theo thiết kế DB. Hiện retriever chỉ trả parent ID và tên bài; không thể suy ra child ID từ nhãn citation. |

Idempotency, status dispatch và incident linking diễn ra trước hoặc ngoài lõi classify–RAG; không ép các nhánh `skip`/`linked_to_incident` vào bốn giá trị routing của câu trả lời.

DB vẫn là nguồn dữ liệu bền vững cho ticket status, graduation, tool trust và review. State không giữ bản sao toàn bộ các bảng hay tự thay đổi quyền thực thi tool.

## 7. Phân biệt state runtime với kết quả eval

`ground_truth_answer`, `ground_truth_contexts`, `human_handoff` của golden dataset và `correctness` thuộc **eval report**, không thuộc `AgentState` runtime.

- `human_handoff` là nhãn chuẩn của testcase, không được dùng như quyết định classify trong production.
- Correctness judge cần golden answer; generator và groundedness judge không được nhận nó.
- File eval vẫn giữ cấu trúc hiện tại: metadata ticket, `kb_contexts`, `generation`, `correctness`, `groundedness`, `errors`. Không bắt buộc thêm confidence/routing vào JSON eval vì đã chủ động loại các trường này trước đó.
- Harness có thể chuyển dữ liệu state sang report khi thực sự tích hợp pipeline chung; chưa cần sửa harness chỉ để dùng class mới.

Ví dụ trạng thái logic khi có lỗi nhãn nguồn:

```json
{
  "generation": {
    "answer": "Hướng dẫn ... [Dong_bo_email_outlock]",
    "citations": ["Dong_bo_email_outlock"]
  },
  "generation_complete": true,
  "confidence_score": null,
  "errors": [
    {"stage": "generation", "message": "Nhãn trích dẫn không có trong evidence"}
  ]
}
```

Đây là một phần state minh họa, không phải output do LLM tạo. Answer vẫn tồn tại để chấm hoặc duyệt; `generation_complete` chỉ xác nhận text không bị cắt, không xác nhận citations hay nội dung đúng.

## 8. Giới hạn và bước triển khai sau này

- Tài liệu này chỉ thêm thiết kế, chưa sửa runtime, DB hoặc công thức confidence.
- Không lưu client ChatOllama/OpenAI, DB connection, API key, model embedding hoặc system prompt trong state. Các đối tượng đó thuộc dependency/config của ứng dụng.
- Không cần `messages` tự cộng dồn hay checkpoint framework cho pipeline tuần tự hiện tại. Nếu sau này triển khai graph, chuyển hợp đồng dữ liệu này sang kiểu state mà framework hỗ trợ; không cần đổi nghiệp vụ chỉ vì đổi công cụ điều phối.
- Khi triển khai class, kiểm tra tối thiểu: hai state không dùng chung list; lỗi citation giữ answer; output bị cắt không đi auto; golden data không lọt vào prompt generation; clarification không vượt một vòng.

Tài liệu tham chiếu: `01_PRD.md`, `02_ARCHITECTURE.md`, `03_DATA_MODEL.md`, `04_CONFIDENCE_EVAL_POLICY.md`, `05_CATEGORY_GRADUATION.md`, `06_TICKET_STATUS_LIFECYCLE.md`, `07_TOOLS.md`, `../workflow.md`, `../to-do-flow.md`, `../eval/EVAL_RETRIEVAL_ANALYSIS.md`.
