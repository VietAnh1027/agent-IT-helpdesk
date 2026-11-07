# Confidence & Evaluation Policy — IT Support Ticket AI Agent

## 1. Mục đích

Doc này định nghĩa: (1) cách tính điểm tin cậy, (2) ngưỡng để quyết định draft / auto-reply / escalate, (3) cách đo chất lượng hệ thống bằng eval set, (4) các trigger escalate bắt buộc.

## 2. Công thức confidence đã triển khai

### Chế độ Full (mặc định — `CONFIDENCE_MODE="full"`)

```text
relevance_score(d) = (1/(60+dense_rank) + 1/(60+bm25_rank)) / (2/61)
# Kênh không có parent này đóng góp 0; rank bắt đầu từ 1.
S_top1 = relevance_score của parent đầu fusion
margin = S_top1 - S_top2; bằng 0 nếu chỉ có một parent
C_retrieval = 0.7*S_top1 + 0.3*min(1, 2*margin)
C_consensus = 1.00 nếu parent đầu fusion thuộc Top-3 cả hai kênh
              0.75 nếu thuộc Top-5 cả hai kênh, chưa đạt mức trên
              0.50 trong các trường hợp còn lại khi hai kênh thành công
C_groundedness = số claim supported / tổng số claim kiểm chứng được
C_RAG = 0.4*C_retrieval + 0.2*C_consensus + 0.4*C_groundedness
```

### Chế độ Lite (`CONFIDENCE_MODE="lite"`)

```text
C_RAG_lite = 0.7*C_retrieval + 0.3*C_consensus
# Không gọi LLM judge cho groundedness; c_groundedness = null; status = ok
```

Chế độ Lite dùng khi cần đánh giá nhanh mà không tốn chi phí compute/thời gian chạy judge; output giữ cùng schema (`c_groundedness` là null). Cấu hình qua `CONFIDENCE_MODE` trong `.env` hoặc tham số `--confidence-mode` ở CLI harness.

Dense và BM25 chạy trên cùng corpus child hợp lệ theo category/active, độc lập chọn candidate. Mỗi kênh gom parent theo điểm child tốt nhất rồi lấy 20 parent (`candidate_pool = 20`); RRF hợp nhất theo parent ID (k = 60). Giữ top-2 trước khi cắt context ngay cả khi generation chỉ dùng top_k=1 (để tính margin). `similarity` vẫn là cosine, `relevance_score` là RRF chuẩn hóa theo trần lý thuyết `2/61`. Không dùng hệ số nguồn hay self-reported confidence.

Groundedness gọi LLM judge qua OpenRouter khi `OPENROUTER_API_KEY` có giá trị trong `.env`, nếu không sẽ fallback sang Ollama local (`qwen3.5:4b`). Cấu hình chung trong `src/core/config.py`:
- `OPENROUTER_API_KEY`: SecretStr cấu hình trong `.env`.
- `OPENROUTER_MODEL`: Model Cloud.
- `OPENROUTER_BASE_URL`: Endpoint OpenRouter (`https://openrouter.ai/api/v1`).
- `GROUNDEDNESS_MODEL`: Model local Ollama (mặc định `qwen3.5:4b`).
- `JUDGE_NUM_CTX`: 32768 tokens.
- `JUDGE_MAX_TOKENS`: 5120 tokens (vượt trội so với giới hạn output của judge).
- `JUDGE_SEED`: 42 (temperature cố định 0 để tăng tính lặp lại).
- `CONFIDENCE_MODE`: `full` hoặc `lite` (mặc định `full`).

Không fallback chéo giữa Cloud và Local khi request giữa chừng bị lỗi (nhằm bảo đảm tính nhất quán của phép đo). Judge không thấy golden answer, chỉ thấy question, answer và toàn bộ evidence thực sự dùng để generation.

Judge tách tối đa 12 claim và cung cấp verdict cùng tối đa 3 `context_id` cho mỗi claim. Trước khi gọi judge, Python gán các context thực tế thành `E1`, `E2`, ...; model không nhận `parent_chunk_id` và không phải chép lại quote. Python kiểm tra schema, claim trùng văn bản, context ID tồn tại và không trùng rồi đếm; unsupported/contradicted/uncertain đóng góp 0. Sau validation, backend ánh xạ ID về source và parent ID thật. Context tồn tại không tự chứng minh entailment; quan hệ ngữ nghĩa vẫn do judge quyết định và cần được kiểm tra bằng nhãn người.

| Trường hợp | Kết quả |
|---|---|
| Judge hợp lệ có claim | score 0–1, status ok |
| Judge lỗi, context ID sai/trùng, không claim, answer rỗng/bị cắt, context vượt ngân sách | score 0.0, status unavailable, giữ error |
| Retrieval hợp lệ nhưng grounding unavailable (Full mode) | C_RAG dùng fallback 0.0, status degraded, không đủ điều kiện auto |
| Chế độ Lite (bỏ qua groundedness judge) | C_RAG tính từ retrieval & consensus (0.7 / 0.3), c_groundedness = null, status ok |
| Retrieval lỗi hoặc không parent | C_RAG null, status unavailable; chặn generation |
| Có claim contradicted | Giữ answer; confidence degraded dù có điểm số |

Module `src/confidence.py` đã triển khai đầy đủ cả hai mode `full` và `lite`; router (`src/router.py`) chưa triển khai. Caller phải đọc status/errors, không chỉ điểm số. Điểm RRF phản ánh thứ hạng, không phải xác suất đúng. Các ngưỡng dưới đây chưa được hiệu chỉnh cho công thức mới, chưa dùng để bật auto-send.

## 3. Ngưỡng xử lý (threshold routing)

| Confidence                                                  | Điều kiện thêm                                                                                                                                             | Hành động                                                                                                                                                                                                                                                                                                                                                                 |
| ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ≥ 0.85                                                     | —                                                                                                                                                             | Chỉ xét khi confidence status = ok, không vi phạm guardrail, metadata nguồn hợp lệ, answer hoàn chỉnh và đủ điều kiện Auto-reply *nếu* category đang ở stage `auto` (xem cơ chế tự động promote/demote category ở `05_CATEGORY_GRADUATION.md`). Kể cả khi đủ điều kiện, 1 phần nhỏ ticket vẫn được lấy mẫu ngẫu nhiên (canary) đi qua Draft mode thay vì auto-gửi, để duy trì tín hiệu giám sát chất lượng — chi tiết ở cùng doc |
| 0.6 – 0.85                                                 | —                                                                                                                                                             | Tạo Draft, chờ người duyệt                                                                                                                                                                                                                                                                                                                                              |
| 0.4 – 0.6                                                  | AI xác định thiếu thông tin cụ thể (thiếu trường bắt buộc theo category, VD: hệ điều hành, mã lỗi)**và** `clarification_rounds = 0` | Sinh 1 câu hỏi làm rõ, gửi public reply, chuyển`status = awaiting_clarification`, tăng `clarification_rounds` lên 1                                                                                                                                                                                                                                              |
| < 0.4, hoặc (0.4–0.6 nhưng`clarification_rounds ≥ 1`) | —                                                                                                                                                             | Escalate, không sinh câu trả lời cho end-user, chỉ ghi log nội bộ                                                                                                                                                                                                                                                                                                     |

> Câu hỏi làm rõ chỉ được sinh khi AI xác định rõ **thiếu trường thông tin cụ thể nào** (không phải "confidence thấp chung chung"). Nếu không xác định được thiếu gì cụ thể, dù confidence ở dải 0.4–0.6 vẫn escalate thẳng — tránh AI hỏi những câu vô nghĩa kiểu "bạn có thể nói rõ hơn không?".

> Các con số trên là điểm khởi đầu để code. Sau khi chạy Shadow mode và Draft mode, dùng dữ liệu approval rate thật để hiệu chỉnh lại ngưỡng — nên vẽ biểu đồ precision/recall theo từng mức ngưỡng để chọn điểm tối ưu, đây là phần rất đáng đưa vào README.
>
> **Công cụ hiệu chỉnh:** Harness `eval/harness/eval_confidence_scorer.py` đã được xây dựng để xuất điểm confidence chi tiết cùng trường `assessment.human_review` (verdict: `acceptable`, `unacceptable`, `uncertain`) nhằm chạy offline calibration trên golden dataset trước khi đưa ngưỡng vào router production.

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

## 5. Eval Set thực tế trong dự án

Hiện tại dự án đã xây dựng và chuẩn hóa các dataset đánh giá trong thư mục `eval/dataset/`:

1. **`golden_dataset.json` (150 tickets)**:
   - Trải đều qua 5 category cốt lõi (mỗi category đúng 30 ticket): `MAIL`, `PRINTER`, `SOFTWARE`, `PASSWORD`, `VPN`.
   - **130 ticket in-domain single-hop** (`query_type: "in_domain_single_hop"`, `human_handoff: false`): có đầy đủ câu hỏi, `ground_truth_answer` chuẩn và `ground_truth_contexts` (evidence trích nguyên văn từ tài liệu KB).
   - **20 ticket out-of-domain / handoff** (`query_type: "out_of_domain"`, `human_handoff: true`): câu hỏi ngoài phạm vi nghiệp vụ hỗ trợ của KB, dùng để kiểm thử khả năng nhận biết và chuyển tiếp người xử lý.
2. **`out_of_scope.json` (20 tickets)**:
   - Chứa 20 ticket ngoài phạm vi để kiểm tra Classifier và cơ chế escalate sớm.
3. **Phân bố độ khó & loại câu hỏi**:
   - **Easy**: Câu hỏi rõ ràng, khớp trực tiếp một đoạn trong bài KB tương ứng.
   - **Medium**: Câu hỏi mô tả triệu chứng, nhiều bước thao tác hoặc cần trích xuất chính xác cấu hình.
   - **Hard / Adversarial**: Chứa mã lỗi, lỗi chính tả, câu hỏi thiếu thông tin ngữ cảnh, hoặc câu hỏi ngoài phạm vi được diễn đạt tương tự câu hỏi IT nội bộ.

## 6. Các Metrics cần đo và cách tính

| Metric                        | Công thức / cách đo                                                                                                                                                                   |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Retrieval Recall@K (K=5)      | Đo qua `eval/harness/eval_retrieval.py` bằng phương pháp **evidence substring containment** (chuẩn hóa khoảng trắng, đối chiếu substring evidence trong chunk thuộc đúng source doc). Bộ 3 chỉ số: **Recall@K** (% evidence được match), **Precision@K** (% chunk top-K chứa evidence), **MRR** (1 / rank chunk đầu tiên chứa evidence). |
| Answer Accuracy               | So sánh draft answer với `ground_truth_answer` qua LLM Judge (`eval_correctness` template, mặc định model `qwen3.5:2b` qua Ollama hoặc Cloud OpenRouter qua cờ `--use-cloud-correctness`). Chỉ tính trên tập ticket answerable (`human_handoff = false`). |
| Hallucination / Ungrounded Rate | Trong harness: `(unsupported + contradicted) / (supported + unsupported + contradicted)`. Đo tỷ lệ claim không có căn cứ từ evidence. Các trường hợp judge bị `uncertain` hoặc `unavailable` được báo cáo riêng biệt. |
| Mean Groundedness             | Điểm groundedness trung bình của tất cả answer có trạng thái chấm hợp lệ (`status = ok`, score từ 0.0 đến 1.0).                                                                        |
| Draft Approval Rate           | Từ bảng `review_actions`: % `approved_as_is` trên tổng draft do AI sinh.                                                                                                                |
| Escalation Precision/Recall   | Trong các case đáng lẽ phải escalate (theo eval set gán sẵn), hệ thống có escalate đúng không.                                                                                           |
| Latency                       | Thời gian trung bình từ nhận ticket đến có draft câu trả lời.                                                                                                                             |
| Clarification Resolution Rate | Trong số ticket được hỏi làm rõ 1 lần, bao nhiêu % sau đó đạt confidence ≥ 0.6 (câu hỏi làm rõ thực sự hữu ích, thu thập đủ thông tin thiếu).                                             |
| Incident Grouping Rate        | % ticket được gộp đúng vào 1 incident thay vì xử lý riêng lẻ trong lúc sự cố diện rộng — đo mức tiết kiệm compute/thời gian duyệt.                                                       |
| Reopened Rate                 | % ticket đã `auto_sent`/`closed` nhưng bị end-user reopen — chỉ số chất lượng thật của auto-reply, quan trọng hơn cả approval rate vì đo hậu quả thật sau khi gửi.                      |
| Recurring Issue Rate          | % ticket gắn `recurring_issue = true` theo category — chỉ ra category nào có vấn đề gốc chưa fix triệt để.                                                                               |

## 7. Cách dùng eval set trong quá trình phát triển

1. Chạy toàn bộ eval set qua hệ thống trước khi thay đổi gì (baseline).
2. Mỗi khi đổi: embedding model, chunking strategy, prompt, model judge, threshold — chạy lại eval set, so sánh số liệu trước/sau.
3. Lưu lại lịch sử các lần chạy (`eval/results/eval_generation_*.json`, `eval/results/eval_confidence_*.json`) để theo dõi tiến độ và hiệu chỉnh hệ thống theo thời gian.

## 8. Guardrail test bắt buộc trước khi coi v1 "hoàn thành"

- [ ] Test với ít nhất 10 ticket adversarial (prompt injection, câu hỏi ngoài scope ngụy trang)
- [ ] Test với ticket cố tình chứa thông tin nhạy cảm giả lập, xác nhận hệ thống escalate đúng
- [ ] Xác nhận không có trường hợp nào confidence cao mà câu trả lời sai trong eval set (nếu có, phải hạ threshold hoặc cải thiện retrieval trước khi tiếp tục)

## 9. Hợp đồng eval hiện tại (schema_version = 2) & Công cụ harness

### 9.1. Định dạng dữ liệu và chuẩn hóa

- Hai harness chính ghi `schema_version: 2` ở cấp report:
  1. `eval/harness/eval_generation.py`: Đo `answer_accuracy` (thông qua correctness judge) và `groundedness` (gọi judge OpenRouter/Ollama), xuất summary gồm `mean_groundedness`, `ungrounded_answer_rate`, `answer_accuracy`.
  2. `eval/harness/eval_confidence_scorer.py`: Chạy pipeline RAG, tính `C_retrieval`, `C_consensus`, `C_groundedness` (nếu ở Full mode) và tổng hợp `C_RAG`. Cung cấp khung `assessment.human_review` để người đánh giá điền verdict nhằm hiệu chỉnh ngưỡng.
  3. `eval/harness/diagnose_groundedness.py`: Công cụ hỗ trợ chẩn đoán chi tiết lỗi schema, rejected evidence và output token của judge.
- Trường `generation` gồm: `answer`, `sources` (danh sách `parent_chunk_id` và `title` từ đúng KB contexts đã cấp cho model) và `complete` (boolean). Không còn dùng format `citations` nhãn inline hay `citation_support`.
- Trường `groundedness` chứa `claims` (tối đa 12 claim), `score` (0.0–1.0), `status` (`ok`/`unavailable`), `error`, `verdict`, `reason`, `problematic_claims`. Mean groundedness chỉ tính trên các kết quả có status `ok`, kể cả score 0.0 hợp lệ.
- `groundedness_unavailable_answers` đếm tất cả ticket chưa có phép chấm hợp lệ (bao gồm lỗi trước generation hoặc lỗi judge/rate limit).
- Accuracy lấy toàn bộ ticket answerable (`human_handoff = false`) làm mẫu số. Không thêm lại các trường cũ như `confidence_score`, `routing_decision`, `raw_outputs`, `latency_seconds`, `citation_check` vào JSON `eval_generation`.

### 9.2. Cơ chế xử lý Rate Limit (HTTP 429) & Delay

Khi chạy harness với Cloud LLM (OpenRouter), cả hai script đều trang bị:
- Tham số `--delay <seconds>` (mặc định 2.0s trong `eval_confidence_scorer.py`) để giãn cách giữa các ticket.
- Cơ chế tự động thử lại (retry với exponential backoff) khi bắt gặp lỗi HTTP 429 (Rate Limit).

### 9.3. Lệnh kiểm thử (Testing)

Khuyến nghị chạy test trong môi trường ảo (.venv) với pytest:
- Chạy toàn bộ test suite chức năng:
  ```powershell
  .\.venv\Scripts\python.exe -m pytest
  ```
  (hoặc `python -m unittest discover -s tests/functions -v`)
- Kiểm tra branch coverage cho các module cốt lõi (`retriever`, `confidence`, `llm_client`):
  ```powershell
  .\.venv\Scripts\python.exe -m pytest --cov=src.rag.retriever --cov=src.confidence --cov=src.agent.llm_client --cov-branch --cov-report=term-missing
  ```
- Chi tiết hướng dẫn xem tại [tests/functions/README.md](file:///d:/python-workspace/agent-IT-helpdesk/tests/functions/README.md) và [09_CONFIDENCE_IMPLEMENTATION_PLAN.md](file:///d:/python-workspace/agent-IT-helpdesk/implementation_plan/09_CONFIDENCE_IMPLEMENTATION_PLAN.md).
