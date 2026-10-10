# Implementation plan — Hybrid retrieval và confidence

Ngày: 11/10/2026. **Kế hoạch để duyệt; chưa triển khai code chức năng.**

## 1. Phạm vi và quyết định đã chốt

- Dense Vector + BM25 truy vấn độc lập trên cùng tập KB hợp lệ; hợp nhất bằng RRF.
- Đơn vị xếp hạng và consensus là parent ID, không phải tên bài viết.
- Hai kênh có trọng số bằng nhau; RRF dùng k = 60, rank bắt đầu từ 1.
- S_top1 lấy từ RRF chuẩn hóa theo mức tối đa lý thuyết, không chuẩn hóa theo max của mỗi query.
- Không dùng A_source hoặc phân cấp uy tín nguồn.
- Groundedness dùng LLM judge MistralAI nếu có `MISTRAL_API_KEY`trong .env, không thì sẽ sử dụng ChatOllama với model Qwen 3.5:4b, đánh giá từng claim; không dùng NLI/CrossEncoder.
- Các hàm có thể import độc lập; tái sử dụng settings, generate, render_prompt, build_answer và retriever hiện có.
- Không triển khai router, auto-send, AgentState hoặc thay chiến lược chunking trong đợt này.

## 2. Công thức chính xác

```text
RRF(d) = dense_contribution(d) + bm25_contribution(d)
contribution(d) = 1 / (60 + rank(d)) nếu có trong danh sách của kênh, ngược lại 0
relevance_score(d) = RRF(d) / (2 / 61)

S_top1 = relevance_score của parent đầu bảng RRF
margin = S_top1 - S_top2; bằng 0 nếu chỉ có một kết quả
C_retrieval = 0.7 * S_top1 + 0.3 * min(1, 2 * margin)

C_consensus = 1.00 nếu parent đầu bảng RRF thuộc Top-3 của cả hai kênh
              0.75 nếu thuộc Top-5 của cả hai kênh nhưng không đạt điều kiện trên
              0.50 trong mọi trường hợp còn lại khi cả hai kênh chạy thành công

C_groundedness = supported_claim_count / total_claim_count
C_RAG = 0.4 * C_retrieval + 0.2 * C_consensus + 0.4 * C_groundedness
```

Kiểm tra dữ liệu đầu vào hữu hạn, score trong [0,1], thứ tự giảm dần và parent không trùng. Không âm thầm clamp dữ liệu lỗi; chỉ có thể giới hạn sai số số thực rất nhỏ ở biên. Giữ số thực đầy đủ khi tính, chỉ làm tròn khi hiển thị.

## 3. Hybrid retrieval

### 3.1. Cùng corpus, hai kênh độc lập

1. Lọc theo category và trạng thái active của KB; giữ parent/child/article hợp lệ.
2. Dense dùng embedding BAAI/bge-m3 hiện có và cosine similarity trên child.
3. BM25 chấm trên toàn bộ child trong tập hợp lệ, không chỉ trên các kết quả Dense đã chọn.
4. Mỗi kênh gom theo parent, lấy điểm child tốt nhất của parent rồi xếp hạng. Giới hạn `candidate_pool` sau khi gom để nhiều child cùng parent không chiếm hết danh sách.
5. Hợp nhất hai danh sách parent bằng RRF, giữ rank từng kênh, cuối cùng trả top_k parent để đưa vào prompt.

**Lựa chọn triển khai ban đầu:** BM25 bằng stdlib (Counter/math), không thêm dịch vụ search hay extension DB. Dùng công thức BM25 với IDF dương, k1 = 1.5, b = 0.75; tính thống kê trên corpus child đã lọc. Tách token bằng Unicode, chuẩn hóa NFC + casefold, giữ dấu tiếng Việt và mã lỗi dạng chữ/số; không bỏ stopword/stem ở bản đầu. Query dùng các token khác nhau, tránh tăng trọng số vô tình do từ lặp.

Chỉ xếp vào kênh BM25 những child có ít nhất một token khớp và score dương. Không có từ khớp là danh sách rỗng thành công, khác lỗi truy vấn. Corpus rỗng không tạo kết quả giả.

BM25 đọc/chấm toàn bộ corpus đã lọc mỗi request: phù hợp KB nhỏ hiện tại nhưng chi phí tăng theo số child. Đo latency trước khi cần chuyển sang index BM25 hoặc cache có cơ chế invalidation. Không dùng PostgreSQL ts_rank rồi gọi đó là BM25.

### 3.2. Thứ hạng ổn định và kết quả trả về

- Khi bằng điểm: dùng parent ID làm tiêu chí phụ cố định; không phụ thuộc thứ tự row ngẫu nhiên từ DB.
- Giữ signature `retrieve_top_k_parents()` và các trường cũ: parent_chunk_id, parent_index, text, article_title, similarity.
- `similarity` tiếp tục là cosine Dense, tuyệt đối không đổi nghĩa thành RRF. Tính cosine cho cả parent chỉ lọt vào danh sách BM25 để caller cũ vẫn có giá trị đúng nghĩa.
- Thêm `dense_rank`, `bm25_rank` (null khi không trong candidate list), `bm25_score`, `rrf_score`, `relevance_score`.
- Giữ hai danh sách candidate đủ dài (mặc định 20 parent/kênh), không cắt theo top_k trước khi tính consensus.
- Khi cần tính confidence và top_k = 1, lấy tối thiểu hai kết quả fusion để biết margin, nhưng chỉ đưa đúng một parent vào generation. Nếu toàn bộ tập fusion thực sự chỉ có một parent thì margin = 0.
- Score retrieval/consensus dùng kết quả trước khi cắt context; groundedness chỉ dùng đúng evidence đã đưa vào generator.
- Một kênh lỗi: báo lỗi có tên kênh, confidence unavailable; không âm thầm chuyển sang single-channel rồi ghi consensus = 0.5.

## 4. Các hàm confidence và groundedness

Đặt trong `src/confidence.py` hiện đang trống; không tạo package scorer mới.

| Hàm dự kiến                     | Đầu vào                                                 | Kết quả                                 |
| ---------------------------------- | ---------------------------------------------------------- | ----------------------------------------- |
| `calculate_retrieval_confidence` | Các parent sau fusion, có ít nhất top-2 nếu tồn tại | C_retrieval                               |
| `calculate_consensus`            | Parent đầu fusion với dense_rank/bm25_rank              | C_consensus                               |
| `evaluate_groundedness`          | question, answer, kb_contexts thực tế                    | claims, score, status, error              |
| `calculate_confidence`           | Hai điểm retrieval/consensus và kết quả groundedness  | Ba thành phần, C_RAG, trạng thái/lỗi |

Các hàm tính điểm không gọi DB hoặc model; chỉ `evaluate_groundedness` gọi LLM. Retriever tiếp tục sở hữu mọi logic search/ranking. Nếu cần helper RRF/BM25, giữ trong `src/rag/retriever.py` để tái sử dụng và kiểm thử.

### 4.1. Judge

- Một request riêng đọc question, toàn bộ answer và evidence; không nhận golden answer.
- Output JSON có claims; mỗi claim gồm claim text, verdict và evidence quotes/source. Giữ thông tin citation_support riêng cho eval, không đồng nhất citation đúng tên với groundedness.
- supported = 1; unsupported/contradicted/uncertain = 0. Tất cả claim kiểm chứng được tham gia mẫu số.
- Tách claim ghép, giữ điều kiện/negation/numbers/commands; loại claim lặp; không chấm lời chào hoặc câu hỏi thuần túy.
- Python validate schema/nguồn/quote trong đúng evidence rồi đếm. Không dùng fuzzy matching để hợp thức hóa quote sai. Claim supported/contradicted phải có evidence tương ứng.
- Dedupe trùng nghĩa do judge thực hiện theo prompt; Python chỉ có thể kiểm tra trùng văn bản rõ ràng. Không tuyên bố code đã xác minh đầy đủ ngữ nghĩa hay độ bao phủ claim.
- Temperature 0, seed cố định; model judge lấy từ Settings chung, mặc định qwen3.5:4b.
- Tiếp tục dùng `generate()`; chỉ bổ sung tham số tối thiểu cần thiết cho seed và ngân sách output của judge. Output claim dài hơn verdict hiện tại nên ngân sách output phải được tính vào kiểm tra context; tuyệt đối không silently truncate evidence.

### 4.2. Lỗi và score fallback

| Trường hợp                                                              | Kết quả                                                                                   |
| -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Judge hoàn tất hợp lệ, có claim                                       | score là tỷ lệ supported; status = ok                                                    |
| Không claim, answer rỗng/bị cắt, JSON lỗi, quote sai hoặc model lỗi | groundedness = 0.0; status = unavailable; giữ lý do                                       |
| Groundedness unavailable nhưng retrieval hợp lệ                         | Có thể tính C_RAG theo fallback 0.0; đánh dấu degraded, không đủ điều kiện auto |
| Dense/BM25 lỗi hoặc không có parent                                    | C_RAG = null; status = unavailable; không gọi generator/judge nếu chưa có answer       |
| Citation trong answer sai nhưng text đầy đủ                           | Giữ answer/lỗi, vẫn chấm groundedness; không coi citation đã được xác minh       |

Score số học không tự quyết định auto. Chưa triển khai router, nhưng hợp đồng phải ghi rõ caller không được chỉ đọc C_RAG rồi bỏ qua status/errors. Groundedness 0.0 vì lỗi khác với 0.0 do mọi claim đều bị đánh giá không supported.

## 5. Các file dự kiến thay đổi

| File                                                                                     | Thay đổi                                                                                             |
| ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `src/rag/retriever.py`                                                                 | BM25, dedupe từng kênh, RRF, metadata điểm/rank; giữ API hiện tại                               |
| `src/confidence.py`                                                                    | Hàm tính ba thành phần, judge claim và tổng hợp confidence                                      |
| `src/core/config.py`                                                                   | Model judge và các cấu hình judge thực sự cần; không đọc .env riêng                         |
| `src/agent/llm_client.py`                                                              | Tham số judge cần thiết; giữ generate trả text và cơ chế lưu partial answer                   |
| `src/agent/prompts/eval_groundedness_system.jinja2`, `eval_groundedness_task.jinja2` | Chuyển sang claim-based judge, dùng chung cả runtime và eval; prompt tiếng Anh                    |
| `eval/harness/eval_generation.py`                                                      | Import judge dùng chung, bỏ schema/logic grounding trùng lặp; correctness giữ riêng              |
| `eval/harness/eval_retrieval.py`                                                       | Kiểm tra caller theo hybrid; giữ cosine riêng cho metric cũ, không dùng ngưỡng cosine lên RRF |
| `eval/harness/test_eval_generation.py` và một file kiểm thử confidence/hybrid nhỏ | Cập nhật fixture và kiểm tra logic mới                                                            |
| `documents/04_CONFIDENCE_EVAL_POLICY.md`, `08_AGENT_STATE.md`                        | Đồng bộ công thức mới, trọng số, fallback và trạng thái unavailable                         |

Không thay judge correctness hiện tại chỉ vì đổi judge groundedness. Không đổi provider của generation hoặc xóa các chỉnh sửa riêng của người dùng.

## 6. Tích hợp và tương thích eval

- Giữ answer và errors kể cả khi citation lỗi; partial answer vẫn được lưu, không gọi judge trên text chưa hoàn chỉnh.
- `groundedness` của JSON eval sẽ thêm claims/score/status để quan sát kết quả mới. Đây là thay đổi cấu trúc có chủ đích để bạn duyệt trong kế hoạch này.
- Không thêm lại trường top-level confidence_score, routing_decision, raw_outputs, latency_seconds hoặc citation_check đã yêu cầu bỏ trước đây. Module confidence có thể import và gọi riêng; tích hợp routing thực tế là việc sau.
- Nếu giữ verdict cấp answer để tương thích summary cũ: contradicted ưu tiên trước, rồi unsupported, rồi uncertain, sau cùng supported khi tất cả claim supported. Kết quả unavailable không bị gán verdict supported.
- Báo mean groundedness trên kết quả status ok, kèm số unavailable; fallback 0.0 không được âm thầm tính như một phép chấm thành công. Giữ answer accuracy và ungrounded answer rate với mẫu số minh bạch.
- Không chạy lại judge lần thứ hai để tính confidence; dùng chính kết quả grounding đã nhận.
- `src/pipeline.py` tiếp tục dùng retriever qua API hiện có. Chưa thêm auto-send hoặc một pipeline confidence mới vào script này.

## 7. Thứ tự triển khai và kiểm thử

1. Viết/test các hàm RRF và phép tính score thuần trước.
2. Thêm BM25, gom parent và metadata; test union kết quả, lọc category/active, tie-break, BM25 rỗng và lỗi kênh.
3. Thêm judge/schema/prompt, validate evidence, tính groundedness và fallback bằng phản hồi mock.
4. Nối eval với hàm dùng chung, sửa fixture và kiểm tra JSON vẫn giữ answer khi lỗi.
5. Đồng bộ docs; chạy kiểm thử local không cần DB/model trước.
6. Khi có DB/Ollama hoạt động: smoke test một ticket, sau đó so sánh các case TC_SYNC_002/009 và case nhiều sự cố với baseline. Ghi rõ kiểm thử live nào thực sự đã chạy; không tự ingest lại KB.

Các kiểm tra bắt buộc:

- Top-1 ở cả hai kênh: relevance_score = 1; một kênh rank 1 duy nhất: relevance_score = 0.5.
- Top-1 hai kênh, top-2 cũng cả hai rank 2: C_retrieval xấp xỉ 0.709677. Không kỳ vọng nó tự bằng 1.
- Consensus đúng cả ba mức và chỉ tính sau khi dedupe parent; top_k = 1 không làm mất rank Top-5 hoặc top-2 dùng tính margin.
- Không có lexical match thì BM25 rỗng, không tạo đồng thuận từ tài liệu score 0.
- 3/4 claim supported cho 0.75; quote bịa, JSON sai, không claim và truncation cho 0.0 + unavailable.
- Trọng số đúng 0.4/0.2/0.4; finite/range validation; không thể gọi kết quả degraded là sẵn sàng auto.
- Không truyền golden data vào grounding/generation; không mất answer khi lỗi; không đổi nghĩa similarity cũ.
- Kiểm thử mock và live tách biệt; kiểm tra nhãn claim bằng người trên một tập nhỏ trước khi coi score là đáng tin.

## 8. Rủi ro cần biết trước khi duyệt

- RRF chuẩn hóa phản ánh thứ hạng, không phải xác suất relevance. Hai kênh cùng xếp một tài liệu kém lên đầu vẫn có thể cho điểm cao.
- RRF và consensus sử dụng chung tín hiệu xếp hạng nên không độc lập; bộ trọng số này là heuristic đã chọn, cần đo bằng eval.
- Với k=60, margin thường nhỏ. Ví dụ consensus = 1 và groundedness = 1 ở trường hợp hai parent cùng rank 1/2 cho C_RAG khoảng 0.883871. Không tự sửa hệ số để làm điểm đẹp hơn.
- C_consensus có sàn 0.5; không được tạo score cho trường hợp không có kết quả hoặc retrieval lỗi chỉ vì có sàn này.
- LLM judge có thể bỏ sót claim và chấm sai. Seed/temperature chỉ tăng tính lặp lại, không biến đánh giá ngữ nghĩa thành deterministic tuyệt đối.
- BM25 tokenization đơn giản có giới hạn với tiếng Việt; đo retrieval trước khi bổ sung bộ tách từ hoặc index ngoài.
- Các ngưỡng routing cũ chưa được hiệu chỉnh cho công thức mới; đợt này không bật tự động gửi dựa trên chúng.

**Tiêu chí hoàn tất:** các hàm import được, hai kênh retrieval thực sự độc lập, công thức đúng theo thỏa thuận, lỗi không bị che bằng score hợp lệ, eval tái sử dụng judge và giữ answer, kiểm thử đạt; không thêm hệ thống ngoài phạm vi trên.
