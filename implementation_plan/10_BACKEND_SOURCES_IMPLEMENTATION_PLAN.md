# Kế hoạch: backend quản lý nguồn, LLM chỉ sinh câu trả lời

Ngày: 15/10/2026. **Trạng thái: chờ duyệt, chưa triển khai code.**

## 1. Mục tiêu và lý do

LLM sinh answer thuần. Backend đính kèm metadata nguồn lấy từ DB của đúng các parent đã đưa vào generation. Không yêu cầu LLM tự viết tên nguồn, không tách citations bằng regex từ answer.

Lần eval `eval_confidence_20261014_173000_870143.json` cho thấy TC_SYNC_001 thiếu citations, TC_SYNC_002 viết sai tên thành `Dong_bo_email_outlink`. Cả hai bị degraded dù judge đánh giá được nội dung. Thay đổi này loại bỏ phụ thuộc vào khả năng sao chép nhãn của generator.

**Ý nghĩa sources:** tài liệu/context backend đã cung cấp cho LLM; không phải bằng chứng rằng LLM thực sự dùng đúng tất cả các nguồn hoặc từng câu đã được nguồn hỗ trợ. Groundedness tiếp tục kiểm tra nội dung câu trả lời.

Tên hiển thị đề xuất: **Tài liệu tham khảo đã cung cấp cho AI**. Không gọi danh sách này là “nguồn đã xác minh cho từng câu”.

## 2. Phạm vi

### Thực hiện trong đợt này

- Đổi output `generation.citations` sang `generation.sources` được dựng hoàn toàn từ metadata retrieval.
- Giữ `generate()` trả text và giữ `build_answer()` trong `src/agent/llm_client.py`; sửa nhiệm vụ hàm hiện có, không tạo module answer mới.
- Sửa prompt generation, schema/prompt groundedness và toàn bộ caller/test liên quan.
- Bỏ tiêu chí thiếu/sai citation inline khỏi confidence vì đây không còn là hợp đồng của generator.
- Giữ evidence source/quote trong judge và tiếp tục validate nghiêm ngặt.
- Đồng bộ tài liệu và ghi phiên bản định dạng JSON mới.

### Ngoài phạm vi

- Không đổi Dense/BM25/RRF, candidate_pool, top_k, trọng số 0.4/0.2/0.4 hay provider/model.
- Không thêm retry, fuzzy matching, tự sửa quote, nâng cấp model, router, UI hoặc auto-send.
- Không xử lý gộp các đề xuất trước đó về câu chuyển IT, claim ghép, debug quote và thống kê theo thành phần. Những việc này cần duyệt riêng.
- Không ingest lại KB, migrate DB hoặc sửa các JSON kết quả cũ đã có đánh giá thủ công.

## 3. Luồng trước và sau

### Hiện tại

```text
Retrieval → parent context {source, content}
          → LLM sinh answer + nhãn [tên nguồn]
          → build_answer tách/validate citations
          → judge nội dung và citation_support
          → confidence có thể degraded vì citation
```

### Sau thay đổi

```text
Hybrid retrieval → toàn bộ parent fusion (dùng tính retrieval/consensus)
                 → chọn đúng parents[:top_k]
                 → tạo kb_contexts có parent_chunk_id, source, content
                 → LLM sinh answer thuần
                 → build_answer ghép answer + sources từ kb_contexts
                 → judge claim + evidence source/quote
                 → confidence dựa trên score, tình trạng judge và mâu thuẫn evidence
```

Nguồn được lấy từ cùng danh sách context đã gửi generation, không từ toàn bộ fusion và không từ các child ứng viên chưa được chọn. Không truy vấn DB lần hai để dựng sources. Nội dung đầy đủ vẫn là parent, dù xếp hạng được tính trên child.

## 4. Hợp đồng dữ liệu mới

### 4.1. Context nội bộ

```json
{
  "parent_chunk_id": "uuid-parent",
  "source": "Dong_bo_email_outlook",
  "content": "Nội dung parent đã cung cấp cho LLM"
}
```

- `parent_chunk_id` và `source` lấy từ `parent_chunk_id` / `article_title` của retriever hiện có.
- `source` vẫn là tên nguồn để judge đối chiếu quote; không phải cột nguồn gốc biên soạn `kb_articles.source`.
- ID dùng truy vết, không yêu cầu generator in ID hoặc tên tài liệu trong answer.
- Hai parent cùng bài có thể có cùng source; vẫn giữ hai parent khác nhau.

### 4.2. Output build_answer

```json
{
  "answer": "Vào tab Send / Receive và tắt Work Offline.",
  "sources": [
    {
      "parent_chunk_id": "uuid-parent",
      "title": "Dong_bo_email_outlook"
    }
  ]
}
```

- Sources giữ thứ tự context và dedupe theo parent ID, không dedupe theo title.
- Không thêm content vào sources vì evidence đã có ở kb_contexts hoặc retrieval.parents trong report.
- Metadata thiếu/rỗng hoặc cùng parent ID có title mâu thuẫn: báo lỗi rõ, không tự tạo ID/title hay coi output hợp lệ.
- Context rỗng cho sources rỗng ở helper; các pipeline vẫn chặn generation khi retrieval không có kết quả.
- `build_answer()` giữ nguyên text, kể cả text rỗng/partial để caller lưu lỗi; không tự chứng nhận generation hoàn tất.
- Ngoặc vuông bất kỳ trong answer không còn được parse thành citations hoặc dùng để tạo sources. Không regex xóa ngoặc vuông vì có thể làm hỏng lệnh hoặc nội dung hợp lệ.
- Cờ `complete` thuộc harness, không phải dữ liệu do LLM hoặc build_answer quyết định. Giữ `complete` của confidence eval; bổ sung cùng cờ này cho generation eval để phân biệt text bị cắt.

### 4.3. Groundedness

Judge input chỉ còn question, answer, kb_contexts. Bỏ input citations và output citation_support.

Giữ claims, reason và mỗi evidence `{source, quote}`. Giữ các field do Python tính: score, status, error, verdict, problematic_claims. Confidence eval tiếp tục có supported_claim_count và total_claim_count.

Backend sources đến từ DB không thay thế validation quote: judge vẫn có thể bịa nguồn hoặc trích sai. Source phải khớp và quote phải tồn tại trong một context thực sự đã dùng; vẫn cho phép nhiều context cùng source như hiện tại. Không tự thêm parent ID vào output judge trong đợt này.

## 5. Vị trí thay đổi chính xác

Các đường dẫn dưới đây tương đối với `D:/python-workspace/agent-IT-helpdesk`. Vị trí được chỉ theo tên hàm/section để không lệ thuộc số dòng sau khi sửa.

### 5.1. `src/agent/llm_client.py`

**`build_answer(answer, kb_contexts)`**

- Bỏ regex lấy nhãn, kiểm tra unknown labels và lỗi citation.
- Dựng sources từ parent_chunk_id/source theo hợp đồng mục 4; validate metadata và dedupe giữ thứ tự.
- Bỏ import `re` nếu không còn nơi dùng.
- Lý do: nguồn là trách nhiệm của backend, không phụ thuộc text model.

**`GenerationError.__init__` và các nơi ném lỗi trong `generate()`**

- Giữ exception cho output bị cắt và khả năng lưu partial answer.
- Bỏ tham số citations; payload exception chỉ giữ answer. Không cho LLM client tự nhận rằng sources rỗng vì client không biết context được chọn.
- Caller ghép sources từ context đang có khi bắt lỗi. Không gọi judge cho partial answer.
- Giữ nguyên lựa chọn OpenAI/Ollama/Mistral, timeout và ngân sách token.

### 5.2. `src/agent/prompts/answer_with_cite_system.jinja2`

- Bỏ toàn bộ section Citations yêu cầu sao chép nhãn, ngoặc vuông và kiểm tra label.
- Thay bằng chỉ dẫn ngắn: trả answer thuần; backend quản lý danh sách tài liệu; không tự thêm citation inline hoặc danh sách nguồn.
- Giữ guardrail, điều kiện áp dụng evidence, giới hạn hành động và quy tắc trả lời đúng trọng tâm.
- Prompt vẫn viết tiếng Anh.
- Giữ tên file cũ trong đợt này để tránh đổi chuỗi prompt ở nhiều caller. Ghi chú tên lịch sử, không còn nghĩa là LLM phải sinh citation.

### 5.3. `src/agent/prompts/answer_with_cite_task.jinja2`

- Cập nhật comment về cấu trúc context có parent ID.
- Giữ subject/body/category/supplied_knowledge và cách serialize JSON; không thêm golden data.

### 5.4. `src/confidence.py`

**`GroundednessReport`**: bỏ field citation_support khỏi schema gửi cả Ollama và Mistral.

**`evaluate_groundedness()`**:

- Bỏ gọi build_answer để lấy citations và bỏ nhánh invalid_citations.
- Không truyền citations vào template, không ép citation_support thành unsupported khi answer thiếu nhãn.
- Bỏ import build_answer/GenerationError nếu không còn sử dụng trong file.
- Giữ kiểm tra answer hoàn chỉnh, evidence, JSON và fallback khi provider/schema/quote lỗi.

**`score_groundedness()`**: giữ validation claim/source/quote và supported/total; output không còn citation_support từ model_dump.

**`unavailable_groundedness()`**: bỏ citation_support, giữ 0.0 + unavailable + error.

**`calculate_confidence()`**: bỏ điều kiện degraded vì citation_support không supported. Giữ degraded khi grounding unavailable hoặc có claim contradicted; giữ unavailable khi retrieval lỗi/rỗng. Không thay đổi công thức hay tuyên bố status ok đồng nghĩa answer đúng.

### 5.5. Hai prompt groundedness trong `src/agent/prompts/`

- `eval_groundedness_system.jinja2`: bỏ phần đánh giá citation inline và field citation_support trong output; giữ yêu cầu evidence source/quote nguyên văn cùng các quy tắc tách claim.
- `eval_groundedness_task.jinja2`: bỏ key citations khỏi JSON input. Dùng đúng kb_contexts đã cấp cho generator.
- Không biến sources tự động thành claim supported. Judge vẫn phải đọc nội dung evidence.

### 5.6. `src/pipeline.py`

- Tại đoạn dựng kb_contexts: bổ sung parent_chunk_id từ chunks.
- Giữ gọi build_answer, in output mới answer/sources.
- Đảm bảo metadata nguồn hợp lệ trước khi gọi model; giữ chặn retrieval rỗng.
- Nếu lưu/in partial answer khi GenerationError, dùng context hiện tại và đánh dấu chưa hoàn chỉnh; không giả định partial answer dùng được.

### 5.7. `eval/harness/eval_generation.py`

**`evaluate_ticket()`**:

- Thêm parent_chunk_id vào đúng contexts từ parents[:top_k].
- Lưu generation với answer/sources/complete; bỏ khởi tạo citations rỗng và logic tiếp tục chấm sau lỗi citation.
- Khi GenerationError do truncation: giữ partial answer, ghép sources từ context và complete=false, không gọi correctness/groundedness.
- Lỗi metadata nguồn phải được ghi rõ theo stage, không được làm mất answer đã sinh hay coi là pass.
- Giữ correctness judge và cách ly golden data như hiện tại.

**`run_evaluation()`**: thêm `schema_version: 2` ở cấp report. Giữ các field hiện có ngoài những thay đổi đã nêu; không thêm lại confidence/routing/raw_outputs/latency/citation_check.

### 5.8. `eval/harness/eval_confidence_scorer.py`

**`evaluate_ticket()`**:

- Dựng contexts có parent ID từ cùng parents[:top_k]; metadata toàn bộ fusion vẫn được giữ để phân tích retrieval.
- Thay generation.citations bằng sources, bỏ try/except riêng cho nhãn sai.
- Giữ complete và cách lưu partial answer; ghép sources cả khi generate bị cắt.
- Sử dụng grounding schema mới; bỏ thông tin citation khỏi confidence.errors theo hàm chung.
- Giữ assessment.human_review null cho người đánh giá, không tự gắn nhãn.

**`run_evaluation()`**: thêm `schema_version: 2` ở cấp report; vẫn tạo file mới, không ghi đè kết quả đã gắn nhãn.

**`summarize()`**: không đổi công thức/mẫu số trong đợt này. Số status ok có thể tăng vì không còn lỗi citation; đây là thay đổi policy, không đủ để kết luận model trả lời tốt hơn.

### 5.9. Các phần không cần thay đổi chức năng

- `src/rag/retriever.py`: đã trả parent_chunk_id/article_title/text, đủ để dựng sources. Không cần thêm truy vấn hay child ID vào output để đáp ứng yêu cầu hiện tại.
- `eval/harness/eval_retrieval.py`: đo retrieval, không phụ thuộc generation.citations.
- `src/core/config.py`: không cần setting mới.
- Prompt correctness: vẫn so sánh answer với golden, không cần thay vì đổi metadata nguồn.
- `src/agent/__init__.py`: hiện chỉ có docstring, không có export cần sửa.
- `src/agent/confidence.py`: hiện không có implementation; logic đang ở `src/confidence.py`, không tạo bản sao.
- Không có `src/agent/prompt_tasks.py` trong checkout hiện tại; không tạo lại chỉ vì lịch sử từng có file này.

## 6. Tài liệu chịu ảnh hưởng khi triển khai

| File/section | Nội dung cần đồng bộ |
|---|---|
| `documents/02_ARCHITECTURE.md`, bước Retrieval/LLM Answer Generation | Backend đính kèm nguồn của context, LLM trả text; phân biệt nguồn context với citation theo claim |
| `documents/03_DATA_MODEL.md`, `ai_responses.cited_kb_chunk_ids` | Ghi rõ trường thiết kế cũ trỏ child không được điền bằng parent ID hoặc gọi là nguồn đã dùng; mapping persistence sẽ xử lý ở đợt tích hợp DB, không migrate trong đợt này |
| `documents/04_CONFIDENCE_EVAL_POLICY.md`, groundedness/fallback/auto condition và hợp đồng eval | Bỏ citation_support và điều kiện citation inline; vẫn giữ các guardrail khác |
| `documents/08_AGENT_STATE.md`, generation/groundedness/errors và ví dụ JSON | Đổi answer/citations thành answer/sources; không còn invalid_citation như lỗi generator; vẫn giữ thiết kế state chưa triển khai |
| `documents/09_CONFIDENCE_IMPLEMENTATION_PLAN.md`, các mục citation | Thêm ghi chú hợp đồng nguồn được thay thế bởi kế hoạch 10; giữ lịch sử quyết định cũ rõ ràng |
| `workflow.md`, node sinh draft + trích dẫn | Ghi rõ LLM sinh answer, backend đính kèm nguồn |
| `to-do-flow.md`, bước generation | Cập nhật trách nhiệm nguồn của backend |
| `tests/functions/README.md` | Hướng dẫn schema mới, tests nguồn, ý nghĩa nguồn và giới hạn |

Không tự đổi các khái niệm citation không liên quan, ví dụ trích dẫn của tài liệu hoặc tracing child trong thiết kế DB. Phải đọc ngữ cảnh trước khi thay, không search/replace toàn bộ từ citations.

## 7. Tương thích và bảo toàn dữ liệu

- Đây là thay đổi cấu trúc có chủ đích: generation.citations bị bỏ, generation.sources được thêm; groundedness.citation_support bị bỏ.
- Report cũ không có schema_version được hiểu là v1; report mới là v2. Không tự chuyển file cũ vì không thể biết chắc nguồn nào đã được dùng chỉ từ các nhãn citation.
- Không giữ hai trường citations/sources song song với cùng ý nghĩa khác nhau.
- Sửa tất cả caller/test hiện có trong cùng đợt. Công cụ bên ngoài đọc JSON cũ cần cập nhật theo schema_version trước khi đọc v2.
- So sánh trước/sau phải ghi rõ policy citation thay đổi. Confidence số học không đổi nếu ba thành phần giống nhau, nhưng status có thể đổi.
- Không đặt parent ID vào trường DB dành cho child ID. Sources là metadata cấp parent; persistence hiện không được triển khai trong harness này.

## 8. Kiểm thử bắt buộc trong `tests/functions`

### `test_eval_generation.py`

- Thay tests tách/validate citation bằng tests build_answer từ context: text không có citation vẫn có sources đúng.
- Text chứa `[invented]` không tạo nguồn giả, không thay đổi danh sách nguồn từ DB và không bị regex xóa.
- Cùng parent ID trùng được dedupe; hai parent cùng title không bị mất; thứ tự khớp context.
- Metadata thiếu/rỗng/mâu thuẫn bị từ chối; context rỗng không tạo nguồn giả.
- Partial answer vẫn được lưu với sources và complete=false; không gọi hai judge.
- JSON mới có schema_version và không còn citations/citation_support; golden chỉ đi correctness.

### `test_confidence.py`

- Fixture/schema judge bỏ citation_support; payload không có citations.
- Answer không có citation vẫn được judge bình thường; điểm số không bị hạ chỉ vì thiếu nhãn.
- Source/quote bịa, không claim, JSON sai và truncation vẫn fallback 0.0/unavailable.
- Contradicted vẫn degraded; retrieval failure vẫn unavailable; các phép tính RRF/consensus/trọng số không đổi.
- Request schema của cả Mistral và Ollama phù hợp schema mới.

### `test_eval_confidence_scorer.py`

- Sources chỉ từ top_k thực sự gửi generation, không lấy toàn bộ candidate fusion.
- Với top_k=1, vẫn giữ top-2 cho tính margin nhưng output chỉ có nguồn của một context.
- Groundedness có thể ok với answer thuần; bỏ test degraded vì typo citation, thay bằng test nguồn không phụ thuộc text.
- JSON partial/error giữ answer và nguồn, fallback vẫn đúng; output hiện có không bị ghi đè.
- Human review vẫn chưa được gán và golden không đi generator/judge groundedness.

### Kiểm tra hoàn tất

```powershell
python -m unittest discover -s tests/functions -v
python -m eval.harness.eval_generation --help
python -m eval.harness.eval_confidence_scorer --help
git diff --check
```

Khi dịch vụ sẵn sàng, chạy TC_SYNC_001/002/003 vào file mới. Xác nhận sources khớp đúng parent context, không có lỗi thiếu/sai nhãn từ generator. Nếu model vẫn viết citation dù prompt cấm, ghi nhận vi phạm prompt riêng; không coi nhãn đó là metadata nguồn và không tự xóa text.

TC_SYNC_003 vẫn có thể lỗi quote hoặc trả lời sai trọng tâm: đợt này không tuyên bố giải quyết những vấn đề đó. Ghi rõ test mock/live nào thực sự chạy, không dùng số test pass cũ làm bằng chứng cho implementation mới.

## 9. Trình tự triển khai sau khi duyệt

1. Viết tests cho hợp đồng sources và cập nhật fixtures parent/context có ID.
2. Sửa build_answer, GenerationError và caller lưu partial answer.
3. Sửa prompt generation, cập nhật context ở pipeline và hai harness.
4. Sửa schema/prompt groundedness, bỏ citation gate trong confidence.
5. Thêm schema_version, đồng bộ tài liệu và tests còn lại.
6. Chạy toàn bộ kiểm thử, kiểm tra caller bỏ sót bằng rg; smoke test khi dịch vụ hoạt động.
7. Báo cáo thay đổi, kết quả test và giới hạn thực tế.

## 10. Tiêu chí nghiệm thu

- Nguồn output do backend dựng từ context đã gửi, không do generator quyết định.
- Không có nguồn từ parent chưa gửi vào prompt; không dedupe nhầm các parent cùng tên bài.
- Answer thuần không bị phạt vì thiếu citation inline.
- Evidence của judge vẫn được kiểm tra, lỗi judge không trở thành phép chấm thành công.
- Partial answer không mất và không được chấm như answer hoàn chỉnh.
- Các caller, prompt, schema, docs và tests nhất quán; report cũ/nhãn thủ công được bảo toàn.
- Không thêm DB migration, dependency hoặc tính năng ngoài phạm vi.

**Giới hạn:** sources chứng minh provenance của context, không chứng minh entailment của từng câu. Việc bỏ citation gate không cải thiện tự động chất lượng answer, tính đúng của judge hoặc mức an toàn của routing.
