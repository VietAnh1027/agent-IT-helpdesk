# Chẩn đoán groundedness

## Chạy riêng judge trên report cũ

Từ thư mục gốc:

```powershell
python -m eval.harness.diagnose_groundedness --report eval/results/eval_confidence_20261020_151432_788336.json --ticket-id TC_SYNC_001 --ticket-id TC_SYNC_003
```

Bỏ các `--ticket-id` để chạy tất cả ticket trong report. Không chạy lại retrieval,
embedding hoặc generation. Answer và context được lấy từ report; chỉ parent có
`used_for_generation=true` được đưa vào judge. Không sửa report gốc.

Output mặc định là file mới `*_judge_debug_<timestamp>.jsonl` nằm cạnh report:
mỗi dòng là một JSON độc lập cho một ticket, được ghi ngay khi ticket hoàn tất.
Có thể dùng `--output path/to/new.jsonl`; file đã tồn tại sẽ bị từ chối.

Judge dùng model/provider/seed/output budget từ Settings hiện tại và context
budget từ report nếu có. `original_judge` ghi cấu hình cũ, `diagnostic` ghi cấu
hình thực tế của lần chạy lại. Đây là phép thử tái hiện, không khôi phục chính xác
response của lần trước; model có thể trả kết quả khác dù seed giống nhau.

## Đọc kết quả

- `diagnostic.response.raw_response`: text thực tế của judge, kể cả JSON dở dang.
- `finish_reason`, `input_tokens`, `output_tokens`: metadata provider trả về;
  giá trị thiếu không được coi là 0.
- `stage`: `input_validation`, `generation`, `schema_validation`,
  `evidence_validation` hoặc `complete`.
- `rejected_evidence`: context reference đầu tiên không vượt qua kiểm tra, gồm
  claim/evidence index bắt đầu từ 1, claim text, context ID và mã lỗi.
- `unknown_context_id`: ID không thuộc danh sách `E1`, `E2`, ... được cấp cho judge.
- Evidence ID lặp trong cùng claim được báo là `Evidence context bị trùng`.
- `schema_errors`: vị trí/lỗi cấu trúc JSON hoặc kiểu dữ liệu, không dump lỗi SDK.

Context nằm trong cùng bản ghi để đối chiếu. Khi bị cắt do `length`, xem cuối
raw_response và các đoạn lặp trước khi quyết định tăng max_tokens. Nếu timeout
trước khi provider trả response thì chưa có raw_response/token metadata;
stage và error_type vẫn được lưu, không tự kết luận model đã lặp.

## Ý nghĩa “từ chối evidence”

Python gán các context đã dùng để generation thành `E1`, `E2`, ... trước khi gọi
judge. Judge chỉ được trả các ID này; không được tự tạo ID, source, parent ID hoặc
chép quote. Sau validation, Python ánh xạ ID hợp lệ về source và parent ID thật.

Chỉ một context ID không hợp lệ cũng khiến phép chấm trả `score=0.0`,
`status=unavailable`, confidence dùng fallback và `degraded`. Điều đó không
chứng minh answer sai; nó chứng minh kết quả judge chưa đạt hợp đồng kiểm tra.
Ngược lại, context ID tồn tại cũng chưa chứng minh judge suy luận entailment đúng.

## Phạm vi và dữ liệu

Script không đổi model, ngân sách output hoặc trọng số. Groundedness schema giới
hạn tối đa 12 claim và 3 context reference mỗi claim để tránh output lặp vô hạn.
Hai harness eval vẫn giữ report schema v2; evidence trong claim dùng `context_id`.
Chẩn đoán được bật bằng script riêng hoặc truyền dict `diagnostics={}` vào
`evaluate_groundedness`; caller tự quản lý việc lưu dict này.

File debug chứa answer, evidence và output model chưa xác minh. Không đưa
headers, API key hoặc toàn bộ Settings vào file; bản thân nội dung ticket vẫn
có thể nhạy cảm, nên chỉ chia sẻ dữ liệu debug sau khi kiểm tra.

Kiểm thử nằm ở `tests/functions/test_groundedness_diagnostics.py`.
