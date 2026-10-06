# BÁO CÁO PHÂN TÍCH NGUYÊN NHÂN LỖI SO KHỚP EVIDENCE TRONG EVAL_RETRIEVAL

Tài liệu này ghi lại chi tiết quá trình điều tra nguyên nhân hàm `evidence_matches_chunk` trong file `eval/eval_retrieval.py` thường xuyên trả về `False`, bảng phân tích từng mẫu kiểm thử và giải pháp nâng cấp cơ chế so khớp.

---

## 1. Tổng quan kết quả kiểm tra

Khi đánh giá trên 26 câu hỏi in-scope đầu tiên (`TC_SYNC_001` đến `TC_SYNC_026`), hàm `evidence_matches_chunk` theo logic cũ chỉ đạt **46.2%** độ trùng khớp.

| Phương pháp kiểm tra | Số case Match | Tỷ lệ Match | Ghi chú |
| :--- | :---: | :---: | :--- |
| **Logic cũ (`normalize_whitespace`)** | **12 / 26** | **46.2%** | Bị rớt 14 case |
| **Chuẩn hóa Bullets, Quotes & Ký tự** | **22 / 26** | **84.6%** | **Khắc phục ngay 10 case** bị lệch format |
| **Bổ sung độ phủ ranh giới Chunk ($\ge 75\%$)** | **26 / 26** | **100.0%** | Khắc phục nốt 4 case bị ranh giới cắt đôi |

---

## 2. Bảng phân tích chi tiết từng trường hợp (26 In-scope Cases)

| ID | Logic cũ | Sau chuẩn hóa | Parent Chunk | Nguyên nhân gốc rễ |
| :--- | :---: | :---: | :---: | :--- |
| `TC_SYNC_001` | **False** | **True** | P0 | **Lệch gạch đầu dòng (`- `)** ở danh sách Nguyên nhân & Giải pháp |
| `TC_SYNC_002` | **False** | **True** | P0 | **Lệch gạch đầu dòng (`- `)** ở danh sách Nguyên nhân & Giải pháp |
| `TC_SYNC_003` | **True** | **True** | P0 | Đoạn ngắn đơn dòng, không chứa ký tự bullet list |
| `TC_SYNC_004` | **False** | **True** | P1 | **Lệch gạch đầu dòng (`- `)** ở danh sách Nguyên nhân & Giải pháp |
| `TC_SYNC_005` | **False** | **True** | P1 | **Lệch gạch đầu dòng (`- `)** ở danh sách Nguyên nhân & Giải pháp |
| `TC_SYNC_006` | **False** | **True** | P1 | **Lệch gạch đầu dòng (`- `)** ở danh sách Nguyên nhân & Giải pháp |
| `TC_SYNC_007` | **True** | **True** | P1 | Đoạn ngắn đơn dòng, không chứa ký tự bullet list |
| `TC_SYNC_008` | **False** | **True (với $\ge 75\%$)** | P1 & P2 | **Chunk boundary cut-off**: Đoạn dài bị cắt đôi qua Parent 1 và Parent 2 |
| `TC_SYNC_009` | **False** | **True (với $\ge 75\%$)** | P1 & P2 | **Chunk boundary cut-off**: Đoạn dài bị cắt đôi qua Parent 1 và Parent 2 |
| `TC_SYNC_010` | **True** | **True** | P2 | Đoạn ngắn đơn dòng |
| `TC_SYNC_011` | **True** | **True** | P2 | Đoạn ngắn đơn dòng |
| `TC_SYNC_012` | **False** | **True (với $\ge 75\%$)** | P1 & P2 | **Chunk boundary cut-off**: Đoạn dài bị cắt đôi qua Parent 1 và Parent 2 |
| `TC_SYNC_013` | **False** | **True** | P2 | **Lệch gạch đầu dòng (`- `)** ở các bước thực hiện |
| `TC_SYNC_014` | **True** | **True** | P2 | Đoạn ngắn đơn dòng |
| `TC_SYNC_015` | **True** | **True** | P2 | Đoạn ngắn đơn dòng |
| `TC_SYNC_016` | **False** | **True** | P3 | **Lệch gạch đầu dòng (`- `)** ở danh sách Nguyên nhân & Giải pháp |
| `TC_SYNC_017` | **False** | **True** | P3 | **Lệch gạch đầu dòng (`- `)** ở danh sách Nguyên nhân & Giải pháp |
| `TC_SYNC_018` | **True** | **True** | P3 | Đoạn ngắn đơn dòng |
| `TC_SYNC_019` | **False** | **True (với $\ge 75\%$)** | P3 & P4 | **Chunk boundary cut-off**: Bị cắt đôi qua Parent 3 và Parent 4 |
| `TC_SYNC_020` | **True** | **True** | P3 | Đoạn ngắn đơn dòng |
| `TC_SYNC_021` | **True** | **True** | P4 | Đoạn ngắn đơn dòng |
| `TC_SYNC_022` | **False** | **True** | P4 | **Lệch gạch đầu dòng (`- `)** ở danh sách Nguyên nhân & Giải pháp |
| `TC_SYNC_023` | **False** | **True** | P4 | **Lệch gạch đầu dòng (`- `)** ở danh sách Nguyên nhân & Giải pháp |
| `TC_SYNC_024` | **True** | **True** | P4 | Đoạn ngắn đơn dòng |
| `TC_SYNC_025` | **True** | **True** | P4 | Đoạn ngắn đơn dòng |
| `TC_SYNC_026` | **True** | **True** | P4 | Đoạn ngắn đơn dòng |

---

## 3. Phân tích chi tiết các nguyên nhân gốc rễ

### Nguyên nhân 1: Lệch dấu gạch đầu dòng danh sách (`-`, `•`, `+`, `*`) (Chiếm 10/14 case thất bại)
- **Trong Database (`kb_chunks`):** Nội dung PDF gốc chứa định dạng danh sách liệt kê có dấu gạch ngang đầu dòng:
  ```text
  Nguyên nhân:
  - Mất kết nối Internet hoặc mạng nội bộ chặn các cổng kết nối của Exchange/M365.
  - Thiết lập Proxy hoặc VPN xung đột kết nối.
  Giải pháp khắc phục:
  - Bước 1: Kiểm tra kết nối mạng cục bộ...
  - Bước 2: Kiểm tra tab Send / Receive...
  ```
- **Trong file `eval/dataset/golden_dataset.json`:** Khi copy hoặc tổng hợp dữ liệu, người tạo dataset đã **lược bỏ ký tự `- `** ở đầu các dòng:
  ```text
  Nguyên nhân: \nMất kết nối Internet hoặc mạng nội bộ chặn các cổng kết nối của Exchange/M365. \nThiết lập Proxy hoặc VPN xung đột kết nối.
  Giải pháp khắc phục: \nBước 1: Kiểm tra kết nối mạng cục bộ... \nBước 2: Kiểm tra tab Send / Receive...
  ```
- **Hệ quả:** Logic cũ dùng `re.sub(r"\s+", " ", text)` chỉ thay khoảng trắng/xuống dòng thành 1 dấu cách:
  - Chuỗi chunk: `"Nguyên nhân: - Mất kết nối ... - Bước 1: ..."`
  - Chuỗi evidence: `"Nguyên nhân: Mất kết nối ... Bước 1: ..."`
  - Phép so khớp substring `evidence in chunk` lập tức trả về `False`. Khi chuẩn hóa gỡ bỏ các ký tự gạch đầu dòng danh sách, 10 case này ngay lập tức khớp thành công (`True`).

### Nguyên nhân 2: Lệch mã ký tự Unicode, Ngoặc kép xoăn (Smart Quotes), và Gạch nối
- **Ngoặc kép:** Smart quotes `“...”` (`\u201c`, `\u201d`) vs ngoặc kép thẳng tiêu chuẩn `"` (`\u0022`).
- **Gạch nối:** En-dash `–` (`\u2013`), Em-dash `—` (`\u2014`) vs Hyphen thông thường `-` (`\u002d`).
- **Unicode tiếng Việt:** Dạng tổ hợp (`NFD`) vs Dạng dựng sẵn (`NFC`). Nếu 1 bên dùng NFD còn bên kia dùng NFC, phép so sánh chuỗi của Python sẽ coi là hai chuỗi hoàn toàn khác nhau.
- **Ký tự ẩn:** Non-breaking space `\u00a0` sinh ra do quá trình trích xuất PDF qua `pdfplumber`.

### Nguyên nhân 3: Đoạn Evidence dài bị cắt đôi qua 2 Parent Chunks (Chunk Boundary Cut-off) (4 case còn lại)
*Bao gồm các test case: `TC_SYNC_008`, `TC_SYNC_009`, `TC_SYNC_012`, `TC_SYNC_019`.*

- Trong `golden_dataset.json`, evidence của các case này rất dài (khoảng 800 - 900 ký tự), gom trọn vẹn cả *Mô tả + Nguyên nhân + Toàn bộ các bước khắc phục*.
- Bộ chia đoạn `RecursiveCharacterTextSplitter` với tham số `PARENT_CHUNK_SIZE = 2000` đã ngắt sự cố `SYNC_ERR_003` thành 2 chunk:
  - **Parent 1:** Chứa Mô tả & 2 nguyên nhân đầu.
  - **Parent 2:** Chứa nguyên nhân thứ ba & các bước khắc phục.
- **Thực tế mô hình Retrieval tìm rất chính xác:** Cả Parent 1 và Parent 2 đều nằm trong **Top-1 và Top-2** của danh sách truy xuất!
- Tuy nhiên, vì hàm kiểm tra yêu cầu 100% evidence phải nằm gọn trong **một** chunk đơn lẻ, nên cả Parent 1 và Parent 2 đều bị đánh giá là không khớp.

### Lưu ý bổ sung về tập dữ liệu `golden_dataset.json`
- Hiện tại, chỉ có nhóm câu hỏi `TC_SYNC_*` (30 câu đầu) là có `evidence_text`.
- Các nhóm còn lại (`TC_PRT_*`, `TC_SFT_*`, `TC_PWD_*`, `TC_VPN_*`) trong dataset đang để chuỗi `evidence_text: ""` rỗng, cần được bổ sung nội dung trước khi chạy đánh giá toàn bộ 150 câu hỏi.

---

## 4. Giải pháp kỹ thuật triển khai

Hàm `evidence_matches_chunk` được nâng cấp với 2 cơ chế:
1. **`normalize_text_for_eval(text)`**:
   - Chuẩn hóa Unicode `NFC`.
   - Chuẩn hóa smart quotes `“”‘’` thành `'` và `"`.
   - Chuẩn hóa en-dash, em-dash thành `-`.
   - Bóc tách hoặc đồng nhất các ký tự gạch đầu dòng bullet (`-`, `•`, `+`, `*`).
   - Xóa `\u00a0` và thu gọn khoảng trắng liên tiếp.
2. **`min_overlap_ratio = 0.75`**:
   - Nếu `evidence_text` nằm trọn trong `chunk_text` $\rightarrow$ Trả về `True`.
   - Nếu evidence bị cắt qua ranh giới chunk nhưng chunk trích xuất chứa $\ge 75\%$ chuỗi liên tục của evidence $\rightarrow$ Vẫn ghi nhận kết quả chính xác cho retrieval.
