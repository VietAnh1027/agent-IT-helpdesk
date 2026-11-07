"""
ingest_kb.py — Nạp PDF vào kb_articles + kb_chunks (parent-child chunking)
theo 03_DATA_MODEL.md, dùng embedding model BAAI/bge-m3.

Yêu cầu:
    pip install pdfplumber sentence-transformers psycopg2-binary python-dotenv langchain-text-splitters

.env cần thêm (ngoài các biến DB đã có trong .env.example):
    PDF_DIR=/duong/dan/toi/thu/muc/chua/pdf
    EMBEDDING_MODEL=BAAI/bge-m3
    EMBEDDING_DIM=1024          # bge-m3 luôn ra vector 1024 chiều
    EMBEDDING_BATCH_SIZE=16     # nhỏ vì GPU 4GB VRAM

Chạy:
    python ingest_kb.py

Idempotent theo content_hash (tính trên nội dung SAU KHI đã bỏ 3 dòng
metadata): file PDF không đổi nội dung thân bài -> bỏ qua, không re-embed.
Nội dung thân bài đổi -> xóa hết chunk cũ, chunk + embed lại. Metadata
(category/is_active/date_created) luôn được cập nhật mỗi lần chạy, kể cả khi
nội dung thân bài không đổi.

===============================================================================
ĐỊNH DẠNG METADATA — BẮT BUỘC 3 dòng đầu tiên của mỗi PDF, đúng thứ tự:
===============================================================================
    category: EMAIL_OUTLOOK
    is_activate: true
    date_created: 22/09/2026

- category: không còn hardcode trong script — đọc trực tiếp từ dòng này,
  mỗi PDF tự khai báo category của mình. Script quét toàn bộ *.pdf trong
  PDF_DIR, không cần danh sách file cố định nữa.
- is_activate: true/false (không phân biệt hoa thường) -> map vào
  kb_articles.is_active.
- date_created: định dạng DD/MM/YYYY -> map vào kb_articles.date_created
  (ngày tài liệu THẬT SỰ được soạn, khác với date_added — ngày record được
  thêm vào hệ thống, do DB tự set).
3 dòng này bị loại bỏ khỏi nội dung trước khi hash/chunk/embed — không xuất
hiện trong kb_articles.content hay bất kỳ kb_chunks nào.
"""

import hashlib
import os
import sys
from datetime import datetime
from pathlib import Path

import pdfplumber
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ---------------------------------------------------------------------------
# Cấu hình
# ---------------------------------------------------------------------------

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}

PDF_DIR = Path(os.getenv("PDF_DIR", "."))
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))

# 3 dòng đầu mỗi PDF phải là metadata theo đúng thứ tự này (xem docstring đầu file)
METADATA_LINE_COUNT = 3
REQUIRED_METADATA_KEYS = ["category", "is_activate", "date_created"]
DATE_CREATED_FORMAT = "%d/%m/%Y"  # vd 22/09/2026

SOURCE_VALUE = "synthetic"  # theo yêu cầu, mọi article ở bước này đều là synthetic

# Chunking params (đơn vị: ký tự, chuẩn langchain_text_splitters) —
# có thể tinh chỉnh sau khi đo Retrieval Recall@5
PARENT_CHUNK_SIZE = 2000       # ~400 từ tiếng Việt
PARENT_CHUNK_OVERLAP = 200
CHILD_CHUNK_SIZE = 400         # ~80 từ tiếng Việt
CHILD_CHUNK_OVERLAP = 75

REQUIRED_ENV = ["DB_NAME", "DB_USER", "DB_PASSWORD"]


def check_env():
    missing = [k for k in REQUIRED_ENV if not os.getenv(k)]
    if missing:
        print(f"[LỖI] Thiếu biến môi trường trong .env: {', '.join(missing)}")
        sys.exit(1)
    if not PDF_DIR.exists():
        print(f"[LỖI] PDF_DIR không tồn tại: {PDF_DIR}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Trích xuất & hash
# ---------------------------------------------------------------------------

def extract_text_from_pdf(path: Path) -> str:
    """Extract toàn bộ text của PDF, nối các trang bằng 2 dòng trống
    để giữ ranh giới trang cho bước chunking parent."""
    pages_text = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages_text.append(text.strip())
    return "\n\n".join(pages_text)


def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_metadata_and_clean(raw_text: str):
    """Tách 3 dòng metadata đầu tiên (category, is_activate, date_created)
    ra khỏi nội dung. Trả về (metadata_dict, cleaned_content).

    Raise ValueError nếu thiếu dòng, sai key, hoặc sai định dạng date —
    để ingest_one_pdf() bắt và bỏ qua file này, không âm thầm ingest
    metadata sai."""
    lines = raw_text.split("\n")
    if len(lines) < METADATA_LINE_COUNT:
        raise ValueError(f"File chỉ có {len(lines)} dòng, cần ít nhất {METADATA_LINE_COUNT} dòng metadata.")

    metadata_lines = lines[:METADATA_LINE_COUNT]
    remaining_lines = lines[METADATA_LINE_COUNT:]

    parsed = {}
    for line in metadata_lines:
        if ":" not in line:
            raise ValueError(f"Dòng metadata sai định dạng (thiếu dấu ':'): '{line}'")
        key, value = line.split(":", 1)
        parsed[key.strip().lower()] = value.strip()

    missing = [k for k in REQUIRED_METADATA_KEYS if k not in parsed]
    if missing:
        raise ValueError(f"Thiếu key metadata bắt buộc: {missing}. Đã đọc được: {list(parsed.keys())}")

    category = parsed["category"]
    if not category:
        raise ValueError("category rỗng.")

    is_active_raw = parsed["is_activate"].strip().lower()
    is_active = is_active_raw in ("true", "1", "yes")
    if is_active_raw not in ("true", "false", "1", "0", "yes", "no"):
        raise ValueError(f"is_activate không hợp lệ: '{parsed['is_activate']}' (mong đợi true/false).")

    try:
        date_created = datetime.strptime(parsed["date_created"], DATE_CREATED_FORMAT).date()
    except ValueError:
        raise ValueError(
            f"date_created '{parsed['date_created']}' không đúng định dạng {DATE_CREATED_FORMAT} (vd 22/09/2026)."
        )

    metadata = {
        "category": category,
        "is_active": is_active,
        "date_created": date_created,
    }
    cleaned_content = "\n".join(remaining_lines).strip()
    # Ví dụ output tuple: ({"category": "MAIL", "is_active": True,
    # "date_created": date(2026, 9, 22)}, "Nội dung sau metadata"). date_created là datetime.date, không phải str.
    return metadata, cleaned_content


# ---------------------------------------------------------------------------
# Parent-child chunking — dùng langchain_text_splitters thay vì tự viết logic
# chia chuỗi thủ công. RecursiveCharacterTextSplitter thử tách theo thứ tự
# separator ưu tiên (đoạn văn -> dòng -> câu -> từ) nên giữ ngữ nghĩa tốt hơn
# cắt cứng theo số từ.
# ---------------------------------------------------------------------------

_SPLIT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=PARENT_CHUNK_SIZE,
    chunk_overlap=PARENT_CHUNK_OVERLAP,
    separators=_SPLIT_SEPARATORS,
)

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHILD_CHUNK_SIZE,
    chunk_overlap=CHILD_CHUNK_OVERLAP,
    separators=_SPLIT_SEPARATORS,
)


def build_parent_child_chunks(full_text: str):
    """Trả về list[dict]: mỗi phần tử là 1 parent kèm list child của nó.
    [{ "parent_text": ..., "parent_index": 0, "children": ["...", "..."] }, ...]

    Chiến lược: parent_splitter chia văn bản gốc thành các đoạn lớn (context
    cho LLM), sau đó mỗi đoạn lớn lại được child_splitter chia tiếp thành các
    đoạn nhỏ (dùng để embed/search) — đúng pattern parent-document retriever.
    """
    parent_texts = parent_splitter.split_text(full_text)
    result = []
    for parent_index, parent_text in enumerate(parent_texts):
        child_texts = child_splitter.split_text(parent_text)
        result.append({
            "parent_text": parent_text,
            "parent_index": parent_index,
            "children": child_texts,
        })
    return result


# ---------------------------------------------------------------------------
# Embedding — BAAI/bge-m3 qua sentence-transformers
# ---------------------------------------------------------------------------

_model = None


def get_embedding_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"Đang load embedding model '{EMBEDDING_MODEL_NAME}' ...")
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        # Kiểm tra dimension khớp với EMBEDDING_DIM trong .env / schema DB —
        # tránh lỗi khó hiểu về sau nếu model/schema lệch nhau.
        test_vec = _model.encode(["test"], normalize_embeddings=True)
        actual_dim = len(test_vec[0])
        if actual_dim != EMBEDDING_DIM:
            print(
                f"[LỖI] EMBEDDING_DIM trong .env = {EMBEDDING_DIM}, "
                f"nhưng model '{EMBEDDING_MODEL_NAME}' thực tế ra {actual_dim} chiều. "
                f"Sửa lại EMBEDDING_DIM={actual_dim} trong .env và chạy lại migrate.py."
            )
            sys.exit(1)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    # Ví dụ texts=["Child A", "Child B"] → [[0.01, -0.02, ...], [0.03, 0.04, ...]].
    # Mỗi hàng là vector của texts[i]; mỗi vector có EMBEDDING_DIM phần tử float.
    if not texts:
        return []
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,  # bắt buộc để khớp cosine similarity (vector_cosine_ops)
        show_progress_bar=False,
    )
    return [vec.tolist() for vec in embeddings]


# ---------------------------------------------------------------------------
# DB logic
# ---------------------------------------------------------------------------

def find_existing_article(cur, title: str):
    cur.execute(
        "SELECT id, content_hash FROM kb_articles WHERE title = %s;",
        (title,),
    )
    # Output là (article_id, content_hash), ví dụ ("uuid-article", "a1b2..."), hoặc None nếu chưa có.
    # Đây là tuple theo thứ tự SELECT, không phải {"id": ..., "content_hash": ...}.
    return cur.fetchone()


def update_metadata_only(cur, article_id, category, is_active, date_created):
    """Cập nhật category/is_active/date_created MÀ KHÔNG đụng tới content/hash
    hay chunk. Dùng khi content_hash không đổi nhưng metadata có thể đã đổi
    (vd ai đó sửa lại category trong file PDF mà không đổi nội dung thân bài)."""
    cur.execute("""
        UPDATE kb_articles
        SET category = %s, is_active = %s, date_created = %s
        WHERE id = %s;
    """, (category, is_active, date_created, article_id))
    # category/is_active denormalize xuống kb_chunks — cập nhật luôn để
    # hybrid filter ở retrieval không bị lệch so với kb_articles.
    cur.execute("""
        UPDATE kb_chunks SET category = %s, is_active = %s
        WHERE kb_article_id = %s;
    """, (category, is_active, article_id))


def upsert_article(cur, title, category, is_active, date_created, content, content_hash, existing):
    if existing:
        article_id, _ = existing
        cur.execute("""
            UPDATE kb_articles
            SET content = %s, content_hash = %s, category = %s,
                source = %s, last_updated = now(), is_active = %s, date_created = %s
            WHERE id = %s;
        """, (content, content_hash, category, SOURCE_VALUE, is_active, date_created, article_id))
        # date_added KHÔNG bị đổi ở đây — giữ nguyên thời điểm article được
        # thêm vào hệ thống lần đầu, chỉ set 1 lần lúc INSERT.
        #
        # Xóa hết chunk cũ của article này — ON DELETE CASCADE lo phần
        # self-referencing (xóa parent -> tự xóa child), nên chỉ cần xóa
        # parent row, nhưng để chắc chắn dọn sạch cả row lẻ, xóa theo
        # kb_article_id trực tiếp là an toàn nhất.
        cur.execute("DELETE FROM kb_chunks WHERE kb_article_id = %s;", (article_id,))
        return article_id
    else:
        cur.execute("""
            INSERT INTO kb_articles (title, category, content, content_hash, source, is_active, date_created)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id;
        """, (title, category, content, content_hash, SOURCE_VALUE, is_active, date_created))
        # date_added dùng DEFAULT now() của cột trong schema — không cần truyền ở đây.
        return cur.fetchone()[0]


def insert_chunks(cur, article_id, category, parent_child_list):
    total_children = 0
    for group in parent_child_list:
        # 1. Insert parent row — embedding NULL, parent_chunk_id NULL
        cur.execute("""
            INSERT INTO kb_chunks (kb_article_id, parent_chunk_id, chunk_index, content, embedding, category, is_active)
            VALUES (%s, NULL, %s, %s, NULL, %s, true)
            RETURNING id;
        """, (article_id, group["parent_index"], group["parent_text"], category))
        parent_id = cur.fetchone()[0]

        if not group["children"]:
            continue

        # 2. Embed toàn bộ child text của parent này 1 lượt (đỡ overhead gọi model nhiều lần)
        # group ví dụ: {"parent_index": 0, "parent_text": "Đoạn lớn", "children": ["Child A", "Child B"]}.
        # zip bên dưới ghép ("Child A", vector_A), ("Child B", vector_B), giữ đúng thứ tự.
        child_embeddings = embed_texts(group["children"])

        # 3. Insert từng child row, trỏ về parent_id vừa tạo
        for child_index, (child_text, embedding) in enumerate(
            zip(group["children"], child_embeddings)
        ):
            cur.execute("""
                INSERT INTO kb_chunks (kb_article_id, parent_chunk_id, chunk_index, content, embedding, category, is_active)
                VALUES (%s, %s, %s, %s, %s, %s, true);
            """, (article_id, parent_id, child_index, child_text, embedding, category))
            total_children += 1

    return total_children


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def ingest_one_pdf(cur, path: Path):
    title = path.stem  # dùng tên file (không đuôi) làm title, khóa tra cứu idempotent
    print(f"-> Xử lý {path.name}")

    raw_text = extract_text_from_pdf(path)
    if not raw_text.strip():
        print(f"  [CẢNH BÁO] PDF rỗng hoặc không extract được text: {path.name}")
        return

    metadata, content = parse_metadata_and_clean(raw_text)
    category = metadata["category"]
    is_active = metadata["is_active"]
    date_created = metadata["date_created"]

    if not content.strip():
        print(f"  [CẢNH BÁO] Sau khi bỏ 3 dòng metadata, nội dung rỗng: {path.name}")
        return

    content_hash = compute_hash(content)
    existing = find_existing_article(cur, title)

    print(f"  category={category}, is_active={is_active}, date_created={date_created}")

    if existing and existing[1] == content_hash:
        # Nội dung thân bài không đổi -> bỏ qua re-chunk/re-embed, nhưng
        # vẫn cập nhật metadata (category/is_active/date_created có thể đổi
        # độc lập với nội dung).
        update_metadata_only(cur, existing[0], category, is_active, date_created)
        print(f"  Nội dung không đổi (hash khớp) — bỏ qua re-embed, đã cập nhật metadata.")
        return

    article_id = upsert_article(cur, title, category, is_active, date_created, content, content_hash, existing)

    parent_child_list = build_parent_child_chunks(content)
    n_parents = len(parent_child_list)
    n_children = insert_chunks(cur, article_id, category, parent_child_list)

    action = "Cập nhật lại" if existing else "Tạo mới"
    print(f"  {action}: {n_parents} parent chunk, {n_children} child chunk (đã embed).")


def run_ingest():
    check_env()
    print(f"Kết nối tới database '{DB_CONFIG['dbname']}' @ {DB_CONFIG['host']}:{DB_CONFIG['port']} ...")
    print(f"PDF_DIR = {PDF_DIR.resolve()}")

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"[CẢNH BÁO] Không tìm thấy file .pdf nào trong {PDF_DIR.resolve()}")
        return

    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False

    try:
        with conn.cursor() as cur:
            for path in pdf_files:
                try:
                    ingest_one_pdf(cur, path)
                    conn.commit()  # commit theo từng file — 1 file lỗi không kéo sập cả batch
                except Exception as e:
                    conn.rollback()
                    print(f"  [LỖI] Xử lý '{path.name}' thất bại, đã rollback riêng file này: {e}")
        print("\n✅ Ingest hoàn tất.")
    finally:
        conn.close()


if __name__ == "__main__":
    run_ingest()
