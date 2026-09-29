"""
eval_retrieval.py — Đánh giá chất lượng retrieval của RAG (không dùng ragas —
ragas đổi API liên tục giữa các version, gây lỗi tương thích khó chịu; thay
bằng metric tự viết dùng difflib, thư viện chuẩn của Python, không có rủi ro
version-lệch-API).

Yêu cầu:
    pip install sentence-transformers psycopg2-binary python-dotenv pandas

.env dùng chung với migrate.py/ingest_kb.py (DB config, EMBEDDING_MODEL,
EMBEDDING_DIM). Thêm:
    GOLDEN_DATASET_PATH=golden_dataset.json
    RETRIEVAL_TOP_K=5
    RETRIEVAL_CANDIDATE_POOL=20   # số child chunk raw lấy trước khi dedupe theo parent

Chạy:
    python eval_retrieval.py

===============================================================================
QUAN TRỌNG — cách điền golden_dataset.json để so khớp đúng
===============================================================================
Mỗi ground_truth_contexts[i] cần:
    "section": parent_index THẬT của kb_chunks (parent row) trong DB
    "text":    content THẬT của parent row đó (copy y nguyên sau khi ingest)
Vì "section"/"text" phải khớp với dữ liệu đã ingest, hãy chạy ingest_kb.py
TRƯỚC, sau đó query bảng kb_chunks (parent_chunk_id IS NULL) để lấy đúng
parent_index + content mà điền vào golden_dataset — không tự bịa ra.

===============================================================================
CÁC CHỈ SỐ ĐƯỢC TÍNH — vì sao thêm ngoài Recall@5
===============================================================================
1. Recall@5 (tự tính, exact-match theo source_doc + parent_index)
   — đúng định nghĩa cổ điển: bao nhiêu % ground-truth context nằm trong
   top-5 kết quả trả về. Đây là chỉ số chính được yêu cầu.
2. MRR@5 (Mean Reciprocal Rank, tự tính)
   — Recall@5 chỉ trả lời "có tìm thấy không", không nói "tìm thấy ở vị trí
   nào". Nếu context đúng luôn nằm ở vị trí #5 (áp chót) thay vì #1, retrieval
   vẫn "đúng" theo Recall@5 nhưng thực ra kém — MRR bắt được sự khác biệt này.
3. Precision@5 (tự tính)
   — trong 5 kết quả trả về, bao nhiêu % thực sự đúng. Recall cao mà
   Precision thấp nghĩa là hệ thống "quăng lưới rộng", tốn context window
   của LLM bằng nội dung không liên quan.
4. Soft Recall@5 / Soft Precision@5 (tự viết, dùng difflib.SequenceMatcher)
   — bản "mềm" của Recall/Precision ở trên, so khớp bằng string similarity
   thay vì đúng tuyệt đối parent_index. Hữu ích khi chunking strategy thay
   đổi (parent boundary lệch đi vài ký tự) — vẫn cho điểm một phần thay vì
   tính là sai hoàn toàn như cách exact-match ở trên. Dùng difflib (built-in
   Python, zero dependency) thay vì ragas để tránh lỗi tương thích version.
5. Phân tách theo category và query_type
   — số liệu trung bình chung dễ che giấu 1 category tệ. Tách riêng để biết
   chính xác category/loại câu hỏi nào cần cải thiện retrieval.
6. False-retrieval rate cho case out-of-scope (ground_truth_contexts rỗng)
   — với ticket cố tình ngoài phạm vi KB, "retrieval tốt" nghĩa là KHÔNG có
   kết quả nào đạt similarity cao, không phải trả về top-5 gì đó. Tính riêng,
   không gộp vào Recall@5 (sẽ làm sai lệch ý nghĩa của Recall@5).
===============================================================================
"""

import json
import os
import sys
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv

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

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
GOLDEN_DATASET_PATH = Path(os.getenv("GOLDEN_DATASET_PATH", "golden_dataset.json"))
TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
CANDIDATE_POOL = int(os.getenv("RETRIEVAL_CANDIDATE_POOL", "20"))

# Ngưỡng similarity để coi 1 kết quả retrieval là "có khả năng liên quan" —
# dùng riêng cho việc tính false-retrieval rate ở case out-of-scope.
NO_MATCH_SIMILARITY_THRESHOLD = float(os.getenv("NO_MATCH_SIMILARITY_THRESHOLD", "0.5"))

# Ngưỡng string similarity (difflib, 0-1) để coi 1 cặp retrieved/reference
# context là "match mềm" — dùng cho Soft Recall/Precision (mục 4 docstring).
SOFT_MATCH_THRESHOLD = float(os.getenv("SOFT_MATCH_THRESHOLD", "0.7"))

REQUIRED_ENV = ["DB_NAME", "DB_USER", "DB_PASSWORD"]


def check_env():
    missing = [k for k in REQUIRED_ENV if not os.getenv(k)]
    if missing:
        print(f"[LỖI] Thiếu biến môi trường trong .env: {', '.join(missing)}")
        sys.exit(1)
    if not GOLDEN_DATASET_PATH.exists():
        print(f"[LỖI] Không tìm thấy golden dataset: {GOLDEN_DATASET_PATH}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

_model = None


def get_embedding_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"Đang load embedding model '{EMBEDDING_MODEL_NAME}' ...")
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_query(text: str) -> list[float]:
    model = get_embedding_model()
    vec = model.encode([text], normalize_embeddings=True)[0]
    return vec.tolist()


# ---------------------------------------------------------------------------
# Retrieval — đúng logic hybrid search đã thiết kế trong 02_ARCHITECTURE.md:
# filter category + is_active trước, rank similarity trong tập đã lọc,
# rồi dedupe theo parent_chunk_id để trả về top-K PARENT (không phải child).
# ---------------------------------------------------------------------------

def retrieve_top_k_parents(cur, question: str, category: str, top_k: int = TOP_K):
    query_embedding = embed_query(question)

    cur.execute("""
        SELECT
            c.parent_chunk_id,
            p.chunk_index AS parent_index,
            p.content AS parent_text,
            a.title AS article_title,
            1 - (c.embedding <=> %s::vector) AS similarity
        FROM kb_chunks c
        JOIN kb_chunks p ON p.id = c.parent_chunk_id
        JOIN kb_articles a ON a.id = c.kb_article_id
        WHERE c.parent_chunk_id IS NOT NULL
          AND c.category = %s
          AND c.is_active = true
        ORDER BY c.embedding <=> %s::vector
        LIMIT %s;
    """, (query_embedding, category, query_embedding, CANDIDATE_POOL))

    rows = cur.fetchall()

    # Dedupe theo parent_chunk_id, giữ nguyên thứ tự (rank cao nhất của mỗi
    # parent xuất hiện trước), lấy đúng top_k parent riêng biệt.
    seen_parents = set()
    deduped = []
    for parent_chunk_id, parent_index, parent_text, article_title, similarity in rows:
        if parent_chunk_id in seen_parents:
            continue
        seen_parents.add(parent_chunk_id)
        deduped.append({
            "parent_chunk_id": str(parent_chunk_id),
            "parent_index": parent_index,
            "text": parent_text,
            "article_title": article_title,
            "similarity": float(similarity),
        })
        if len(deduped) >= top_k:
            break

    return deduped


# ---------------------------------------------------------------------------
# Custom exact-match metrics: Recall@5, MRR@5, Precision@5
# ---------------------------------------------------------------------------

def match_key(source_doc: str, section) -> tuple:
    # if section is None or (isinstance(section, str) and section.strip() == ""):
    #     raise ValueError(
    #         f"section rỗng cho source_doc='{source_doc}' — ground_truth_contexts "
    #         f"có thể còn placeholder chưa điền, hoặc record này lẽ ra phải để "
    #         f"ground_truth_contexts=[] (case human_handoff)."
    #     )
    stem = Path(source_doc).stem
    return (stem, int(section))


def compute_exact_match_metrics(retrieved_parents, ground_truth_contexts):
    """Trả về (recall, precision, reciprocal_rank) cho 1 sample.

    Bỏ qua TỪNG context rỗng/chưa điền riêng lẻ (không raise lỗi cho cả
    record) — 1 record multi-hop có nhiều context, vài cái chưa điền xong
    không có nghĩa toàn bộ record phải loại khỏi eval."""
    gt_keys = set()
    for gt in ground_truth_contexts:
        try:
            gt_keys.add(match_key(gt["source_doc"], gt["section"]))
        except ValueError as e:
            print(f"    [CẢNH BÁO] Bỏ qua 1 context rỗng/chưa điền: {e}")

    if not gt_keys:
        raise ValueError("Không có context nào hợp lệ trong ground_truth_contexts sau khi lọc.")

    retrieved_keys = [
        (r["article_title"], r["parent_index"]) for r in retrieved_parents
    ]

    hit_flags = [key in gt_keys for key in retrieved_keys]
    n_hits = sum(hit_flags)

    recall = n_hits / len(gt_keys) if gt_keys else None
    precision = n_hits / len(retrieved_parents) if retrieved_parents else 0.0

    # Precision@1 — top-1 accuracy: chunk đứng đầu có đúng không. Có ý nghĩa
    # hơn Precision@5 khi ground truth chỉ có 1 context đúng (trường hợp phổ
    # biến), vì range đầy đủ 0-1, không bị trần toán học như Precision@5.
    precision_at_1 = 1.0 if (hit_flags and hit_flags[0]) else 0.0

    # Precision@5 CHUẨN HÓA theo trần lý thuyết: nếu chỉ có n_gt context đúng
    # trong khi lấy về k kết quả, precision tối đa có thể đạt được là
    # min(n_gt, k) / k, KHÔNG BAO GIỜ là 1.0 trừ khi n_gt >= k. Chia lại theo
    # trần này để so sánh công bằng giữa các sample có số lượng ground truth
    # khác nhau (vd sample multi-hop 2 context vs sample 1 context).
    if retrieved_parents:
        theoretical_max = min(len(gt_keys), len(retrieved_parents)) / len(retrieved_parents)
        precision_normalized = precision / theoretical_max if theoretical_max > 0 else None
    else:
        precision_normalized = None

    reciprocal_rank = 0.0
    for rank, is_hit in enumerate(hit_flags, start=1):
        if is_hit:
            reciprocal_rank = 1.0 / rank
            break

    return recall, precision, precision_at_1, precision_normalized, reciprocal_rank


# ---------------------------------------------------------------------------
# Soft Recall/Precision — tự viết bằng difflib.SequenceMatcher (built-in
# Python, không dependency ngoài, không rủi ro lệch API giữa các version như
# ragas). So khớp bằng string similarity thay vì đúng tuyệt đối parent_index.
# ---------------------------------------------------------------------------

def text_similarity(a: str, b: str) -> float:
    """Trả về tỉ lệ giống nhau 0-1 giữa 2 đoạn text (SequenceMatcher.ratio())."""
    return SequenceMatcher(None, a, b).ratio()


def compute_soft_match_metrics(retrieved_texts: list[str], reference_texts: list[str]):
    """Soft Recall: % reference context có ít nhất 1 retrieved context giống
    đủ ngưỡng SOFT_MATCH_THRESHOLD. Soft Precision: % retrieved context giống
    đủ ngưỡng với ít nhất 1 reference context."""
    if not reference_texts:
        return None, None

    ref_matched = [
        any(text_similarity(ref, ret) >= SOFT_MATCH_THRESHOLD for ret in retrieved_texts)
        for ref in reference_texts
    ]
    soft_recall = sum(ref_matched) / len(reference_texts)

    if not retrieved_texts:
        soft_precision = 0.0
    else:
        ret_matched = [
            any(text_similarity(ref, ret) >= SOFT_MATCH_THRESHOLD for ref in reference_texts)
            for ret in retrieved_texts
        ]
        soft_precision = sum(ret_matched) / len(retrieved_texts)

    return soft_recall, soft_precision


def run_soft_match_metrics(samples: list[dict]):
    """samples: list[{"question", "retrieved_texts", "reference_texts"}]
    Trả về DataFrame per-sample soft_recall/soft_precision, hoặc None nếu
    không có sample nào đủ điều kiện."""
    rows = []
    for s in samples:
        recall, precision = compute_soft_match_metrics(s["retrieved_texts"], s["reference_texts"])
        if recall is None:
            continue
        rows.append({
            "question": s["question"],
            "soft_recall": recall,
            "soft_precision": precision,
        })

    if not rows:
        print("[CẢNH BÁO] Không có sample nào đủ điều kiện tính soft-match (thiếu reference_texts).")
        return None

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_evaluation(num_sample: int = 0):
    check_env()

    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        if not num_sample:
            golden_dataset = json.load(f)
        else:
            golden_dataset = json.load(f)[:num_sample]
    print("=========================================")
    print(f"SỐ SAMPLE ĐỂ TEST: {len(golden_dataset)}")
    print("=========================================")

    print(f"Đã load {len(golden_dataset)} câu hỏi từ {GOLDEN_DATASET_PATH}")

    conn = psycopg2.connect(**DB_CONFIG)
    per_sample_rows = []
    soft_match_input_samples = []

    with conn.cursor() as cur:
        for record in golden_dataset:
            question = record["question"]
            category = record["category"]
            gt_contexts = record.get("ground_truth_contexts", [])
            # Dùng thẳng field human_handoff làm tiêu chí chính — KHÔNG suy
            # luận qua gt_contexts[0] (crash IndexError nếu list rỗng, đúng
            # trường hợp phổ biến nhất của case human_handoff).
            is_out_of_scope = record.get("human_handoff", False)

            retrieved = retrieve_top_k_parents(cur, question, category, TOP_K)
            max_similarity = max((r["similarity"] for r in retrieved), default=0.0)

            if is_out_of_scope:
                false_retrieval = max_similarity >= NO_MATCH_SIMILARITY_THRESHOLD
                per_sample_rows.append({
                    "id": record["id"],
                    "category": category,
                    "query_type": record.get("query_type"),
                    "recall_at_k": None,
                    "precision_at_k": None,
                    "reciprocal_rank": None,
                    "max_similarity": max_similarity,
                    "out_of_scope_case": True,
                    "false_retrieval": false_retrieval,
                })
                continue

            if not gt_contexts:
                print(f"  [BỎ QUA] {record['id']}: human_handoff=false nhưng "
                      f"ground_truth_contexts rỗng — dữ liệu không nhất quán, cần sửa golden_dataset.")
                continue

            try:
                recall, precision, precision_at_1, precision_normalized, rr = \
                    compute_exact_match_metrics(retrieved, gt_contexts)
            except ValueError as e:
                print(f"  [BỎ QUA] {record['id']}: {e}")
                continue

            per_sample_rows.append({
                "id": record["id"],
                "category": category,
                "query_type": record.get("query_type"),
                "n_ground_truth": len(gt_contexts),
                "recall_at_k": recall,
                "precision_at_k": precision,
                "precision_at_1": precision_at_1,
                "precision_at_k_normalized": precision_normalized,
                "reciprocal_rank": rr,
                "max_similarity": max_similarity,
                "out_of_scope_case": False,
            })
            soft_match_input_samples.append({
                "question": question,
                "retrieved_texts": [r["text"] for r in retrieved],
                "reference_texts": [gt["text"] for gt in gt_contexts],
            })

    conn.close()

    df = pd.DataFrame(per_sample_rows)

    # --- Báo cáo tổng hợp ---
    in_scope = df[df["out_of_scope_case"] == False]  # noqa: E712
    out_of_scope = df[df["out_of_scope_case"] == True]  # noqa: E712

    print("\n" + "=" * 70)
    print("KẾT QUẢ TỔNG HỢP (exact-match, trên toàn bộ dataset)")
    print("=" * 70)
    if not in_scope.empty:
        avg_n_gt = in_scope["n_ground_truth"].mean()
        print(f"Số ground-truth context trung bình / câu hỏi: {avg_n_gt:.2f}  "
              f"(=1.0 nghĩa là toàn bộ câu hỏi single-hop, Precision@{TOP_K} bị trần {1/TOP_K:.2f})")
        print(f"Recall@{TOP_K}:              {in_scope['recall_at_k'].mean():.3f}")
        print(f"Precision@{TOP_K} (raw):     {in_scope['precision_at_k'].mean():.3f}  "
              f"<- LƯU Ý: có trần lý thuyết min(n_gt, {TOP_K})/{TOP_K}, xem cột normalized")
        print(f"Precision@{TOP_K} (chuẩn hóa theo trần): {in_scope['precision_at_k_normalized'].mean():.3f}")
        print(f"Precision@1:                 {in_scope['precision_at_1'].mean():.3f}  "
              f"<- top-1 accuracy, không bị trần, đáng tin cậy nhất khi 1 câu hỏi thường chỉ cần 1 context")
        print(f"MRR@{TOP_K}:                 {in_scope['reciprocal_rank'].mean():.3f}")
        print(f"(n = {len(in_scope)} câu hỏi in-scope)")

    if not out_of_scope.empty:
        false_rate = out_of_scope["false_retrieval"].mean()
        print(f"\nFalse-retrieval rate (out-of-scope): {false_rate:.3f}  "
              f"(n = {len(out_of_scope)}) — càng gần 0 càng tốt")

    METRIC_COLS = ["recall_at_k", "precision_at_k", "precision_at_k_normalized",
                   "precision_at_1", "reciprocal_rank"]

    if not in_scope.empty:
        print("\n--- Phân tách theo category ---")
        print(in_scope.groupby("category")[METRIC_COLS].mean().round(3))

        print("\n--- Phân tách theo query_type ---")
        print(in_scope.groupby("query_type")[METRIC_COLS].mean().round(3))

    # --- Soft-match metrics (bản mềm, string similarity) ---
    print("\n" + "=" * 70)
    print(f"KẾT QUẢ SOFT-MATCH (difflib, threshold={SOFT_MATCH_THRESHOLD})")
    print("=" * 70)
    soft_df = run_soft_match_metrics(soft_match_input_samples)
    if soft_df is not None:
        print(f"Soft Recall:    {soft_df['soft_recall'].mean():.3f}")
        print(f"Soft Precision: {soft_df['soft_precision'].mean():.3f}")
        print(f"(n = {len(soft_df)} câu hỏi)")

    # --- Lưu kết quả chi tiết ---
    out_path = Path("eval_retrieval_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nĐã lưu chi tiết từng sample vào {out_path.resolve()}")


if __name__ == "__main__":
    run_evaluation(30)