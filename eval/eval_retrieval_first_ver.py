"""
eval_retrieval.py — Đánh giá chất lượng retrieval của RAG bằng ragas.

Yêu cầu:
    pip install ragas sentence-transformers psycopg2-binary python-dotenv pandas

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
4. ragas NonLLMContextRecall / NonLLMContextPrecisionWithReference
   — bản "mềm" của Recall/Precision ở trên, so khớp bằng string similarity
   (Levenshtein) thay vì đúng tuyệt đối parent_index. Hữu ích khi chunking
   strategy thay đổi (parent boundary lệch đi vài ký tự) — vẫn cho điểm một
   phần thay vì tính là sai hoàn toàn như cách exact-match ở trên.
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
from pathlib import Path

import numpy as np
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
    """Chuẩn hóa khóa so khớp: (tên file KHÔNG đuôi .pdf, parent_index)."""
    stem = Path(source_doc).stem
    return (stem, int(section))


def compute_exact_match_metrics(retrieved_parents, ground_truth_contexts):
    """Trả về (recall, precision, reciprocal_rank, hit_ranks) cho 1 sample."""
    gt_keys = {
        match_key(gt["source_doc"], gt["section"])
        for gt in ground_truth_contexts
    }
    retrieved_keys = [
        (r["article_title"], r["parent_index"]) for r in retrieved_parents
    ]

    hit_flags = [key in gt_keys for key in retrieved_keys]
    n_hits = sum(hit_flags)

    recall = n_hits / len(gt_keys) if gt_keys else None
    precision = n_hits / len(retrieved_parents) if retrieved_parents else 0.0

    reciprocal_rank = 0.0
    for rank, is_hit in enumerate(hit_flags, start=1):
        if is_hit:
            reciprocal_rank = 1.0 / rank
            break

    return recall, precision, reciprocal_rank


# ---------------------------------------------------------------------------
# ragas — NonLLMContextRecall / NonLLMContextPrecisionWithReference
# (bản "mềm", so khớp bằng string similarity thay vì đúng tuyệt đối parent_index)
# ---------------------------------------------------------------------------

def run_ragas_metrics(samples: list[dict]):
    """samples: list[{"question", "retrieved_texts", "reference_texts"}]"""
    try:
        from ragas import evaluate
        from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
        from ragas.metrics import NonLLMContextPrecisionWithReference, NonLLMContextRecall
    except ImportError as e:
        print(f"[CẢNH BÁO] Không import được ragas ({e}) — bỏ qua ragas metrics, "
              f"chỉ báo cáo exact-match metrics tự tính.")
        return None

    ragas_samples = []
    for s in samples:
        if not s["reference_texts"]:
            continue  # ragas cần reference_contexts khác rỗng
        ragas_samples.append(SingleTurnSample(
            user_input=s["question"],
            retrieved_contexts=s["retrieved_texts"],
            reference_contexts=s["reference_texts"],
        ))

    if not ragas_samples:
        print("[CẢNH BÁO] Không có sample nào đủ điều kiện chạy ragas (thiếu reference_contexts).")
        return None

    dataset = EvaluationDataset(samples=ragas_samples)

    try:
        result = evaluate(
            dataset=dataset,
            metrics=[NonLLMContextRecall(), NonLLMContextPrecisionWithReference()],
        )
        return result
    except Exception as e:
        # API ragas đổi khá thường xuyên giữa các version — nếu lỗi ở đây,
        # kiểm tra lại version đã cài (`pip show ragas`) so với code mẫu
        # trong docs chính thức của ragas tại thời điểm bạn chạy.
        print(f"[LỖI] ragas.evaluate() thất bại: {e}")
        return None


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
    ragas_input_samples = []

    with conn.cursor() as cur:
        for record in golden_dataset:
            question = record["question"]
            category = record["category"]
            gt_contexts = record.get("ground_truth_contexts", [])
            retrieved = retrieve_top_k_parents(cur, question, category, TOP_K)

            if gt_contexts[0].get("section"):
                recall, precision, rr = compute_exact_match_metrics(retrieved, gt_contexts)
                max_similarity = max((r["similarity"] for r in retrieved), default=0.0)
                per_sample_rows.append({
                    "id": record["id"],
                    "category": category,
                    "query_type": record.get("query_type"),
                    "recall_at_k": recall,
                    "precision_at_k": precision,
                    "reciprocal_rank": rr,
                    "max_similarity": max_similarity,
                    "out_of_scope_case": False,
                })
                ragas_input_samples.append({
                    "question": question,
                    "retrieved_texts": [r["text"] for r in retrieved],
                    "reference_texts": [gt["text"] for gt in gt_contexts],
                })
            else:
                # Case out-of-scope / human_handoff — không có ground truth
                # context nào đúng nghĩa, đánh giá theo hướng khác (mục 6
                # trong docstring đầu file).
                max_similarity = max((r["similarity"] for r in retrieved), default=0.0)
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

    conn.close()

    df = pd.DataFrame(per_sample_rows)

    # --- Báo cáo tổng hợp ---
    in_scope = df[df["out_of_scope_case"] == False]  # noqa: E712
    out_of_scope = df[df["out_of_scope_case"] == True]  # noqa: E712

    print("\n" + "=" * 70)
    print("KẾT QUẢ TỔNG HỢP (exact-match, trên toàn bộ dataset)")
    print("=" * 70)
    if not in_scope.empty:
        print(f"Recall@{TOP_K}:    {in_scope['recall_at_k'].mean():.3f}")
        print(f"Precision@{TOP_K}: {in_scope['precision_at_k'].mean():.3f}")
        print(f"MRR@{TOP_K}:       {in_scope['reciprocal_rank'].mean():.3f}")
        print(f"(n = {len(in_scope)} câu hỏi in-scope)")

    if not out_of_scope.empty:
        false_rate = out_of_scope["false_retrieval"].mean()
        print(f"\nFalse-retrieval rate (out-of-scope): {false_rate:.3f}  "
              f"(n = {len(out_of_scope)}) — càng gần 0 càng tốt")

    if not in_scope.empty:
        print("\n--- Phân tách theo category ---")
        print(in_scope.groupby("category")[["recall_at_k", "precision_at_k", "reciprocal_rank"]]
              .mean().round(3))

        print("\n--- Phân tách theo query_type ---")
        print(in_scope.groupby("query_type")[["recall_at_k", "precision_at_k", "reciprocal_rank"]]
              .mean().round(3))

    # --- ragas metrics (bản mềm) ---
    print("\n" + "=" * 70)
    print("KẾT QUẢ ragas (NonLLMContextRecall / NonLLMContextPrecisionWithReference)")
    print("=" * 70)
    ragas_result = run_ragas_metrics(ragas_input_samples)
    if ragas_result is not None:
        print(ragas_result)

    # --- Lưu kết quả chi tiết ---
    out_path = Path("eval_retrieval_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nĐã lưu chi tiết từng sample vào {out_path.resolve()}")


if __name__ == "__main__":
    run_evaluation(30)