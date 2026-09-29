"""
eval_retrieval.py — Đánh giá retrieval bằng 3 chỉ số: Recall@K, Precision@K, MRR.
So khớp bằng EVIDENCE SUBSTRING, không dùng parent_index — nhờ vậy 1 golden
dataset dùng được để so sánh NHIỀU chiến lược chunking khác nhau, không cần
gán nhãn lại mỗi lần đổi chunk size/overlap.

Yêu cầu:
    pip install sentence-transformers psycopg2-binary python-dotenv pandas

.env dùng chung với migrate.py/ingest_kb.py (DB config, EMBEDDING_MODEL,
EMBEDDING_DIM). Thêm:
    GOLDEN_DATASET_PATH=golden_dataset.json
    RETRIEVAL_TOP_K=5
    RETRIEVAL_CANDIDATE_POOL=20
    NO_MATCH_SIMILARITY_THRESHOLD=0.5

Chạy:
    python eval_retrieval.py

===============================================================================
SCHEMA golden_dataset.json — mỗi entry:
===============================================================================
{
  "id": "TC_SYNC_026",
  "category": "EMAIL_OUTLOOK",
  "query_type": "in_domain_single_hop",
  "question": "...",
  "ground_truth_answer": "... (không dùng cho retrieval eval, giữ cho answer eval sau này)",
  "evidence": [
    { "source_doc": "Dong_bo_email_outlook.pdf", "evidence_text": "ĐOẠN NGUYÊN VĂN NGẮN NHẤT chứa đủ thông tin trả lời, copy-paste y nguyên từ PDF, KHÔNG cắt giữa (chỉ được cắt đầu/cuối)" }
  ],
  "human_handoff": false
}
Case out-of-scope: "human_handoff": true, "evidence": [].

===============================================================================
CÁCH CHẤM: substring containment, không phải similarity mờ
===============================================================================
1 chunk được tính là "chứa đúng evidence" nếu evidence_text (sau khi chuẩn
hóa khoảng trắng) là 1 substring liên tục của content chunk đó, VÀ chunk đó
thuộc đúng source_doc. Deterministic, không cần threshold tùy chỉnh, và
KHÔNG phụ thuộc chunk boundary cụ thể — nếu chiến lược chunk cắt đứt evidence
ra làm 2 mảnh, kết quả đúng là miss (chunking đó thực sự có vấn đề).

===============================================================================
3 CHỈ SỐ — không tính Hit@K vì nó là tập con thông tin của Recall@K
(khi 1 câu hỏi chỉ có 1 evidence, Recall@K tự động trùng Hit@K; khi
multi-hop, Recall@K cho nhiều thông tin hơn Hit@K) — xem thảo luận trước.
1. Recall@K    = số evidence được match / tổng số evidence của câu hỏi
2. Precision@K = số chunk trong top-K có chứa >=1 evidence / K
                 (kèm bản chuẩn hóa theo trần lý thuyết min(n_evidence, K)/K,
                  vì Precision@K thô bị trần thấp khi n_evidence=1)
3. MRR         = 1 / rank của chunk đầu tiên trong top-K có chứa evidence
===============================================================================
"""

import json
import os
import re
import sys
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
# Retrieval — hybrid search: filter category + is_active, rank similarity
# trong tập đã lọc, dedupe theo parent_chunk_id -> trả về top-K PARENT.
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
# Evidence matching — substring containment, chunk-config-agnostic
# ---------------------------------------------------------------------------

def normalize_whitespace(text: str) -> str:
    """Gộp mọi chuỗi khoảng trắng/xuống dòng thành 1 space, strip 2 đầu.
    Cần thiết vì pdfplumber có thể tách dòng khác với cách evidence_text
    được copy-paste thủ công."""
    return re.sub(r"\s+", " ", text).strip()


def evidence_matches_chunk(evidence_text: str, evidence_source_doc: str,
                            chunk_text: str, chunk_article_title: str) -> bool:
    """1 chunk được tính là chứa đúng evidence nếu (a) cùng article,
    (b) evidence_text (đã chuẩn hóa) là substring của chunk_text (đã chuẩn hóa)."""
    if Path(evidence_source_doc).stem != chunk_article_title:
        return False
    return normalize_whitespace(evidence_text) in normalize_whitespace(chunk_text)


def compute_metrics(retrieved_parents: list[dict], evidence_list: list[dict]):
    """Trả về (recall, precision, precision_normalized, reciprocal_rank)
    cho 1 sample. evidence_list: list[{"source_doc", "evidence_text"}]."""
    n_evidence = len(evidence_list)

    # matched_evidence[i] = True nếu evidence thứ i được tìm thấy ở >=1 chunk trong top-K
    matched_evidence = [False] * n_evidence
    # relevant_chunk[j] = True nếu chunk thứ j (rank j) chứa >=1 evidence
    relevant_chunk = [False] * len(retrieved_parents)

    for chunk_idx, chunk in enumerate(retrieved_parents):
        for ev_idx, ev in enumerate(evidence_list):
            if evidence_matches_chunk(ev["evidence_text"], ev["source_doc"],
                                       chunk["text"], chunk["article_title"]):
                matched_evidence[ev_idx] = True
                relevant_chunk[chunk_idx] = True

    recall = sum(matched_evidence) / n_evidence if n_evidence else None
    precision = sum(relevant_chunk) / len(retrieved_parents) if retrieved_parents else 0.0

    if retrieved_parents:
        theoretical_max = min(n_evidence, len(retrieved_parents)) / len(retrieved_parents)
        precision_normalized = precision / theoretical_max if theoretical_max > 0 else None
    else:
        precision_normalized = None

    reciprocal_rank = 0.0
    for rank, is_relevant in enumerate(relevant_chunk, start=1):
        if is_relevant:
            reciprocal_rank = 1.0 / rank
            break

    return recall, precision, precision_normalized, reciprocal_rank


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_evaluation():
    check_env()

    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        golden_dataset = json.load(f)

    print(f"Đã load {len(golden_dataset)} câu hỏi từ {GOLDEN_DATASET_PATH}")

    conn = psycopg2.connect(**DB_CONFIG)
    per_sample_rows = []

    with conn.cursor() as cur:
        for record in golden_dataset:
            question = record["question"]
            category = record["category"]
            evidence_list = record.get("evidence", [])
            is_out_of_scope = record.get("human_handoff", False)

            retrieved = retrieve_top_k_parents(cur, question, category, TOP_K)
            max_similarity = max((r["similarity"] for r in retrieved), default=0.0)

            if is_out_of_scope:
                false_retrieval = max_similarity >= NO_MATCH_SIMILARITY_THRESHOLD
                per_sample_rows.append({
                    "id": record["id"],
                    "category": category,
                    "query_type": record.get("query_type"),
                    "n_evidence": 0,
                    "recall_at_k": None,
                    "precision_at_k": None,
                    "precision_at_k_normalized": None,
                    "reciprocal_rank": None,
                    "max_similarity": max_similarity,
                    "out_of_scope_case": True,
                    "false_retrieval": false_retrieval,
                })
                continue

            if not evidence_list:
                print(f"  [BỎ QUA] {record['id']}: human_handoff=false nhưng "
                      f"evidence rỗng — dữ liệu không nhất quán, cần sửa golden_dataset.")
                continue

            recall, precision, precision_normalized, rr = compute_metrics(retrieved, evidence_list)

            per_sample_rows.append({
                "id": record["id"],
                "category": category,
                "query_type": record.get("query_type"),
                "n_evidence": len(evidence_list),
                "recall_at_k": recall,
                "precision_at_k": precision,
                "precision_at_k_normalized": precision_normalized,
                "reciprocal_rank": rr,
                "max_similarity": max_similarity,
                "out_of_scope_case": False,
            })

    conn.close()

    df = pd.DataFrame(per_sample_rows)
    in_scope = df[df["out_of_scope_case"] == False]  # noqa: E712
    out_of_scope = df[df["out_of_scope_case"] == True]  # noqa: E712

    print("\n" + "=" * 70)
    print("KẾT QUẢ TỔNG HỢP")
    print("=" * 70)

    if not in_scope.empty:
        avg_n_evidence = in_scope["n_evidence"].mean()
        print(f"Số evidence trung bình / câu hỏi: {avg_n_evidence:.2f}  "
              f"(=1.0 nghĩa là toàn bộ single-hop, Precision@{TOP_K} bị trần {1/TOP_K:.2f})")
        print(f"Recall@{TOP_K}:                          {in_scope['recall_at_k'].mean():.3f}")
        print(f"Precision@{TOP_K} (raw):                 {in_scope['precision_at_k'].mean():.3f}  "
              f"<- có trần lý thuyết, xem cột normalized")
        print(f"Precision@{TOP_K} (chuẩn hóa theo trần): {in_scope['precision_at_k_normalized'].mean():.3f}")
        print(f"MRR@{TOP_K}:                              {in_scope['reciprocal_rank'].mean():.3f}")
        print(f"(n = {len(in_scope)} câu hỏi in-scope)")

        METRIC_COLS = ["recall_at_k", "precision_at_k", "precision_at_k_normalized", "reciprocal_rank"]

        print("\n--- Phân tách theo category ---")
        print(in_scope.groupby("category")[METRIC_COLS].mean().round(3))

        print("\n--- Phân tách theo query_type ---")
        print(in_scope.groupby("query_type")[METRIC_COLS].mean().round(3))

    if not out_of_scope.empty:
        false_rate = out_of_scope["false_retrieval"].mean()
        print(f"\nFalse-retrieval rate (out-of-scope): {false_rate:.3f}  "
              f"(n = {len(out_of_scope)}) — càng gần 0 càng tốt")

    out_path = Path("eval_retrieval_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nĐã lưu chi tiết từng sample vào {out_path.resolve()}")


if __name__ == "__main__":
    run_evaluation()
