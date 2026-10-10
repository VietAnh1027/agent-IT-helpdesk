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

Chạy từ thư mục gốc dự án:
    python -m eval.harness.eval_retrieval

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

import difflib
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv

from src.rag.retriever import retrieve_top_k_parents

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

GOLDEN_DATASET_PATH = Path(os.getenv("GOLDEN_DATASET_PATH", "golden_dataset.json"))
TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
CANDIDATE_POOL = int(os.getenv("RETRIEVAL_CANDIDATE_POOL", "20"))
NO_MATCH_SIMILARITY_THRESHOLD = float(os.getenv("NO_MATCH_SIMILARITY_THRESHOLD", "0.5"))

REQUIRED_ENV = ["DB_NAME", "DB_USER", "DB_PASSWORD"]
NUM_TEST = int(os.getenv("NUM_TEST", 0))


def check_env():
    missing = [k for k in REQUIRED_ENV if not os.getenv(k)]
    if missing:
        print(f"[LỖI] Thiếu biến môi trường trong .env: {', '.join(missing)}")
        sys.exit(1)
    if not GOLDEN_DATASET_PATH.exists():
        print(f"[LỖI] Không tìm thấy golden dataset: {GOLDEN_DATASET_PATH}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Evidence matching — substring containment, chunk-config-agnostic
# ---------------------------------------------------------------------------

def normalize_text_for_eval(text: str) -> str:
    """Chuẩn hóa để so khớp evidence_text và chunk_text:
    1. Unicode NFC (tránh lệch dựng sẵn / tổ hợp tiếng Việt).
    2. Đồng nhất/thay thế smart quotes, dấu nháy kép, nháy đơn thành khoảng trắng.
    3. Thay thế các loại bullet / gạch nối (- , • , + , * , – , —) thành khoảng trắng.
    4. Non-breaking space (\\u00a0) và thu gọn khoảng trắng liên tiếp.
    """
    if not text:
        return ""
    # Unicode chuẩn dựng sẵn
    text = unicodedata.normalize("NFC", text)
    # Bỏ dấu ngoặc kép / nháy đơn để tránh lệch kiểu quote
    text = re.sub(r'[\u201c\u201d\u201e\u201f\u2033\u2036"«»\'\u2018\u2019\u201a\u201b\u2032\u2035`]', " ", text)
    # Bỏ các loại bullet / gạch nối / dash
    text = re.sub(r"[-•*+–—]", " ", text)
    # Non-breaking spaces và gộp whitespace
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def evidence_matches_chunk(evidence_text: str, evidence_source_doc: str,
                            chunk_text: str, chunk_article_title: str,
                            min_overlap_ratio: float = 0.35,
                            min_matched_chars: int = 150) -> bool:
    """1 chunk được tính là chứa đúng evidence nếu:
    (a) Cùng article (source_doc).
    (b) TH1: evidence_text nằm trọn vẹn trong chunk (áp dụng cho mọi độ dài).
    (c) TH2: Đối với evidence dài bị ranh giới chunking cắt đôi, chunk phải chứa một phần
        liên tục đáng kể (>= 35% VÀ >= 150 ký tự), đồng thời phần khớp phải neo vào ĐẦU hoặc CUỐI
        của evidence (đảm bảo đúng là mảnh cắt của đoạn văn, chống bắt nhầm câu generic).
    """
    if Path(evidence_source_doc).stem != chunk_article_title:
        return False

    ev_norm = normalize_text_for_eval(evidence_text)
    chunk_norm = normalize_text_for_eval(chunk_text)

    # 1. Exact substring match sau khi chuẩn hóa
    if ev_norm in chunk_norm:
        return True

    # 2. Xử lý evidence dài bị cắt ranh giới (chống False Positive cho evidence ngắn)
    ev_len = len(ev_norm)
    if ev_len >= 200:
        # Bắt buộc autojunk=False để difflib không bỏ sót đoạn văn dài
        matcher = difflib.SequenceMatcher(None, ev_norm, chunk_norm, autojunk=False)
        match = matcher.find_longest_match(0, ev_len, 0, len(chunk_norm))
        if (match.size / ev_len >= min_overlap_ratio) and (match.size >= min_matched_chars):
            # Khóa ranh giới: Đoạn khớp phải bắt đầu từ đầu evidence HOẶC kết thúc ở cuối evidence
            is_start_piece = (match.a <= 10)
            is_end_piece = (match.a + match.size >= ev_len - 10)
            if is_start_piece or is_end_piece:
                return True

    return False


def compute_cumulative_coverage(retrieved_parents: list[dict], evidence_list: list[dict]) -> float:
    """Tính tỷ lệ % ký tự của evidence_list được bao phủ lũy kế bởi các chunk trong Top-K.
    Với mỗi evidence, hợp nhất (set union) các chỉ số ký tự được các chunk cùng source_doc phủ.
    Nếu evidence_list có nhiều phần tử, lấy trung bình độ phủ của các evidence."""
    if not evidence_list:
        return 0.0

    coverages = []
    for ev in evidence_list:
        ev_norm = normalize_text_for_eval(ev["evidence_text"])
        ev_len = len(ev_norm)
        if ev_len == 0:
            continue

        covered_indices = set()
        ev_doc_stem = Path(ev["source_doc"]).stem

        for chunk in retrieved_parents:
            if ev_doc_stem != chunk["article_title"]:
                continue
            chunk_norm = normalize_text_for_eval(chunk["text"])

            matcher = difflib.SequenceMatcher(None, ev_norm, chunk_norm, autojunk=False)
            for block in matcher.get_matching_blocks():
                # Chỉ lấy các khối khớp có độ dài >= 25 ký tự để tránh nhiễu từ ngữ đơn lẻ
                if block.size >= 25:
                    for i in range(block.a, block.a + block.size):
                        covered_indices.add(i)

        ev_coverage = min(1.0, len(covered_indices) / ev_len)
        coverages.append(ev_coverage)

    return sum(coverages) / len(coverages) if coverages else 0.0


def compute_metrics(retrieved_parents: list[dict], evidence_list: list[dict]):
    """Trả về (hit_rate, recall_coverage, ap, mrr) cho 1 sample.
    - hit_rate: 1.0 nếu có ít nhất 1 chunk liên quan trong Top-K, ngược lại 0.0.
    - recall_coverage: Tỷ lệ % ký tự/nội dung evidence được Top-K bao phủ lũy kế.
    - ap: Average Precision@K (thưởng điểm cao khi các chunk đúng nằm ở Rank 1, 2).
    - mrr: 1.0 / rank của chunk đúng đầu tiên trong Top-K.
    """
    k = len(retrieved_parents)

    # 1. Xác định relevant_chunk cho từng vị trí rank
    relevant_chunk = [False] * k
    for chunk_idx, chunk in enumerate(retrieved_parents):
        for ev in evidence_list:
            if evidence_matches_chunk(ev["evidence_text"], ev["source_doc"],
                                       chunk["text"], chunk["article_title"]):
                relevant_chunk[chunk_idx] = True
                break

    # 2. Hit Rate (nhị phân: 1.0 hoặc 0.0)
    hit_rate = 1.0 if any(relevant_chunk) else 0.0

    # 3. Recall theo Cumulative Evidence Coverage
    recall_coverage = compute_cumulative_coverage(retrieved_parents, evidence_list)

    # 4. Average Precision (AP@K)
    n_rel = sum(relevant_chunk)
    if n_rel > 0 and k > 0:
        # Số chunk target tối thiểu cần tìm (ít nhất là số evidence hoặc số relevant chunk lấy được)
        n_target = min(k, max(len(evidence_list), n_rel))
        cum_hits = 0
        sum_p = 0.0
        for rank, is_rel in enumerate(relevant_chunk, start=1):
            if is_rel:
                cum_hits += 1
                sum_p += cum_hits / rank
        ap = sum_p / n_target
    else:
        ap = 0.0 if k > 0 else None

    # 5. MRR (Reciprocal Rank của chunk đúng đầu tiên)
    mrr = 0.0
    for rank, is_rel in enumerate(relevant_chunk, start=1):
        if is_rel:
            mrr = 1.0 / rank
            break

    return hit_rate, recall_coverage, ap, mrr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

LIST_TC = []  # Đặt ID các case cần test nhanh (vd: ["TC_SYNC_008"]), hoặc để rỗng [] để chạy toàn bộ


def run_evaluation():
    check_env()

    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        all_records = json.load(f)
        if LIST_TC:
            golden_dataset = [r for r in all_records if r["id"] in LIST_TC]
        elif NUM_TEST:
            golden_dataset = all_records[:NUM_TEST]
        else:
            golden_dataset = all_records

    print(f"Đã load {len(golden_dataset)} câu hỏi từ {GOLDEN_DATASET_PATH}")

    conn = psycopg2.connect(**DB_CONFIG)
    per_sample_rows = []

    with conn.cursor() as cur:
        for record in golden_dataset:
            question = record["question"]
            category = record["category"]
            evidence_list = record.get("ground_truth_contexts", [])
            is_out_of_scope = record.get("human_handoff", False)

            retrieved = retrieve_top_k_parents(
                cur, question, category, top_k=TOP_K, candidate_pool=CANDIDATE_POOL
            )
            max_similarity = max((r["similarity"] for r in retrieved), default=0.0)

            if is_out_of_scope:
                false_retrieval = max_similarity >= NO_MATCH_SIMILARITY_THRESHOLD
                per_sample_rows.append({
                    "id": record["id"],
                    "category": category,
                    "query_type": record.get("query_type"),
                    "n_evidence": 0,
                    "hit_rate_at_k": None,
                    "recall_coverage_at_k": None,
                    "ap_at_k": None,
                    "mrr_at_k": None,
                    "max_similarity": max_similarity,
                    "out_of_scope_case": True,
                    "false_retrieval": false_retrieval,
                })
                continue

            if not evidence_list:
                print(f"  [BỎ QUA] {record['id']}: human_handoff=false nhưng "
                      f"evidence rỗng — dữ liệu không nhất quán, cần sửa golden_dataset.")
                continue

            hit_rate, recall_cov, ap, mrr = compute_metrics(retrieved, evidence_list)

            per_sample_rows.append({
                "id": record["id"],
                "category": category,
                "query_type": record.get("query_type"),
                "n_evidence": len(evidence_list),
                "hit_rate_at_k": hit_rate,
                "recall_coverage_at_k": recall_cov,
                "ap_at_k": ap,
                "mrr_at_k": mrr,
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
        print(f"Số evidence trung bình / câu hỏi:            {avg_n_evidence:.2f}")
        print(f"Hit Rate@{TOP_K}:                           {in_scope['hit_rate_at_k'].mean():.3f}")
        print(f"Recall (Cumulative Coverage)@{TOP_K}:       {in_scope['recall_coverage_at_k'].mean():.3f}")
        print(f"MAP@{TOP_K} (Average Precision):            {in_scope['ap_at_k'].mean():.3f}")
        print(f"MRR@{TOP_K}:                                {in_scope['mrr_at_k'].mean():.3f}")
        print(f"(n = {len(in_scope)} câu hỏi in-scope)")

        METRIC_COLS = ["hit_rate_at_k", "recall_coverage_at_k", "ap_at_k", "mrr_at_k"]

        print("\n--- Phân tách theo category ---")
        print(in_scope.groupby("category")[METRIC_COLS].mean().round(3))

        print("\n--- Phân tách theo query_type ---")
        print(in_scope.groupby("query_type")[METRIC_COLS].mean().round(3))

    if not out_of_scope.empty:
        false_rate = out_of_scope["false_retrieval"].mean()
        print(f"\nFalse-retrieval rate (out-of-scope): {false_rate:.3f}  "
              f"(n = {len(out_of_scope)}) — càng gần 0 càng tốt")

    out_path = os.path.join(os.path.dirname(__file__), 'results', 'eval_retrieval_results.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        df.to_csv(f, index=False)
        f.write("\n\n")
        f.write("KẾT QUẢ TỔNG HỢP,\n")
        f.write("Chỉ số,Giá trị\n")
        if not in_scope.empty:
            avg_n_evidence = in_scope["n_evidence"].mean()
            f.write(f"Số câu hỏi in-scope,{len(in_scope)}\n")
            f.write(f"Số evidence trung bình / câu hỏi,{avg_n_evidence:.2f}\n")
            f.write(f"Hit Rate@{TOP_K},{in_scope['hit_rate_at_k'].mean():.3f}\n")
            f.write(f"Recall (Cumulative Coverage)@{TOP_K},{in_scope['recall_coverage_at_k'].mean():.3f}\n")
            f.write(f"MAP@{TOP_K} (Average Precision),{in_scope['ap_at_k'].mean():.3f}\n")
            f.write(f"MRR@{TOP_K},{in_scope['mrr_at_k'].mean():.3f}\n")

        if not out_of_scope.empty:
            false_rate = out_of_scope["false_retrieval"].mean()
            f.write(f"Số câu hỏi out-of-scope,{len(out_of_scope)}\n")
            f.write(f"False-retrieval rate (out-of-scope),{false_rate:.3f}\n")

        if not in_scope.empty:
            METRIC_COLS = ["hit_rate_at_k", "recall_coverage_at_k", "ap_at_k", "mrr_at_k"]
            f.write("\nPHÂN TÁCH THEO CATEGORY,\n")
            cat_df = in_scope.groupby("category")[METRIC_COLS].mean().round(3)
            cat_df.to_csv(f)

            f.write("\nPHÂN TÁCH THEO QUERY_TYPE,\n")
            query_df = in_scope.groupby("query_type")[METRIC_COLS].mean().round(3)
            query_df.to_csv(f)

    print(f"\nĐã lưu chi tiết từng sample và kết quả tổng hợp vào {out_path}")


if __name__ == "__main__":
    run_evaluation()
