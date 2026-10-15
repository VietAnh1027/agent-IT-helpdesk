"""Truy xuất child bằng Dense + BM25, gom parent rồi hợp nhất RRF."""

from collections import Counter
import math
import re
import unicodedata

from src.core.config import get_settings

DEFAULT_CANDIDATE_POOL = 20
RRF_K = 60
_model = None


def get_embedding_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(get_settings().embedding_model, local_files_only=True)
    return _model


def embed_query(text: str) -> list[float]:
    return get_embedding_model().encode([text], normalize_embeddings=True)[0].tolist()


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())


def bm25_scores(question: str, documents: list[str]) -> list[float]:
    """BM25 IDF dương; từ lặp trong query không tăng trọng số."""
    counts = [Counter(tokenize(document)) for document in documents]
    lengths = [sum(count.values()) for count in counts]
    average = sum(lengths) / len(lengths) if lengths else 0
    if not average:
        return [0.0] * len(documents)
    frequency = Counter(term for count in counts for term in count)
    terms = set(tokenize(question))
    scores = []
    for count, length in zip(counts, lengths):
        score = 0.0
        for term in sorted(terms):
            tf = count[term]
            if tf:
                idf = math.log1p((len(documents) - frequency[term] + 0.5) / (frequency[term] + 0.5))
                score += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * length / average))
        scores.append(score)
    return scores


def fuse_rankings(dense: list[dict], bm25: list[dict]) -> list[dict]:
    """Nhận hai bảng parent đã dedupe; rank bắt đầu từ 1."""
    fused = {}
    for channel, ranking in (("dense", dense), ("bm25", bm25)):
        seen = set()
        for rank, parent in enumerate(ranking, 1):
            parent_id = parent["parent_chunk_id"]
            if parent_id in seen:
                raise ValueError(f"Parent trùng trong kênh {channel}: {parent_id}")
            seen.add(parent_id)
            entry = fused.setdefault(parent_id, dict(parent, dense_rank=None, bm25_rank=None, rrf_score=0.0))
            entry[f"{channel}_rank"] = rank
            entry["rrf_score"] += 1 / (RRF_K + rank)
    for entry in fused.values():
        entry["relevance_score"] = entry["rrf_score"] / (2 / (RRF_K + 1))
    return sorted(fused.values(), key=lambda p: (-p["rrf_score"], p["parent_chunk_id"]))


def retrieve_hybrid_parents(cur, question: str, category: str,
                            candidate_pool: int = DEFAULT_CANDIDATE_POOL) -> list[dict]:
    """Trả toàn bộ fusion để tính margin trước khi cắt context top_k."""
    if type(candidate_pool) is not int or candidate_pool < 1:
        raise ValueError("candidate_pool must be a positive integer")
    if not question.strip() or not category.strip():
        raise ValueError("question and category must not be blank")
    try:
        embedding = embed_query(question)
        # shortcut: quét corpus đã lọc; chuyển sang index BM25/ANN khi KB lớn và latency tăng.
        cur.execute("""
            SELECT c.parent_chunk_id, p.chunk_index, p.content, a.title,
                   1 - (c.embedding <=> %s::vector), c.content
            FROM kb_chunks c
            JOIN kb_chunks p ON p.id = c.parent_chunk_id
            JOIN kb_articles a ON a.id = c.kb_article_id
            WHERE c.parent_chunk_id IS NOT NULL AND c.embedding IS NOT NULL
              AND p.parent_chunk_id IS NULL AND p.kb_article_id = c.kb_article_id
              AND c.category = %s AND p.category = %s AND a.category = %s
              AND c.is_active = true AND p.is_active = true AND a.is_active = true
        """, (embedding, category, category, category))
        rows = cur.fetchall()
        parents = {}
        for parent_id, index, text, title, similarity, _ in rows:
            similarity = float(similarity)
            if not math.isfinite(similarity) or not -1.000000001 <= similarity <= 1.000000001:
                raise ValueError("Invalid cosine similarity")
            parent_id = str(parent_id)
            parent = parents.setdefault(parent_id, {
                "parent_chunk_id": parent_id, "parent_index": index, "text": text,
                "article_title": title, "similarity": similarity, "bm25_score": 0.0,
            })
            parent["similarity"] = max(parent["similarity"], similarity)
        dense = sorted(parents.values(), key=lambda p: (-p["similarity"], p["parent_chunk_id"]))[:candidate_pool]
    except Exception as exc:
        raise RuntimeError(f"Dense/corpus retrieval failed ({type(exc).__name__})") from exc
    try:
        lexical_scores = bm25_scores(question, [row[5] for row in rows])
        for row, score in zip(rows, lexical_scores):
            parent = parents[str(row[0])]
            parent["bm25_score"] = max(parent["bm25_score"], score)
        bm25 = sorted((p for p in parents.values() if p["bm25_score"] > 0),
                      key=lambda p: (-p["bm25_score"], p["parent_chunk_id"]))[:candidate_pool]
    except Exception as exc:
        raise RuntimeError(f"BM25 retrieval failed ({type(exc).__name__})") from exc
    return fuse_rankings(dense, bm25)


def retrieve_top_k_parents(cur, question: str, category: str, top_k: int | None = None,
                           candidate_pool: int = DEFAULT_CANDIDATE_POOL):
    top_k = get_settings().top_k if top_k is None else top_k
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    return retrieve_hybrid_parents(cur, question, category, candidate_pool)[:top_k]
