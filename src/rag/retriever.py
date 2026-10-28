"""Hybrid retrieval: Dense + BM25 trên child chunks, sau đó gom parent bằng RRF.

Luồng dữ liệu chính:
1. PostgreSQL trả từng child chunk cùng parent của nó và cosine similarity.
2. ``_group_parents`` gom các child cùng parent, giữ similarity cao nhất.
3. Dense và BM25 tạo hai bảng xếp hạng parent độc lập.
4. ``fuse_rankings`` hợp nhất hai bảng bằng Reciprocal Rank Fusion (RRF).
5. ``retrieve_top_k_parents`` cắt kết quả cuối còn top-k parent cho LLM.
"""

from collections import Counter
import math
import re
import unicodedata

from src.core.config import get_settings

# Số parent tối đa giữ lại ở MỖI kênh Dense và BM25 trước khi hợp nhất.
DEFAULT_CANDIDATE_POOL = 20
# Hằng số làm giảm chênh lệch giữa các vị trí rank trong công thức 1 / (RRF_K + rank).
RRF_K = 60
# Cache model trong process để không phải load model embedding cho từng ticket.
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
    """Tính một điểm BM25 cho mỗi document theo đúng thứ tự đầu vào.

    Args:
        question: Query người dùng, ví dụ ``"outlook disconnected"``.
        documents: Danh sách nội dung child chunk, ví dụ
            ``["Outlook mất kết nối", "Email bị kẹt trong Outbox"]``.

    Returns:
        ``list[float]`` có cùng số phần tử và cùng thứ tự với ``documents``.
        Ví dụ ``[0.82, 0.0]`` nghĩa là document đầu khớp query, document sau
        không chứa query term. Danh sách document rỗng trả ``[]``.

    Logic:
        - ``token_counts[i]`` đếm token của ``documents[i]``.
        - ``document_frequency[t]`` đếm số document có token ``t``.
        - Mỗi query term chỉ được tính một lần dù người dùng lặp từ.
        - ``k1=1.5`` làm tần suất từ bão hòa; ``b=0.75`` chuẩn hóa độ dài.
    """
    # Ví dụ: [Counter({"mail": 2, "lỗi": 1}), Counter({"vpn": 1})].
    token_counts = [Counter(tokenize(document)) for document in documents]
    # lengths[i] là tổng số token của documents[i], bao gồm token lặp.
    lengths = [sum(count.values()) for count in token_counts]
    average_length = sum(lengths) / len(lengths) if lengths else 0
    if not average_length:
        return [0.0] * len(documents)

    # Duyệt key của từng Counter nên mỗi term chỉ đóng góp một lần/document.
    # Ví dụ "mail" có trong 3 document thì document_frequency["mail"] == 3.
    document_frequency = Counter(term for count in token_counts for term in count)
    query_terms = set(tokenize(question))
    scores = []
    for count, length in zip(token_counts, lengths):
        score = 0.0
        for term in sorted(query_terms):
            term_frequency = count[term]
            if term_frequency:
                # Từ hiếm trong corpus có inverse_frequency lớn hơn từ phổ biến.
                inverse_frequency = math.log1p(
                    (len(documents) - document_frequency[term] + 0.5)
                    / (document_frequency[term] + 0.5)
                )
                # 2.5 = k1 + 1; 0.25 = 1 - b; k1=1.5 và b=0.75.
                score += inverse_frequency * term_frequency * 2.5 / (
                    term_frequency + 1.5 * (0.25 + 0.75 * length / average_length)
                )
        scores.append(score)
    return scores


def fuse_rankings(dense: list[dict], bm25: list[dict]) -> list[dict]:
    """Hợp nhất hai bảng xếp hạng parent bằng Reciprocal Rank Fusion.

    Args:
        dense: Parent đã xếp giảm dần theo ``similarity``. Ví dụ rút gọn::

            [{"parent_chunk_id": "p1", "similarity": 0.82},
             {"parent_chunk_id": "p2", "similarity": 0.70}]

        bm25: Parent đã xếp giảm dần theo ``bm25_score``, cùng cấu trúc dict
            với ``dense``. Một parent có thể xuất hiện ở cả hai list.

    Returns:
        List parent sắp theo ``rrf_score`` giảm dần. Mỗi dict được bổ sung::

            {"dense_rank": 1, "bm25_rank": 2,
             "rrf_score": 1/(60+1) + 1/(60+2),
             "relevance_score": 0.9919, ...}

        Rank bắt đầu từ 1. ``None`` nghĩa là parent không nằm trong candidate
        pool của kênh đó. ``relevance_score`` chuẩn hóa để parent đứng đầu cả
        hai kênh nhận 1.0.

    Raises:
        ValueError: Nếu một parent xuất hiện hai lần trong cùng một ranking.
    """
    # fused tra cứu theo parent ID. Giá trị là bản sao dict parent đầu tiên gặp,
    # sau đó được cập nhật rank/điểm khi parent xuất hiện trong kênh còn lại.
    fused = {}
    for channel, ranking in (("dense", dense), ("bm25", bm25)):
        seen = set()
        for rank, parent in enumerate(ranking, 1):
            parent_id = parent["parent_chunk_id"]
            if parent_id in seen:
                raise ValueError(f"Parent trùng trong kênh {channel}: {parent_id}")
            seen.add(parent_id)
            entry = fused.setdefault(
                parent_id, dict(parent, dense_rank=None, bm25_rank=None, rrf_score=0.0)
            )
            entry[f"{channel}_rank"] = rank
            entry["rrf_score"] += 1 / (RRF_K + rank)

    # Điểm RRF lớn nhất xảy ra khi parent có rank=1 ở cả Dense và BM25.
    best_possible_score = 2 / (RRF_K + 1)
    for parent in fused.values():
        parent["relevance_score"] = parent["rrf_score"] / best_possible_score
    # Tie-break theo ID để sort ổn định giữa các lần chạy.
    return sorted(fused.values(), key=lambda parent: (-parent["rrf_score"], parent["parent_chunk_id"]))


def _fetch_child_rows(cur, question: str, category: str):
    """Lấy mọi child chunk hợp lệ trong category cùng parent và cosine score.

    Args:
        cur: PostgreSQL cursor hỗ trợ ``execute(sql, params)`` và ``fetchall()``.
        question: Câu hỏi người dùng; được chuyển thành ``list[float]`` để so
            sánh với cột pgvector ``c.embedding``.
        category: Nhãn lọc chính xác, ví dụ ``"MAIL"``. Child, parent và article
            đều phải cùng category và đang active.

    Returns:
        Danh sách tuple 6 phần tử theo đúng thứ tự câu SELECT::

            [(parent_id, parent_index, parent_text, article_title,
              cosine_similarity, child_text), ...]

        Ví dụ:

            [(UUID("..."), 0, "Nội dung parent", "Dong_bo_email_outlook",
              0.82, "Child nói về Work Offline")]

        Một parent có thể xuất hiện ở nhiều row vì nó có nhiều child chunk.
    """
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
    return cur.fetchall()


def _group_parents(rows) -> dict[str, dict]:
    """Gom nhiều child rows về parent và giữ cosine similarity cao nhất.

    Args:
        rows: Iterable các tuple 6 phần tử do ``_fetch_child_rows`` trả về.
            Phần tử cuối ``child_text`` không được lưu ở parent; nó vẫn còn
            trong ``rows`` để ``_rank_bm25`` sử dụng sau đó.

    Returns:
        Dict dùng parent ID dạng chuỗi làm key. Ví dụ::

            {
              "p1": {
                "parent_chunk_id": "p1",
                "parent_index": 0,
                "text": "Nội dung parent",
                "article_title": "Dong_bo_email_outlook",
                "similarity": 0.82,
                "bm25_score": 0.0
              }
            }

        ``bm25_score`` khởi tạo bằng 0.0 và sẽ được ``_rank_bm25`` sửa trực
        tiếp trên cùng dict. Nếu nhiều child thuộc cùng parent, ``similarity``
        là giá trị lớn nhất trong các child đó.

    Raises:
        ValueError: Nếu cosine similarity không hữu hạn hoặc nằm ngoài [-1, 1]
        sau khi cho phép sai số floating-point rất nhỏ.
    """
    parents = {}
    for parent_id, index, text, title, similarity, _ in rows:
        similarity = float(similarity)
        if not math.isfinite(similarity) or not -1.000000001 <= similarity <= 1.000000001:
            raise ValueError("Invalid cosine similarity")

        parent_id = str(parent_id)
        entry = parents.setdefault(parent_id, {
            "parent_chunk_id": parent_id,
            "parent_index": index,
            "text": text,
            "article_title": title,
            "similarity": similarity,
            "bm25_score": 0.0,
        })
        # Nếu parent đã tồn tại, setdefault trả entry cũ; max giữ cosine cao nhất trong các child.
        entry["similarity"] = max(entry["similarity"], similarity)
    return parents


def _rank_bm25(question: str, rows, parents: dict[str, dict], candidate_pool: int) -> list[dict]:
    """Chấm BM25 ở child level rồi tạo ranking ở parent level.

    Args:
        question: Query gốc dạng chuỗi.
        rows: Các tuple child row từ ``_fetch_child_rows``; ``row[5]`` là
            ``child_text`` được đưa vào BM25.
        parents: Dict do ``_group_parents`` tạo. Hàm này thay đổi trực tiếp
            ``parents[parent_id]["bm25_score"]``.
        candidate_pool: Số parent BM25 tối đa được trả về.

    Returns:
        List chứa chính các dict nằm trong ``parents``, sắp theo
        ``bm25_score`` giảm dần. Parent có nhiều child giữ điểm child cao nhất;
        parent có điểm 0 không được đưa vào ranking.

        Ví dụ child scores ``p1=[0.4, 1.2]`` và ``p2=[0.8]`` tạo ranking
        ``[{p1, bm25_score: 1.2}, {p2, bm25_score: 0.8}]``.
    """
    # scores[i] luôn thuộc rows[i] vì bm25_scores giữ nguyên thứ tự documents.
    scores = bm25_scores(question, [row[5] for row in rows])
    for row, score in zip(rows, scores):
        parent = parents[str(row[0])]
        parent["bm25_score"] = max(parent["bm25_score"], score)
    ranked = (parent for parent in parents.values() if parent["bm25_score"] > 0)
    return sorted(ranked, key=lambda parent: (-parent["bm25_score"], parent["parent_chunk_id"]))[:candidate_pool]


def retrieve_hybrid_parents(
    cur, question: str, category: str, candidate_pool: int = DEFAULT_CANDIDATE_POOL
) -> list[dict]:
    """Chạy Dense, BM25 và RRF; trả toàn bộ tập parent đã hợp nhất.

    Args:
        cur: PostgreSQL cursor đang mở.
        question: Ticket/câu hỏi khác khoảng trắng.
        category: Category khác khoảng trắng, ví dụ ``"MAIL"``.
        candidate_pool: Số parent tối đa của từng ranking trước RRF. Đây không
            phải số context cuối cùng gửi vào LLM.

    Returns:
        ``list[dict]`` đã sắp theo RRF. Một phần tử đầy đủ có dạng::

            {
              "parent_chunk_id": "p1",
              "parent_index": 0,
              "text": "Nội dung parent",
              "article_title": "Dong_bo_email_outlook",
              "similarity": 0.82,
              "bm25_score": 1.2,
              "dense_rank": 1,
              "bm25_rank": 2,
              "rrf_score": 0.0325,
              "relevance_score": 0.9919
            }

        Hàm cố ý chưa cắt ``top_k`` để confidence có thể dùng top-1/top-2 và
        rank của toàn bộ candidate fusion.

    Raises:
        ValueError: Input rỗng hoặc candidate_pool không phải số nguyên dương.
        RuntimeError: Lỗi embedding/SQL/cosine được gắn nhãn ``Dense/corpus``;
        lỗi tính BM25 được gắn nhãn ``BM25`` và giữ exception gốc qua ``from``.
    """
    if type(candidate_pool) is not int or candidate_pool < 1:
        raise ValueError("candidate_pool must be a positive integer")
    if not question.strip() or not category.strip():
        raise ValueError("question and category must not be blank")

    try:
        rows = _fetch_child_rows(cur, question, category)
        parents = _group_parents(rows)
        # Dense parent score = cosine similarity cao nhất trong các child của parent.
        dense = sorted(
            parents.values(),
            key=lambda parent: (-parent["similarity"], parent["parent_chunk_id"]),
        )[:candidate_pool]
    except Exception as exc:
        raise RuntimeError(f"Dense/corpus retrieval failed ({type(exc).__name__})") from exc

    try:
        bm25 = _rank_bm25(question, rows, parents, candidate_pool)
    except Exception as exc:
        raise RuntimeError(f"BM25 retrieval failed ({type(exc).__name__})") from exc
    return fuse_rankings(dense, bm25)


def retrieve_top_k_parents(
    cur, question: str, category: str, top_k: int | None = None,
    candidate_pool: int = DEFAULT_CANDIDATE_POOL,
) -> list[dict]:
    """Lấy các parent RRF tốt nhất để đưa vào bước generation.

    Args:
        cur: PostgreSQL cursor đang mở.
        question: Ticket/câu hỏi dạng chuỗi.
        category: Category dùng để lọc corpus.
        top_k: Số parent cuối cùng cần trả. ``None`` dùng ``settings.top_k``.
        candidate_pool: Số candidate mỗi kênh trước khi RRF; thông thường phải
            lớn hơn hoặc bằng ``top_k`` để hai ranking có đủ lựa chọn.

    Returns:
        ``list[dict]`` cùng định dạng với ``retrieve_hybrid_parents`` nhưng chỉ
        chứa tối đa ``top_k`` phần tử đầu tiên. Nếu corpus không có kết quả thì
        trả ``[]``.

    Raises:
        ValueError: Nếu top_k không phải số nguyên dương. ``bool`` cũng bị từ
        chối dù trong Python ``bool`` là subclass của ``int``.
    """
    top_k = get_settings().top_k if top_k is None else top_k
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    return retrieve_hybrid_parents(cur, question, category, candidate_pool)[:top_k]
