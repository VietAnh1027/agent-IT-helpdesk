from src.core.config import get_settings

settings = get_settings()
DEFAULT_CANDIDATE_POOL = 20
_model = None

def get_embedding_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        print(f"Đang load embedding model '{settings.embedding_model}' ...")
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def embed_query(text: str) -> list[float]:
    model = get_embedding_model()
    vec = model.encode([text], normalize_embeddings=True)[0]
    return vec.tolist()


# ---------------------------------------------------------------------------
# Retrieval — hybrid search: filter category + is_active, rank similarity
# trong tập đã lọc, dedupe theo parent_chunk_id -> trả về top-K PARENT.
# ---------------------------------------------------------------------------

def retrieve_top_k_parents(
    cur,
    question: str,
    category: str,
    top_k: int = settings.top_k,
    candidate_pool: int = DEFAULT_CANDIDATE_POOL,
):
    if top_k < 1 or candidate_pool < 1:
        raise ValueError("top_k and candidate_pool must be positive")
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
    """, (query_embedding, category, query_embedding, candidate_pool))

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
