from src.rag.retriever import retrieve_top_k_parents
from src.agent.llm_client import render_prompt, generate, build_answer, GenerationError
from src.core.config import get_settings
import json
import psycopg2

settings = get_settings()
db_config = settings.db_connect_kwargs

try:
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()

    print("="*40 ,"HUMAN PHASE", "="*40)
    user_problem = input("Input your ticket: ")
    cate_problem = input("Category: ")
    chunks = retrieve_top_k_parents(cur, user_problem, cate_problem, settings.top_k)
    if not chunks:
        raise ValueError("Không tìm thấy KB hợp lệ; chuyển IT xử lý.")
    # Ví dụ context: [{"parent_chunk_id": "p1", "source": "mail", "content": "Tắt Outlook."}].
    # Context chứa nội dung cho LLM; sources trong output chỉ chứa ID/title để truy vết.
    kb_contexts = [
        {"parent_chunk_id": chunk["parent_chunk_id"], "source": chunk["article_title"], "content": chunk["text"]}
        for chunk in chunks
    ]
    system_prompt = render_prompt("answer_with_cite_system.jinja2")
    task_prompt = render_prompt(
        "answer_with_cite_task.jinja2",
        ticket_subject=cate_problem,
        ticket_body=user_problem,
        category=cate_problem,
        kb_contexts=kb_contexts,
    )

    answer = build_answer("", kb_contexts)
    try:
        response = generate(task_prompt, system_prompt=system_prompt)
        answer.update(answer=response, complete=True)
    except GenerationError as exc:
        answer.update(exc.generation, complete=False)
        answer["error"] = str(exc)
    print("="*40 ,"AI PHASE", "="*40)
    print("AI Message:", json.dumps(answer, ensure_ascii=False))

except (KeyError, ValueError) as e:
    print("Có lỗi xảy ra, lỗi: ", e)
