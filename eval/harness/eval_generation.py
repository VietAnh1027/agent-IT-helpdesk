"""Đánh giá generation bằng LLM và lưu file JSON kết quả.

Chạy từ thư mục gốc: python -m eval.harness.eval_generation --limit 3
Không truyền đáp án chuẩn cho generator hoặc judge groundedness.
"""

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict

from src.agent.llm_client import generate, render_prompt, build_answer, GenerationError
from src.core.config import PROJECT_ROOT, get_settings
from src.confidence import evaluate_groundedness
from src.core.database import close_pool, get_connection
from src.rag.retriever import retrieve_top_k_parents


GENERATION_MODEL = "qwen3.5:4b"
JUDGE_MODEL = "qwen3.5:2b"


class CorrectnessReport(BaseModel):
    model_config = ConfigDict(strict=True)
    verdict: Literal["correct", "incorrect", "uncertain"]
    reason: str


def call_prompt(
    model_name: str,
    prompt_name: str,
    schema=None,
    num_ctx: int = 32768,
    use_cloud_model: bool = False,
    seed: int | None = None,
    max_tokens: int | None = None,
    **values,
):
    """Generation trả text; chỉ parse và kiểm tra JSON khi judge cần schema."""
    # values là dict tham số template, ví dụ {"question": "...", "answer": "...",
    # "ground_truth_answer": "...", "ground_truth_contexts": [...]}.
    # schema=None → trả str; schema=CorrectnessReport → dict {"verdict": "correct", "reason": "..."}.
    system = render_prompt(f"{prompt_name}_system.jinja2")
    task = render_prompt(f"{prompt_name}_task.jinja2", **values)
    # Giới hạn trên bảo thủ theo byte UTF-8 để tránh cắt mất evidence.
    if len((system + task).encode("utf-8")) + 256 + 2048 > num_ctx:
        raise ValueError("Context có thể không đủ; tăng --num-ctx để giữ nguyên evidence.")

    settings = get_settings()
    actual_seed = settings.judge_seed if seed is None else seed
    actual_max_tokens = settings.judge_max_tokens if max_tokens is None else max_tokens

    response = generate(
        task,
        model_name=model_name,
        system_prompt=system,
        num_ctx=num_ctx,
        response_schema=schema.model_json_schema() if schema else None,
        use_cloud_model=use_cloud_model,
        seed=actual_seed,
        max_tokens=actual_max_tokens,
    )
    return schema.model_validate_json(response).model_dump() if schema else response


def _do_retrieval(ticket: dict, top_k: int) -> list[dict]:
    """Truy xuất tài liệu từ DB và format thành kb_contexts."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            parents = retrieve_top_k_parents(cur, ticket["question"], ticket["category"], top_k)
            if not parents:
                raise ValueError("Không tìm thấy KB hợp lệ")
    return [
        {
            "parent_chunk_id": parent["parent_chunk_id"],
            "source": parent["article_title"],
            "content": parent["text"],
        }
        for parent in parents[:top_k]
    ]


def _do_generation(
    ticket: dict,
    kb_contexts: list[dict],
    generation_model: str,
    num_ctx: int,
) -> str:
    """Sinh câu trả lời từ LLM với context KB."""
    return call_prompt(
        generation_model,
        "answer_with_cite",
        num_ctx=num_ctx,
        ticket_subject=ticket["category"],
        ticket_body=ticket["question"],
        category=ticket["category"],
        kb_contexts=kb_contexts,
    )


def _do_correctness(
    ticket: dict,
    answer: str,
    judge_model: str,
    num_ctx: int,
    use_cloud_model: bool = False,
) -> dict:
    """Gọi LLM Judge để chấm tính đúng đắn so với ground truth."""
    return call_prompt(
        judge_model,
        "eval_correctness",
        schema=CorrectnessReport,
        num_ctx=num_ctx,
        use_cloud_model=use_cloud_model,
        question=ticket["question"],
        answer=answer,
        ground_truth_answer=ticket["ground_truth_answer"],
        ground_truth_contexts=ticket["ground_truth_contexts"],
    )


def _do_groundedness(
    ticket: dict,
    answer: str,
    kb_contexts: list[dict],
    num_ctx: int,
    max_retries: int = 2,
) -> dict:
    """Gọi LLM Judge đánh giá groundedness kèm cơ chế retry khi gặp HTTP 429."""
    for attempt in range(max_retries + 1):
        grounding = evaluate_groundedness(
            ticket["question"], answer, kb_contexts, num_ctx=num_ctx,
        )
        if grounding.get("status") == "unavailable" and "429" in str(grounding.get("error")):
            if attempt < max_retries:
                wait_time = 3.0 * (attempt + 1)
                print(
                    f"[{ticket['id']}] Gặp HTTP 429 (Rate Limit), đang chờ {wait_time}s "
                    f"rồi thử lại (lần {attempt + 1}/{max_retries})..."
                )
                time.sleep(wait_time)
                continue
        return grounding
    return grounding


def evaluate_ticket(
    ticket: dict,
    top_k: int,
    num_ctx: int = 32768,
    generation_model: str = GENERATION_MODEL,
    judge_model: str = JUDGE_MODEL,
    use_cloud_correctness: bool = False,
) -> dict:
    """Retrieve -> generate -> chấm hai tiêu chí (correctness & groundedness)."""
    result = {key: ticket.get(key) for key in (
        "id", "question", "category", "query_type", "human_handoff",
        "ground_truth_answer", "ground_truth_contexts",
    )}
    result.update(kb_contexts=[], generation=None, correctness=None, groundedness=None, errors=[])

    source_output = build_answer("", [])
    stage = "retrieval"
    generated = None

    try:
        result["kb_contexts"] = _do_retrieval(ticket, top_k)
        source_output = build_answer("", result["kb_contexts"])
        stage = "generation"
        generated = _do_generation(ticket, result["kb_contexts"], generation_model, num_ctx)
        result["generation"] = source_output | {"answer": generated, "complete": True}
    except Exception as exc:
        if isinstance(exc, GenerationError):
            result["generation"] = source_output | exc.generation | {"complete": False}
        result["errors"].append({"stage": stage, "message": str(exc)})
        # Chỉ chấm khi generate đã trả về đầy đủ.
        if generated is None:
            return result

    answer = result["generation"]["answer"]
    if not answer.strip():
        if not ticket["human_handoff"]:
            result["correctness"] = {"verdict": "incorrect", "reason": "Answer rỗng."}
        return result

    # 1. Chấm Correctness (chỉ áp dụng cho ticket answerable, không phải handoff)
    if not ticket["human_handoff"]:
        stage = "correctness"
        try:
            result["correctness"] = _do_correctness(
                ticket, answer, judge_model, num_ctx, use_cloud_model=use_cloud_correctness,
            )
        except Exception as exc:
            result["errors"].append({"stage": stage, "message": str(exc)})

    # 2. Chấm Groundedness
    stage = "groundedness"
    result["groundedness"] = _do_groundedness(ticket, answer, result["kb_contexts"], num_ctx)
    if result["groundedness"]["status"] != "ok":
        result["errors"].append({"stage": stage, "message": result["groundedness"]["error"]})

    return result


def summarize(results: list[dict]) -> dict:
    """Tính tỷ lệ và xuất số lượng verdict để không che lỗi hoặc uncertain."""
    answerable = [r for r in results if not r["human_handoff"]]
    answered = [r for r in results if r["generation"] and r["generation"]["answer"].strip()]
    correctness = Counter((r["correctness"] or {}).get("verdict", "unavailable") for r in answerable)
    groundedness = Counter((r["groundedness"] or {}).get("verdict", "unavailable") for r in answered)
    assessed = sum(groundedness[v] for v in ("supported", "unsupported", "contradicted"))
    valid_grounding = [r["groundedness"] for r in results
                       if r["groundedness"] and r["groundedness"].get("status") == "ok"]
    return {
        "mean_groundedness": sum(g["score"] for g in valid_grounding) / len(valid_grounding) if valid_grounding else None,
        "groundedness_scored_answers": len(valid_grounding),
        "groundedness_unavailable_answers": len(results) - len(valid_grounding),
        "total_tickets": len(results),
        "answerable_tickets": len(answerable),
        "answer_accuracy": correctness["correct"] / len(answerable) if answerable else None,
        "correctness_counts": dict(correctness),
        "nonempty_answers": len(answered),
        "groundedness_assessed_answers": assessed,
        "ungrounded_answer_rate": (groundedness["unsupported"] + groundedness["contradicted"]) / assessed if assessed else None,
        "groundedness_counts": dict(groundedness),
        "tickets_with_errors": sum(bool(r["errors"]) for r in results),
    }


def run_evaluation(
    dataset,
    output,
    limit: int = 0,
    num_ctx: int = 32768,
    ticket_id: str | None = None,
    generation_model: str = GENERATION_MODEL,
    judge_model: str = JUDGE_MODEL,
    use_cloud_correctness: bool = False,
):
    """Chạy tuần tự; cập nhật một file JSON sau mỗi ticket để giữ tiến độ."""
    tickets = json.loads(Path(dataset).read_text(encoding="utf-8-sig"))
    if not isinstance(tickets, list) or not tickets:
        raise ValueError("Dataset phải là danh sách ticket không rỗng.")
    # Lọc ID trước limit để tìm được cả ticket nằm cuối dataset.
    if ticket_id is not None:
        tickets = [t for t in tickets if isinstance(t, dict) and t.get("id") == ticket_id]
        if not tickets:
            raise ValueError(f"Không tìm thấy ticket ID: {ticket_id}")
        if len(tickets) > 1:
            raise ValueError(f"Dataset có nhiều ticket cùng ID: {ticket_id}")
    if limit:
        tickets = tickets[:limit]
    for ticket in tickets:
        if not isinstance(ticket, dict) or not all(key in ticket for key in (
            "id", "question", "category", "human_handoff", "ground_truth_answer", "ground_truth_contexts",
        )):
            raise ValueError("Ticket thiếu trường bắt buộc.")
        if not isinstance(ticket["human_handoff"], bool):
            raise ValueError("human_handoff phải là boolean.")
        if not ticket["human_handoff"] and not (
            ticket["ground_truth_answer"] and ticket["ground_truth_contexts"]
        ):
            raise ValueError(f"{ticket['id']}: thiếu đáp án hoặc evidence chuẩn.")

    settings = get_settings()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    has_cloud = bool(
        getattr(settings, "openrouter_api_key", None)
        and settings.openrouter_api_key.get_secret_value().strip()
    )
    report = {
        "schema_version": 2,
        "generation_model": generation_model,
        "judge_model": judge_model,
        "groundedness_model": settings.openrouter_model if has_cloud else settings.groundedness_model,
        "selected_tickets": len(tickets),
        "completed": False,
        "summary": {},
        "results": [],
    }
    try:
        for index, ticket in enumerate(tickets, 1):
            result = evaluate_ticket(
                ticket,
                settings.top_k,
                num_ctx=num_ctx,
                generation_model=generation_model,
                judge_model=judge_model,
                use_cloud_correctness=use_cloud_correctness,
            )
            report["results"].append(result)
            report["summary"] = summarize(report["results"])
            report["completed"] = index == len(tickets)
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[{index}/{len(tickets)}] {ticket['id']}: {len(result['errors'])} lỗi")
    finally:
        close_pool()
    print(f"Đã lưu kết quả: {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "eval/dataset/golden_dataset.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "eval/results" / f"eval_generation_{datetime.now():%Y%m%d_%H%M%S}.json")
    parser.add_argument("--limit", type=int, default=0, help="0: toàn bộ; số dương: chạy N ticket")
    parser.add_argument("--ticket-id", help="Chỉ đánh giá ticket có ID khớp chính xác")
    parser.add_argument("--num-ctx", type=int, default=32768)
    parser.add_argument("--generation-model", default=GENERATION_MODEL, help=f"Model sinh câu trả lời (mặc định: {GENERATION_MODEL})")
    parser.add_argument("--judge-model", default=JUDGE_MODEL, help=f"Model chấm correctness (mặc định: {JUDGE_MODEL})")
    args = parser.parse_args()
    if args.limit < 0 or args.num_ctx <= 2304:
        parser.error("--limit phải >= 0; --num-ctx phải > 2304.")
    run_evaluation(
        args.dataset,
        args.output,
        limit=args.limit,
        num_ctx=args.num_ctx,
        ticket_id=args.ticket_id,
        generation_model=args.generation_model,
        judge_model=args.judge_model,
        use_cloud_correctness=False,
    )
