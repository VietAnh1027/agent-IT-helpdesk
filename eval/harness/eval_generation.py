"""Đánh giá generation bằng Qwen local và lưu một file JSON kết quả.

Chạy từ thư mục gốc: python -m eval.harness.eval_generation --limit 3
Không truyền đáp án chuẩn cho generator hoặc judge groundedness.
"""

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from src.agent.llm_client import generate, render_prompt, build_answer
from src.core.config import PROJECT_ROOT, get_settings
from src.core.database import close_pool, get_connection
from src.rag.retriever import retrieve_top_k_parents


GENERATION_MODEL = "qwen3.5:4b"
JUDGE_MODEL = "qwen3.5:4b"


class CorrectnessReport(BaseModel):
    model_config = ConfigDict(strict=True)
    verdict: Literal["correct", "incorrect", "uncertain"]
    reason: str


class GroundednessReport(BaseModel):
    model_config = ConfigDict(strict=True)
    verdict: Literal["supported", "unsupported", "contradicted", "uncertain"]
    reason: str
    problematic_claims: list[str]
    citation_support: Literal["supported", "unsupported", "uncertain"]


def call_prompt(model_name, prompt_name, schema=None, num_ctx=32768, **values):
    """Generation trả text; chỉ parse và kiểm tra JSON khi judge cần schema."""
    system = render_prompt(f"{prompt_name}_system.jinja2")
    task = render_prompt(f"{prompt_name}_task.jinja2", **values)
    # Giới hạn trên bảo thủ theo byte UTF-8 để tránh cắt mất evidence.
    # Context gồm input, token đặc biệt và output; không chia nhỏ parent chunk.
    if len((system + task).encode("utf-8")) + 256 + 2048 > num_ctx:
        raise ValueError("Context có thể không đủ; tăng --num-ctx để giữ nguyên evidence.")
    response = generate(task, model_name=model_name, system_prompt=system,
                        num_ctx=num_ctx, response_schema=schema.model_json_schema() if schema else None)
    return schema.model_validate_json(response).model_dump() if schema else response


def evaluate_ticket(ticket, top_k, num_ctx=32768):
    """Retrieve -> generate -> chấm hai tiêu chí; lỗi được giữ theo giai đoạn."""
    result = {key: ticket.get(key) for key in (
        "id", "question", "category", "query_type", "human_handoff",
        "ground_truth_answer", "ground_truth_contexts",
    )}
    result.update(kb_contexts=[], generation=None, correctness=None, groundedness=None, errors=[])
    stage = "retrieval"
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                parents = retrieve_top_k_parents(cur, ticket["question"], ticket["category"], top_k)
        result["kb_contexts"] = [
            {"source": parent["article_title"], "content": parent["text"]}
            for parent in parents
        ]
        stage = "generation"
        generated = call_prompt(
            GENERATION_MODEL, "answer_with_cite", num_ctx=num_ctx,
            ticket_subject=ticket["category"], ticket_body=ticket["question"],
            category=ticket["category"], kb_contexts=result["kb_contexts"],
        )
        result["generation"] = build_answer(generated, result["kb_contexts"])
    except Exception as exc:
        result["errors"].append({"stage": stage, "message": str(exc)})
        return result

    answer = result["generation"]["answer"]
    if not answer.strip():
        # Answer rỗng trên ticket answerable là thất bại; không cần gọi judge.
        if not ticket["human_handoff"]:
            result["correctness"] = {"verdict": "incorrect", "reason": "Answer rỗng."}
        return result

    jobs = []
    if not ticket["human_handoff"]:
        jobs.append(("correctness", CorrectnessReport, {
            "question": ticket["question"], "answer": answer,
            "ground_truth_answer": ticket["ground_truth_answer"],
            "ground_truth_contexts": ticket["ground_truth_contexts"],
        }))
    # Groundedness chỉ thấy evidence generator thực sự nhận; không thấy đáp án
    # chuẩn. Mỗi invoke là request mới, không mang lịch sử của lần chấm trước.
    jobs.append(("groundedness", GroundednessReport, {
        "question": ticket["question"], "answer": answer,
        "citations": result["generation"]["citations"], "kb_contexts": result["kb_contexts"],
    }))
    for stage, schema, values in jobs:
        try:
            result[stage] = call_prompt(JUDGE_MODEL, f"eval_{stage}", schema, num_ctx=num_ctx, **values)
        except Exception as exc:
            result["errors"].append({"stage": stage, "message": str(exc)})
    return result


def summarize(results):
    """Tính tỷ lệ và xuất số lượng verdict để không che lỗi hoặc uncertain."""
    answerable = [r for r in results if not r["human_handoff"]]
    answered = [r for r in results if r["generation"] and r["generation"]["answer"].strip()]
    correctness = Counter((r["correctness"] or {}).get("verdict", "unavailable") for r in answerable)
    groundedness = Counter((r["groundedness"] or {}).get("verdict", "unavailable") for r in answered)
    assessed = sum(groundedness[v] for v in ("supported", "unsupported", "contradicted"))
    return {
        "total_tickets": len(results), "answerable_tickets": len(answerable),
        # Lỗi và uncertain vẫn nằm trong mẫu số accuracy; không loại để tăng điểm.
        "answer_accuracy": correctness["correct"] / len(answerable) if answerable else None,
        "correctness_counts": dict(correctness),
        "nonempty_answers": len(answered), "groundedness_assessed_answers": assessed,
        "ungrounded_answer_rate": (groundedness["unsupported"] + groundedness["contradicted"]) / assessed if assessed else None,
        "groundedness_counts": dict(groundedness),
        "tickets_with_errors": sum(bool(r["errors"]) for r in results),
    }


def run_evaluation(dataset, output, limit=0, num_ctx=32768):
    """Chạy tuần tự; cập nhật một file JSON sau mỗi ticket để giữ tiến độ."""
    tickets = json.loads(Path(dataset).read_text(encoding="utf-8-sig"))
    if not isinstance(tickets, list) or not tickets:
        raise ValueError("Dataset phải là danh sách ticket không rỗng.")
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
    report = {"generation_model": GENERATION_MODEL, "judge_model": JUDGE_MODEL,
              "selected_tickets": len(tickets), "completed": False, "summary": {}, "results": []}
    try:
        for index, ticket in enumerate(tickets, 1):
            result = evaluate_ticket(ticket, settings.top_k, num_ctx)
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
    parser.add_argument("--num-ctx", type=int, default=32768)
    args = parser.parse_args()
    if args.limit < 0 or args.num_ctx <= 2304:
        parser.error("--limit phải >= 0; --num-ctx phải > 2304.")
    run_evaluation(args.dataset, args.output, args.limit, args.num_ctx)
