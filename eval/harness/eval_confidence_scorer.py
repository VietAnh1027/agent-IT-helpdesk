"""Chạy golden dataset, lưu confidence và chỗ điền đánh giá thủ công.

python -m eval.harness.eval_confidence_scorer --limit 3
Không gọi correctness judge; golden chỉ được lưu để người đọc đối chiếu.
"""

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path

from eval.harness.eval_generation import GENERATION_MODEL, call_prompt
from src.agent.llm_client import GenerationError, build_answer
from src.confidence import (
    CONFIDENCE_WEIGHTS, calculate_confidence, calculate_consensus,
    calculate_retrieval_confidence, evaluate_groundedness, unavailable_groundedness,
)
from src.core.config import PROJECT_ROOT, get_settings
from src.core.database import close_pool, get_connection
from src.rag.retriever import DEFAULT_CANDIDATE_POOL, RRF_K, retrieve_hybrid_parents


def evaluate_ticket(ticket, top_k, num_ctx):
    """Giữ toàn bộ fusion để tính margin, chỉ gửi top_k context cho LLM."""
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k phải là số nguyên dương")
    result = {
        "ticket_id": ticket["id"], "question": ticket["question"], "category": ticket["category"],
        "reference": {key: ticket.get(key) for key in (
            "human_handoff", "ground_truth_answer", "ground_truth_contexts")},
        "retrieval": {"parents": [], "calculation": None}, "generation": None,
        "groundedness": None, "confidence": None,
        "assessment": {"human_review": {"verdict": None, "reason": ""}}, "errors": [],
    }
    retrieval_score = consensus_score = None
    stage = "retrieval"
    complete = False
    grounding = unavailable_groundedness("Chưa có answer hoàn chỉnh")
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                parents = retrieve_hybrid_parents(cur, ticket["question"], ticket["category"])
        if not parents:
            raise ValueError("Không tìm thấy KB hợp lệ")
        retrieval_score = calculate_retrieval_confidence(parents)
        consensus_score = calculate_consensus(parents[0])
        top1 = parents[0]["relevance_score"]
        top2 = parents[1]["relevance_score"] if len(parents) > 1 else None
        result["retrieval"]["calculation"] = {
            "s_top1": top1, "s_top2": top2, "margin": top1 - top2 if top2 is not None else 0.0,
            "retrieval_score": retrieval_score, "consensus_score": consensus_score,
            "consensus_reason": {1.0: "Top-3 cả hai kênh", 0.75: "Top-5 cả hai kênh",
                                 0.5: "Không đạt đồng thuận Top-5 cả hai kênh"}[consensus_score],
        }
        for index, parent in enumerate(parents):
            entry = {key: value for key, value in parent.items() if key not in ("text", "article_title")}
            entry.update(source=parent["article_title"], used_for_generation=index < top_k)
            if index < top_k:
                entry["content"] = parent["text"]
            result["retrieval"]["parents"].append(entry)
        contexts = [{"source": p["article_title"], "content": p["text"]} for p in parents[:top_k]]
        stage = "generation"
        answer = call_prompt(GENERATION_MODEL, "answer_with_cite", num_ctx=num_ctx,
                             ticket_subject=ticket["category"], ticket_body=ticket["question"],
                             category=ticket["category"], kb_contexts=contexts)
        if not answer.strip():
            raise ValueError("Answer rỗng")
        complete = True
        result["generation"] = {"answer": answer, "citations": [], "complete": True}
        try:
            result["generation"].update(build_answer(answer, contexts))
        except GenerationError as exc:
            result["generation"].update(exc.generation)
            result["errors"].append({"stage": stage, "message": str(exc)})
        stage = "groundedness"
        grounding = evaluate_groundedness(ticket["question"], answer, contexts, num_ctx=num_ctx)
        if grounding["status"] != "ok":
            result["errors"].append({"stage": stage, "message": grounding["error"]})
    except Exception as exc:
        # Không chép raw lỗi SDK/DB có thể chứa credential vào JSON.
        message = str(exc) if isinstance(exc, GenerationError) else type(exc).__name__
        result["errors"].append({"stage": stage, "message": message})
        if isinstance(exc, GenerationError):
            result["generation"] = exc.generation | {"complete": complete}
        grounding = unavailable_groundedness(f"{stage}: {message}")
        if stage == "retrieval":
            retrieval_score = consensus_score = None

    result["groundedness"] = grounding | {
        "supported_claim_count": sum(c["verdict"] == "supported" for c in grounding["claims"]),
        "total_claim_count": len(grounding["claims"]),
    }
    confidence = calculate_confidence(retrieval_score, consensus_score, grounding)
    components = {name: confidence[f"c_{name}"] for name in CONFIDENCE_WEIGHTS}
    result["confidence"] = {
        "status": confidence["status"], "components": components,
        "weighted_contributions": {name: value * CONFIDENCE_WEIGHTS[name] if value is not None else None
                                   for name, value in components.items()},
        "score": confidence["c_rag"], "errors": confidence["errors"],
    }
    return result


def summarize(results):
    """Chỉ lấy status ok vào trung bình; không che số ticket bị lỗi."""
    counts = Counter(r["confidence"]["status"] for r in results)
    valid = [r["confidence"] for r in results if r["confidence"]["status"] == "ok"]
    means = {name: sum(c["components"][name] for c in valid) / len(valid) if valid else None
             for name in CONFIDENCE_WEIGHTS}
    means["confidence"] = sum(c["score"] for c in valid) / len(valid) if valid else None
    return {"total_tickets": len(results), "status_counts": {s: counts[s] for s in ("ok", "degraded", "unavailable")},
            "valid_score_count": len(valid), "mean_scores": means}


def run_evaluation(dataset, output, limit=0, num_ctx=None, ticket_id=None):
    """Lưu sau mỗi ticket; không ghi đè file có thể đã được người dùng gắn nhãn."""
    if type(limit) is not int or limit < 0:
        raise ValueError("limit phải là số nguyên >= 0")
    tickets = json.loads(Path(dataset).read_text(encoding="utf-8-sig"))
    if not isinstance(tickets, list) or not tickets:
        raise ValueError("Dataset phải là danh sách ticket không rỗng")
    ids = set()
    for ticket in tickets:
        if not isinstance(ticket, dict) or any(not isinstance(ticket.get(k), str) or not ticket[k].strip()
                                              for k in ("id", "question", "category")):
            raise ValueError("Ticket phải có id, question, category không rỗng")
        if not isinstance(ticket.get("human_handoff"), bool) or "ground_truth_answer" not in ticket or "ground_truth_contexts" not in ticket:
            raise ValueError("Ticket thiếu thông tin golden để đánh giá thủ công")
        if ticket["id"] in ids:
            raise ValueError(f"Ticket ID trùng: {ticket['id']}")
        ids.add(ticket["id"])
    if ticket_id is not None:
        tickets = [t for t in tickets if t["id"] == ticket_id]
        if not tickets:
            raise ValueError(f"Không tìm thấy ticket ID: {ticket_id}")
    if limit:
        tickets = tickets[:limit]
    settings = get_settings()
    num_ctx = settings.judge_num_ctx if num_ctx is None else num_ctx
    if type(num_ctx) is not int or num_ctx <= max(2048, settings.judge_max_tokens) + 256:
        raise ValueError("num_ctx phải lớn hơn ngân sách output + 256")
    mistral = bool(settings.mistral_api_key and settings.mistral_api_key.get_secret_value().strip())
    report = {"run": {
        "created_at": datetime.now().astimezone().isoformat(), "dataset": str(Path(dataset).resolve()),
        "weights": dict(CONFIDENCE_WEIGHTS), "selected_tickets": len(tickets), "completed": False,
        "retrieval_config": {"top_k": settings.top_k, "candidate_pool": DEFAULT_CANDIDATE_POOL,
                             "rrf_k": RRF_K, "embedding_model": settings.embedding_model},
        "generation_model": GENERATION_MODEL, "num_ctx": num_ctx,
        "groundedness_judge": {"provider": "mistral" if mistral else "ollama",
            "model": settings.mistral_model if mistral else settings.groundedness_model,
            "temperature": 0, "seed": settings.judge_seed, "max_tokens": settings.judge_max_tokens},
    }, "summary": summarize([]), "results": []}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create bảo vệ cả dataset và các đánh giá thủ công đã lưu.
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    try:
        for ticket in tickets:
            report["results"].append(evaluate_ticket(ticket, settings.top_k, num_ctx))
            report["summary"] = summarize(report["results"])
            report["run"]["completed"] = len(report["results"]) == len(tickets)
            temporary = output.with_suffix(output.suffix + ".tmp")
            temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            temporary.replace(output)
            print(f"[{len(report['results'])}/{len(tickets)}] {ticket['id']}: {report['results'][-1]['confidence']}")
    finally:
        close_pool()
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "eval/dataset/golden_dataset.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "eval/results" / f"eval_confidence_{datetime.now():%Y%m%d_%H%M%S_%f}.json")
    parser.add_argument("--limit", type=int, default=0, help="0: toàn bộ dataset")
    parser.add_argument("--ticket-id", help="Chỉ chạy ticket có ID này")
    parser.add_argument("--num-ctx", type=int, help="Mặc định dùng JUDGE_NUM_CTX")
    args = parser.parse_args()
    print(f"Đã lưu: {run_evaluation(args.dataset, args.output, args.limit, args.num_ctx, args.ticket_id)}")
