"""Chạy golden dataset, lưu confidence và chỗ điền đánh giá thủ công.

python -m eval.harness.eval_confidence_scorer
--confidence-mode lite/full
--limit 3/number
Không gọi correctness judge; golden chỉ được lưu để người đọc đối chiếu.
"""

import argparse
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import time

from eval.harness.eval_generation import GENERATION_MODEL, call_prompt
from src.agent.llm_client import GenerationError, build_answer
from src.confidence import (
    CONFIDENCE_WEIGHTS, CONFIDENCE_LITE_WEIGHTS,
    calculate_confidence, calculate_confidence_lite,
    calculate_consensus, calculate_retrieval_confidence,
    evaluate_groundedness, unavailable_groundedness,
)
from src.core.config import PROJECT_ROOT, get_settings
from src.core.database import close_pool, get_connection
from src.rag.retriever import DEFAULT_CANDIDATE_POOL, RRF_K, retrieve_hybrid_parents

# Map consensus score -> mô tả lý do; dùng .get() để guard nếu giá trị ngoài dự kiến.
_CONSENSUS_REASON = {
    1.0: "Top-3 cả hai kênh",
    0.75: "Top-5 cả hai kênh",
    0.5: "Không đạt đồng thuận Top-5 cả hai kênh",
}


def _do_retrieval(ticket: dict, result: dict, top_k: int):
    """Chạy retrieval, điền result["retrieval"], trả (parents, retrieval_score, consensus_score, contexts)."""
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
        "consensus_reason": _CONSENSUS_REASON.get(consensus_score, f"Unknown ({consensus_score})"),
    }
    for index, parent in enumerate(parents):
        # entry trong retrieval.parents đổi article_title→source, text→content (chỉ khi được chọn).
        # Ví dụ rút gọn: {"parent_chunk_id": "p1", "source": "mail", "dense_rank": 1,
        # "bm25_rank": 2, "used_for_generation": True, "content": "..."}. Parent ngoài top_k không có content.
        entry = {key: value for key, value in parent.items() if key not in ("text", "article_title")}
        entry.update(source=parent["article_title"], used_for_generation=index < top_k)
        if index < top_k:
            entry["content"] = parent["text"]
        result["retrieval"]["parents"].append(entry)
    # contexts ví dụ: [{"parent_chunk_id": "p1", "source": "mail", "content": "Tắt Outlook."}].
    contexts = [
        {"parent_chunk_id": p["parent_chunk_id"], "source": p["article_title"], "content": p["text"]}
        for p in parents[:top_k]
    ]
    return parents, retrieval_score, consensus_score, contexts


def _do_generation(ticket: dict, contexts: list, num_ctx: int) -> str:
    """Gọi LLM để sinh answer; trả answer string. Raise GenerationError hoặc ValueError."""
    answer = call_prompt(
        GENERATION_MODEL, "answer_with_cite", num_ctx=num_ctx,
        ticket_subject=ticket["category"], ticket_body=ticket["question"],
        category=ticket["category"], kb_contexts=contexts,
    )
    if not answer.strip():
        raise ValueError("Answer rỗng")
    return answer


def _do_groundedness(ticket: dict, answer: str, contexts: list, num_ctx: int, max_retries: int = 2) -> dict:
    """Đánh giá groundedness; luôn trả grounding dict (unavailable nếu lỗi). Tự động retry nếu dính HTTP 429."""
    for attempt in range(max_retries + 1):
        grounding = evaluate_groundedness(ticket["question"], answer, contexts, num_ctx=num_ctx, use_cloud_model=True)
        if grounding.get("status") == "unavailable" and "429" in str(grounding.get("error")):
            if attempt < max_retries:
                wait_time = 3.0 * (attempt + 1)
                print(f"[{ticket['id']}] Gặp HTTP 429 (Rate Limit), đang chờ {wait_time}s rồi thử lại (lần {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
                continue
        return grounding
    return grounding


def evaluate_ticket(ticket: dict, top_k: int, num_ctx: int, confidence_mode: str = "full") -> dict:
    """Giữ toàn bộ fusion để tính margin, chỉ gửi top_k context cho LLM."""
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k phải là số nguyên dương")
    # result giữ reference riêng; assessment.human_review={"verdict": None, "reason": ""}
    # là chỗ người đánh giá điền, không được truyền vào generator/groundedness judge.
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
    # Lite mode bỏ qua groundedness judge; full mode gọi judge sau generation.
    grounding = unavailable_groundedness(
        "Lite mode: groundedness skipped" if confidence_mode == "lite" else "Chưa có answer hoàn chỉnh"
    )
    source_output = build_answer("", [])  # khởi tạo sớm để tránh NameError nếu GenerationError xảy ra trước dòng gán thực sự
    try:
        _, retrieval_score, consensus_score, contexts = _do_retrieval(ticket, result, top_k)
        # source_output={"answer": "", "sources": [...]}; phép | tạo dict mới, thay answer rỗng bằng text LLM.
        source_output = build_answer("", contexts)
        stage = "generation"
        answer = _do_generation(ticket, contexts, num_ctx)
        complete = True
        result["generation"] = source_output | {"answer": answer, "complete": True}
        if confidence_mode == "full":
            stage = "groundedness"
            grounding = _do_groundedness(ticket, answer, contexts, num_ctx)
            if grounding["status"] != "ok":
                result["errors"].append({"stage": stage, "message": grounding["error"]})
    except Exception as exc:
        # Không chép raw lỗi SDK/DB có thể chứa credential vào JSON.
        message = str(exc) if isinstance(exc, GenerationError) else type(exc).__name__
        result["errors"].append({"stage": stage, "message": message})
        if isinstance(exc, GenerationError):
            result["generation"] = source_output | exc.generation | {"complete": complete}
        grounding = unavailable_groundedness(f"{stage}: {message}")
        if stage == "retrieval":
            retrieval_score = consensus_score = None

    if confidence_mode == "lite":
        result["groundedness"] = None
    else:
        result["groundedness"] = grounding | {
            "supported_claim_count": sum(c["verdict"] == "supported" for c in grounding["claims"]),
            "total_claim_count": len(grounding["claims"]),
        }
    weights = CONFIDENCE_LITE_WEIGHTS if confidence_mode == "lite" else CONFIDENCE_WEIGHTS
    confidence = (
        calculate_confidence_lite(retrieval_score, consensus_score)
        if confidence_mode == "lite"
        else calculate_confidence(retrieval_score, consensus_score, grounding)
    )
    # components chỉ gồm các thành phần theo weights của mode đang dùng.
    # Full: {retrieval, consensus, groundedness}; Lite: {retrieval, consensus}.
    components = {name: confidence[f"c_{name}"] for name in weights}
    result["confidence"] = {
        "status": confidence["status"], "components": components,
        "weighted_contributions": {name: value * weights[name] if value is not None else None
                                   for name, value in components.items()},
        "score": confidence["c_rag"], "errors": confidence["errors"],
    }
    return result



def summarize(results, weights=None):
    """Chỉ lấy status ok vào trung bình; không che số ticket bị lỗi."""
    if weights is None:
        weights = CONFIDENCE_WEIGHTS
    counts = Counter(r["confidence"]["status"] for r in results)
    valid = [r["confidence"] for r in results if r["confidence"]["status"] == "ok"]
    means = {name: sum(c["components"][name] for c in valid) / len(valid) if valid else None
             for name in weights}
    means["confidence"] = sum(c["score"] for c in valid) / len(valid) if valid else None
    return {
        "total_tickets": len(results),
        "status_counts": {s: counts.get(s, 0) for s in ("ok", "degraded", "unavailable")},
        "valid_score_count": len(valid),
        "mean_scores": means,
    }


def run_evaluation(dataset, output, limit=0, num_ctx=None, ticket_id=None, confidence_mode=None, delay_seconds: float = 2.0):
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
    # CLI --confidence-mode ưu tiên hơn CONFIDENCE_MODE trong .env.
    confidence_mode = confidence_mode or getattr(settings, "confidence_mode", "full")
    weights = CONFIDENCE_LITE_WEIGHTS if confidence_mode == "lite" else CONFIDENCE_WEIGHTS
    num_ctx = settings.judge_num_ctx if num_ctx is None else num_ctx
    if confidence_mode == "full":
        if type(num_ctx) is not int or num_ctx <= max(2048, settings.judge_max_tokens) + 256:
            raise ValueError("num_ctx phải lớn hơn ngân sách output judge + 256")
    elif type(num_ctx) is not int or num_ctx < 512:
        raise ValueError("num_ctx phải là số nguyên >= 512")
    cloud_active = bool(getattr(settings, "openrouter_api_key", None) and settings.openrouter_api_key.get_secret_value().strip())
    # Khung JSON (rút gọn): {"schema_version": 2, "run": {"confidence_mode": "full",
    # "confidence_function": "calculate_confidence", "weights": {...}, ...}, ...}.
    # confidence_function ghi rõ hàm thực sự được dùng để không nhầm lẫn khi so sánh kết quả.
    confidence_function = "calculate_confidence_lite" if confidence_mode == "lite" else "calculate_confidence"
    report = {"schema_version": 2, "run": {
        "created_at": datetime.now().astimezone().isoformat(), "dataset": str(Path(dataset).resolve()),
        "confidence_mode": confidence_mode, "confidence_function": confidence_function,
        "weights": dict(weights), "selected_tickets": len(tickets), "completed": False,
        "retrieval_config": {"top_k": settings.top_k, "candidate_pool": DEFAULT_CANDIDATE_POOL,
                             "rrf_k": RRF_K, "embedding_model": settings.embedding_model},
        "generation_model": settings.openrouter_model if cloud_active else GENERATION_MODEL, "num_ctx": num_ctx,
        "groundedness_judge": None if confidence_mode == "lite" else {
            "provider": "openrouter" if cloud_active else "ollama",
            "model": settings.openrouter_model if cloud_active else settings.groundedness_model,
            "temperature": 0, "seed": settings.judge_seed, "max_tokens": settings.judge_max_tokens},
    }, "summary": summarize([], weights), "results": []}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create bảo vệ cả dataset và các đánh giá thủ công đã lưu.
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    try:
        for index, ticket in enumerate(tickets):
            if index > 0 and delay_seconds > 0:
                time.sleep(delay_seconds)
            report["results"].append(evaluate_ticket(ticket, settings.top_k, num_ctx, confidence_mode))
            report["summary"] = summarize(report["results"], weights)
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
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "eval/results" / f"eval_confidence_{datetime.now():%Y%m%d_%H%M%S_%f}")
    parser.add_argument("--limit", type=int, default=0, help="0: toàn bộ dataset")
    parser.add_argument("--ticket-id", help="Chỉ chạy ticket có ID này")
    parser.add_argument("--num-ctx", type=int, help="Mặc định dùng JUDGE_NUM_CTX")
    parser.add_argument("--confidence-mode", choices=["full", "lite"],
                        help="Ghi đè CONFIDENCE_MODE trong .env: full=có groundedness judge, lite=chỉ retrieval+consensus")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Thời gian giãn cách (giây) giữa các ticket để tránh rate limit HTTP 429 (mặc định 2.0s)")
    args = parser.parse_args()
    mode = args.confidence_mode or getattr(get_settings(), "confidence_mode", "full")
    out_file = Path(f"{args.output}_{mode}.json") if not str(args.output).endswith(".json") else args.output
    print(f"Đã lưu: {run_evaluation(args.dataset, out_file, args.limit, args.num_ctx, args.ticket_id, args.confidence_mode, delay_seconds=args.delay)}")

