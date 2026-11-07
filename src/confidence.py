"""Tính confidence RAG và kiểm tra groundedness bằng LLM judge."""

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.agent.llm_client import generate, render_prompt
from src.core.config import get_settings

CONFIDENCE_WEIGHTS = {"retrieval": 0.4, "consensus": 0.2, "groundedness": 0.4}
# Lite mode: chỉ dùng retrieval + consensus, không gọi groundedness judge.
CONFIDENCE_LITE_WEIGHTS = {"retrieval": 0.7, "consensus": 0.3}


class EvidenceReference(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    context_id: str = Field(pattern=r"^E[1-9][0-9]*$")


class Claim(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    claim: str = Field(min_length=1)
    verdict: Literal["supported", "unsupported", "contradicted", "uncertain"]
    evidence: list[EvidenceReference] = Field(max_length=3)


class GroundednessReport(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    claims: list[Claim] = Field(max_length=12)
    reason: str = Field(min_length=1, max_length=500)


class EvidenceValidationError(ValueError):
    """Lỗi evidence có thêm dữ liệu phục vụ chẩn đoán."""

    def __init__(self, details):
        message = (
            f"Evidence bị từ chối: {details['code']} "
            f"(claim {details['claim_index']}, evidence {details['evidence_index']})"
        )
        super().__init__(message)
        self.details = details


def _unit_score(value: float) -> float:
    valid_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if not valid_number or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Score phải là số hữu hạn trong [0, 1]")
    return value


def _error_message(exc: Exception) -> str:
    """Trả message cho ValueError thuần; type name cho exception có schema errors hoặc không phải ValueError."""
    return str(exc) if isinstance(exc, ValueError) and not hasattr(exc, "errors") else type(exc).__name__


def calculate_retrieval_confidence(parents: list[dict]) -> float:
    """Tính độ mạnh của top-1 và khoảng cách giữa top-1 với top-2."""
    if not parents:
        raise ValueError("Không có parent để tính confidence")

    ids = [parent["parent_chunk_id"] for parent in parents]
    if len(ids) != len(set(ids)):
        raise ValueError("Parent bị trùng")

    scores = [_unit_score(parent["relevance_score"]) for parent in parents]
    if scores != sorted(scores, reverse=True):
        raise ValueError("Parent phải xếp giảm dần theo relevance_score")

    margin = scores[0] - scores[1] if len(scores) > 1 else 0.0
    return 0.7 * scores[0] + 0.3 * min(1.0, margin * 2.0)


def calculate_consensus(top_parent: dict) -> float:
    """Chấm mức đồng thuận dựa trên rank Dense và BM25 của top parent."""
    ranks = [top_parent["dense_rank"], top_parent["bm25_rank"]]
    if any(rank is not None and (type(rank) is not int or rank < 1) for rank in ranks):
        raise ValueError("Rank phải là số nguyên dương hoặc None")
    if all(rank is None for rank in ranks):
        raise ValueError("Parent không thuộc kênh retrieval nào")
    if all(rank is not None and rank <= 3 for rank in ranks):
        return 1.0
    if all(rank is not None and rank <= 5 for rank in ranks):
        return 0.75
    return 0.5


def _judge_contexts(kb_contexts: list[dict]) -> list[dict]:
    """Chỉ đưa ID ngắn và content cho judge; metadata backend không phải evidence."""
    return [
        {"context_id": f"E{index}", "content": context["content"]}
        for index, context in enumerate(kb_contexts, 1)
    ]


def _validate_claim_references(claims: list[Claim], contexts_by_id: dict[str, dict]) -> None:
    """Kiểm tra tính hợp lệ của claims và các reference evidence trong context."""
    seen_claims = set()
    for claim_index, item in enumerate(claims, 1):
        normalized_claim = " ".join(item.claim.casefold().split())
        if not normalized_claim or normalized_claim in seen_claims:
            raise ValueError("Claim rỗng hoặc trùng lặp")
        seen_claims.add(normalized_claim)

        if item.verdict in ("supported", "contradicted") and not item.evidence:
            raise ValueError("Claim supported/contradicted cần evidence")

        references = [evidence.context_id for evidence in item.evidence]
        if len(references) != len(set(references)):
            raise ValueError("Evidence context bị trùng")

        for evidence_index, context_id in enumerate(references, 1):
            if context_id not in contexts_by_id:
                raise EvidenceValidationError({
                    "code": "unknown_context_id",
                    "context_id": context_id,
                    "allowed_context_ids": list(contexts_by_id),
                    "claim_index": claim_index,
                    "evidence_index": evidence_index,
                    "claim": item.claim,
                    "verdict": item.verdict,
                })


def _determine_overall_verdict(claims: list[Claim]) -> str:
    """Xác định verdict tổng thể theo thứ tự ưu tiên worst-case:
    contradicted > unsupported > uncertain > supported.
    """
    verdicts = {item.verdict for item in claims}
    return next(
        candidate for candidate in ("contradicted", "unsupported", "uncertain", "supported")
        if candidate in verdicts
    )


def _build_groundedness_result(parsed: GroundednessReport, contexts_by_id: dict[str, dict]) -> dict:
    """Gắn metadata backend vào evidence và tính score groundedness."""
    result = parsed.model_dump()
    for item in result["claims"]:
        for evidence in item["evidence"]:
            context = contexts_by_id[evidence["context_id"]]
            evidence.update(
                source=context["source"],
                parent_chunk_id=context.get("parent_chunk_id"),
            )

    supported_count = sum(item.verdict == "supported" for item in parsed.claims)
    return result | {
        "score": supported_count / len(parsed.claims),
        "status": "ok",
        "error": None,
        "raw_response": None,
        "verdict": _determine_overall_verdict(parsed.claims),
        "problematic_claims": [
            item.claim for item in parsed.claims if item.verdict != "supported"
        ],
    }


def score_groundedness(report: dict, kb_contexts: list[dict]) -> dict:
    """Validate context reference rồi tính tỷ lệ claim được hỗ trợ."""
    parsed = GroundednessReport.model_validate(report)
    if not parsed.claims:
        raise ValueError("Không có claim kiểm chứng được")

    contexts_by_id = {
        f"E{index}": context for index, context in enumerate(kb_contexts, 1)
    }
    _validate_claim_references(parsed.claims, contexts_by_id)
    return _build_groundedness_result(parsed, contexts_by_id)


def unavailable_groundedness(error: str, raw_response: str | None = None) -> dict:
    """Trả fallback khi groundedness không tính được, lưu lại raw_response nếu có lỗi."""
    return {
        "claims": [],
        "score": 0.0,
        "status": "unavailable",
        "error": error,
        "raw_response": raw_response,
        "verdict": "unavailable",
        "reason": error,
        "problematic_claims": [],
    }


def _record_judge_error(diagnostics: dict, stage: str, exc: Exception) -> None:
    message = _error_message(exc)
    diagnostics.update(
        validation="failed", stage=stage, error_type=type(exc).__name__, error=message
    )
    if isinstance(exc, EvidenceValidationError):
        diagnostics["rejected_evidence"] = exc.details
    elif hasattr(exc, "errors"):
        diagnostics["schema_errors"] = exc.errors(
            include_input=False, include_context=False, include_url=False
        )


def _validate_judge_inputs(answer: str, answer_complete: bool, kb_contexts: list[dict]) -> None:
    """Kiểm tra điều kiện đầu vào cho judge groundedness."""
    if not answer_complete or not answer.strip():
        raise ValueError("Answer rỗng hoặc chưa hoàn chỉnh")
    if not kb_contexts or any(
        not context.get("source", "").strip() or not context.get("content", "").strip()
        for context in kb_contexts
    ):
        raise ValueError("Evidence rỗng hoặc không hợp lệ")


def evaluate_groundedness(
    question: str, answer: str, kb_contexts: list[dict], *,
    answer_complete: bool = True, num_ctx: int | None = None,
    diagnostics: dict | None = None, use_cloud_model: bool | None = None,
) -> dict:
    """Gọi đúng một judge; mọi lỗi đều fallback về groundedness unavailable."""
    stage = "input_validation"
    response_diagnostics = {}
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update(
            question=question,
            answer=answer,
            kb_contexts=kb_contexts,
            response=response_diagnostics,
            validation="not_checked",
        )

    response = None
    try:
        _validate_judge_inputs(answer, answer_complete, kb_contexts)

        settings = get_settings()
        if use_cloud_model is None:
            use_cloud_model = bool(
                getattr(settings, "openrouter_api_key", None)
                and settings.openrouter_api_key.get_secret_value().strip()
            )

        model = settings.openrouter_model if use_cloud_model else settings.groundedness_model
        context_size = settings.judge_num_ctx if num_ctx is None else num_ctx
        system_prompt = render_prompt("eval_groundedness_system.jinja2")
        task_prompt = render_prompt(
            "eval_groundedness_task.jinja2",
            question=question,
            answer=answer,
            kb_contexts=_judge_contexts(kb_contexts),
        )

        stage = "generation"
        generate_options = {
            "system_prompt": system_prompt,
            "model_name": model,
            "use_cloud_model": use_cloud_model,
            "num_ctx": context_size,
            "response_schema": GroundednessReport.model_json_schema(),
            "seed": settings.judge_seed,
            "max_tokens": settings.judge_max_tokens,
            "diagnostics": response_diagnostics,
        }
        if diagnostics is not None:
            diagnostics.update(
                provider="openrouter" if use_cloud_model else "ollama",
                model=model,
                num_ctx=context_size,
                max_tokens=settings.judge_max_tokens,
                seed=settings.judge_seed,
                system_prompt=system_prompt,
            )

        response = generate(task_prompt, **generate_options)
        if diagnostics is not None:
            response_diagnostics.setdefault("raw_response", response)

        stage = "schema_validation"
        parsed = GroundednessReport.model_validate_json(response)

        stage = "evidence_validation"
        result = score_groundedness(parsed.model_dump(), kb_contexts)
        if diagnostics is not None:
            diagnostics.update(validation="passed", stage="complete")
        return result
    except Exception as exc:
        message = _error_message(exc)
        if diagnostics is not None:
            _record_judge_error(diagnostics, stage, exc)

        raw_err = response
        if raw_err is None:
            if hasattr(exc, "generation") and isinstance(exc.generation, dict):
                raw_err = exc.generation.get("answer")
            if not raw_err and "raw_response" in response_diagnostics:
                raw_err = response_diagnostics["raw_response"]

        return unavailable_groundedness(message, raw_response=raw_err)


def calculate_confidence(
    c_retrieval: float | None,
    c_consensus: float | None,
    groundedness: dict,
    *,
    retrieval_error: str | None = None,
) -> dict:
    """Kết hợp ba thành phần; status phải được kiểm tra trước khi routing."""
    if retrieval_error or c_retrieval is None or c_consensus is None:
        return {
            "c_retrieval": None,
            "c_consensus": None,
            "c_groundedness": None,
            "c_rag": None,
            "status": "unavailable",
            "errors": [retrieval_error or "Retrieval unavailable"],
        }

    _unit_score(c_retrieval)
    _unit_score(c_consensus)
    if c_consensus not in (0.5, 0.75, 1.0):
        raise ValueError("Consensus phải thuộc {0.5, 0.75, 1.0}")
    if groundedness.get("status") not in ("ok", "unavailable"):
        raise ValueError("Groundedness status không hợp lệ")

    errors = []
    if groundedness["status"] == "unavailable":
        groundedness_score = 0.0
        errors.append(groundedness.get("error") or "Groundedness unavailable")
    else:
        groundedness_score = _unit_score(groundedness["score"])
        if groundedness.get("error"):
            raise ValueError("Groundedness ok không được có error")
        if groundedness.get("verdict") == "contradicted":
            errors.append("Answer có claim mâu thuẫn evidence")

    total = (
        CONFIDENCE_WEIGHTS["retrieval"] * c_retrieval
        + CONFIDENCE_WEIGHTS["consensus"] * c_consensus
        + CONFIDENCE_WEIGHTS["groundedness"] * groundedness_score
    )
    return {
        "c_retrieval": c_retrieval,
        "c_consensus": c_consensus,
        "c_groundedness": groundedness_score,
        "c_rag": total,
        "status": "degraded" if errors else "ok",
        "errors": errors,
    }


def calculate_confidence_lite(
    c_retrieval: float | None,
    c_consensus: float | None,
    *,
    retrieval_error: str | None = None,
) -> dict:
    """Tính confidence chỉ từ retrieval và consensus; không gọi groundedness judge.

    Dùng khi muốn đánh giá nhanh mà không tốn compute/thời gian chạy judge.
    Trọng số: retrieval=0.7, consensus=0.3 — tổng bằng 1, đặt nhiều vào tín hiệu dense.
    Output giữ cùng shape với calculate_confidence (đều có c_groundedness) để
    caller xử lý nhất quán — c_groundedness là None thay vì 0.0.
    """
    if retrieval_error or c_retrieval is None or c_consensus is None:
        return {
            "c_retrieval": None,
            "c_consensus": None,
            "c_groundedness": None,
            "c_rag": None,
            "status": "unavailable",
            "errors": [retrieval_error or "Retrieval unavailable"],
        }
    _unit_score(c_retrieval)
    _unit_score(c_consensus)
    if c_consensus not in (0.5, 0.75, 1.0):
        raise ValueError("Consensus phải thuộc {0.5, 0.75, 1.0}")
    total = (
        CONFIDENCE_LITE_WEIGHTS["retrieval"] * c_retrieval
        + CONFIDENCE_LITE_WEIGHTS["consensus"] * c_consensus
    )
    return {
        "c_retrieval": c_retrieval,
        "c_consensus": c_consensus,
        "c_groundedness": None,  # không tính trong lite mode
        "c_rag": total,
        "status": "ok",
        "errors": [],
    }
