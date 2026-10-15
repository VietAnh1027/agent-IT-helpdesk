"""Confidence RAG: hàm tính thuần và groundedness judge dùng chung."""

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.agent.llm_client import generate, render_prompt, build_answer, GenerationError
from src.core.config import get_settings

CONFIDENCE_WEIGHTS = {"retrieval": 0.4, "consensus": 0.2, "groundedness": 0.4}

class EvidenceQuote(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    source: str = Field(min_length=1)
    quote: str = Field(min_length=1)


class Claim(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    claim: str = Field(min_length=1)
    verdict: Literal["supported", "unsupported", "contradicted", "uncertain"]
    evidence: list[EvidenceQuote]


class GroundednessReport(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    claims: list[Claim]
    reason: str = Field(min_length=1)
    citation_support: Literal["supported", "unsupported", "uncertain"]


def _unit_score(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Score phải là số hữu hạn trong [0, 1]")
    return value


def calculate_retrieval_confidence(parents: list[dict]) -> float:
    """Truyền fusion chưa cắt top_k; một parent thật sự duy nhất có margin = 0."""
    if not parents:
        raise ValueError("Không có parent để tính confidence")
    seen, scores = set(), []
    for parent in parents:
        parent_id = parent["parent_chunk_id"]
        if parent_id in seen:
            raise ValueError("Parent bị trùng")
        seen.add(parent_id)
        scores.append(_unit_score(parent["relevance_score"]))
    if scores != sorted(scores, reverse=True):
        raise ValueError("Parent phải xếp giảm dần theo relevance_score")
    margin = scores[0] - scores[1] if len(scores) > 1 else 0.0
    return 0.7 * scores[0] + 0.3 * min(1.0, margin * 2.0)


def calculate_consensus(top_parent: dict) -> float:
    ranks = [top_parent["dense_rank"], top_parent["bm25_rank"]]
    for rank in ranks:
        if rank is not None and (type(rank) is not int or rank < 1):
            raise ValueError("Rank phải là số nguyên dương hoặc None")
    if all(rank is None for rank in ranks):
        raise ValueError("Parent không thuộc kênh retrieval nào")
    if all(rank is not None and rank <= 3 for rank in ranks):
        return 1.0
    if all(rank is not None and rank <= 5 for rank in ranks):
        return 0.75
    return 0.5


def score_groundedness(report: dict, kb_contexts: list[dict]) -> dict:
    """Validate quote chính xác rồi đếm; không tự xác minh entailment bằng substring."""
    parsed = GroundednessReport.model_validate(report)
    if not parsed.claims:
        raise ValueError("Không có claim kiểm chứng được")
    seen = set()
    for claim in parsed.claims:
        normalized = " ".join(claim.claim.casefold().split())
        if not normalized or normalized in seen:
            raise ValueError("Claim rỗng hoặc trùng lặp")
        seen.add(normalized)
        if claim.verdict in ("supported", "contradicted") and not claim.evidence:
            raise ValueError("Claim supported/contradicted cần evidence")
        for quote in claim.evidence:
            if not quote.quote.strip() or not any(
                ctx["source"] == quote.source and quote.quote in ctx["content"] for ctx in kb_contexts
            ):
                raise ValueError("Nguồn hoặc quote không tồn tại trong evidence")
    verdicts = {claim.verdict for claim in parsed.claims}
    verdict = next(v for v in ("contradicted", "unsupported", "uncertain", "supported") if v in verdicts)
    return parsed.model_dump() | {
        "score": sum(c.verdict == "supported" for c in parsed.claims) / len(parsed.claims),
        "status": "ok", "error": None, "verdict": verdict,
        "problematic_claims": [c.claim for c in parsed.claims if c.verdict != "supported"],
    }


def unavailable_groundedness(error: str) -> dict:
    """Fallback số 0 phải kèm trạng thái chưa đánh giá được."""
    return {"claims": [], "score": 0.0, "status": "unavailable", "error": error,
            "verdict": "unavailable", "reason": error, "problematic_claims": [],
            "citation_support": "uncertain"}


def evaluate_groundedness(question: str, answer: str, kb_contexts: list[dict], *,
                          answer_complete: bool = True, num_ctx: int | None = None) -> dict:
    """Một request judge; lỗi provider không âm thầm gọi lại bằng model khác."""
    try:
        if not answer_complete or not answer.strip():
            raise ValueError("Answer rỗng hoặc chưa hoàn chỉnh")
        if not kb_contexts or any(not c["source"].strip() or not c["content"].strip() for c in kb_contexts):
            raise ValueError("Evidence rỗng hoặc không hợp lệ")
        try:
            citations = build_answer(answer, kb_contexts)["citations"]
            invalid_citations = False
        except GenerationError as exc:
            citations = exc.generation["citations"]
            invalid_citations = True
        settings = get_settings()
        use_mistral = bool(settings.mistral_api_key and settings.mistral_api_key.get_secret_value().strip())
        system = render_prompt("eval_groundedness_system.jinja2")
        task = render_prompt("eval_groundedness_task.jinja2", question=question, answer=answer,
                             citations=citations, kb_contexts=kb_contexts)
        response = generate(task, system_prompt=system,
                            model_name=settings.mistral_model if use_mistral else settings.groundedness_model,
                            provider="mistral" if use_mistral else "auto",
                            num_ctx=settings.judge_num_ctx if num_ctx is None else num_ctx,
                            response_schema=GroundednessReport.model_json_schema(),
                            seed=settings.judge_seed, max_tokens=settings.judge_max_tokens)
        parsed = GroundednessReport.model_validate_json(response)
        result = score_groundedness(parsed.model_dump(), kb_contexts)
        if invalid_citations or not citations:
            result["citation_support"] = "unsupported"
        return result
    except Exception as exc:
        # Không ghi raw output/credential từ lỗi SDK vào report.
        message = str(exc) if isinstance(exc, ValueError) and not hasattr(exc, "errors") else type(exc).__name__
        return unavailable_groundedness(message)


def calculate_confidence(c_retrieval: float | None, c_consensus: float | None,
                         groundedness: dict, *, retrieval_error: str | None = None) -> dict:
    """Caller phải kiểm tra status/errors; score không phải quyết định auto-send."""
    if retrieval_error or c_retrieval is None or c_consensus is None:
        return {"c_retrieval": None, "c_consensus": None, "c_groundedness": None,
                "c_rag": None, "status": "unavailable", "errors": [retrieval_error or "Retrieval unavailable"]}
    _unit_score(c_retrieval)
    _unit_score(c_consensus)
    if c_consensus not in (0.5, 0.75, 1.0):
        raise ValueError("Consensus phải thuộc {0.5, 0.75, 1.0}")
    if groundedness.get("status") not in ("ok", "unavailable"):
        raise ValueError("Groundedness status không hợp lệ")
    errors = []
    if groundedness["status"] == "unavailable":
        score = 0.0
        errors.append(groundedness.get("error") or "Groundedness unavailable")
    else:
        score = _unit_score(groundedness["score"])
        if groundedness.get("error"):
            raise ValueError("Groundedness ok không được có error")
        if groundedness.get("citation_support") != "supported":
            errors.append("Citations chưa được xác minh")
        if groundedness.get("verdict") == "contradicted":
            errors.append("Answer có claim mâu thuẫn evidence")
    return {"c_retrieval": c_retrieval, "c_consensus": c_consensus, "c_groundedness": score,
            "c_rag": CONFIDENCE_WEIGHTS["retrieval"] * c_retrieval
                     + CONFIDENCE_WEIGHTS["consensus"] * c_consensus
                     + CONFIDENCE_WEIGHTS["groundedness"] * score,
            "status": "degraded" if errors else "ok", "errors": errors}
