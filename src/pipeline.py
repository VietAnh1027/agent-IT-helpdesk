"""Pipeline Orchestrator hoàn chỉnh của IT Helpdesk AI Agent xây dựng bằng LangGraph.

Sâu chuỗi toàn bộ luồng xử lý theo documents/02_ARCHITECTURE.md (Mục 4) & documents/08_AGENT_STATE.md:
1. Similarity & Duplicate / Incident Detection
2. Fast Guardrails & Intent Classification
3. Diagnostic Tools (Read-Only)
4. Hybrid Retrieval (Dense + BM25 + RRF)
5. LLM Answer Generation
6. Confidence Scorer (c_retrieval + c_consensus + c_groundedness)
7. Router (Auto, Draft, Clarify, Escalate)
"""

import json
import logging
from typing import Literal

from langgraph.graph import END, START, StateGraph

from src.agent.classifiers.intent_classifier import classify_state
from src.agent.guardrails.input_guardrails import (
    check_fast_security_concern,
    check_prompt_injection,
)
from src.agent.llm_client import GenerationError, build_answer, generate, render_prompt
from src.agent.similarity.repository import InMemoryTicketRepository, TicketRepository
from src.agent.similarity.ticket_matcher import match_ticket_state
from src.agent.state import AgentState
from src.agent.tools.diagnostic_adapter import MockDiagnosticBackendAdapter
from src.agent.tools.diagnostic_tools import (
    check_license_availability,
    check_printer_status,
    check_vpn_gateway_status,
)
from src.agent.tools.guardrails import validate_tool_call
from src.agent.tools.interfaces import DiagnosticBackendAdapter
from src.confidence import (
    calculate_confidence,
    calculate_confidence_lite,
    calculate_consensus,
    calculate_retrieval_confidence,
    evaluate_groundedness,
)
from src.core.config import get_settings
from src.rag.retriever import embed_query, retrieve_top_k_parents
from src.router import route_ticket

logger = logging.getLogger(__name__)


# ==============================================================================
# CÁC NODE CỦA LANGGRAPH WORKFLOW
# ==============================================================================


def similarity_node(state: AgentState, repo: TicketRepository | None = None) -> AgentState:
    """Node 1: Kiểm tra spam dồn dập, câu hỏi lặp lại và sự cố diện rộng (Mass incident)."""
    active_repo = repo or InMemoryTicketRepository()
    try:
        # Tạo vector embedding của ticket
        text = f"{state.subject} {state.body}".strip()
        embedding = embed_query(text)
        state = match_ticket_state(state=state, embedding=embedding, repo=active_repo)
    except Exception as exc:
        logger.warning("Similarity check gặp lỗi: %s", exc)
        state.add_error("similarity", str(exc))

    return state


def classify_node(state: AgentState) -> AgentState:
    """Node 2: Kiểm tra bảo mật sớm (Fast-path) và phân loại Intent/Sentiment."""
    # Kiểm tra dấu hiệu hack / bảo mật nhanh (< 1ms)
    if check_fast_security_concern(state.subject, state.body):
        state.classification = {
            "category": "OUT_OF_SCOPE",
            "multi_issue_suspected": False,
            "prompt_injection_suspected": False,
            "sentiment_flag": "neutral",
            "missing_fields": [],
            "security_concern": True,
            "reason": "security_concern_keyword",
        }
        return state

    # Kiểm tra prompt injection
    if check_prompt_injection(f"{state.subject} {state.body}"):
        state.classification = {
            "category": "OUT_OF_SCOPE",
            "multi_issue_suspected": False,
            "prompt_injection_suspected": True,
            "sentiment_flag": "neutral",
            "missing_fields": [],
            "security_concern": False,
            "reason": "prompt_injection_suspected",
        }
        return state

    # Phân loại intent bằng LLM Classifier
    return classify_state(state)


def diagnostic_tool_node(
    state: AgentState,
    adapter: DiagnosticBackendAdapter | None = None,
) -> AgentState:
    """Node 3: Gọi các tool chẩn đoán read-only theo phân vùng Category."""
    category = (state.classification or {}).get("category", "").upper()
    active_adapter = adapter or MockDiagnosticBackendAdapter()

    if category == "VPN" and validate_tool_call("VPN", "check_vpn_gateway_status"):
        state.diagnostic_data = check_vpn_gateway_status(gateway_id="default", adapter=active_adapter)

    elif category == "PRINTER" and validate_tool_call("PRINTER", "check_printer_status"):
        # Trích xuất mã máy in nếu có trong nội dung, mặc định 'PRINTER_01'
        body_upper = f"{state.subject} {state.body}".upper()
        printer_id = "PRINTER_02" if "PRINTER_02" in body_upper or "TẦNG 4" in body_upper else "PRINTER_01"
        state.diagnostic_data = check_printer_status(printer_id=printer_id, adapter=active_adapter)

    elif category == "SOFTWARE" and validate_tool_call("SOFTWARE", "check_license_availability"):
        # Trích xuất tên phần mềm phổ biến
        body_lower = f"{state.subject} {state.body}".lower()
        software_name = "Photoshop" if "photoshop" in body_lower else "Figma" if "figma" in body_lower else "General"
        state.diagnostic_data = check_license_availability(software_name=software_name, adapter=active_adapter)

    return state


def retrieval_node(state: AgentState) -> AgentState:
    """Node 4: Hybrid retrieval trên kho tri thức (KB) với Dense + BM25 + RRF."""
    category = (state.classification or {}).get("category", "")
    query = f"{state.subject} {state.body}".strip()
    settings = get_settings()

    try:
        from src.core.database import get_connection

        with get_connection() as conn:
            with conn.cursor() as cur:
                parents = retrieve_top_k_parents(
                    cur=cur,
                    question=query,
                    category=category,
                    top_k=settings.top_k,
                )
                state.retrieved_parents = parents
    except Exception as exc:
        logger.warning("Retrieval gặp lỗi (DB offline hoặc query fail): %s", exc)
        state.add_error("retrieval", str(exc))
        state.retrieved_parents = []

    return state


def build_augmented_contexts(state: AgentState) -> list[dict]:
    """Tạo danh sách ngữ cảnh tri thức (KB chunks + dữ liệu chẩn đoán kỹ thuật thời gian thực)."""
    contexts = [
        {
            "parent_chunk_id": chunk.get("parent_chunk_id", ""),
            "source": chunk.get("article_title", "KB Article"),
            "content": chunk.get("text", ""),
        }
        for chunk in state.retrieved_parents
    ]
    if state.diagnostic_data:
        diag_str = json.dumps(state.diagnostic_data, ensure_ascii=False)
        contexts.append(
            {
                "parent_chunk_id": "diagnostic_tool",
                "source": "Hệ thống chẩn đoán kỹ thuật thời gian thực",
                "content": f"Dữ liệu chẩn đoán thời gian thực từ hệ thống: {diag_str}",
            }
        )
    return contexts


def generation_node(state: AgentState) -> AgentState:
    """Node 5: LLM sinh câu trả lời dựa trên tài liệu KB và dữ liệu chẩn đoán,
    hoặc sinh câu hỏi làm rõ (clarify) nếu thiếu thông tin."""
    # Nhánh 1: Nếu có missing_fields -> sinh câu hỏi làm rõ (Clarify)
    if state.missing_fields and state.clarification_rounds < 1:
        system_prompt = render_prompt("clarify_system.jinja2")
        task_prompt = render_prompt(
            "clarify_task.jinja2",
            ticket_subject=state.subject,
            ticket_body=state.body,
            missing_fields=state.missing_fields,
        )
        answer_obj = build_answer("", [])
        try:
            raw_text = generate(task_prompt, system_prompt=system_prompt)
            answer_obj.update(answer=raw_text, complete=True)
            state.generation = answer_obj
            state.generation_complete = True
        except GenerationError as exc:
            answer_obj.update(exc.generation, complete=False)
            answer_obj["error"] = str(exc)
            state.generation = answer_obj
            state.generation_complete = False
            state.add_error("generation", str(exc))
        return state

    # Nhánh 2: Luồng giải đáp kỹ thuật thông thường
    category = (state.classification or {}).get("category", "")
    kb_contexts = build_augmented_contexts(state)

    system_prompt = render_prompt("answer_with_cite_system.jinja2")
    task_prompt = render_prompt(
        "answer_with_cite_task.jinja2",
        ticket_subject=state.subject,
        ticket_body=state.body,
        category=category,
        kb_contexts=kb_contexts,
    )

    answer_obj = build_answer("", kb_contexts)
    try:
        raw_text = generate(task_prompt, system_prompt=system_prompt)
        answer_obj.update(answer=raw_text, complete=True)
        state.generation = answer_obj
        state.generation_complete = True
    except GenerationError as exc:
        answer_obj.update(exc.generation, complete=False)
        answer_obj["error"] = str(exc)
        state.generation = answer_obj
        state.generation_complete = False
        state.add_error("generation", str(exc))

    return state


def confidence_node(state: AgentState) -> AgentState:
    """Node 6: Tính toán confidence score tổng hợp (Retrieval + Consensus + Groundedness)."""
    settings = get_settings()

    if not state.retrieved_parents:
        state.confidence = {
            "c_retrieval": None,
            "c_consensus": None,
            "c_groundedness": None,
            "c_rag": None,
            "status": "unavailable",
            "errors": ["Không có tài liệu KB để tính confidence"],
        }
        return state

    try:
        c_retrieval = calculate_retrieval_confidence(state.retrieved_parents)
        top_parent = dict(state.retrieved_parents[0])
        if top_parent.get("dense_rank") is None and top_parent.get("bm25_rank") is None:
            top_parent["dense_rank"] = 1
            top_parent["bm25_rank"] = 1
        c_consensus = calculate_consensus(top_parent)

        if settings.confidence_mode == "lite":
            state.confidence = calculate_confidence_lite(c_retrieval, c_consensus)
        else:
            # Full mode: đánh giá groundedness qua LLM judge
            answer_text = (state.generation or {}).get("answer", "")
            if state.generation_complete and answer_text.strip() and state.retrieved_parents:
                kb_contexts = build_augmented_contexts(state)
                groundedness_report = evaluate_groundedness(
                    question=f"{state.subject} {state.body}",
                    answer=answer_text,
                    kb_contexts=kb_contexts,
                )
                print("=== ĐÂY LÀ GROUNDEDNESS REPORT ===")
                print(groundedness_report)
                print("==================================")
                state.groundedness = groundedness_report
                state.groundedness_score = groundedness_report.get("score", 0.0)
                state.confidence = calculate_confidence(c_retrieval, c_consensus, groundedness_report)
            else:
                state.confidence = calculate_confidence_lite(c_retrieval, c_consensus)
    except Exception as exc:
        logger.warning("Confidence scorer gặp lỗi: %s", exc)
        state.add_error("confidence", str(exc))
        state.confidence = {
            "c_retrieval": None,
            "c_consensus": None,
            "c_groundedness": None,
            "c_rag": None,
            "status": "unavailable",
            "errors": [str(exc)],
        }

    return state


def router_node(state: AgentState) -> AgentState:
    """Node 7: Ra quyết định routing cuối cùng (auto, draft, clarify, escalate)."""
    return route_ticket(state)


# ==============================================================================
# CONDITIONAL EDGES (CƠ CHẾ NGẮT LUỒNG SỚM - FAIL-SAFE SHORT CIRCUITING)
# ==============================================================================


def check_early_exit_after_similarity(state: AgentState) -> Literal["router", "classify"]:
    """Nếu là spam dồn dập hoặc đã được xử lý bởi Incident -> ngắt luồng sớm."""
    if state.is_spam_duplicate or state.repeat_ticket_suspected:
        return "router"
    if state.incident_id and state.routing_decision in ("auto", "escalate"):
        return "router"
    return "classify"


def check_early_exit_after_classify(state: AgentState) -> Literal["router", "generation", "diagnostic_tool"]:
    """Nếu ticket bị out-of-scope, vi phạm an ninh, prompt injection hoặc tức giận -> ngắt luồng sớm sang Router.
    Nếu có missing_fields -> chuyển thẳng sang Generation (dùng clarify prompt).
    Nếu đầy đủ thông tin -> chuyển sang Diagnostic Tool."""
    cls = state.classification or {}
    category = str(cls.get("category", "")).upper()
    sentiment = str(cls.get("sentiment_flag", "")).lower()

    if (
        category == "OUT_OF_SCOPE"
        or cls.get("security_concern") is True
        or cls.get("prompt_injection_suspected") is True
        or cls.get("multi_issue_suspected") is True
        or sentiment == "angry"
    ):
        return "router"

    if state.missing_fields:
        if state.clarification_rounds >= 1:
            return "router"
        return "generation"

    return "diagnostic_tool"


def check_retrieval_exit(state: AgentState) -> Literal["router", "generation"]:
    """Nếu không tìm thấy tài liệu KB nào hợp lệ -> chuyển thẳng sang Router để Escalate."""
    if not state.retrieved_parents:
        return "router"
    return "generation"


def check_generation_exit(state: AgentState) -> Literal["router", "confidence"]:
    """Nếu là luồng clarify (có missing_fields) -> chuyển thẳng Router, bỏ qua Confidence."""
    if state.missing_fields:
        return "router"
    return "confidence"


# ==============================================================================
# KHỞI TẠO STATEGRAPH WORKFLOW
# ==============================================================================


def build_pipeline_graph(
    repo: TicketRepository | None = None,
    adapter: DiagnosticBackendAdapter | None = None,
) -> StateGraph:
    """Xây dựng StateGraph của LangGraph cho toàn bộ pipeline Helpdesk."""
    workflow = StateGraph(AgentState)

    # 1. Thêm các node
    workflow.add_node("similarity", lambda s: similarity_node(s, repo=repo))
    workflow.add_node("classify", classify_node)
    workflow.add_node("diagnostic_tool", lambda s: diagnostic_tool_node(s, adapter=adapter))
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("generation", generation_node)
    workflow.add_node("confidence", confidence_node)
    workflow.add_node("router", router_node)

    # 2. Thêm các cạnh và rẽ nhánh có điều kiện
    workflow.add_edge(START, "similarity")
    workflow.add_conditional_edges(
        "similarity",
        check_early_exit_after_similarity,
        {"router": "router", "classify": "classify"},
    )
    workflow.add_conditional_edges(
        "classify",
        check_early_exit_after_classify,
        {"router": "router", "generation": "generation", "diagnostic_tool": "diagnostic_tool"},
    )
    workflow.add_edge("diagnostic_tool", "retrieval")
    workflow.add_conditional_edges(
        "retrieval",
        check_retrieval_exit,
        {"router": "router", "generation": "generation"},
    )
    workflow.add_conditional_edges(
        "generation",
        check_generation_exit,
        {"router": "router", "confidence": "confidence"},
    )
    workflow.add_edge("confidence", "router")
    workflow.add_edge("router", END)

    return workflow


# Compiled application singleton
_pipeline_app = None


def get_pipeline_app(
    repo: TicketRepository | None = None,
    adapter: DiagnosticBackendAdapter | None = None,
):
    """Lấy hoặc biên dịch pipeline ứng dụng từ LangGraph."""
    global _pipeline_app
    if _pipeline_app is None or repo is not None or adapter is not None:
        graph = build_pipeline_graph(repo=repo, adapter=adapter)
        _pipeline_app = graph.compile()
    return _pipeline_app


def process_ticket(
    subject: str,
    body: str,
    requester_email: str = "user@company.com",
    ticket_id: str = "TICK_001",
    *,
    automation_stage: str = "draft",
    clarification_rounds: int = 0,
    repo: TicketRepository | None = None,
    adapter: DiagnosticBackendAdapter | None = None,
) -> AgentState:
    """Entrypoint chính xử lý 1 ticket qua toàn bộ pipeline hoàn chỉnh."""
    initial_state = AgentState(
        ticket_id=ticket_id,
        requester_email=requester_email,
        subject=subject,
        body=body,
        automation_stage=automation_stage,
        clarification_rounds=clarification_rounds,
    )

    app = get_pipeline_app(repo=repo, adapter=adapter)
    final_output = app.invoke(initial_state)

    # LangGraph invoke trả về dict hoặc AgentState instance
    if isinstance(final_output, dict):
        # Đồng bộ ngược vào instance AgentState
        for key, val in final_output.items():
            if hasattr(initial_state, key):
                setattr(initial_state, key, val)
        return initial_state

    return final_output


