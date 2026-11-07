"""Contract tests for every decision path in retriever, confidence and LLM client."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import SecretStr

from src.agent import llm_client
from src import confidence
from src.rag import retriever


def settings(**overrides):
    values = {
        "openrouter_api_key": None,
        "openrouter_model": "openrouter-test",
        "openrouter_base_url": "https://openrouter.ai/api/v1",
        "ollama_model": "ollama-test",
        "ollama_host": "http://ollama",
        "llm_timeout_seconds": 10,
        "groundedness_model": "judge-test",
        "judge_num_ctx": 20_000,
        "judge_max_tokens": 100,
        "judge_seed": 42,
        "embedding_model": "embedding-test",
        "top_k": 2,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def context(source="kb", content="Exact quote."):
    return [{"parent_chunk_id": "p1", "source": source, "content": content}]


def claim(verdict="supported", evidence=None, text="Claim"):
    if evidence is None:
        evidence = [{"context_id": "E1"}]
    return {"claim": text, "verdict": verdict, "evidence": evidence}


def report(*claims):
    return {"claims": list(claims), "reason": "reason"}


class FakeMessage:
    def __init__(self, content=" answer ", **metadata):
        self.content = content
        self.response_metadata = metadata


def test_embedding_model_cache_and_creation(monkeypatch):
    model = Mock()
    constructor = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=constructor))
    monkeypatch.setattr(retriever, "get_settings", lambda: settings())
    monkeypatch.setattr(retriever, "_model", None)

    assert retriever.get_embedding_model() is model
    assert retriever.get_embedding_model() is model
    constructor.assert_called_once_with("embedding-test", local_files_only=True)

    model.encode.return_value = [SimpleNamespace(tolist=lambda: [0.1, 0.2])]
    assert retriever.embed_query("question") == [0.1, 0.2]


def test_bm25_and_rrf_all_paths():
    assert retriever.tokenize("LỖI 01") == retriever.tokenize("lỗi 01")
    assert retriever.bm25_scores("x", []) == []
    assert retriever.bm25_scores("x", ["", ""]) == [0.0, 0.0]
    scores = retriever.bm25_scores("mail absent", ["mail mail", "other"])
    assert scores[0] > 0 and scores[1] == 0

    a = {"parent_chunk_id": "a", "text": "A"}
    b = {"parent_chunk_id": "b", "text": "B"}
    assert retriever.fuse_rankings([], []) == []
    fused = retriever.fuse_rankings([a, b], [b, a])
    assert [(p["parent_chunk_id"], p["dense_rank"], p["bm25_rank"]) for p in fused] == [
        ("a", 1, 2), ("b", 2, 1)
    ]
    assert retriever.fuse_rankings([a], [])[0]["bm25_rank"] is None
    with pytest.raises(ValueError, match="dense"):
        retriever.fuse_rankings([a, a], [])
    with pytest.raises(ValueError, match="bm25"):
        retriever.fuse_rankings([], [a, a])


@pytest.mark.parametrize("candidate", [0, True, 1.5])
def test_retrieve_rejects_invalid_candidate_pool(candidate):
    with pytest.raises(ValueError, match="candidate_pool"):
        retriever.retrieve_hybrid_parents(Mock(), "q", "MAIL", candidate)


@pytest.mark.parametrize(("question", "category"), [(" ", "MAIL"), ("q", " ")])
def test_retrieve_rejects_blank_input(question, category):
    with pytest.raises(ValueError, match="must not be blank"):
        retriever.retrieve_hybrid_parents(Mock(), question, category)


def test_retrieve_groups_children_and_keeps_independent_channels(monkeypatch):
    rows = [
        (1, 0, "Parent A", "A", 0.4, "ordinary"),
        (1, 0, "Parent A", "A", 0.9, "ordinary"),
        (2, 1, "Parent B", "B", -1.0, "rare"),
    ]
    cursor = Mock()
    cursor.fetchall.return_value = rows
    monkeypatch.setattr(retriever, "embed_query", lambda _: [1.0])
    monkeypatch.setattr(retriever, "bm25_scores", lambda *_: [0.0, 0.0, 2.0])

    result = retriever.retrieve_hybrid_parents(cursor, "q", "MAIL", candidate_pool=1)
    by_id = {p["parent_chunk_id"]: p for p in result}
    assert by_id["1"]["similarity"] == 0.9
    assert by_id["1"]["dense_rank"] == 1 and by_id["1"]["bm25_rank"] is None
    assert by_id["2"]["dense_rank"] is None and by_id["2"]["bm25_rank"] == 1


@pytest.mark.parametrize("similarity", [float("nan"), float("inf"), -1.1, 1.1, "bad"])
def test_retrieve_wraps_invalid_dense_results(monkeypatch, similarity):
    cursor = Mock()
    cursor.fetchall.return_value = [("p", 0, "text", "title", similarity, "child")]
    monkeypatch.setattr(retriever, "embed_query", lambda _: [1.0])
    with pytest.raises(RuntimeError, match="Dense/corpus retrieval failed"):
        retriever.retrieve_hybrid_parents(cursor, "q", "MAIL")


def test_retrieve_wraps_dense_and_bm25_failures(monkeypatch):
    cursor = Mock()
    monkeypatch.setattr(retriever, "embed_query", Mock(side_effect=OSError("model")))
    with pytest.raises(RuntimeError, match=r"\(OSError\)"):
        retriever.retrieve_hybrid_parents(cursor, "q", "MAIL")

    monkeypatch.setattr(retriever, "embed_query", lambda _: [1.0])
    cursor.fetchall.return_value = [("p", 0, "text", "title", 0.5, "child")]
    monkeypatch.setattr(retriever, "bm25_scores", Mock(side_effect=KeyError("term")))
    with pytest.raises(RuntimeError, match=r"BM25 retrieval failed \(KeyError\)"):
        retriever.retrieve_hybrid_parents(cursor, "q", "MAIL")


def test_top_k_default_explicit_and_invalid(monkeypatch):
    parents = [{"parent_chunk_id": str(i)} for i in range(3)]
    retrieve = Mock(return_value=parents)
    monkeypatch.setattr(retriever, "retrieve_hybrid_parents", retrieve)
    monkeypatch.setattr(retriever, "get_settings", lambda: settings(top_k=2))
    assert retriever.retrieve_top_k_parents(Mock(), "q", "MAIL") == parents[:2]
    assert retriever.retrieve_top_k_parents(Mock(), "q", "MAIL", 1, 7) == parents[:1]
    assert retrieve.call_args.args[-1] == 7
    for invalid in (0, True, 1.5):
        with pytest.raises(ValueError, match="top_k"):
            retriever.retrieve_top_k_parents(Mock(), "q", "MAIL", invalid)


@pytest.mark.parametrize("value", [True, "1", float("nan"), float("inf"), -0.1, 1.1])
def test_unit_score_rejects_every_invalid_kind(value):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        confidence._unit_score(value)
    assert confidence._unit_score(0) == 0
    assert confidence._unit_score(1.0) == 1.0


def test_retrieval_confidence_all_paths():
    with pytest.raises(ValueError, match="Không có"):
        confidence.calculate_retrieval_confidence([])
    with pytest.raises(ValueError, match="trùng"):
        confidence.calculate_retrieval_confidence([
            {"parent_chunk_id": "p", "relevance_score": 1.0},
            {"parent_chunk_id": "p", "relevance_score": 0.5},
        ])
    with pytest.raises(ValueError, match="xếp giảm"):
        confidence.calculate_retrieval_confidence([
            {"parent_chunk_id": "a", "relevance_score": 0.2},
            {"parent_chunk_id": "b", "relevance_score": 0.8},
        ])
    assert confidence.calculate_retrieval_confidence([{"parent_chunk_id": "a", "relevance_score": 0.5}]) == 0.35
    assert confidence.calculate_retrieval_confidence([
        {"parent_chunk_id": "a", "relevance_score": 1.0},
        {"parent_chunk_id": "b", "relevance_score": 0.0},
    ]) == 1.0


@pytest.mark.parametrize(("ranks", "expected"), [((1, 3), 1.0), ((4, 5), 0.75), ((1, 6), 0.5), ((None, 1), 0.5)])
def test_consensus_levels(ranks, expected):
    assert confidence.calculate_consensus({"dense_rank": ranks[0], "bm25_rank": ranks[1]}) == expected


@pytest.mark.parametrize("ranks", [(None, None), (0, 1), (True, 1), (1.5, 1)])
def test_consensus_rejects_invalid_ranks(ranks):
    with pytest.raises(ValueError):
        confidence.calculate_consensus({"dense_rank": ranks[0], "bm25_rank": ranks[1]})


def test_groundedness_scoring_validation_and_verdict_precedence():
    with pytest.raises(ValueError, match="Không có claim"):
        confidence.score_groundedness(report(), context())
    with pytest.raises(ValueError, match="trùng lặp"):
        confidence.score_groundedness(report(claim(text=" Same "), claim(text="same")), context())
    with pytest.raises(ValueError, match="trùng lặp"):
        confidence.score_groundedness(report(claim(text=" ")), context())
    for verdict in ("supported", "contradicted"):
        with pytest.raises(ValueError, match="cần evidence"):
            confidence.score_groundedness(report(claim(verdict, [])), context())
    with pytest.raises(confidence.EvidenceValidationError) as error:
        confidence.score_groundedness(report(claim(evidence=[{"context_id": "E2"}])), context())
    assert error.value.details["code"] == "unknown_context_id"
    with pytest.raises(ValueError, match="Evidence context bị trùng"):
        confidence.score_groundedness(
            report(claim(evidence=[{"context_id": "E1"}, {"context_id": "E1"}])), context()
        )

    unsupported = claim("unsupported", [], "u")
    uncertain = claim("uncertain", [], "n")
    supported = claim(text="s")
    contradicted = claim("contradicted", text="c")
    for claims, verdict in [
        ([supported], "supported"),
        ([supported, uncertain], "uncertain"),
        ([supported, unsupported], "unsupported"),
        ([supported, contradicted], "contradicted"),
    ]:
        result = confidence.score_groundedness(report(*claims), context())
        assert result["verdict"] == verdict
        assert result["score"] == sum(c["verdict"] == "supported" for c in claims) / len(claims)
        assert result["claims"][0]["evidence"][0]["source"] == "kb"

    with pytest.raises(ValueError):
        confidence.score_groundedness(report(*(claim(text=str(i)) for i in range(13))), context())
    with pytest.raises(ValueError):
        confidence.score_groundedness(
            report(claim(evidence=[{"context_id": f"E{i}"} for i in range(1, 5)])),
            context(),
        )


def test_evaluate_groundedness_input_and_diagnostics(monkeypatch):
    generate = Mock()
    monkeypatch.setattr(confidence, "generate", generate)
    for answer, complete, contexts in [
        ("", True, context()), ("answer", False, context()), ("answer", True, []),
        ("answer", True, [{"source": " ", "content": "x"}]),
        ("answer", True, [{"source": "x", "content": " "}]),
    ]:
        diagnostic = {"old": True}
        result = confidence.evaluate_groundedness("q", answer, contexts, answer_complete=complete, diagnostics=diagnostic)
        assert result["status"] == "unavailable"
        assert diagnostic["stage"] == "input_validation" and "old" not in diagnostic
    generate.assert_not_called()


@pytest.mark.parametrize(("key", "diagnostic_provider", "use_cloud", "model"), [
    (None, "ollama", False, "judge-test"),
    (SecretStr(" "), "ollama", False, "judge-test"),
    (SecretStr("secret"), "openrouter", True, "openrouter-test"),
])
def test_evaluate_groundedness_provider_success(
    monkeypatch, key, diagnostic_provider, use_cloud, model
):
    generated = Mock(return_value=json.dumps(report(claim())))
    monkeypatch.setattr(confidence, "get_settings", lambda: settings(openrouter_api_key=key))
    monkeypatch.setattr(confidence, "generate", generated)
    diagnostic = {}
    result = confidence.evaluate_groundedness("q", "answer", context(), num_ctx=12345, diagnostics=diagnostic)
    assert result["status"] == "ok" and diagnostic["stage"] == "complete"
    assert diagnostic["provider"] == diagnostic_provider and diagnostic["model"] == model
    assert diagnostic["response"]["raw_response"] == generated.return_value
    assert generated.call_args.kwargs["use_cloud_model"] == use_cloud
    assert generated.call_args.kwargs["num_ctx"] == 12345
    assert "diagnostics" in generated.call_args.kwargs
    schema = generated.call_args.kwargs["response_schema"]
    assert schema["properties"]["claims"]["maxItems"] == 12
    assert schema["$defs"]["Claim"]["properties"]["evidence"]["maxItems"] == 3
    task_data = json.loads(generated.call_args.args[0])
    assert task_data["kb_contexts"] == [{"context_id": "E1", "content": "Exact quote."}]


def test_evaluate_groundedness_failure_stages(monkeypatch):
    monkeypatch.setattr(confidence, "get_settings", lambda: settings())

    diagnostic = {}
    monkeypatch.setattr(confidence, "generate", Mock(side_effect=ValueError("provider failed")))
    result = confidence.evaluate_groundedness("q", "answer", context(), diagnostics=diagnostic)
    assert result["error"] == "provider failed" and diagnostic["stage"] == "generation"

    diagnostic = {}
    monkeypatch.setattr(confidence, "generate", Mock(return_value="not json"))
    result = confidence.evaluate_groundedness("q", "answer", context(), diagnostics=diagnostic)
    assert result["error"] == "ValidationError" and diagnostic["stage"] == "schema_validation"
    assert diagnostic["schema_errors"]

    bad = report(claim(evidence=[{"context_id": "E2"}]))
    diagnostic = {}
    monkeypatch.setattr(confidence, "generate", Mock(return_value=json.dumps(bad)))
    result = confidence.evaluate_groundedness("q", "answer", context(), diagnostics=diagnostic)
    assert result["status"] == "unavailable" and diagnostic["stage"] == "evidence_validation"
    assert diagnostic["rejected_evidence"]["code"] == "unknown_context_id"

    monkeypatch.setattr(confidence, "generate", Mock(side_effect=RuntimeError("secret details")))
    result = confidence.evaluate_groundedness("q", "answer", context())
    assert result["error"] == "RuntimeError"


def test_calculate_confidence_all_paths():
    ok = {"status": "ok", "score": 0.5, "error": None, "verdict": "supported"}
    assert confidence.calculate_confidence(None, 1.0, ok)["status"] == "unavailable"
    assert confidence.calculate_confidence(1.0, None, ok)["errors"] == ["Retrieval unavailable"]
    assert confidence.calculate_confidence(1.0, 1.0, ok, retrieval_error="db")["errors"] == ["db"]
    with pytest.raises(ValueError, match="Consensus"):
        confidence.calculate_confidence(1.0, 0.6, ok)
    with pytest.raises(ValueError, match="status"):
        confidence.calculate_confidence(1.0, 1.0, {"status": "broken"})
    with pytest.raises(ValueError, match="không được có error"):
        confidence.calculate_confidence(1.0, 1.0, ok | {"error": "bad"})
    unavailable = confidence.unavailable_groundedness("")
    result = confidence.calculate_confidence(1.0, 1.0, unavailable)
    assert result["status"] == "degraded" and result["errors"] == ["Groundedness unavailable"]
    contradicted = ok | {"verdict": "contradicted"}
    assert confidence.calculate_confidence(1.0, 1.0, contradicted)["status"] == "degraded"
    assert confidence.calculate_confidence(1.0, 1.0, ok)["status"] == "ok"


def test_api_key_prompt_generation_error_and_build_answer(monkeypatch):
    monkeypatch.setattr(llm_client, "get_settings", lambda: settings(openrouter_api_key=SecretStr(" key ")))
    assert llm_client.check_api_key()
    assert not llm_client.check_api_key(settings(openrouter_api_key=None))
    assert not llm_client.check_api_key(settings(openrouter_api_key=SecretStr(" ")))
    assert llm_client.render_prompt(
        "clarify_task.jinja2", ticket_subject="subject", ticket_body="body", missing_fields=["field"]
    )

    error = llm_client.GenerationError("bad", "partial")
    assert str(error) == "bad" and error.generation == {"answer": "partial"}

    contexts = [
        {"parent_chunk_id": "p1", "source": "A"},
        {"parent_chunk_id": "p1", "source": "A"},
        {"parent_chunk_id": "p2", "source": "B"},
    ]
    assert llm_client.build_answer("answer", contexts) == {
        "answer": "answer",
        "sources": [{"parent_chunk_id": "p1", "title": "A"}, {"parent_chunk_id": "p2", "title": "B"}],
    }
    for bad in [
        {}, {"parent_chunk_id": 1, "source": "A"}, {"parent_chunk_id": " ", "source": "A"},
        {"parent_chunk_id": "p", "source": 1}, {"parent_chunk_id": "p", "source": " "},
    ]:
        with pytest.raises(ValueError, match="Source metadata"):
            llm_client.build_answer("answer", [bad])
    with pytest.raises(ValueError, match="Conflicting"):
        llm_client.build_answer("answer", [
            {"parent_chunk_id": "p", "source": "A"}, {"parent_chunk_id": "p", "source": "B"}
        ])


@pytest.mark.parametrize(("kwargs", "message"), [
    ({"prompt": " "}, "prompt"),
    ({"prompt": "q", "max_tokens": 0}, "max_tokens"),
    ({"prompt": "q", "max_tokens": True}, "max_tokens"),
    ({"prompt": "q", "model_name": " "}, "model_name"),
    ({"prompt": "q", "num_ctx": 300, "max_tokens": 100}, "Context"),
])
def test_generate_rejects_invalid_arguments(monkeypatch, kwargs, message):
    diagnostic = {"old": True}
    with pytest.raises(ValueError, match=message):
        llm_client.generate(**kwargs, diagnostics=diagnostic)
    assert diagnostic == {}


def make_openrouter_mock(content=" result ", finish="stop", prompt_tokens=3, completion_tokens=4, error=None):
    if error:
        create = Mock(side_effect=error)
    else:
        choice = Mock(message=Mock(content=content), finish_reason=finish)
        usage = Mock(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        completion = Mock(choices=[choice], usage=usage)
        create = Mock(return_value=completion)
    client = Mock(chat=Mock(completions=Mock(create=create)))
    return client, create


def test_generate_openrouter_missing_key_and_api_error(monkeypatch):
    monkeypatch.setattr(llm_client, "get_settings", lambda: settings())
    mock_ollama = Mock(return_value="ollama_fallback")
    monkeypatch.setattr(llm_client, "_ollama_generate", mock_ollama)
    with pytest.warns(UserWarning, match="OPENROUTER_API_KEY"):
        assert llm_client.generate("q", use_cloud_model=True) == "ollama_fallback"
    monkeypatch.setattr(
        llm_client, "get_settings", lambda: settings(openrouter_api_key=SecretStr(" "))
    )
    with pytest.warns(UserWarning, match="OPENROUTER_API_KEY"):
        assert llm_client.generate("q", use_cloud_model=True) == "ollama_fallback"

    monkeypatch.setattr(llm_client, "get_settings", lambda: settings(openrouter_api_key=SecretStr("key")))
    client, _ = make_openrouter_mock(error=Exception("429 rate limit"))
    monkeypatch.setattr(llm_client, "OpenAI", Mock(return_value=client))
    with pytest.raises(ValueError, match="429"):
        llm_client.generate("q", use_cloud_model=True)


def test_generate_openrouter_requests_diagnostics_and_truncation(monkeypatch):
    monkeypatch.setattr(llm_client, "get_settings", lambda: settings(openrouter_api_key=SecretStr(" key ")))
    client, create = make_openrouter_mock(" json ", prompt_tokens=3, completion_tokens=4)
    monkeypatch.setattr(llm_client, "OpenAI", Mock(return_value=client))
    diagnostic = {}
    result = llm_client.generate(
        "q", use_cloud_model=True, model_name=" model ", system_prompt="system", seed=7,
        response_schema={"type": "object"}, diagnostics=diagnostic,
    )
    assert result == "json"
    kwargs = create.call_args.kwargs
    assert kwargs["model"] == "model" and kwargs["messages"][0]["role"] == "system"
    assert kwargs["seed"] == 7 and kwargs["response_format"]["type"] == "json_schema"
    assert diagnostic == {"raw_response": " json ", "finish_reason": "stop", "input_tokens": 3, "output_tokens": 4}

    client, create = make_openrouter_mock("partial", finish="length")
    monkeypatch.setattr(llm_client, "OpenAI", Mock(return_value=client))
    with pytest.raises(llm_client.GenerationError) as error:
        llm_client.generate("q", use_cloud_model=True)
    assert error.value.generation["answer"] == "partial"

    client, create = make_openrouter_mock(None, finish="length")
    monkeypatch.setattr(llm_client, "OpenAI", Mock(return_value=client))
    with pytest.raises(llm_client.GenerationError) as error:
        llm_client.generate("q", use_cloud_model=True)
    assert error.value.generation["answer"] == ""


def test_generate_openrouter_minimal_request_and_empty_results(monkeypatch):
    monkeypatch.setattr(llm_client, "get_settings", lambda: settings(openrouter_api_key=SecretStr("key")))
    client, create = make_openrouter_mock()
    monkeypatch.setattr(llm_client, "OpenAI", Mock(return_value=client))
    assert llm_client.generate("q", use_cloud_model=True) == "result"
    kwargs = create.call_args.kwargs
    assert "seed" not in kwargs and "response_format" not in kwargs
    assert kwargs["messages"] == [{"role": "user", "content": "q"}]
    for empty in (" ", None):
        client, create = make_openrouter_mock(empty)
        monkeypatch.setattr(llm_client, "OpenAI", Mock(return_value=client))
        with pytest.raises(ValueError, match="empty response"):
            llm_client.generate("q", use_cloud_model=True)


def test_generate_openrouter_auto_selection(monkeypatch):
    config = settings(openrouter_api_key=SecretStr(" key "))
    monkeypatch.setattr(llm_client, "get_settings", lambda: config)
    mock_openrouter = Mock(return_value="answer")
    monkeypatch.setattr(llm_client, "_openrouter_generate", mock_openrouter)
    assert llm_client.generate("q", system_prompt="sys", use_cloud_model=True) == "answer"
    assert mock_openrouter.call_args.args[0] == "q"


def test_generate_ollama_options_messages_diagnostics_and_failures(monkeypatch):
    monkeypatch.setattr(llm_client, "get_settings", lambda: settings(openrouter_api_key=None))
    client = Mock()
    client.invoke.return_value = FakeMessage(" answer ", done_reason="stop", prompt_eval_count=3, eval_count=4)
    constructor = Mock(return_value=client)
    monkeypatch.setattr(llm_client, "ChatOllama", constructor)
    diagnostic = {}
    result = llm_client.generate(
        "q", model_name=" local ", system_prompt="sys", num_ctx=10_000,
        response_schema={"type": "object"}, seed=1, max_tokens=10, diagnostics=diagnostic,
    )
    assert result == "answer" and client.invoke.call_args.args[0] == [("system", "sys"), ("human", "q")]
    options = constructor.call_args.kwargs
    assert options["model"] == "local" and options["temperature"] == 0 and options["num_predict"] == 10
    assert diagnostic["finish_reason"] == "stop" and diagnostic["input_tokens"] == 3

    client.invoke.return_value = FakeMessage("partial", done_reason="length")
    with pytest.raises(llm_client.GenerationError) as error:
        llm_client.generate("q", model_name="local")
    assert error.value.generation["answer"] == "partial"
    client.invoke.return_value = FakeMessage(None, done_reason="length")
    with pytest.raises(llm_client.GenerationError) as error:
        llm_client.generate("q", model_name="local")
    assert error.value.generation["answer"] == ""

    client.invoke.return_value = FakeMessage(" ", done_reason="stop")
    with pytest.raises(ValueError, match="empty response"):
        llm_client.generate("q", model_name="local")


def test_generate_ollama_default_model_no_system(monkeypatch):
    monkeypatch.setattr(llm_client, "get_settings", lambda: settings())
    client = Mock()
    client.invoke.return_value = FakeMessage("ok", done_reason="stop")
    constructor = Mock(return_value=client)
    monkeypatch.setattr(llm_client, "ChatOllama", constructor)
    assert llm_client.generate("q") == "ok"
    assert constructor.call_args.kwargs["model"] == "ollama-test"
    assert constructor.call_args.kwargs["temperature"] == 0.2
    assert client.invoke.call_args.args[0] == [("human", "q")]
