"""Kiểm thử hybrid và confidence, không cần DB/model/API thật."""
import copy
import json
import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pydantic import SecretStr
from src.rag import retriever as rag
from src import confidence as conf
from src.agent import llm_client
from src.core.config import Settings


def parent(pid, score=0.8):
    return {"parent_chunk_id": pid, "parent_index": 0, "text": pid,
            "article_title": "mail", "similarity": score, "bm25_score": 0.0}


CONTEXT = [{"source": "mail", "content": "Tắt Outlook. Mở Control Panel. Tạo profile. Khởi động lại."}]
REPORT = {"claims": [
    {"claim": "Tắt Outlook.", "verdict": "supported", "evidence": [{"source": "mail", "quote": "Tắt Outlook."}]},
    {"claim": "Mở Control Panel.", "verdict": "supported", "evidence": [{"source": "mail", "quote": "Mở Control Panel."}]},
    {"claim": "Tạo profile.", "verdict": "supported", "evidence": [{"source": "mail", "quote": "Tạo profile."}]},
    {"claim": "Không mất dữ liệu.", "verdict": "unsupported", "evidence": []},
], "reason": "Một claim thiếu căn cứ.", "citation_support": "supported"}
SETTINGS = SimpleNamespace(mistral_api_key=None, mistral_model="mistral-small-latest",
    groundedness_model="qwen3.5:4b", judge_num_ctx=32768, judge_max_tokens=4096, judge_seed=42,
    ollama_host="http://localhost:11434", ollama_model="qwen3.5:4b", llm_timeout_seconds=10,
    openai_api_key=None)


class HybridTests(unittest.TestCase):
    def test_bm25_unicode_positive_idf_empty_and_query_duplicates(self):
        self.assertEqual(rag.tokenize("LỖI 0x80040600"), rag.tokenize("lỗi 0x80040600"))
        self.assertEqual(rag.bm25_scores("a a", ["a", "b"]), rag.bm25_scores("a", ["a", "b"]))
        self.assertGreater(rag.bm25_scores("a", ["a", "a"])[0], 0)
        self.assertEqual(rag.bm25_scores("x", ["a", "b"]), [0, 0])
        self.assertEqual(rag.bm25_scores("a", ["", ""]), [0, 0])
        self.assertEqual(rag.bm25_scores("a", []), [])
        self.assertAlmostEqual(rag.bm25_scores("a", ["a", "b"])[0], math.log(2))

    def test_rrf_normalization_and_consensus(self):
        a, b = parent("a"), parent("b")
        fused = rag.fuse_rankings([a, b], [a, b])
        self.assertEqual(fused[0]["relevance_score"], 1)
        self.assertAlmostEqual(conf.calculate_retrieval_confidence(fused), 0.7096774193548386)
        self.assertEqual(rag.fuse_rankings([a], [])[0]["relevance_score"], 0.5)
        for ranks, expected in [((1, 3), 1), ((4, 5), .75), ((1, 6), .5), ((None, 1), .5)]:
            self.assertEqual(conf.calculate_consensus(dict(zip(("dense_rank", "bm25_rank"), ranks))), expected)
        self.assertEqual(conf.calculate_retrieval_confidence([fused[0]]), .7)
        with self.assertRaises(ValueError):
            rag.fuse_rankings([a, a], [])
        self.assertEqual([p["parent_chunk_id"] for p in rag.fuse_rankings([b], [a])], ["a", "b"])

    def test_independent_channels_group_before_limit_and_filters(self):
        rows = [("a", 0, "A", "mail", .9, "ordinary"),
                ("a", 0, "A", "mail", .8, "ordinary"),
                ("b", 1, "B", "mail", .7, "other"),
                ("c", 2, "C", "mail", .1, "0x80040600")]
        cur = Mock()
        cur.fetchall.return_value = rows
        with patch.object(rag, "embed_query", return_value=[1.0]):
            fused = rag.retrieve_hybrid_parents(cur, "0x80040600", "MAIL", candidate_pool=2)
            self.assertEqual({p["parent_chunk_id"] for p in fused}, {"a", "b", "c"})
            c = next(p for p in fused if p["parent_chunk_id"] == "c")
            self.assertIsNone(c["dense_rank"])
            self.assertEqual(c["bm25_rank"], 1)
            self.assertEqual(c["similarity"], .1)
            b = next(p for p in fused if p["parent_chunk_id"] == "b")
            self.assertEqual(b["dense_rank"], 2)
            self.assertEqual(len(rag.retrieve_top_k_parents(cur, "0x80040600", "MAIL", top_k=1)), 1)
        query, params = cur.execute.call_args.args
        for alias in ("c", "p", "a"):
            self.assertIn(f"{alias}.is_active = true", query)
            self.assertIn(f"{alias}.category = %s", query)
        self.assertEqual(params[1:], ("MAIL", "MAIL", "MAIL"))

    def test_empty_lexical_and_channel_failure(self):
        cur = Mock()
        cur.fetchall.return_value = [("a", 0, "A", "mail", .5, "ordinary")]
        with patch.object(rag, "embed_query", return_value=[1.]):
            result = rag.retrieve_hybrid_parents(cur, "missing", "MAIL")
            self.assertIsNone(result[0]["bm25_rank"])
            self.assertEqual(conf.calculate_consensus(result[0]), .5)
            with patch.object(rag, "bm25_scores", side_effect=RuntimeError("broken")):
                with self.assertRaisesRegex(RuntimeError, "BM25"):
                    rag.retrieve_hybrid_parents(cur, "missing", "MAIL")
            cur.fetchall.return_value = []
            self.assertEqual(rag.retrieve_hybrid_parents(cur, "missing", "MAIL"), [])
            cur.execute.side_effect = RuntimeError("database down")
            with self.assertRaisesRegex(RuntimeError, "Dense/corpus"):
                rag.retrieve_hybrid_parents(cur, "missing", "MAIL")

    def test_embedding_cache_and_validation(self):
        model = Mock()
        model.encode.return_value = [Mock(tolist=lambda: [1.0])]
        with patch.object(rag, "_model", model):
            self.assertIs(rag.get_embedding_model(), model)
            self.assertEqual(rag.embed_query("q"), [1.0])
        for kwargs in ({"top_k": 0}, {"top_k": True}, {"candidate_pool": 0}):
            with self.assertRaises(ValueError):
                rag.retrieve_top_k_parents(Mock(), "q", "MAIL", **kwargs)


class ConfidenceTests(unittest.TestCase):
    def test_weights_errors_and_validation(self):
        grounding = conf.score_groundedness(REPORT, CONTEXT)
        self.assertEqual(grounding["score"], .75)
        result = conf.calculate_confidence(.7, 1., grounding)
        self.assertAlmostEqual(result["c_rag"], .78)
        self.assertEqual(result["status"], "ok")
        result = conf.calculate_confidence(.7, 1., conf.unavailable_groundedness("timeout"))
        self.assertEqual(result["status"], "degraded")
        self.assertEqual(result["c_groundedness"], 0)
        self.assertAlmostEqual(result["c_rag"], .48)
        self.assertIsNone(conf.calculate_confidence(None, None, grounding, retrieval_error="BM25 failed")["c_rag"])
        for value in (float("nan"), float("inf"), -1, 2, True):
            with self.assertRaises(ValueError):
                conf.calculate_confidence(value, 1., grounding)
        for parents in ([], [{"parent_chunk_id": "a", "relevance_score": 2}],
                        [{"parent_chunk_id": "a", "relevance_score": .5}] * 2,
                        [{"parent_chunk_id": "a", "relevance_score": .1}, {"parent_chunk_id": "b", "relevance_score": .5}]):
            with self.assertRaises(ValueError):
                conf.calculate_retrieval_confidence(parents)
        for ranks in ((None, None), (0, 1), (True, 2)):
            with self.assertRaises(ValueError):
                conf.calculate_consensus(dict(zip(("dense_rank", "bm25_rank"), ranks)))

    def test_claim_quotes_duplicates_and_verdict_precedence(self):
        for modification in ("source", "quote", "empty", "duplicate"):
            report = copy.deepcopy(REPORT)
            if modification in ("source", "quote"):
                report["claims"][0]["evidence"][0][modification] = "invented"
            elif modification == "empty":
                report["claims"][0]["evidence"] = []
            else:
                report["claims"].append(report["claims"][0])
            with self.assertRaises(ValueError):
                conf.score_groundedness(report, CONTEXT)
        report = copy.deepcopy(REPORT)
        report["claims"][0]["verdict"] = "contradicted"
        report["claims"][1]["verdict"] = "uncertain"
        result = conf.score_groundedness(report, CONTEXT)
        self.assertEqual(result["verdict"], "contradicted")
        self.assertEqual(result["score"], .25)
        self.assertEqual(conf.calculate_confidence(.7, 1., result)["status"], "degraded")

    def test_judge_provider_selection_payload_and_fallback(self):
        for key, provider in ((None, "auto"), (SecretStr(" "), "auto"), (SecretStr("test-key"), "mistral")):
            settings = copy.copy(SETTINGS)
            settings.mistral_api_key = key
            with patch.object(conf, "get_settings", return_value=settings), patch.object(conf, "generate", return_value=json.dumps(REPORT)) as generate:
                result = conf.evaluate_groundedness("Question", "Answer [mail]", CONTEXT)
                self.assertEqual(result["score"], .75)
                self.assertEqual(generate.call_args.kwargs["provider"], provider)
                self.assertEqual(generate.call_args.kwargs["seed"], 42)
                self.assertEqual(generate.call_args.kwargs["max_tokens"], 4096)
                self.assertNotIn("ground_truth", generate.call_args.args[0])
                self.assertEqual(json.loads(generate.call_args.args[0])["kb_contexts"], CONTEXT)
        with patch.object(conf, "get_settings", return_value=SETTINGS), patch.object(conf, "generate") as generate:
            for output in ("bad json", '{"claims": [], "reason":"none", "citation_support":"uncertain"}',
                           RuntimeError("network"), llm_client.GenerationError("truncated", "partial")):
                generate.side_effect = output if isinstance(output, Exception) else None
                generate.return_value = output
                result = conf.evaluate_groundedness("q", "answer", CONTEXT)
                self.assertEqual((result["score"], result["status"]), (0., "unavailable"))
            generate.reset_mock()
            for answer, complete in (("", True), ("partial", False)):
                self.assertEqual(conf.evaluate_groundedness("q", answer, CONTEXT, answer_complete=complete)["status"], "unavailable")
            generate.assert_not_called()

    def test_invalid_citation_does_not_change_claim_score(self):
        with patch.object(conf, "get_settings", return_value=SETTINGS), patch.object(conf, "generate", return_value=json.dumps(REPORT)):
            result = conf.evaluate_groundedness("q", "answer [typo]", CONTEXT)
        self.assertEqual(result["score"], .75)
        self.assertEqual(result["citation_support"], "unsupported")
        self.assertEqual(conf.calculate_confidence(.7, 1., result)["status"], "degraded")


class ProviderTests(unittest.TestCase):
    def test_mistral_schema_seed_budget_and_failure(self):
        settings = copy.copy(SETTINGS)
        settings.mistral_api_key = SecretStr("test-key")
        response = Mock(is_error=False)
        response.json.return_value = {"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}], "usage": {}}
        with patch.object(llm_client, "get_settings", return_value=settings), patch.object(llm_client.httpx, "post", return_value=response) as post, patch.object(llm_client, "ChatOllama") as ollama:
            self.assertEqual(llm_client.generate("q", provider="mistral", system_prompt="sys", response_schema={"type":"object"}, seed=42, max_tokens=4096), "{}")
            payload = post.call_args.kwargs["json"]
            self.assertEqual(payload["random_seed"], 42)
            self.assertEqual(payload["response_format"]["json_schema"]["schema"], {"type":"object"})
            self.assertEqual(payload["max_tokens"], 4096)
            self.assertEqual(payload["messages"][0]["role"], "system")
            ollama.assert_not_called()
            response.json.return_value["choices"][0]["finish_reason"] = "length"
            with self.assertRaises(llm_client.GenerationError):
                llm_client.generate("q", provider="mistral")
            response.is_error, response.status_code = True, 401
            with self.assertRaisesRegex(ValueError, "401"):
                llm_client.generate("q", provider="mistral")

    def test_context_budget_and_settings(self):
        with patch.object(llm_client, "ChatOllama") as ollama:
            with self.assertRaisesRegex(ValueError, "Context"):
                llm_client.generate("x" * 100, model_name="local", num_ctx=4300, max_tokens=4096)
            ollama.assert_not_called()
        settings = Settings(_env_file=None, db_name="test", db_user="test", db_password="test", mistral_api_key="secret")
        self.assertEqual(settings.groundedness_model, "qwen3.5:4b")
        self.assertNotIn("secret", repr(settings))
        with self.assertRaises(ValueError):
            Settings(_env_file=None, db_name="test", db_user="test", db_password="test", judge_num_ctx=2048)


if __name__ == "__main__":
    unittest.main()
