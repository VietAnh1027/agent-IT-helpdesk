"""Kiểm tra provider, cách ly prompt và JSON kết quả bằng dữ liệu giả."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from eval.harness import eval_generation as harness
from src.agent import llm_client
from src import confidence
from src.agent.llm_client import build_answer

TICKET = {
    "id": "T1", "question": "Outlook offline", "category": "MAIL",
    "query_type": "in_domain_single_hop", "human_handoff": False,
    "ground_truth_answer": "REFERENCE_SECRET",
    "ground_truth_contexts": [{"source_doc": "mail.pdf", "evidence_text": "GOLD_SECRET"}],
}
ANSWER = "Tắt Work Offline."
CORRECT = {"verdict": "correct", "reason": "Đúng thao tác."}
SUPPORTED = {"claims": [{"claim": "Tắt Work Offline.", "verdict": "supported",
              "evidence": [{"context_id": "E1"}]}],
             "reason": "Có căn cứ."}
SETTINGS = SimpleNamespace(top_k=3, openrouter_api_key=None, groundedness_model="qwen3.5:4b",
                           openrouter_model="openrouter-test", judge_num_ctx=32768,
                           judge_max_tokens=4096, judge_seed=42)
PARENTS = [{"parent_chunk_id": "p1", "article_title": "mail", "text": "Tắt Work Offline."}]
REMOVED = {"citation_check", "confidence_score", "routing_decision", "raw_outputs", "latency_seconds"}


class GenerationTests(unittest.TestCase):
    def setUp(self):
        client_settings = SimpleNamespace(openrouter_api_key=None, openrouter_model="openrouter-test",
            llm_timeout_seconds=10, ollama_host="http://localhost:11434", ollama_model="qwen3.5:4b")
        patcher = patch.object(llm_client, "get_settings", return_value=client_settings)
        patcher.start()
        self.addCleanup(patcher.stop)

    def evaluate(self, outputs, ticket=TICKET):
        with patch.object(harness, "get_connection"), patch.object(
            harness, "retrieve_top_k_parents", return_value=PARENTS
        ), patch.object(harness, "generate", side_effect=[
            json.dumps(o) if isinstance(o, dict) else o for o in outputs
        ]) as generate, patch.object(confidence, "generate", generate), patch.object(confidence, "get_settings", return_value=SETTINGS):
            result = harness.evaluate_ticket(ticket, top_k=5)
            return result, generate.call_args_list

    def test_prompts_and_reference_isolation(self):
        result, calls = self.evaluate([ANSWER, CORRECT, SUPPORTED])
        self.assertFalse(result["errors"])
        self.assertEqual(result["generation"], {"answer": ANSWER, "sources": [{"parent_chunk_id": "p1", "title": "mail"}], "complete": True})
        self.assertIsNone(calls[0].kwargs["response_schema"])
        self.assertIsNotNone(calls[1].kwargs["response_schema"])
        self.assertIsNotNone(calls[2].kwargs["response_schema"])
        self.assertFalse(REMOVED & result.keys())
        self.assertIn("REFERENCE_SECRET", calls[1].args[0])
        for i in (0, 2):
            self.assertNotIn("REFERENCE_SECRET", calls[i].args[0])
            self.assertNotIn("GOLD_SECRET", calls[i].args[0])
        self.assertEqual([c.kwargs["model_name"] for c in calls], [harness.GENERATION_MODEL, harness.JUDGE_MODEL, SETTINGS.groundedness_model])
        self.assertNotEqual(calls[0].kwargs["system_prompt"], calls[1].kwargs["system_prompt"])
        self.assertNotEqual(calls[1].kwargs["system_prompt"], calls[2].kwargs["system_prompt"])

    def test_error_does_not_skip_other_judge(self):
        result, _ = self.evaluate([ANSWER, "invalid JSON", SUPPORTED])
        self.assertIsNone(result["correctness"])
        self.assertEqual(result["errors"][0]["stage"], "correctness")
        self.assertEqual(result["groundedness"]["verdict"], "supported")

    def test_empty_answer_and_handoff(self):
        result, calls = self.evaluate([""])
        self.assertEqual(result["correctness"]["verdict"], "incorrect")
        self.assertEqual(len(calls), 1)
        result, calls = self.evaluate([ANSWER, SUPPORTED], TICKET | {"human_handoff": True})
        self.assertIsNone(result["correctness"])
        self.assertNotIn("human_handoff", calls[0].args[0])

    def test_failures_stay_in_denominator(self):
        good, _ = self.evaluate([ANSWER, CORRECT, SUPPORTED])
        bad, _ = self.evaluate([ValueError("Đầu ra bị cắt")])
        summary = harness.summarize([good, bad])
        self.assertEqual(summary["answer_accuracy"], 0.5)
        self.assertEqual(summary["correctness_counts"]["unavailable"], 1)
        self.assertEqual(summary["mean_groundedness"], 1.0)
        self.assertEqual(summary["groundedness_unavailable_answers"], 1)
        self.assertIsNone(harness.summarize([])["answer_accuracy"])

    def test_top_k_one_keeps_only_one_context_and_empty_retrieval_stops(self):
        with patch.object(harness, "get_connection"), patch.object(
            harness, "retrieve_top_k_parents", return_value=PARENTS * 2
        ) as retrieve, patch.object(harness, "generate", return_value=ANSWER), patch.object(
            harness, "evaluate_groundedness", return_value={"status": "ok"}
        ) as judge:
            result = harness.evaluate_ticket(TICKET | {"human_handoff": True}, top_k=1)
            self.assertEqual(retrieve.call_args.args[-1], 2)
            self.assertEqual(len(result["kb_contexts"]), 1)
            self.assertEqual(judge.call_args.args[2], result["kb_contexts"])
            retrieve.return_value = []
            judge.reset_mock()
            result = harness.evaluate_ticket(TICKET, top_k=1)
            self.assertIsNone(result["generation"])
            judge.assert_not_called()

    def test_sources_order_dedupe_and_validation(self):
        contexts = [{"parent_chunk_id": "p1", "source": "mail"},
                    {"parent_chunk_id": "p2", "source": "mail"}]
        answer = "Text [invented]"
        output = build_answer(answer, contexts + contexts[:1])
        self.assertEqual(output, {"answer": answer, "sources": [
            {"parent_chunk_id": "p1", "title": "mail"}, {"parent_chunk_id": "p2", "title": "mail"}]})
        self.assertEqual(build_answer("", []), {"answer": "", "sources": []})
        for bad in ([{}], [{"parent_chunk_id": " ", "source": "mail"}],
                    contexts + [{"parent_chunk_id": "p1", "source": "other"}]):
            with self.assertRaises(ValueError):
                build_answer(answer, bad)

    def test_labels_do_not_create_sources(self):
        result, calls = self.evaluate(["Text [invented]", CORRECT, SUPPORTED])
        self.assertEqual(result["generation"]["sources"], [{"parent_chunk_id": "p1", "title": "mail"}])
        self.assertFalse(result["errors"])
        self.assertEqual(len(calls), 3)

    def test_partial_answer_is_saved_without_judging(self):
        result, calls = self.evaluate([llm_client.GenerationError("Đầu ra bị cắt", "Một phần answer")])
        self.assertEqual(result["generation"]["answer"], "Một phần answer")
        self.assertEqual(result["errors"][0]["stage"], "generation")
        self.assertIsNone(result["correctness"])
        self.assertFalse(result["generation"]["complete"])
        self.assertEqual(len(result["generation"]["sources"]), 1)
        self.assertEqual(len(calls), 1)

    def test_retrieval_error_does_not_reference_unassigned_answer(self):
        with patch.object(harness, "get_connection", side_effect=RuntimeError("DB unavailable")):
            result = harness.evaluate_ticket(TICKET, top_k=3)
        self.assertIsNone(result["generation"])
        self.assertEqual(result["errors"], [{"stage": "retrieval", "message": "DB unavailable"}])

    def test_plain_answer_has_backend_sources(self):
        result, calls = self.evaluate([ANSWER, CORRECT, SUPPORTED])
        self.assertEqual(len(result["generation"]["sources"]), 1)
        self.assertNotIn("citations", result["generation"])
        self.assertNotIn("citation_support", result["groundedness"])
        self.assertEqual(len(calls), 3)

    def test_truncated_text_is_rejected(self):
        with patch.object(llm_client, "ChatOllama") as ollama:
            ollama.return_value.invoke.return_value.content = "Câu trả lời chưa xong"
            ollama.return_value.invoke.return_value.response_metadata = {"done_reason": "length"}
            with self.assertRaisesRegex(ValueError, "giới hạn token"):
                llm_client.generate("task", model_name="qwen3.5:4b")
            self.assertIsNone(ollama.call_args.kwargs["format"])
            self.assertEqual(ollama.call_args.kwargs["num_predict"], 2048)

    def test_local_model_override_and_fresh_clients(self):
        first, second = MagicMock(), MagicMock()
        first.invoke.return_value.content = "first"
        second.invoke.return_value.content = "second"
        with patch.object(llm_client, "ChatOllama", side_effect=[first, second]) as constructor, patch.object(
            llm_client, "_openrouter_generate"
        ) as openrouter, patch.object(llm_client, "check_api_key", return_value=False):
            self.assertEqual(llm_client.generate("task", model_name="qwen3.5:4b", system_prompt="generator"), "first")
            self.assertEqual(llm_client.generate("task", model_name="qwen3.5:4b", system_prompt="judge"), "second")
        self.assertEqual(constructor.call_count, 2)
        openrouter.assert_not_called()
        first.invoke.assert_called_once_with([("system", "generator"), ("human", "task")])
        second.invoke.assert_called_once_with([("system", "judge"), ("human", "task")])

    def test_default_provider_behavior(self):
        settings = SimpleNamespace(openrouter_api_key=MagicMock(), openrouter_model="openrouter-test",
                                   llm_timeout_seconds=10, ollama_host="http://localhost:11434",
                                   ollama_model="default-local")
        with patch.object(llm_client, "get_settings", return_value=settings), patch.object(
            llm_client, "check_api_key", return_value=True
        ), patch.object(llm_client, "_openrouter_generate", return_value="cloud") as cloud_gen, patch.object(llm_client, "ChatOllama") as ollama:
            self.assertEqual(llm_client.generate("task", use_cloud_model=True), "cloud")
            ollama.assert_not_called()
            cloud_gen.assert_called_once()
        with patch.object(llm_client, "get_settings", return_value=settings), patch.object(
            llm_client, "check_api_key", return_value=False
        ), patch.object(llm_client, "ChatOllama") as ollama:
            ollama.return_value.invoke.return_value.content = "local"
            self.assertEqual(llm_client.generate("task"), "local")
            self.assertEqual(ollama.call_args.kwargs["model"], "default-local")
            import warnings
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always")
                self.assertEqual(llm_client.generate("task", use_cloud_model=True), "local")
                self.assertEqual(len(recorded), 1)
                self.assertIn("OPENROUTER_API_KEY", str(recorded[0].message))

    def test_json_report(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            dataset, output = Path(directory) / "dataset.json", Path(directory) / "results.json"
            dataset.write_text(json.dumps([TICKET]), encoding="utf-8")
            with patch.object(harness, "get_connection"), patch.object(
                harness, "retrieve_top_k_parents", return_value=PARENTS
            ), patch.object(harness, "generate", side_effect=[ANSWER + " [typo]", json.dumps(CORRECT), json.dumps(SUPPORTED)]) as generate, patch.object(harness, "close_pool"), patch.object(confidence, "generate", generate), patch.object(confidence, "get_settings", return_value=SETTINGS), patch.object(harness, "get_settings", return_value=SETTINGS):
                harness.run_evaluation(dataset, output)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["completed"])
            self.assertEqual(report["summary"]["answer_accuracy"], 1.0)
            self.assertEqual(report["results"][0]["generation"]["answer"], ANSWER + " [typo]")
            self.assertFalse(report["results"][0]["errors"])
            self.assertEqual(report["schema_version"], 2)
            self.assertFalse(REMOVED & report["results"][0].keys())


if __name__ == "__main__":
    unittest.main()
