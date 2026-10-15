"""Kiểm thử harness confidence bằng dữ liệu giả, không gọi DB hoặc LLM."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from eval.harness import eval_confidence_scorer as harness
from src.agent.llm_client import GenerationError
from src.confidence import unavailable_groundedness
from src.rag.retriever import fuse_rankings


TICKET = {"id": "T1", "question": "Outlook offline", "category": "MAIL", "human_handoff": False,
          "ground_truth_answer": "GOLD_SECRET", "ground_truth_contexts": ["GOLD_CONTEXT"]}
PARENTS = fuse_rankings([
    {"parent_chunk_id": str(i), "article_title": "mail", "text": f"Evidence {i}",
     "similarity": 0.8, "bm25_score": 2.0} for i in range(3)
], [{"parent_chunk_id": str(i)} for i in range(3)])
GROUNDING = {"status": "ok", "score": 1.0, "error": None, "verdict": "supported",
             "citation_support": "supported", "claims": [{"claim": "x", "verdict": "supported"}]}
SETTINGS = SimpleNamespace(top_k=1, judge_num_ctx=32768, judge_max_tokens=4096, judge_seed=42,
                           embedding_model="test", mistral_api_key=None, mistral_model="mistral",
                           groundedness_model="qwen3.5:4b")


class ConfidenceHarnessTests(unittest.TestCase):
    def evaluate(self, answer="Answer [mail]", parents=PARENTS, grounding=GROUNDING):
        with patch.object(harness, "get_connection"), patch.object(
            harness, "retrieve_hybrid_parents", return_value=parents
        ), patch.object(harness, "call_prompt", side_effect=answer if isinstance(answer, Exception) else None,
                        return_value=answer) as generate, patch.object(
            harness, "evaluate_groundedness", return_value=grounding
        ) as judge:
            result = harness.evaluate_ticket(TICKET, 1, 32768)
            return result, generate, judge

    def test_scores_full_fusion_and_reference_isolation(self):
        result, generate, judge = self.evaluate()
        self.assertAlmostEqual(result["confidence"]["score"], 0.8838709677419355)
        self.assertAlmostEqual(sum(result["confidence"]["weighted_contributions"].values()),
                               result["confidence"]["score"])
        self.assertEqual(len(result["retrieval"]["parents"]), 3)
        self.assertNotIn("content", result["retrieval"]["parents"][1])
        self.assertEqual(len(judge.call_args.args[2]), 1)
        self.assertEqual(generate.call_args.kwargs["kb_contexts"], judge.call_args.args[2])
        self.assertNotIn("GOLD_SECRET", str(generate.call_args) + str(judge.call_args))
        self.assertEqual(result["reference"]["ground_truth_answer"], "GOLD_SECRET")
        self.assertIsNone(result["assessment"]["human_review"]["verdict"])

    def test_retrieval_empty_or_failure_stops_models(self):
        result, generate, judge = self.evaluate(parents=[])
        self.assertIsNone(result["confidence"]["score"])
        generate.assert_not_called()
        judge.assert_not_called()
        with patch.object(harness, "get_connection", side_effect=RuntimeError("private connection string")):
            result = harness.evaluate_ticket(TICKET, 1, 32768)
        self.assertNotIn("private", json.dumps(result))
        self.assertEqual(result["confidence"]["status"], "unavailable")

    def test_partial_and_judge_failure_fallback(self):
        result, _, judge = self.evaluate(GenerationError("length", "Partial answer"))
        self.assertEqual(result["generation"]["answer"], "Partial answer")
        self.assertFalse(result["generation"]["complete"])
        self.assertEqual(result["confidence"]["status"], "degraded")
        judge.assert_not_called()
        result, _, _ = self.evaluate(grounding=unavailable_groundedness("Judge failed"))
        self.assertEqual(result["confidence"]["components"]["groundedness"], 0.0)
        self.assertEqual(result["errors"][0]["stage"], "groundedness")

    def test_citation_error_preserves_answer_and_judging(self):
        result, _, judge = self.evaluate("Answer [typo]", grounding=GROUNDING | {"citation_support": "unsupported"})
        self.assertEqual(result["generation"]["answer"], "Answer [typo]")
        self.assertTrue(result["generation"]["complete"])
        self.assertEqual(result["confidence"]["status"], "degraded")
        judge.assert_called_once()

    def test_summary_excludes_fallback_but_includes_valid_zero(self):
        good, _, _ = self.evaluate()
        bad, _, _ = self.evaluate(grounding=unavailable_groundedness("error"))
        zero, _, _ = self.evaluate(grounding=GROUNDING | {"score": 0.0, "verdict": "unsupported"})
        summary = harness.summarize([good, bad, zero])
        self.assertEqual(summary["mean_scores"]["groundedness"], 0.5)
        self.assertEqual(summary["valid_score_count"], 2)
        self.assertEqual(summary["status_counts"]["degraded"], 1)
        self.assertIsNone(harness.summarize([])["mean_scores"]["confidence"])

    def test_json_selection_and_preserve_existing_reviews(self):
        result, _, _ = self.evaluate()
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            dataset, output = Path(directory) / "data.json", Path(directory) / "out.json"
            dataset.write_text(json.dumps([TICKET | {"id": "T0"}, TICKET]), encoding="utf-8")
            with patch.object(harness, "get_settings", return_value=SETTINGS), patch.object(
                harness, "evaluate_ticket", return_value=result
            ) as evaluate, patch.object(harness, "close_pool") as close:
                harness.run_evaluation(dataset, output, limit=1, ticket_id="T1")
                report = json.loads(output.read_text(encoding="utf-8"))
                self.assertTrue(report["run"]["completed"])
                self.assertEqual(evaluate.call_args.args[0]["id"], "T1")
                self.assertEqual(report["run"]["weights"], harness.CONFIDENCE_WEIGHTS)
                close.assert_called_once()
                original = output.read_bytes()
                with self.assertRaises(FileExistsError):
                    harness.run_evaluation(dataset, output)
                self.assertEqual(output.read_bytes(), original)
                with self.assertRaisesRegex(ValueError, "Không tìm thấy"):
                    harness.run_evaluation(dataset, output, ticket_id="unknown")
                with self.assertRaises(ValueError):
                    harness.run_evaluation(dataset, output, limit=-1)


if __name__ == "__main__":
    unittest.main()
