"""Chẩn đoán không được làm mất context reference hoặc output lỗi."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock
from pydantic import SecretStr

from src import confidence as conf
from src.agent import llm_client
from eval.harness.diagnose_groundedness import replay


SETTINGS = SimpleNamespace(
    openrouter_api_key=None, openrouter_model="test-openrouter", groundedness_model="qwen3.5:4b",
    judge_num_ctx=32768, judge_max_tokens=4096, judge_seed=42,
    ollama_host="http://localhost:11434", llm_timeout_seconds=10,
)
CONTEXT = [{"parent_chunk_id": "p1", "source": "mail", "content": "Tắt\nOutlook."}]


class DiagnosticTests(unittest.TestCase):
    def test_unknown_context_reference_is_diagnosed(self):
        report = {"claims": [{"claim": "Tắt Outlook.", "verdict": "supported",
                  "evidence": [{"context_id": "E2"}]}], "reason": "test"}
        trace = {}
        with patch.object(conf, "get_settings", return_value=SETTINGS), patch.object(
            conf, "generate", return_value=json.dumps(report)
        ):
            result = conf.evaluate_groundedness("q", "a", CONTEXT, diagnostics=trace)
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["score"], 0)
        self.assertEqual(trace["stage"], "evidence_validation")
        self.assertEqual(trace["rejected_evidence"]["code"], "unknown_context_id")
        self.assertEqual(trace["rejected_evidence"]["claim_index"], 1)
        self.assertEqual(json.loads(trace["response"]["raw_response"]), report)
        self.assertNotIn("diagnostic", result)

    def test_truncation_retains_partial_response_and_metadata(self):
        response = Mock(content='{"claims": [', response_metadata={
            "done_reason": "length", "prompt_eval_count": 1500, "eval_count": 4096,
            "unused_sensitive_field": "DO_NOT_LOG",
        })
        trace = {}
        with patch.object(conf, "get_settings", return_value=SETTINGS), patch.object(
            llm_client, "get_settings", return_value=SETTINGS
        ), patch.object(llm_client, "ChatOllama") as model:
            model.return_value.invoke.return_value = response
            result = conf.evaluate_groundedness("q", "a", CONTEXT, diagnostics=trace)
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(trace["response"]["raw_response"], '{"claims": [')
        self.assertEqual(trace["response"]["output_tokens"], 4096)
        self.assertEqual(trace["response"]["finish_reason"], "length")
        self.assertNotIn("DO_NOT_LOG", json.dumps(trace))

    def test_schema_failure_and_valid_result(self):
        for raw, expected in [('{"claims":', "schema_validation"),
                              ('{"claims":[{"claim":"Tắt Outlook.","verdict":"supported","evidence":'
                               '[{"context_id":"E1"}]}],"reason":"test"}', "complete")]:
            trace = {}
            with patch.object(conf, "get_settings", return_value=SETTINGS), patch.object(conf, "generate", return_value=raw):
                conf.evaluate_groundedness("q", "a", CONTEXT, diagnostics=trace)
            self.assertEqual(trace["stage"], expected)
            self.assertEqual(trace["response"]["raw_response"], raw)

    def test_openrouter_truncation_metadata(self):
        settings = SimpleNamespace(**(vars(SETTINGS) | {"openrouter_api_key": SecretStr("fake-key")}))
        mock_choice = Mock(message=Mock(content='{"claims":'), finish_reason="length")
        mock_usage = Mock(prompt_tokens=200, completion_tokens=4096)
        mock_completion = Mock(choices=[mock_choice], usage=mock_usage)
        mock_client = Mock(chat=Mock(completions=Mock(create=Mock(return_value=mock_completion))))
        trace = {}
        with patch.object(llm_client, "get_settings", return_value=settings), patch.object(
            llm_client, "OpenAI", return_value=mock_client
        ):
            with self.assertRaises(llm_client.GenerationError):
                llm_client.generate("q", use_cloud_model=True, diagnostics=trace)
        self.assertEqual(trace["finish_reason"], "length")
        self.assertEqual(trace["output_tokens"], 4096)
        self.assertEqual(trace["raw_response"], '{"claims":')
        self.assertNotIn("fake-key", json.dumps(trace))

    def test_replay_uses_saved_context_and_does_not_overwrite(self):
        ticket = {"ticket_id": "T1", "question": "q",
                  "generation": {"answer": "a", "complete": True},
                  "retrieval": {"parents": [
                      {"parent_chunk_id": "p1", "source": "mail", "content": "evidence", "used_for_generation": True},
                      {"parent_chunk_id": "p2", "source": "mail", "used_for_generation": False}]}}
        with TemporaryDirectory(dir=Path(__file__).parent) as directory:
            src, dst = Path(directory) / "original.json", Path(directory) / "debug.jsonl"
            src.write_text(json.dumps({"results": [ticket]}), encoding="utf-8")
            original = src.read_bytes()
            with patch("eval.harness.diagnose_groundedness.evaluate_groundedness",
                       return_value=conf.unavailable_groundedness("test")) as judge:
                replay(src, dst, ["T1"])
            self.assertEqual(len(judge.call_args.args[2]), 1)
            self.assertEqual(judge.call_args.args[:2], ("q", "a"))
            self.assertEqual(json.loads(dst.read_text())["ticket_id"], "T1")
            self.assertEqual(src.read_bytes(), original)
            with self.assertRaises(FileExistsError):
                replay(src, dst)


if __name__ == "__main__":
    unittest.main()
