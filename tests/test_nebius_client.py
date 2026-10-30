from __future__ import annotations

import json
import os
import unittest
from unittest.mock import Mock, patch

import requests
from pydantic import BaseModel

import embeddings
from nebius_client import ModelUnavailable, backend, generate_json, is_live, nebius_post
import workflow


class SampleAnswer(BaseModel):
    answer: str


def completion_response(data):
    response = Mock(ok=True, status_code=200)
    events = []
    for choice in data["choices"]:
        events.append({"choices": [{"index": 0, "delta": choice["message"],
                                    "finish_reason": choice.get("finish_reason", "stop")}]})
    if "usage" in data:
        events.append({"choices": [], "usage": data["usage"]})
    response.iter_lines.return_value = ["data: " + json.dumps(event) for event in events] + ["data: [DONE]"]
    return response


class NebiusClientTests(unittest.TestCase):
    def test_nebius_is_the_explicit_live_backend(self):
        with patch.dict(os.environ, {
            "AI_PROVIDER": "nebius",
            "NEBIUS_API_KEY": "test-key",
            "NEBIUS_MODEL": "nvidia/Nemotron-3_5-Lightning",
            "FORCE_MOCK": "0",
        }, clear=True):
            self.assertTrue(is_live())
            self.assertEqual(backend(), "Nebius Token Factory · nvidia/Nemotron-3_5-Lightning")

    def test_generate_json_uses_token_factory_and_validates_the_schema(self):
        response = completion_response({"choices": [{"message": {"content": '{"answer":"voltage sag"}'}}]})
        with patch.dict(os.environ, {
            "AI_PROVIDER": "nebius",
            "NEBIUS_API_KEY": "test-key",
            "NEBIUS_MODEL": "nvidia/Nemotron-3_5-Lightning",
            "FORCE_MOCK": "0",
        }, clear=True), patch("nebius_client.requests.post", return_value=response) as post:
            result = generate_json("Diagnose this synthetic incident.", SampleAnswer)

        self.assertEqual(result["answer"], "voltage sag")
        self.assertEqual(result["_source"], "nebius:nvidia/Nemotron-3_5-Lightning")
        self.assertIn("Nebius Token Factory", result["_backend"])
        args, kwargs = post.call_args
        self.assertEqual(args[0], "https://api.tokenfactory.nebius.com/v1/chat/completions")
        self.assertEqual(kwargs["json"]["model"], "nvidia/Nemotron-3_5-Lightning")
        self.assertEqual(kwargs["json"]["response_format"], {
            "type": "json_schema",
            "json_schema": {
                "name": "SampleAnswer",
                "strict": True,
                "schema": SampleAnswer.model_json_schema(),
            },
        })
        self.assertIn("SampleAnswer", kwargs["json"]["messages"][0]["content"])

    def test_missing_required_fields_remain_an_explicit_failure(self):
        response = completion_response({"choices": [{"message": {"content": "{}"}}]})
        with patch.dict(os.environ, {
            "AI_PROVIDER": "nebius", "NEBIUS_API_KEY": "test-key", "FORCE_MOCK": "0",
        }, clear=True), patch("nebius_client.requests.post", return_value=response):
            with self.assertRaisesRegex(ModelUnavailable, "required response schema"):
                generate_json("Brief these public sources.", SampleAnswer)

    def test_generation_can_repair_invalid_schema_once_without_weakening_validation(self):
        from dataset_gen import DatasetDraft

        asset = {"id": "A", "metrics": [{"name": "x", "unit": "mm", "normal_min": 1, "normal_max": 2}]}
        invalid = json.dumps({"spec": {"assets": [asset, asset]}})
        corrected = json.dumps({"spec": {"assets": [asset]}})
        with patch.dict(os.environ, {
            "AI_PROVIDER": "nebius", "NEBIUS_API_KEY": "test-key", "FORCE_MOCK": "0",
        }, clear=True), patch("nebius_client._nebius_generate_text", side_effect=[invalid, corrected]) as model:
            result = generate_json("One synthetic asset.", DatasetDraft, repair_schema=True, reasoning=False)
        self.assertEqual(len(result["spec"]["assets"]), 1)
        self.assertEqual(model.call_count, 2)
        self.assertIn("Asset IDs must be unique", model.call_args.args[0])
        self.assertIn("previous output is not evidence", model.call_args.args[0])
        self.assertIs(model.call_args.kwargs["schema"], DatasetDraft)
        self.assertTrue(all(call.kwargs["reasoning"] is False for call in model.call_args_list))

    def test_schema_repair_stops_after_one_attempt_and_reports_failure(self):
        with patch.dict(os.environ, {
            "AI_PROVIDER": "nebius", "NEBIUS_API_KEY": "test-key", "FORCE_MOCK": "0",
        }, clear=True), patch("nebius_client._nebius_generate_text", return_value="{}") as model:
            with self.assertRaises(ModelUnavailable):
                generate_json("Answer.", SampleAnswer, repair_schema=True)
        self.assertEqual(model.call_count, 2)

    def test_empty_monitor_plan_is_repaired_instead_of_reasking_supplied_facts(self):
        from monitor_gen import MonitorPlan

        invalid = '{"poll_interval_s":60,"params":[],"missing_details":[],"answer":"Plan ready."}'
        corrected = '{"poll_interval_s":60,"params":[{"pid":"TEST_000","unit":"mm"}]}'
        with patch.dict(os.environ, {
            "AI_PROVIDER": "nebius", "NEBIUS_API_KEY": "test-key", "FORCE_MOCK": "0",
        }, clear=True), patch("nebius_client._nebius_generate_text", side_effect=[invalid, corrected]) as model:
            result = generate_json("Monitor TEST_000 in mm every 60 seconds.", MonitorPlan, repair_schema=True)
        self.assertEqual(len(result["params"]), 1)
        self.assertEqual(model.call_count, 2)
        self.assertIn("nonempty params", model.call_args.args[0])

    def test_rejected_schema_does_not_fall_back_to_unconstrained_output(self):
        response = Mock(ok=False, status_code=400)
        with patch.dict(os.environ, {
            "AI_PROVIDER": "nebius", "NEBIUS_API_KEY": "test-key", "FORCE_MOCK": "0",
        }, clear=True), patch("nebius_client.requests.post", return_value=response) as post:
            with self.assertRaisesRegex(ModelUnavailable, "HTTP 400"):
                generate_json("Brief these public sources.", SampleAnswer)
        post.assert_called_once()

    def test_malformed_model_json_is_an_explicit_failure(self):
        response = completion_response({"choices": [{"message": {"content": "not json"}}]})
        with patch.dict(os.environ, {
            "AI_PROVIDER": "nebius",
            "NEBIUS_API_KEY": "test-key",
            "FORCE_MOCK": "0",
        }, clear=True), patch("nebius_client.requests.post", return_value=response):
            with self.assertRaisesRegex(ModelUnavailable, "malformed or truncated JSON"):
                generate_json("Diagnose this synthetic incident.", SampleAnswer)

    def test_reasoning_budget_exhaustion_is_an_explicit_failure(self):
        data = {"choices": [{
            "finish_reason": "length",
            "message": {"content": "", "reasoning_content": "unfinished reasoning"},
        }]}
        response = completion_response(data)
        with patch.dict(os.environ, {
            "AI_PROVIDER": "nebius", "NEBIUS_API_KEY": "test-key", "FORCE_MOCK": "0",
        }, clear=True), patch("nebius_client.requests.post", return_value=response):
            with self.assertRaisesRegex(ModelUnavailable, "exhausted the output token budget"):
                generate_json("Brief these public sources.", SampleAnswer)

    def test_budget_recovery_is_once_bounded_and_keeps_schema_and_usage(self):
        exhausted = completion_response({"choices": [{"finish_reason": "length", "message": {"content": ""}}],
                                         "usage": {"completion_tokens": 16384, "total_tokens": 18000}})
        completed = completion_response({"choices": [{"finish_reason": "stop",
                                                      "message": {"content": '{"answer":"Validated answer"}'}}],
                                         "usage": {"completion_tokens": 1400, "total_tokens": 3000}})
        with patch.dict(os.environ, {
            "AI_PROVIDER": "nebius", "NEBIUS_API_KEY": "test-key", "FORCE_MOCK": "0",
            "LANGSMITH_TRACING": "false",
        }, clear=True), patch("nebius_client.requests.post", side_effect=[exhausted, completed]) as post:
            with workflow.Workflow("specta") as run:
                result = generate_json("Supplied facts only.", SampleAnswer, max_output_tokens=16384, reasoning=False)
        self.assertEqual(result["answer"], "Validated answer")
        self.assertEqual([call.kwargs["json"]["max_tokens"] for call in post.call_args_list], [16384, 32768])
        self.assertEqual(post.call_args_list[0].kwargs["json"]["response_format"],
                         post.call_args_list[1].kwargs["json"]["response_format"])
        self.assertTrue(all(call.kwargs["json"]["reasoning_effort"] == "none" for call in post.call_args_list))
        self.assertEqual(run.report["status"], "complete")
        self.assertEqual([event["status"] for event in run.report["stages"]], ["recovered", "succeeded"])
        self.assertEqual(run.report["stages"][0]["usage"]["completion_tokens"], 16384)

    def test_exhaustion_does_not_loop_or_export_http_success_as_model_success(self):
        data = {"choices": [{"finish_reason": "length", "message": {"content": ""}}]}
        with patch.dict(os.environ, {
            "AI_PROVIDER": "nebius", "NEBIUS_API_KEY": "test-key", "FORCE_MOCK": "0",
            "LANGSMITH_TRACING": "false",
        }, clear=True), patch("nebius_client.requests.post", side_effect=lambda *a, **k: completion_response(data)) as post:
            with workflow.Workflow("specta") as run:
                with self.assertRaises(ModelUnavailable):
                    generate_json("Supplied facts only.", SampleAnswer, max_output_tokens=16384)
        self.assertEqual(post.call_count, 2)
        self.assertEqual(run.report["status"], "failed")
        self.assertTrue(all(event["status"] == "failed" for event in run.report["stages"]))

    def test_length_finished_partial_json_is_never_salvaged_as_a_completed_answer(self):
        response = completion_response({"choices": [{"finish_reason": "length",
                                                      "message": {"content": '{"answer":"Partial'}}]})
        with patch.dict(os.environ, {
            "AI_PROVIDER": "nebius", "NEBIUS_API_KEY": "test-key", "FORCE_MOCK": "0",
        }, clear=True), patch("nebius_client.requests.post", return_value=response) as post:
            with self.assertRaises(ModelUnavailable):
                generate_json("Answer.", SampleAnswer, max_output_tokens=32768)
        post.assert_called_once()

    def test_recovery_still_rejects_missing_required_fields(self):
        exhausted = completion_response({"choices": [{"finish_reason": "length", "message": {"content": ""}}]})
        invalid = completion_response({"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]})
        with patch.dict(os.environ, {
            "AI_PROVIDER": "nebius", "NEBIUS_API_KEY": "test-key", "FORCE_MOCK": "0",
        }, clear=True), patch("nebius_client.requests.post", side_effect=[exhausted, invalid]) as post:
            with self.assertRaisesRegex(ModelUnavailable, "required response schema"):
                generate_json("Answer.", SampleAnswer)
        self.assertEqual(post.call_count, 2)

    def test_transport_failures_are_classified_without_leaking_request_details(self):
        cases = [
            (requests.exceptions.ReadTimeout, "120-second read timeout"),
            (requests.exceptions.ConnectTimeout, "10-second connection timeout"),
            (requests.exceptions.SSLError, "TLS certificate verification"),
            (requests.exceptions.ProxyError, "proxy connection"),
            (requests.exceptions.Timeout, "request timed out"),
            (requests.exceptions.ConnectionError, "could not connect"),
            (requests.exceptions.RequestException, "transport failure"),
        ]
        for error_type, expected in cases:
            with self.subTest(error_type=error_type.__name__):
                cause = error_type("private-request-details")
                with patch.dict(os.environ, {"NEBIUS_API_KEY": "test-key"}, clear=True), \
                        patch("nebius_client.netsetup.configure_tls"), \
                        patch("nebius_client.requests.post", side_effect=cause) as post, \
                        self.assertLogs("nebius_client", level="WARNING") as logs:
                    with self.assertRaises(ModelUnavailable) as caught:
                        nebius_post("chat/completions", {})
                self.assertIn(expected, str(caught.exception))
                self.assertIs(caught.exception.__cause__, cause)
                self.assertNotIn("private-request-details", str(caught.exception))
                self.assertNotIn("private-request-details", "\n".join(logs.output))
                self.assertIn(error_type.__name__, "\n".join(logs.output))
                post.assert_called_once()

    def test_read_timeout_reports_the_configured_deadline_not_a_credit_diagnosis(self):
        with patch.dict(os.environ, {
            "NEBIUS_API_KEY": "test-key", "NEBIUS_TIMEOUT_SECONDS": "45",
        }, clear=True), patch("nebius_client.netsetup.configure_tls"), \
                patch("nebius_client.requests.post", side_effect=requests.exceptions.ReadTimeout):
            with self.assertRaisesRegex(ModelUnavailable, "45-second read timeout") as caught:
                nebius_post("chat/completions", {})
        self.assertIn("not a confirmed credit error", str(caught.exception))

    def test_http_credit_authentication_and_rate_errors_remain_distinct(self):
        cases = [(401, "API key"), (402, "payment or credit"), (403, "permission"), (429, "rate or quota")]
        for status, expected in cases:
            with self.subTest(status=status):
                response = Mock(ok=False, status_code=status)
                with patch.dict(os.environ, {"NEBIUS_API_KEY": "test-key"}, clear=True), \
                        patch("nebius_client.netsetup.configure_tls"), \
                        patch("nebius_client.requests.post", return_value=response) as post, \
                        patch("nebius_client.time.sleep"):
                    with self.assertRaises(ModelUnavailable) as caught:
                        nebius_post("chat/completions", {})
                self.assertIn(f"HTTP {status}", str(caught.exception))
                self.assertIn(expected, str(caught.exception))
                self.assertEqual(post.call_count, 3 if status == 429 else 1)

    def test_nebius_api_key_does_not_enable_vertex_embeddings(self):
        with patch.dict(os.environ, {
            "AI_PROVIDER": "nebius",
            "NEBIUS_API_KEY": "test-key",
            "USE_VERTEX": "0",
            "GEMINI_API_KEY": "",
            "FORCE_MOCK": "0",
        }, clear=True):
            self.assertTrue(is_live())
            self.assertEqual(embeddings.backend(), "nebius:Qwen/Qwen3-Embedding-8B")


if __name__ == "__main__":
    unittest.main()
