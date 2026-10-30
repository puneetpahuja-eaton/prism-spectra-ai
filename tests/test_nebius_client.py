from __future__ import annotations

import json
import os
import unittest
from unittest.mock import Mock, patch

import requests
from pydantic import BaseModel

import embeddings
from gemini_client import ModelUnavailable, backend, generate_json, is_live, nebius_post


class SampleAnswer(BaseModel):
    answer: str


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
        response = Mock(ok=True, status_code=200)
        response.json.return_value = {"choices": [{"message": {"content": '{"answer":"voltage sag"}'}}]}
        with patch.dict(os.environ, {
            "AI_PROVIDER": "nebius",
            "NEBIUS_API_KEY": "test-key",
            "NEBIUS_MODEL": "nvidia/Nemotron-3_5-Lightning",
            "FORCE_MOCK": "0",
        }, clear=True), patch("gemini_client.requests.post", return_value=response) as post:
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
        response = Mock(ok=True, status_code=200)
        response.json.return_value = {"choices": [{"message": {"content": "{}"}}]}
        with patch.dict(os.environ, {
            "AI_PROVIDER": "nebius", "NEBIUS_API_KEY": "test-key", "FORCE_MOCK": "0",
        }, clear=True), patch("gemini_client.requests.post", return_value=response):
            with self.assertRaisesRegex(ModelUnavailable, "required response schema"):
                generate_json("Brief these public sources.", SampleAnswer)

    def test_generation_can_repair_invalid_schema_once_without_weakening_validation(self):
        from dataset_gen import DatasetDraft

        asset = {"id": "A", "metrics": [{"name": "x", "unit": "mm", "normal_min": 1, "normal_max": 2}]}
        invalid = json.dumps({"spec": {"assets": [asset, asset]}})
        corrected = json.dumps({"spec": {"assets": [asset]}})
        with patch.dict(os.environ, {
            "AI_PROVIDER": "nebius", "NEBIUS_API_KEY": "test-key", "FORCE_MOCK": "0",
        }, clear=True), patch("gemini_client._nebius_generate_text", side_effect=[invalid, corrected]) as model:
            result = generate_json("One synthetic asset.", DatasetDraft, repair_schema=True)
        self.assertEqual(len(result["spec"]["assets"]), 1)
        self.assertEqual(model.call_count, 2)
        self.assertIn("Asset IDs must be unique", model.call_args.args[0])
        self.assertIn("previous output is not evidence", model.call_args.args[0])
        self.assertIs(model.call_args.kwargs["schema"], DatasetDraft)

    def test_schema_repair_stops_after_one_attempt_and_reports_failure(self):
        with patch.dict(os.environ, {
            "AI_PROVIDER": "nebius", "NEBIUS_API_KEY": "test-key", "FORCE_MOCK": "0",
        }, clear=True), patch("gemini_client._nebius_generate_text", return_value="{}") as model:
            with self.assertRaises(ModelUnavailable):
                generate_json("Answer.", SampleAnswer, repair_schema=True)
        self.assertEqual(model.call_count, 2)

    def test_rejected_schema_does_not_fall_back_to_unconstrained_output(self):
        response = Mock(ok=False, status_code=400)
        with patch.dict(os.environ, {
            "AI_PROVIDER": "nebius", "NEBIUS_API_KEY": "test-key", "FORCE_MOCK": "0",
        }, clear=True), patch("gemini_client.requests.post", return_value=response) as post:
            with self.assertRaisesRegex(ModelUnavailable, "HTTP 400"):
                generate_json("Brief these public sources.", SampleAnswer)
        post.assert_called_once()

    def test_malformed_model_json_is_an_explicit_failure(self):
        response = Mock(ok=True, status_code=200)
        response.json.return_value = {"choices": [{"message": {"content": "not json"}}]}
        with patch.dict(os.environ, {
            "AI_PROVIDER": "nebius",
            "NEBIUS_API_KEY": "test-key",
            "FORCE_MOCK": "0",
        }, clear=True), patch("gemini_client.requests.post", return_value=response):
            with self.assertRaisesRegex(ModelUnavailable, "malformed or truncated JSON"):
                generate_json("Diagnose this synthetic incident.", SampleAnswer)

    def test_reasoning_budget_exhaustion_is_an_explicit_failure(self):
        response = Mock(ok=True, status_code=200)
        response.json.return_value = {"choices": [{
            "finish_reason": "length",
            "message": {"content": "", "reasoning_content": "unfinished reasoning"},
        }]}
        with patch.dict(os.environ, {
            "AI_PROVIDER": "nebius", "NEBIUS_API_KEY": "test-key", "FORCE_MOCK": "0",
        }, clear=True), patch("gemini_client.requests.post", return_value=response):
            with self.assertRaisesRegex(ModelUnavailable, "exhausted the output token budget"):
                generate_json("Brief these public sources.", SampleAnswer)

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
                        patch("gemini_client.netsetup.configure_tls"), \
                        patch("gemini_client.requests.post", side_effect=cause) as post, \
                        self.assertLogs("gemini_client", level="WARNING") as logs:
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
        }, clear=True), patch("gemini_client.netsetup.configure_tls"), \
                patch("gemini_client.requests.post", side_effect=requests.exceptions.ReadTimeout):
            with self.assertRaisesRegex(ModelUnavailable, "45-second read timeout") as caught:
                nebius_post("chat/completions", {})
        self.assertIn("not a confirmed credit error", str(caught.exception))

    def test_http_credit_authentication_and_rate_errors_remain_distinct(self):
        cases = [(401, "API key"), (402, "payment or credit"), (403, "permission"), (429, "rate or quota")]
        for status, expected in cases:
            with self.subTest(status=status):
                response = Mock(ok=False, status_code=status)
                with patch.dict(os.environ, {"NEBIUS_API_KEY": "test-key"}, clear=True), \
                        patch("gemini_client.netsetup.configure_tls"), \
                        patch("gemini_client.requests.post", return_value=response) as post, \
                        patch("gemini_client.time.sleep"):
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
