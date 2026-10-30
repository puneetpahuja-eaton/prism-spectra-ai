import json
import os
import unittest
from unittest.mock import Mock, patch

import requests
from pydantic import BaseModel
from urllib3.exceptions import ReadTimeoutError

import nebius_client
import workflow


class StreamAnswer(BaseModel):
    answer: str


def stream_response(events):
    response = Mock(ok=True, status_code=200)
    response.iter_lines.return_value = iter(
        "data: " + (event if isinstance(event, str) else json.dumps(event)) for event in events)
    return response


def chunk(content="", *, finish=None, reasoning=None):
    delta = {"content": content}
    if reasoning:
        delta["reasoning_content"] = reasoning
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


class NebiusStreamingTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {
            "AI_PROVIDER": "nebius", "NEBIUS_API_KEY": "test-key", "FORCE_MOCK": "0",
            "NEBIUS_TIMEOUT_SECONDS": "240", "NEBIUS_GENERATION_DEADLINE_SECONDS": "600",
            "LANGSMITH_TRACING": "false",
        }, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_chat_uses_streaming_and_validates_final_json_without_exposing_reasoning(self):
        response = stream_response([
            chunk(reasoning="private internal reasoning"),
            chunk('{"answer":"'),
            chunk("complete"),
            chunk('"}', finish="stop"),
            {"choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50}},
            "[DONE]",
        ])
        with patch("nebius_client.requests.post", return_value=response) as post:
            with workflow.Workflow("specta") as run:
                result = nebius_client.generate_json("Synthetic test only.", StreamAnswer)
        self.assertEqual(result["answer"], "complete")
        self.assertTrue(post.call_args.kwargs["stream"])
        self.assertTrue(post.call_args.kwargs["json"]["stream"])
        self.assertEqual(post.call_args.kwargs["json"]["stream_options"], {"include_usage": True})
        self.assertEqual(post.call_args.kwargs["timeout"], (10, 240))
        self.assertEqual(run.report["stages"][0]["usage"]["completion_tokens"], 30)
        self.assertNotIn("private internal reasoning", json.dumps(run.report))
        response.close.assert_called_once()

    def test_active_stream_can_finish_after_the_old_240_second_total_wait(self):
        response = stream_response([chunk(reasoning="still working"), chunk('{"answer":"ok"}', finish="stop"),
                                    "[DONE]"])
        with patch("nebius_client.requests.post", return_value=response), \
                patch("nebius_client._generation_clock", side_effect=[0, 130, 245, 246]):
            self.assertEqual(nebius_client.generate_json("Synthetic test.", StreamAnswer)["answer"], "ok")

    def test_active_stream_is_still_bounded_by_absolute_generation_deadline(self):
        response = stream_response([chunk(reasoning="still working")])
        with patch("nebius_client.requests.post", return_value=response) as post, \
                patch("nebius_client._generation_clock", side_effect=[0, 601]):
            with self.assertRaisesRegex(nebius_client.ModelUnavailable, "600-second generation deadline"):
                nebius_client.generate_json("Synthetic test.", StreamAnswer)
        post.assert_called_once()
        response.close.assert_called_once()

    def test_disconnected_or_unfinished_stream_never_becomes_a_completed_artifact(self):
        for events in ([chunk('{"answer":"partial"}'), "[DONE]"],
                       [chunk('{"answer":"partial"}', finish="stop")],
                       ['{"malformed":', "[DONE]"],
                       [{"error": {"message": "private provider detail"}}, "[DONE]"]):
            with self.subTest(events=events):
                response = stream_response(events)
                with patch("nebius_client.requests.post", return_value=response) as post:
                    with self.assertRaises(nebius_client.ModelUnavailable) as error:
                        nebius_client.generate_json("Synthetic test.", StreamAnswer)
                self.assertNotIn("private provider detail", str(error.exception))
                post.assert_called_once()
                response.close.assert_called_once()

    def test_midstream_idle_timeout_is_not_retried_and_preserves_no_partial_answer(self):
        response = stream_response([])

        def disconnected():
            yield "data: " + json.dumps(chunk('{"answer":"partial'))
            raise requests.ConnectionError(ReadTimeoutError(None, None, "private transport detail"))

        response.iter_lines.return_value = disconnected()
        with patch("nebius_client.requests.post", return_value=response) as post:
            with self.assertRaisesRegex(nebius_client.ModelUnavailable, "240-second read timeout") as error:
                nebius_client.generate_json("Synthetic test.", StreamAnswer)
        self.assertNotIn("private transport detail", str(error.exception))
        post.assert_called_once()
        response.close.assert_called_once()

    def test_truncated_stream_records_usage_but_is_not_accepted_as_json(self):
        response = stream_response([
            chunk('{"answer":"partial"}', finish="length"),
            {"choices": [], "usage": {"completion_tokens": 32768, "total_tokens": 32800}},
            "[DONE]",
        ])
        with patch("nebius_client.requests.post", return_value=response):
            with workflow.Workflow("specta") as run:
                with self.assertRaises(nebius_client.OutputBudgetExceeded):
                    nebius_client.generate_json("Synthetic test.", StreamAnswer, max_output_tokens=32768)
        self.assertEqual(run.report["status"], "failed")
        self.assertEqual(run.report["stages"][0]["usage"]["completion_tokens"], 32768)
