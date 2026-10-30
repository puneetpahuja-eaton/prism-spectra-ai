from __future__ import annotations

import copy
import os
import unittest
from unittest.mock import Mock, patch

import requests

import embeddings
from gemini_client import ModelUnavailable


MODEL = "Qwen/Qwen3-Embedding-8B"


def response_for(rows, *, model=MODEL):
    response = Mock(ok=True, status_code=200)
    response.json.return_value = {"model": model, "data": rows}
    return response


class NebiusEmbeddingTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {
            "AI_PROVIDER": "nebius", "EMBEDDING_PROVIDER": "nebius",
            "NEBIUS_API_KEY": "test-key", "FORCE_MOCK": "0",
        }, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def test_inherited_google_credentials_do_not_select_google(self):
        with patch.dict(os.environ, {
            "EMBEDDING_PROVIDER": "", "GEMINI_API_KEY": "google-test-key", "USE_VERTEX": "1",
            "GOOGLE_CLOUD_PROJECT": "test-project",
        }):
            self.assertEqual(embeddings.backend(), f"nebius:{MODEL}")

    def test_batched_response_is_reordered_and_progress_is_reported(self):
        first = response_for([
            {"index": 1, "embedding": [0.0, 1.0]},
            {"index": 0, "embedding": [1.0, 0.0]},
        ])
        second = response_for([{"index": 0, "embedding": [0.5, 0.5]}])
        progress = Mock()
        with patch("gemini_client.requests.post", side_effect=[first, second]) as post, \
                patch.object(embeddings, "_BATCH", 2):
            result = embeddings.embed_texts(["first", "second", "third"], on_progress=progress)
        self.assertEqual(result, [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        self.assertEqual(progress.call_args_list, [unittest.mock.call(2, 3), unittest.mock.call(3, 3)])
        self.assertEqual(post.call_args_list[0].args[0],
                         "https://api.tokenfactory.nebius.com/v1/embeddings")
        self.assertEqual(post.call_args_list[0].kwargs["json"], {
            "model": MODEL, "input": ["first", "second"], "encoding_format": "float",
        })
        self.assertEqual(post.call_args_list[0].kwargs["headers"]["Authorization"], "Bearer test-key")

    def test_invalid_vectors_are_rejected_without_fallback(self):
        cases = [
            [],
            [{"index": 0, "embedding": [1.0]}, {"index": 0, "embedding": [1.0]}],
            [{"index": 0, "embedding": [1.0]}, {"index": 2, "embedding": [1.0]}],
            [{"index": True, "embedding": [1.0]}, {"index": 0, "embedding": [1.0]}],
            [{"index": 0, "embedding": [1.0]}, {"index": 1, "embedding": []}],
            [{"index": 0, "embedding": [1.0]}, {"index": 1, "embedding": [float("nan")]}],
            [{"index": 0, "embedding": [1.0]}, {"index": 1, "embedding": [float("inf")]}],
            [{"index": 0, "embedding": [1.0]}, {"index": 1, "embedding": ["1.0"]}],
            [{"index": 0, "embedding": [1.0]}, {"index": 1, "embedding": [True]}],
            [{"index": 0, "embedding": [1.0]}, {"index": 1, "embedding": [1.0, 0.0]}],
            [{"index": 0, "embedding": [1.0]}, "invalid"],
        ]
        for rows in cases:
            with self.subTest(rows=rows), \
                    patch("gemini_client.requests.post", return_value=response_for(copy.deepcopy(rows))), \
                    patch("embeddings._offline_embed") as local:
                with self.assertRaises(ModelUnavailable):
                    embeddings.embed_texts(["one", "two"])
                local.assert_not_called()

    def test_different_returned_model_is_rejected(self):
        response = response_for([{"index": 0, "embedding": [1.0]}], model="different-model")
        with patch("gemini_client.requests.post", return_value=response):
            with self.assertRaisesRegex(ModelUnavailable, "different model"):
                embeddings.embed_text("synthetic text")

    def test_inconsistent_dimensions_across_batches_are_rejected(self):
        responses = [
            response_for([{"index": 0, "embedding": [1.0]}]),
            response_for([{"index": 0, "embedding": [1.0, 0.0]}]),
        ]
        with patch("gemini_client.requests.post", side_effect=responses), \
                patch.object(embeddings, "_BATCH", 1):
            with self.assertRaisesRegex(ModelUnavailable, "inconsistent vector dimensions"):
                embeddings.embed_texts(["one", "two"])

    def test_auth_failure_is_not_retried_or_replaced(self):
        response = Mock(ok=False, status_code=401)
        with patch("gemini_client.requests.post", return_value=response) as post, \
                patch("embeddings._offline_embed") as local:
            with self.assertRaisesRegex(ModelUnavailable, "HTTP 401"):
                embeddings.embed_text("synthetic text")
        post.assert_called_once()
        local.assert_not_called()

    def test_transient_failure_is_retried(self):
        success = response_for([{"index": 0, "embedding": [1.0, 0.0]}])
        with patch("gemini_client.requests.post",
                   side_effect=[Mock(ok=False, status_code=429), success]) as post, \
                patch("gemini_client.time.sleep"):
            self.assertEqual(embeddings.embed_text("synthetic text"), [1.0, 0.0])
        self.assertEqual(post.call_count, 2)

    def test_missing_key_and_connectivity_failure_are_explicit(self):
        with patch.dict(os.environ, {"NEBIUS_API_KEY": ""}), \
                patch("gemini_client.requests.post") as post:
            with self.assertRaisesRegex(ModelUnavailable, "not configured"):
                embeddings.embed_text("synthetic text")
            post.assert_not_called()
        with patch("gemini_client.requests.post", side_effect=requests.ConnectionError):
            with self.assertRaisesRegex(ModelUnavailable, "Check connectivity"):
                embeddings.embed_text("synthetic text")

    def test_local_and_mock_modes_do_not_call_remote_services(self):
        for config in ({"EMBEDDING_PROVIDER": "local"}, {"FORCE_MOCK": "1"}):
            with self.subTest(config=config), patch.dict(os.environ, config), \
                    patch("gemini_client.requests.post") as post:
                self.assertEqual(embeddings.backend(), "offline-hashed")
                self.assertEqual(len(embeddings.embed_text("synthetic text")), 256)
                post.assert_not_called()

    def test_google_is_explicit_and_credentials_are_required(self):
        with patch.dict(os.environ, {"EMBEDDING_PROVIDER": "google"}):
            with self.assertRaisesRegex(ModelUnavailable, "credentials are not configured"):
                embeddings.embed_text("synthetic text")
        with patch.dict(os.environ, {
            "EMBEDDING_PROVIDER": "google", "GEMINI_API_KEY": "test-google-key", "USE_VERTEX": "0",
        }):
            self.assertTrue(embeddings.backend().startswith("google-ai:"))

    def test_invalid_provider_is_an_error(self):
        with patch.dict(os.environ, {"EMBEDDING_PROVIDER": "invalid"}):
            with self.assertRaisesRegex(ModelUnavailable, "EMBEDDING_PROVIDER"):
                embeddings.embed_text("synthetic text")


if __name__ == "__main__":
    unittest.main()
