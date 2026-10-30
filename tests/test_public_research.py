from __future__ import annotations

import copy
import json
import os
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import requests
from streamlit.testing.v1 import AppTest

import devices
from gemini_client import ModelUnavailable
from integrations.tavily import Client, TavilyUnavailable
import public_research as research


URL = "https://docs.vendor.com/manual"
TEXT = "The battery alarm requires a qualified engineer to check product-specific documentation."
ARGS = {"product": "Vendor UPS", "symptom": "battery alarm", "domains_text": "vendor.com", "consent": True}


def brief():
    return {
        "findings": [{"statement": "The public source calls for qualified review.",
                      "evidence": [{"source_id": "E1", "quote": TEXT}]}],
        "open_questions": ["Does this manual apply to this exact public model?"],
        "engineer_review_questions": ["Is the product revision covered by the cited documentation?"],
        "_source": "nebius:test-model",
    }


def draft_brief():
    result = brief()
    result["findings"][0]["evidence"] = [{"passage_id": "E1.P1"}]
    return result


def search_result():
    return {"results": [{"url": URL, "title": "Public UPS manual", "content": TEXT}]}


class TavilyClientTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}, clear=True)
        env.start()
        self.addCleanup(env.stop)

    def test_live_request_trace_contains_metadata_not_credentials(self):
        response = Mock(status_code=200)
        response.json.return_value = {"results": [], "request_id": "request-1", "usage": {"credits": 2}}
        client = Client()
        with patch("integrations.tavily.requests.post", return_value=response) as post:
            client.search("Vendor manual", ["vendor.com"], recent=True)
        args, kwargs = post.call_args
        self.assertEqual(args[0], "https://api.tavily.com/search")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(kwargs["json"]["search_depth"], "advanced")
        self.assertEqual(kwargs["json"]["time_range"], "year")
        self.assertFalse(kwargs["json"]["include_answer"])
        self.assertFalse(kwargs["allow_redirects"])
        self.assertEqual(client.trace[0]["request_id"], "request-1")
        self.assertEqual(client.trace[0]["credits"], 2)
        self.assertNotIn("test-key", json.dumps(client.trace))

    def test_extract_and_map_payloads_are_bounded(self):
        response = Mock(status_code=200)
        response.json.return_value = {"results": []}
        client = Client()
        with patch("integrations.tavily.requests.post", return_value=response) as post:
            client.map(URL, "Vendor UPS")
            client.extract([URL], "battery alarm")
        self.assertEqual(post.call_args_list[0].kwargs["json"]["limit"], 5)
        self.assertFalse(post.call_args_list[0].kwargs["json"]["allow_external"])
        self.assertEqual(post.call_args_list[1].kwargs["json"]["extract_depth"], "advanced")
        with self.assertRaises(ValueError):
            client.extract([URL] * 5, "battery alarm")

    def test_errors_are_explicit_no_body_key_or_silent_retries(self):
        for status in (301, 401, 403, 429, 432, 500):
            with self.subTest(status=status):
                client = Client()
                response = Mock(status_code=status, text="PRIVATE_GATEWAY_DATA")
                with patch("integrations.tavily.requests.post", return_value=response) as post:
                    with self.assertRaises(TavilyUnavailable) as error:
                        client.search("Vendor UPS", ["vendor.com"])
                self.assertIn(str(status), str(error.exception))
                self.assertNotIn("PRIVATE_GATEWAY_DATA", str(error.exception))
                self.assertNotIn("test-key", str(error.exception))
                self.assertEqual(client.trace[0]["status"], "failed")
                post.assert_called_once()

    def test_connection_and_invalid_responses_are_explicit(self):
        for response in (None, [], {"unexpected": "shape"}):
            client = Client()
            mocked = Mock(status_code=200)
            mocked.json.return_value = response
            with self.subTest(response=response), \
                    patch("integrations.tavily.requests.post", return_value=mocked):
                with self.assertRaises(TavilyUnavailable):
                    client.search("Vendor", ["vendor.com"])
            self.assertEqual(client.trace[0]["status"], "failed")
        client = Client()
        with patch("integrations.tavily.requests.post", side_effect=requests.Timeout):
            with self.assertRaisesRegex(TavilyUnavailable, "connection failed"):
                client.search("Vendor", ["vendor.com"])
        self.assertEqual(client.trace[0]["status"], "failed")

    def test_missing_key_and_request_limit_prevent_calls(self):
        client = Client()
        with patch.dict(os.environ, {"TAVILY_API_KEY": ""}), \
                patch("integrations.tavily.secretmgr.get_secret", return_value=None), \
                patch("integrations.tavily.requests.post") as post:
            with self.assertRaisesRegex(TavilyUnavailable, "Set TAVILY_API_KEY"):
                client.search("Vendor", ["vendor.com"])
            post.assert_not_called()
        client.trace = [{}] * 5
        with patch("integrations.tavily.requests.post") as post:
            with self.assertRaisesRegex(TavilyUnavailable, "five-request"):
                client.search("Vendor", ["vendor.com"])
            post.assert_not_called()


class PublicResearchTests(unittest.TestCase):
    def setUp(self):
        self.search = patch("public_research.Client.search", side_effect=lambda *a, **k: search_result())
        self.extract = patch("public_research.Client.extract",
                             return_value={"results": [{"url": URL, "raw_content": TEXT}]})
        self.model = patch("public_research.generate_json", side_effect=lambda *a, **k: draft_brief())
        self.search_mock = self.search.start()
        self.extract_mock = self.extract.start()
        self.model_mock = self.model.start()
        self.addCleanup(self.search.stop)
        self.addCleanup(self.extract.stop)
        self.addCleanup(self.model.stop)

    def test_requires_consent_before_any_outbound_request(self):
        with self.assertRaisesRegex(ValueError, "Confirm"):
            research.run_research(**{**ARGS, "consent": False})
        self.search_mock.assert_not_called()
        self.model_mock.assert_not_called()

    def test_public_terms_and_domain_validation(self):
        for value in ("https://internal/path", "secret api_key value", "x@y.com", "10.1.2.3",
                      "ghp_credential", "a" * 121, "private\nsource"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                research.research_plan(value, "battery", "vendor.com")
        for domains in ("", "localhost", "*.vendor.com", "https://vendor.com", "vendor.internal",
                        "internal.lan", "127.0.0.1", "vendor.com:443"):
            with self.subTest(domains=domains), self.assertRaises(ValueError):
                research.research_plan("Vendor UPS", "battery", domains)

    def test_url_filter_blocks_unapproved_private_and_tokenized_links(self):
        for url in ("http://vendor.com", "https://vendor.com.evil.com", "https://localhost/",
                    "https://user:password@vendor.com", "https://vendor.com:8443",
                    "https://vendor.com/manual?token=secret", None, "javascript:alert(1)"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                research.public_url(url, ["vendor.com"])
        self.assertEqual(research.public_url(URL + "#section", ["vendor.com"]), URL)

    def test_three_intents_deduplication_extraction_and_provenance(self):
        packet = research.run_research(**ARGS)
        self.assertEqual(packet["status"], "complete")
        self.assertEqual(self.search_mock.call_count, 3)
        self.assertEqual(self.search_mock.call_args_list[-1].kwargs["recent"], True)
        self.assertEqual(len(packet["sources"]), 1)
        self.assertEqual(packet["sources"][0]["kind"], "extracted page")
        self.assertEqual(len(packet["sources"][0]["intents"]), 3)
        self.assertEqual(packet["brief"]["_source"], "nebius:test-model")
        prompt = self.model_mock.call_args.args[0]
        self.assertIn("UNTRUSTED DATA", prompt)
        self.assertIn("No recent advisory results", prompt)
        self.assertEqual(self.model_mock.call_args.kwargs["max_output_tokens"], 16384)

    def test_map_discovery_is_used_in_extraction_but_external_links_are_filtered(self):
        discovered = "https://docs.vendor.com/service-bulletin"
        with patch("public_research.Client.map",
                   return_value={"results": ["https://vendor.com/", discovered,
                                             "https://evil.com/advisory"]}) as map_call:
            packet = research.run_research(**ARGS, map_url="https://vendor.com/")
        map_call.assert_called_once()
        self.assertEqual(packet["discovered_urls"], [discovered])
        self.assertIn(discovered, self.extract_mock.call_args.args[0])
        self.assertEqual(packet["status"], "partial")  # discovered page was not returned by Extract

    def test_search_failure_is_not_a_success_shaped_packet(self):
        self.search_mock.side_effect = TavilyUnavailable("Tavily search returned HTTP 401.")
        packet = research.run_research(**ARGS)
        self.assertEqual(packet["status"], "failed")
        self.assertTrue(packet["errors"])
        self.extract_mock.assert_not_called()
        self.model_mock.assert_not_called()

    def test_one_failed_search_retains_successful_evidence_as_partial(self):
        self.search_mock.side_effect = [search_result(), TavilyUnavailable("Tavily search returned HTTP 429."),
                                        search_result()]
        packet = research.run_research(**ARGS)
        self.assertEqual(packet["status"], "partial")
        self.assertEqual(len(packet["sources"]), 1)
        self.assertTrue(packet["brief"])
        self.assertEqual(packet["searches"][1]["status"], "failed")
        self.assertIn("HTTP 429", packet["errors"][0])

    def test_no_evidence_is_not_a_no_risk_claim(self):
        self.search_mock.side_effect = None
        self.search_mock.return_value = {"results": []}
        packet = research.run_research(**ARGS)
        self.assertEqual(packet["status"], "no_evidence")
        self.assertIn("do not infer safety", packet["errors"][0])
        self.model_mock.assert_not_called()

    def test_extract_failure_retains_labeled_snippets_and_marks_partial(self):
        self.extract_mock.side_effect = TavilyUnavailable("Extract failed")
        packet = research.run_research(**ARGS)
        self.assertEqual(packet["status"], "partial")
        self.assertEqual(packet["sources"][0]["kind"], "search excerpt")
        self.assertTrue(packet["brief"])

    def test_map_failure_does_not_hide_failure_or_erase_search_evidence(self):
        with patch("public_research.Client.map", side_effect=TavilyUnavailable("Map failed")):
            packet = research.run_research(**ARGS, map_url="https://vendor.com/")
        self.assertEqual(packet["status"], "partial")
        self.assertIn("Map failed", packet["errors"])

    def test_fabricated_passage_rejects_entire_model_brief(self):
        for passage_id in ("E99.P1", "E1.P99", "E1.P1\ninjected"):
            with self.subTest(passage_id=passage_id):
                result = draft_brief()
                result["findings"][0]["evidence"] = [{"passage_id": passage_id}]
                self.model_mock.side_effect = None
                self.model_mock.return_value = result
                packet = research.run_research(**ARGS)
                self.assertEqual(packet["status"], "partial")
                self.assertIsNone(packet["brief"])
                self.assertIn("citation validation failed", packet["errors"][-1])

    def test_model_cannot_override_the_referenced_source_quote(self):
        result = draft_brief()
        result["findings"][0]["evidence"][0]["quote"] = "Invented model quotation."
        self.model_mock.side_effect = None
        self.model_mock.return_value = result
        packet = research.run_research(**ARGS)
        self.assertEqual(packet["status"], "complete")
        self.assertEqual(packet["brief"]["findings"][0]["evidence"],
                         [{"source_id": "E1", "quote": TEXT}])

    def test_passages_are_bounded_and_whitespace_normalized(self):
        text = (TEXT.replace(" ", "\n") + "\n") * 20
        self.extract_mock.return_value = {"results": [{"url": URL, "raw_content": text}]}
        packet = research.run_research(**ARGS)
        self.assertEqual(packet["status"], "complete")
        quote = packet["brief"]["findings"][0]["evidence"][0]["quote"]
        self.assertTrue(12 <= len(quote) <= 600)
        self.assertIn(quote, " ".join(text.split()))
        self.assertNotIn("\n", quote)
        self.assertIn('"passage_id": "E1.P1"', self.model_mock.call_args.args[0])

    def test_uncitable_evidence_does_not_trigger_a_model_request(self):
        self.extract_mock.return_value = {"results": [{"url": URL, "raw_content": "tiny"}]}
        packet = research.run_research(**ARGS)
        self.assertEqual(packet["status"], "partial")
        self.assertIsNone(packet["brief"])
        self.assertIn("no citable passages", packet["errors"][-1])
        self.model_mock.assert_not_called()

    def test_model_outage_retains_evidence_but_no_fake_brief(self):
        self.model_mock.side_effect = ModelUnavailable("Model unavailable")
        packet = research.run_research(**ARGS)
        self.assertEqual(packet["status"], "partial")
        self.assertIsNone(packet["brief"])

    def test_unapproved_or_unsolicited_extract_urls_never_become_evidence(self):
        self.extract_mock.return_value = {"results": [
            {"url": "https://evil.com/", "raw_content": TEXT},
            {"url": "https://vendor.com/not-requested", "raw_content": TEXT},
        ]}
        packet = research.run_research(**ARGS)
        self.assertEqual(len(packet["sources"]), 1)
        self.assertEqual(packet["sources"][0]["kind"], "search excerpt")
        self.assertEqual(packet["status"], "partial")

    def test_attachment_changes_only_supplemental_evidence(self):
        packet = research.run_research(**ARGS)
        engine = SimpleNamespace(cases=[SimpleNamespace(case_id="C1", decision="escalate", status="open")],
                                 cloud_cases={"C1": {"diagnosis": {"root_cause": "unverified"}}})
        before = copy.deepcopy(engine)
        research.attach_to_case(engine, "C1", packet)
        self.assertEqual(engine.cases[0].__dict__, before.cases[0].__dict__)
        self.assertEqual(engine.cloud_cases["C1"]["diagnosis"], before.cloud_cases["C1"]["diagnosis"])
        self.assertEqual(engine.cloud_cases["C1"]["public_research"], packet)
        with self.assertRaises(ValueError):
            research.attach_to_case(engine, "missing", packet)
        with self.assertRaises(ValueError):
            research.attach_to_case(engine, "C1", {"status": "failed"})


class ResearchPageTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        environment = patch.dict(os.environ, {
            "AEGIS_LOCAL_STORE": tmp.name, "USE_FIRESTORE": "0", "USE_SECRET_MANAGER": "0",
            "FORCE_MOCK": "1", "TAVILY_API_KEY": "",
        })
        environment.start()
        self.addCleanup(environment.stop)

    def test_engineer_page_preview_never_calls_api_and_requires_consent(self):
        devices.create_device(name="PRIVATE_DEVICE_NAME", type="METER", ip="10.1.2.3")
        app = AppTest.from_string(
            "import streamlit as st\n"
            "from views.public_research import render\n"
            "st.session_state['user'] = {'email':'engineer@example.com','role':'engineer'}\n"
            "render()\n", default_timeout=30)
        with patch("views.public_research.tavily.configured", return_value=True), \
                patch("integrations.tavily.requests.post") as post:
            app.run()
            app.text_input(key="research_product").set_value("Vendor UPS")
            app.text_input(key="research_symptom").set_value("battery alarm")
            app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(app.button[0].disabled)
        self.assertNotIn("PRIVATE_DEVICE_NAME", str(app.json[0].value))
        post.assert_not_called()

    def test_viewer_is_stopped(self):
        app = AppTest.from_string(
            "import streamlit as st\n"
            "from views.public_research import render\n"
            "st.session_state['user'] = {'email':'viewer@example.com','role':'user'}\n"
            "render()\n", default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(app.error)
        self.assertEqual(len(app.text_input), 0)

    def test_empty_map_has_explicit_coverage_warning_and_trace(self):
        with patch("public_research.Client.search", return_value=search_result()), \
                patch("public_research.Client.extract",
                      return_value={"results": [{"url": URL, "raw_content": TEXT}]}), \
                patch("public_research.Client.map", return_value={"results": ["https://vendor.com/"]}), \
                patch("public_research.generate_json", return_value=brief()):
            packet = research.run_research(**ARGS, map_url="https://vendor.com/")
        app = AppTest.from_string(
            "import streamlit as st\n"
            "from views.public_research import _show_packet\n"
            "_show_packet(st.session_state['packet'])\n", default_timeout=30)
        app.session_state["packet"] = packet
        with patch("views.public_research.st.download_button") as download:
            app.run()
        self.assertEqual(download.call_args.args[2], "prism-spectra-ai-public-evidence.json")
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("Map found no approved documentation links" in warning.value
                            for warning in app.warning))
        self.assertTrue(any(title.value == "Runtime API trace" for title in app.subheader))


if __name__ == "__main__":
    unittest.main()
