"""Partner integrations: source boundaries, metadata privacy and responsive polling."""
from __future__ import annotations

from concurrent.futures import Future
import copy
import json
import os
from unittest.mock import Mock, patch

from langsmith.utils import LangSmithError
import numpy as np
from streamlit.testing.v1 import AppTest

import devices
import inline_research
from nebius_client import ModelUnavailable
from observation_stats import MetricStatistics
from prism_cloud import Engine
import specta
from tests.test_platform import PlatformTestBase
from tests.test_specta import spec_for
import dataset_gen
import workflow


def packet() -> dict:
    return {"id": "public-1", "status": "complete", "created_at": "2026-10-30",
            "sources": [{"id": "E1", "title": "Public manual", "url": "https://docs.python.org/3/library/json.html",
                         "text": "Public evidence is supplementary."}],
            "errors": [], "trace": [{"credits": 2}], "brief": None}


class PartnerWorkflowTests(PlatformTestBase):
    def setUp(self):
        super().setUp()
        self.device = devices.create_device(name="PRIVATE_DEVICE", type="CUSTOM")
        self.did = self.device["id"]
        self.request = inline_research.PublicRequest(product="Python", symptom="json",
                                                    domains_text="docs.python.org")

    def test_trace_payload_excludes_private_inputs_outputs_and_exception_messages(self):
        client = Mock()
        with patch.dict(os.environ, {"LANGSMITH_TRACING": "true", "FORCE_MOCK": "0"}), \
                patch("workflow.trace_client", return_value=client):
            with workflow.Workflow("specta") as run:
                result = workflow.run_step("retrieve", lambda: {"private": "SECRET_SOURCE"})
                self.assertEqual(result["private"], "SECRET_SOURCE")
                with self.assertRaisesRegex(ValueError, "SECRET_ERROR"):
                    workflow.run_step("diagnosis", lambda: (_ for _ in ()).throw(ValueError("SECRET_ERROR")))
        sent = str(client.create_run.call_args)
        self.assertNotIn("SECRET_SOURCE", sent)
        self.assertNotIn("SECRET_ERROR", sent)
        self.assertNotIn("PRIVATE_DEVICE", sent)
        self.assertEqual(client.create_run.call_args.kwargs["inputs"], {})
        self.assertEqual(client.create_run.call_args.kwargs["outputs"], {})
        self.assertEqual(run.report["tracing"], "submitted")
        self.assertEqual(run.report["status"], "failed")

    def test_trace_outage_does_not_discard_successful_result(self):
        with patch.dict(os.environ, {"LANGSMITH_TRACING": "true", "FORCE_MOCK": "0"}), \
                patch("workflow.trace_client", side_effect=LangSmithError("do not expose this")):
            with workflow.Workflow("specta") as run:
                self.assertEqual(workflow.run_step("dataset", lambda: 72), 72)
        self.assertEqual(run.report["tracing"], "failed")
        self.assertNotIn("do not expose", json.dumps(run.report))

    def test_trace_is_disabled_by_default_and_rejects_unknown_stages(self):
        with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}), patch("workflow.trace_client") as client:
            with workflow.Workflow("specta"):
                workflow.run_step("dataset", lambda: None)
            client.assert_not_called()
        with self.assertRaises(ValueError):
            workflow.run_step("SECRET_DEVICE_NAME", lambda: None)

    def test_global_auto_tracing_is_suppressed_inside_langchain_stage(self):
        from langsmith.run_helpers import get_tracing_context
        with workflow.Workflow("specta"):
            self.assertFalse(workflow.run_step("dataset", lambda: get_tracing_context()["enabled"]))

    def test_partial_provider_result_is_not_reported_as_complete(self):
        with workflow.Workflow("public_research") as run:
            workflow.run_step("public_research", lambda: {"status": "partial"})
        self.assertEqual(run.report["status"], "partial")

    def test_failed_new_request_never_relabels_cached_old_packet(self):
        script = """
import hashlib
from concurrent.futures import Future
import streamlit as st
import research_ui
st.session_state["user"] = {"role": "engineer", "email": "engineer@example.test"}
key = "inline_research_" + hashlib.sha256(b"selected:case-one").hexdigest()[:20]
if "initialized" not in st.session_state:
    future = Future()
    future.set_exception(ValueError("new request failed"))
    st.session_state[key] = {"future": future, "request": "old",
                            "pending_request": "new", "packet": None}
    st.session_state["initialized"] = True
research_ui.render("selected", "case-one", surface="test", device_evidence=[])
"""
        app = AppTest.from_string(script, default_timeout=30).run()
        self.assertFalse(app.exception)
        import hashlib
        key = "inline_research_" + hashlib.sha256(b"selected:case-one").hexdigest()[:20]
        self.assertEqual(app.session_state[key]["request"], "old")
        self.assertNotIn("pending_request", app.session_state[key])
        self.assertIn("new request failed", app.session_state[key]["error"])

    def test_consent_role_and_domain_validation_happen_before_network(self):
        with patch("inline_research._tool") as tool:
            for role, consent in (("engineer", False), ("user", True)):
                with self.assertRaises((ValueError, PermissionError)):
                    inline_research.run(self.did, self.request, role=role, consent=consent, device_evidence=[])
            bad = self.request.model_copy(update={"product": "password=secret"})
            with self.assertRaises(ValueError):
                inline_research.run(self.did, bad, role="engineer", consent=True, device_evidence=[])
        tool.invoke.assert_not_called()

    def test_only_approved_public_fields_reach_the_tavily_tool(self):
        with patch("inline_research._tool") as tool, \
                patch("inline_research.compare", return_value={"findings": [], "uncertainties": ["Review required"]}):
            tool.invoke.return_value = packet()
            result = inline_research.run(self.did, self.request, role="engineer", consent=True,
                                         device_evidence=[{"text": "SECRET_TELEMETRY"}])
        self.assertEqual(tool.invoke.call_args.args[0], self.request.model_dump())
        self.assertNotIn("SECRET_TELEMETRY", str(tool.invoke.call_args))
        self.assertEqual(result["device_id"], self.did)
        self.assertEqual(result["workflow"]["stages"][0]["usage"]["credits"], 2)

    def test_unknown_comparison_citation_rejects_the_assessment(self):
        with patch("inline_research.generate_json", return_value={
            "findings": [{"statement": "Unsupported", "passage_ids": ["invented"]}], "uncertainties": ["?"]}):
            with self.assertRaises(ModelUnavailable):
                inline_research.compare(packet(), [])

    def test_comparison_attaches_exact_source_text_and_keeps_source_classes_separate(self):
        with patch("inline_research.generate_json", return_value={
            "findings": [{"statement": "Review", "passage_ids": ["D1", "PE1"]}], "uncertainties": ["?"]}):
            result = inline_research.compare(packet(), [{"text": "Exact supplied text"}])
        evidence = result["findings"][0]["evidence"]
        self.assertEqual(evidence[0]["text"], "Exact supplied text")
        self.assertEqual(evidence[0]["kind"], "supplied device evidence")
        self.assertEqual(evidence[1]["kind"], "supplementary public evidence")

    def test_comparison_bounds_excerpts_and_preserves_unicode_without_escape_expansion(self):
        public = packet()
        public["sources"] = [
            {"id": f"E{i}", "text": "\u6e29" * 2000, "url": "https://docs.python.org"}
            for i in range(1, 10)
        ]
        supplied = [{"text": "x" * 2000} for _ in range(10)]
        with patch("inline_research.generate_json", return_value={
            "findings": [{"statement": "Review", "passage_ids": ["D6", "PE6"]}],
            "uncertainties": ["Bounded excerpts only"],
        }) as model:
            result = inline_research.compare(public, supplied)
        prompt = model.call_args.args[0]
        self.assertIn("\u6e29", prompt)
        self.assertNotIn("\\u6e29", prompt)
        self.assertNotIn('"D7"', prompt)
        self.assertNotIn('"PE7"', prompt)
        self.assertEqual(model.call_args.kwargs["max_output_tokens"], 16384)
        self.assertTrue(all(len(item["text"]) == 1200 for item in result["findings"][0]["evidence"]))

    def test_comparison_failure_retains_raw_evidence_as_partial_not_success(self):
        with patch("inline_research._tool") as tool, \
                patch("inline_research.compare", side_effect=ModelUnavailable("citation rejected")):
            tool.invoke.return_value = packet()
            result = inline_research.run(self.did, self.request, role="engineer", consent=True, device_evidence=[])
        self.assertEqual(result["status"], "partial")
        self.assertNotIn("comparison", result)
        self.assertTrue(any("citation rejected" in error for error in result["errors"]))

    def test_stale_device_index_retains_public_packet_with_explicit_partial_status(self):
        with patch("inline_research._tool") as tool, \
                patch("inline_research.rag.retrieve", side_effect=ValueError("Re-embed stale chunks")):
            tool.invoke.return_value = packet()
            result = inline_research.run(self.did, self.request, role="engineer", consent=True, device_evidence=[])
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["sources"][0]["id"], "E1")
        self.assertTrue(any("Re-embed stale chunks" in error for error in result["errors"]))

    def test_reset_discards_old_run_interpretation_even_if_worker_cannot_be_cancelled(self):
        data = dataset_gen.build_dataset(dataset_gen.DatasetSpec.model_validate(spec_for(self.did)))
        engine = Engine(dataset=data, offline=False, device_id=self.did, defer_ai=True)
        future = Future()
        future.set_running_or_notify_cancel()
        case = Mock(case_id="C1", root_asset_id=self.did)
        engine.cases = [case]
        with patch("prism_cloud.jobs.submit", return_value=future):
            engine._enrich_device(case, {"metrics": {"x": 99}, "codes": []}, {})
        old_run = engine.run_id
        engine.reset()
        future.set_result({"diagnosis": {"confidence": .99, "_citation_check": "validated"}, "errors": []})
        engine.drain_ai()
        self.assertNotEqual(old_run, engine.run_id)
        self.assertFalse(engine.cloud_cases)
        self.assertFalse(engine._pending_ai)

    def test_failed_live_interpretation_is_explicit_and_observations_are_retained(self):
        data = dataset_gen.build_dataset(dataset_gen.DatasetSpec.model_validate(spec_for(self.did)))
        engine = Engine(dataset=data, offline=False, device_id=self.did, defer_ai=True)
        future = Future()
        case = Mock(case_id="C1", root_asset_id=self.did)
        engine.cases = [case]
        with patch("prism_cloud.jobs.submit", return_value=future):
            engine._enrich_device(case, {"metrics": {"x": 99}, "codes": []}, {})
        future.set_exception(ModelUnavailable("provider unavailable"))
        engine.drain_ai()
        self.assertEqual(engine.cloud_cases["C1"]["ai_status"], "failed")
        self.assertEqual(engine.cloud_cases["C1"]["evidence"]["metrics"], {"x": 99})
        self.assertIn("provider unavailable", engine.cloud_cases["C1"]["errors"][0])

    def test_public_evidence_is_not_added_to_dataset_generation_sources(self):
        conv = specta.Conversation(self.did, public_evidence={**packet(), "device_id": self.did})
        specta.begin_task(conv, "dataset", "Generate data", days=3, interval_min=60)
        with patch("specta.rag.retrieve", return_value=[]), \
                patch("dataset_gen.generate_spec", return_value=spec_for(self.did)) as gen:
            response = specta.respond(conv, "Generate data", start_task=True)
        self.assertEqual(response["status"], "complete")
        self.assertNotIn("Public evidence is supplementary", str(gen.call_args))

    def test_observation_statistics_do_not_fit_local_ml(self):
        stats = MetricStatistics()
        stats.fit({"A": {"features": ["x"], "X": [[10 + i % 2] for i in range(20)]}})
        self.assertFalse(stats.uses_iforest)
        self.assertTrue(stats.score("A", {"x": 50})["is_anomaly"])
        self.assertIsNone(stats.score("A", {"x": 10})["iforest"])
        self.assertNotIn("forest", str(stats.assets.keys()))

    def test_live_ai_is_pending_and_completed_only_on_the_session_thread(self):
        data = dataset_gen.build_dataset(dataset_gen.DatasetSpec.model_validate(spec_for(self.did)))
        engine = Engine(dataset=data, offline=False, device_id=self.did, defer_ai=True)
        future = Future()
        case = Mock(case_id="C1", root_asset_id=self.did)
        engine.cases = [case]
        with patch("prism_cloud.jobs.submit", return_value=future), patch("prism_cloud.assess_device") as assess:
            engine._enrich_device(case, {"metrics": {"x": 99}, "codes": []}, {})
        assess.assert_not_called()
        self.assertEqual(engine.cloud_cases["C1"]["ai_status"], "pending")
        engine.drain_ai()
        self.assertEqual(len(engine._pending_ai), 1)
        future.set_result({"diagnosis": {"confidence": .2, "_citation_check": "validated"}, "errors": []})
        engine.drain_ai()
        self.assertEqual(engine.cloud_cases["C1"]["ai_status"], "complete")
        self.assertEqual(case.confidence, .2)

    def test_new_chat_has_a_new_research_scope(self):
        first = specta.Conversation(self.did)
        second = specta.Conversation(self.did)
        self.assertNotEqual(first.id, second.id)
        self.assertIsNone(second.public_evidence)

    def test_research_ui_has_public_preview_and_no_private_prefill(self):
        script = """
import streamlit as st
import research_ui
st.session_state["user"] = {"role": "engineer", "email": "engineer@example.test"}
research_ui.render("selected", "case-one", surface="test", device_evidence=[{"text":"SECRET_DATA"}])
"""
        app = AppTest.from_string(script, default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertTrue(all(not field.value for field in app.text_input))
        self.assertNotIn("SECRET_DATA", str(app))
