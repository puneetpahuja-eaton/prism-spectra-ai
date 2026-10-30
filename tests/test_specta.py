"""Regression coverage for conversational generation and the shared chat surfaces."""
from __future__ import annotations

import os
from unittest.mock import patch

import requests

os.environ["FORCE_MOCK"] = "1"

from streamlit.testing.v1 import AppTest

import datastore
import dataset_gen
import devices
import monitor_gen
import specta
from device_grounding import ClarificationNeeded, verify_passages
from gemini_client import ModelUnavailable
from prism_cloud import Engine
from tests.test_platform import PlatformTestBase


def spec_for(device_id: str) -> dict:
    return {"assets": [{"id": device_id, "metrics": [
        {"name": "channel_x", "unit": "mm", "normal_min": 10, "normal_max": 11,
         "evidence_passages": ["channel_x unit mm", "channel_x synthetic range 10 to 11"]}]}],
        "duration_days": 3, "sample_interval_min": 5}


class SpectaTests(PlatformTestBase):
    def setUp(self):
        super().setUp()
        self.device = devices.create_device(name="Supplied sensor", type="CUSTOM")
        self.device_id = self.device["id"]
        self.conv = specta.Conversation(self.device_id)

    def test_dataset_questions_resume_with_original_request_settings_and_reference(self):
        specta.begin_task(self.conv, "dataset", "Generate channel_x samples", reference_path="docs/params.txt",
                          days=3, interval_min=60)
        with patch("specta.rag.retrieve", return_value=[]), \
                patch("specta.knowledge.fetch_repo_file_text", return_value={"text": "channel_x unit mm"}) as reference, \
                patch("dataset_gen.is_live", return_value=True), \
                patch("dataset_gen.generate_json", side_effect=[
                    {"missing_details": ["What synthetic range should channel_x use?"], "answer": "I found channel_x in mm."},
                    {"spec": spec_for(self.device_id)},
                ]) as model:
            question = specta.respond(self.conv, "Generate channel_x samples", start_task=True)
            self.assertEqual(question["status"], "clarification")
            self.assertIn("I found channel_x", question["content"])
            self.assertNotIn("failed", question["content"].lower())
            self.assertIsNotNone(self.conv.task)
            answer = specta.respond(self.conv, "channel_x synthetic range 10 to 11")
        reference.assert_called_once_with(self.device_id, "docs/params.txt")
        self.assertIn("duration 3 days", model.call_args.args[0])
        self.assertIn("sample interval 60 minutes", model.call_args.args[0])
        self.assertIn("Generate channel_x samples", model.call_args.args[0])
        self.assertIn("channel_x synthetic range 10 to 11", model.call_args.args[0])
        self.assertEqual(answer["status"], "complete")
        self.assertIsNone(self.conv.task)
        self.assertEqual(len(answer["dataset"].telemetry), 72)
        self.assertEqual(answer["spec"]["duration_days"], 3)
        self.assertEqual(answer["spec"]["sample_interval_min"], 60)
        engine = Engine(dataset=answer["dataset"], offline=True)
        engine.run_all()
        self.assertGreaterEqual(engine.summary()["auto_resolved"], 1)
        self.assertGreaterEqual(engine.summary()["escalated"], 1)
        self.assertEqual([m["role"] for m in self.conv.messages], ["user", "assistant", "user", "assistant"])

    def test_split_passages_are_supported_but_invented_quotes_or_values_are_not(self):
        verify_passages("channel_x unit mm", ["channel_x synthetic range 10 to 11"],
                        "channel_x unit mm\nchannel_x synthetic range 10 to 11",
                        ["channel_x", "mm", 10., 11.], unit="mm")
        for passages, values in ((["fabricated range 10 to 11"], [10.]),
                                  (["channel_x synthetic range 10 to 11"], [999.])):
            with self.assertRaises(ValueError):
                verify_passages("channel_x unit mm", passages,
                                "channel_x unit mm\nchannel_x synthetic range 10 to 11", values)

    def test_selected_asset_count_is_enforced_by_provider_schema(self):
        for count in (1, 3):
            schema = dataset_gen._draft_schema(count).model_json_schema()
            assets = schema["$defs"][f"DatasetSpec{count}Assets"]["properties"]["assets"]
            self.assertEqual(assets["minItems"], count)
            self.assertEqual(assets["maxItems"], count)
        one = dataset_gen._draft_schema(1)
        duplicate = spec_for(self.device_id)
        duplicate["assets"] *= 2
        with self.assertRaises(ValueError):
            one.model_validate({"spec": duplicate})

    def test_bad_model_evidence_is_corrected_without_reasking_supplied_facts(self):
        broken = spec_for(self.device_id)
        broken["assets"][0]["metrics"][0]["evidence_passages"] = ["User provided channel_x in mm from 10 to 11"]
        with patch("dataset_gen.is_live", return_value=True), \
                patch("dataset_gen.generate_json", side_effect=[
                    {"spec": broken}, {"spec": spec_for(self.device_id)},
                ]) as model:
            result = dataset_gen.generate_spec("", "channel_x unit mm. channel_x synthetic range 10 to 11")
        self.assertEqual(result["assets"][0]["metrics"][0]["unit"], "mm")
        self.assertEqual(model.call_count, 2)
        self.assertIn("Correct the MODEL OUTPUT", model.call_args.args[0])

    def test_evidence_retry_is_bounded_and_reports_model_failure_not_more_questions(self):
        with patch("dataset_gen.is_live", return_value=True), \
                patch("dataset_gen.generate_json", return_value={"spec": spec_for(self.device_id)}) as model:
            with self.assertRaisesRegex(ValueError, "generated evidence after one correction"):
                dataset_gen.generate_spec("", "No metric guidance is available.")
        self.assertEqual(model.call_count, 2)

    def test_provider_error_is_a_persisted_reply_and_can_retry(self):
        specta.begin_task(self.conv, "dataset", "Generate data", days=1, interval_min=60)
        with patch("specta.rag.retrieve", return_value=[]), \
                patch("specta.dataset_gen.generate_spec", side_effect=ModelUnavailable("provider timed out")):
            response = specta.respond(self.conv, "Generate data", start_task=True)
        self.assertEqual(response["status"], "error")
        self.assertIn("provider timed out", response["content"])
        self.assertIn("retry", response["content"])
        self.assertEqual(self.conv.messages[-1], response)
        self.assertIsNotNone(self.conv.task)
        with patch("specta.rag.retrieve", return_value=[]), \
                patch("specta.dataset_gen.generate_spec", return_value=spec_for(self.device_id)):
            self.assertEqual(specta.respond(self.conv, "retry")["status"], "complete")

    def test_real_client_timeout_retains_task_without_claiming_generation_or_empty_credit(self):
        specta.begin_task(self.conv, "dataset", "Generate data", days=3, interval_min=60)
        task = self.conv.task
        with patch.dict(os.environ, {
            "AI_PROVIDER": "nebius", "FORCE_MOCK": "0", "NEBIUS_API_KEY": "test-key",
            "NEBIUS_TIMEOUT_SECONDS": "120",
        }), patch("specta.rag.retrieve", return_value=[]), \
                patch("gemini_client.netsetup.configure_tls"), \
                patch("gemini_client.requests.post", side_effect=requests.exceptions.ReadTimeout) as post:
            response = specta.respond(self.conv, "Generate data", start_task=True)
        self.assertEqual(response["status"], "error")
        self.assertIn("120-second read timeout", response["content"])
        self.assertIn("not a confirmed credit error", response["content"])
        self.assertIn("wait before retrying", response["content"])
        self.assertNotIn("dataset", response)
        self.assertIs(self.conv.task, task)
        self.assertEqual(self.conv.task.interval_min, 60)
        self.assertEqual(self.conv.messages[-1], response)
        post.assert_called_once()

    def test_monitor_questions_resume_and_never_apply_implicitly(self):
        specta.begin_task(self.conv, "monitor", "Monitor parameter 17")
        with patch("specta.rag.retrieve", return_value=[]), patch("monitor_gen.is_live", return_value=True), \
                patch("monitor_gen.generate_json", side_effect=[
                    {"missing_details": ["What unit and interval should parameter 17 use?"]},
                    {"poll_interval_s": 30, "params": [
                        {"pid": "17", "unit": "mm", "path": "data.channel_x",
                         "evidence": "17 unit mm path data.channel_x"}]},
                ]):
            self.assertEqual(specta.respond(self.conv, "Monitor parameter 17", start_task=True)["status"], "clarification")
            response = specta.respond(self.conv, "17 unit mm path data.channel_x. Poll every 30 seconds.")
        self.assertEqual(response["status"], "complete")
        self.assertIsNone(datastore.get_doc(devices.live_path(self.device_id), "config"))
        with self.assertRaises(PermissionError):
            specta.apply_monitor_plan(self.device_id, response, "https://device.example/api", role="user", confirmed=True)
        with self.assertRaisesRegex(ValueError, "Verify"):
            specta.apply_monitor_plan(self.device_id, response, "https://device.example/api", role="engineer", confirmed=False)
        with self.assertRaisesRegex(ValueError, "Stop"):
            specta.apply_monitor_plan(self.device_id, response, "https://device.example/api",
                                     role="engineer", confirmed=True, running=True)
        specta.apply_monitor_plan(self.device_id, response, "https://device.example/api", role="engineer", confirmed=True)
        cfg = datastore.get_doc(devices.live_path(self.device_id), "config")
        self.assertFalse(cfg["simulated"])
        self.assertEqual(cfg["field_map"], {"p17": "data.channel_x"})
        self.assertIsNone(cfg["hex_as_float"])

    def test_dataset_loading_is_permission_and_confirmation_gated(self):
        with patch("specta.rag.retrieve", return_value=[]), \
                patch("specta.dataset_gen.generate_spec", return_value=spec_for(self.device_id)):
            specta.begin_task(self.conv, "dataset", "Generate data", days=1, interval_min=60)
            response = specta.respond(self.conv, "Generate data", start_task=True)
        with patch("specta.device_data.set_dataset") as save:
            with self.assertRaises(PermissionError):
                specta.load_dataset(self.device_id, response, role="engineer", confirmed=True)
            with self.assertRaises(ValueError):
                specta.load_dataset(self.device_id, response, role="admin", confirmed=False)
            save.assert_not_called()
            specta.load_dataset(self.device_id, response, role="admin", confirmed=True)
            save.assert_called_once_with(self.device_id, response["dataset"])

    def test_ask_and_plan_do_not_execute_generation(self):
        for mode in ("ask", "plan"):
            with patch("specta.rag.converse", return_value={"answer": "A plan.", "task": "dataset"}), \
                    patch("specta._run_task") as task:
                self.assertEqual(specta.respond(self.conv, "generate data", mode=mode)["content"], "A plan.")
                task.assert_not_called()
        specta.begin_task(self.conv, "dataset", "Generate data")
        with patch("specta._run_task") as task:
            self.assertEqual(specta.respond(self.conv, "Here are details", mode="ask")["status"], "paused")
            task.assert_not_called()
        self.assertIsNotNone(self.conv.task)

    def test_agent_chat_routes_to_real_dataset_tool(self):
        with patch("specta.rag.converse", return_value={"answer": "", "task": "dataset"}), \
                patch("specta.rag.retrieve", return_value=[]), \
                patch("specta.dataset_gen.generate_spec", return_value=spec_for(self.device_id)):
            result = specta.respond(self.conv, "Generate a synthetic dataset for channel_x.")
        self.assertIn("dataset", result)

    def test_new_task_reuses_user_guidance_but_not_assistant_assumptions(self):
        self.conv.messages = [
            {"role": "user", "content": "channel_x unit mm; channel_x synthetic range 10 to 11"},
            {"role": "assistant", "content": "INVENTED_ASSISTANT_FACT"},
        ]
        specta.begin_task(self.conv, "dataset", "Generate data", days=1, interval_min=60)
        with patch("specta.rag.retrieve", return_value=[]), \
                patch("specta.dataset_gen.generate_spec", return_value=spec_for(self.device_id)) as generate:
            response = specta.respond(self.conv, "Generate data", start_task=True)
        self.assertEqual(response["status"], "complete")
        self.assertIn("channel_x unit mm", generate.call_args.args[0])
        self.assertNotIn("INVENTED_ASSISTANT_FACT", generate.call_args.args[0])

    def test_attachments_are_atomic_device_scoped_and_sent_to_the_tool(self):
        specta.add_attachments(self.conv, [("units.txt", b"channel_x unit mm")])
        with self.assertRaisesRegex(ValueError, "No readable text"):
            specta.add_attachments(self.conv, [("range.txt", b"channel_x range 10 to 11"), ("empty.txt", b"")])
        self.assertEqual(len(self.conv.attachments), 1)
        other = specta.Conversation("another-device")
        self.assertFalse(other.attachments)
        specta.begin_task(self.conv, "dataset", "Generate data")
        with patch("specta.rag.retrieve", return_value=[]), \
                patch("specta.dataset_gen.generate_spec", side_effect=ClarificationNeeded(["What range?"])) as generate:
            specta.respond(self.conv, "Generate data", start_task=True)
        self.assertIn("channel_x unit mm", generate.call_args.args[2])


UI_SCRIPT = """
import streamlit as st
import specta_ui
st.session_state.setdefault("user", {"email": "engineer@example.test", "role": "engineer"})
st.session_state["specta_page"] = "Ask the device"
specta_ui.mount_dock()
specta_ui.render_workspace()
"""


class SpectaUITests(PlatformTestBase):
    def setUp(self):
        super().setUp()
        self.a = devices.create_device(name="Sensor A", type="CUSTOM")["id"]
        self.b = devices.create_device(name="Sensor B", type="CUSTOM")["id"]

    def app(self):
        return AppTest.from_string(UI_SCRIPT, default_timeout=30).run()

    def test_open_close_device_modes_and_new_chat(self):
        app = self.app()
        self.assertFalse(app.exception)
        self.assertIn("Specta", [button.label for button in app.button])
        app.selectbox(key="specta_device_full").select(self.a).run()
        mode = f"specta_mode_full_{self.a}"
        app.selectbox(key=mode).select("Ask").run()
        self.assertEqual(app.session_state[f"specta_conversation_{self.a}"].mode, "ask")
        app.button(key="specta_toggle").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.selectbox(key="specta_device_dock").value, self.a)
        app.selectbox(key=f"specta_mode_dock_{self.a}").select("Plan").run()
        self.assertEqual(app.selectbox(key=mode).value, "Plan")
        conv = app.session_state[f"specta_conversation_{self.a}"]
        conv.messages.append({"role": "assistant", "content": "Remember me."})
        app.run()
        self.assertEqual(sum(m.value == "Remember me." for m in app.markdown), 2)
        app.selectbox(key="specta_device_dock").select(self.b).run()
        self.assertEqual(app.selectbox(key="specta_device_full").value, self.b)
        self.assertFalse(app.session_state[f"specta_conversation_{self.b}"].messages)
        app.selectbox(key="specta_device_dock").select(self.a).run()
        self.assertEqual(len(app.session_state[f"specta_conversation_{self.a}"].messages), 1)
        app.button(key="specta_new_full").click().run()
        self.assertFalse(app.session_state[f"specta_conversation_{self.a}"].messages)
        app.button(key="specta_toggle").click().run()
        self.assertNotIn("specta_device_dock", [s.key for s in app.selectbox])
        self.assertFalse(app.exception)

    def test_missing_dataset_details_appear_in_chat_and_reply_finishes(self):
        app = self.app()
        app.selectbox(key="specta_device_full").select(self.a).run()
        app.text_area(key=f"ds_req_{self.a}").input("Generate channel_x samples").run()
        with patch("specta.rag.retrieve", return_value=[]), \
                patch("specta.dataset_gen.generate_spec", side_effect=ClarificationNeeded(["What unit and range?"])):
            app.button(key=f"ds_gen_{self.a}").click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertFalse(any("Ask clarifying" in button.label for button in app.button))
        self.assertIn("What unit and range?", "\n".join(m.value for m in app.markdown))
        self.assertIsNotNone(app.session_state[f"specta_conversation_{self.a}"].task)
        with patch("specta.rag.retrieve", return_value=[]), \
                patch("specta.dataset_gen.generate_spec", return_value=spec_for(self.a)):
            app.chat_input(key=f"specta_input_full_{self.a}").set_value("Use mm and 10 to 11.").run()
        self.assertFalse(app.exception)
        self.assertIsNone(app.session_state[f"specta_conversation_{self.a}"].task)
        self.assertIn("synthetic dataset is ready", "\n".join(m.value for m in app.markdown))
        self.assertNotIn("Load into Analysis", [b.label for b in app.button])  # engineer, not admin

    def test_monitor_builder_uses_same_conversation_without_manual_question_button(self):
        script = """
import streamlit as st
import devices
import specta_ui
from views.live_monitoring import _plan_builder
st.session_state["user"] = {"email": "engineer@example.test", "role": "engineer"}
specta_ui.mount_dock()
_plan_builder(devices.list_devices()[0])
"""
        app = AppTest.from_string(script, default_timeout=30).run()
        app.text_area(key=f"mp_req_{self.a}").input("Monitor channel_x").run()
        with patch("specta.rag.retrieve", return_value=[]), \
                patch("specta.monitor_gen.generate_plan", side_effect=ClarificationNeeded(["Which polling interval?"])):
            app.button(key=f"mp_gen_{self.a}").click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertFalse(any("Ask clarifying" in b.label for b in app.button))
        self.assertIn("Which polling interval?", "\n".join(m.value for m in app.markdown))
        self.assertEqual(app.session_state[f"specta_conversation_{self.a}"].task.kind, "monitor")
