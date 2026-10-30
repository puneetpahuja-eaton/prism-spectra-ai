"""Regression boundaries between supplied device facts and synthetic fixtures."""
from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest

import dataset_gen
import device_data
import devices
from device_grounding import verify_quote
from fleet_data import FleetAsset, FleetDataset, Reading
from nebius_client import ModelUnavailable
from live.inference import suggest_field_map
from live.monitor import LiveMonitor
from live.rest_adapter import poll
from live_engine import LiveFleetEngine
import monitor_gen
from prism_cloud import Engine
from tests.test_platform import PlatformTestBase


def small_dataset():
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    history = [Reading(timestamp=base + timedelta(minutes=i), asset_id="A",
                       metrics={"channel_x": 10 + (i % 3) * .1}, event_codes=["DEVICE_READY"])
               for i in range(20)]
    replay = [Reading(timestamp=base + timedelta(minutes=20 + i), asset_id="A",
                      metrics={"channel_x": value}, event_codes=["DEVICE_READY"] if i != 1 else ["BREAKER_OPEN"])
              for i, value in enumerate((10, 90, 10, 10))]
    return FleetDataset(assets=[FleetAsset(id="A", type="CUSTOM", criticality="low")],
                        history=history, telemetry=replay)


class DeviceGroundingTests(PlatformTestBase):
    def test_generation_schema_requires_explicit_sampling_without_changing_manual_defaults(self):
        for asset_count in (None, 1):
            with self.subTest(asset_count=asset_count):
                schema = dataset_gen._draft_schema(asset_count)
                raw = {"spec": {"assets": [{"id": "A", "metrics": [
                    {"name": "x", "unit": "mm", "normal_min": 10, "normal_max": 11}
                ]}]}}
                with self.assertRaises(ValueError):
                    schema.model_validate(raw)
                raw["spec"].update(duration_days=1, sample_interval_min=5)
                spec = schema.model_validate(raw).spec
                self.assertEqual(spec.duration_days, 1)
                self.assertEqual(spec.sample_interval_min, 5)
        manual = dataset_gen.DatasetSpec.model_validate(raw["spec"])
        self.assertEqual(manual.duration_days, 1)
        self.assertEqual(dataset_gen.DatasetSpec.model_fields["duration_days"].default, 3)

    def test_bad_model_quote_is_replaced_only_by_a_complete_verbatim_source_row(self):
        source = "channel_unique unit mm; channel_unique synthetic range 10 to 11."
        raw = {"spec": {"assets": [{"id": "A", "metrics": [
            {"name": "channel_unique", "unit": "mm", "normal_min": 10, "normal_max": 11,
             "evidence": "Model invented this quote."}]}]}}
        with patch("dataset_gen.is_live", return_value=True), \
                patch("dataset_gen.generate_json", return_value=raw) as model:
            result = dataset_gen.generate_spec("", source)
        self.assertEqual(model.call_count, 1)
        metric = result["assets"][0]["metrics"][0]
        self.assertEqual(metric["evidence"], source)
        self.assertEqual(metric["evidence_passages"], [])

    def test_source_row_repair_cannot_supply_a_fabricated_range(self):
        source = "channel_unique unit mm; channel_unique synthetic range 10 to 11."
        raw = {"spec": {"assets": [{"id": "A", "metrics": [
            {"name": "channel_unique", "unit": "mm", "normal_min": 10, "normal_max": 999,
             "evidence": "Model invented range 10 to 999."}]}]}}
        with patch("dataset_gen.is_live", return_value=True), \
                patch("dataset_gen.generate_json", return_value=raw) as model:
            with self.assertRaisesRegex(ValueError, "generated evidence after one correction"):
                dataset_gen.generate_spec("", source)
        self.assertEqual(model.call_count, 2)

    def test_generation_bounds_large_sources_but_preserves_user_request_and_relevant_tail(self):
        irrelevant = "unrelated placeholder documentation\n" * 8000
        target = "channel_unique unit mm; channel_unique synthetic range 10 to 11"
        request = "Generate a synthetic dataset for channel_unique."
        with patch("dataset_gen.is_live", return_value=True), \
                patch("dataset_gen.generate_json", return_value={"missing_details": ["Which asset?"]}) as model:
            with self.assertRaises(ValueError):
                dataset_gen.generate_spec(irrelevant, request, irrelevant + target)
        prompt = model.call_args.args[0]
        self.assertEqual(model.call_args.kwargs["max_output_tokens"], 32768)
        self.assertFalse(model.call_args.kwargs["reasoning"])
        self.assertLess(len(prompt), 38000)
        self.assertIn(request, prompt)
        self.assertIn(target, prompt)
        self.assertIn("selected excerpts", prompt)

    def test_shared_source_budget_keeps_small_reference_and_bounds_large_context(self):
        from device_grounding import generation_sources
        context, reference = generation_sources(
            "irrelevant placeholder\n" * 10000, "channel_unique unit mm range 10 to 11",
            "Monitor channel_unique", "")
        self.assertLessEqual(len(context) + len(reference), 30000)
        self.assertIn("channel_unique unit mm range 10 to 11", reference)

    def test_source_selection_preserves_all_requested_parameter_facts_at_tail(self):
        from device_grounding import generation_sources
        filler = "\n".join(f"Unrelated synthetic placeholder record {i}: no device facts." for i in range(6000))
        ids = [f"TEST_{i:03d}" for i in range(30)]
        facts = [f"Parameter {pid} label channel_{i:03d} unit mm path value." for i, pid in enumerate(ids)]
        request = "Build a monitoring plan for exactly these parameter IDs: " + ", ".join(ids)
        context, reference = generation_sources(filler, filler + "\n" + "\n".join(facts), request, "")
        self.assertLessEqual(len(context) + len(reference) + len(request), 30000)
        for fact in facts:
            self.assertIn(fact, reference)

    def test_trimmed_monitor_material_is_not_available_as_validation_evidence(self):
        omitted = "HIDDEN_PARAMETER unit mm"
        context = "\n".join(f"irrelevant row {index} other text" for index in range(12000))
        context = context[:120000] + omitted + context[120000:]
        raw = {"poll_interval_s": 60, "params": [
            {"pid": "HIDDEN_PARAMETER", "unit": "mm", "evidence": omitted}]}
        with patch("monitor_gen.is_live", return_value=True), \
                patch("monitor_gen.generate_json", return_value=raw) as model:
            with self.assertRaisesRegex(ValueError, "generated evidence after one correction"):
                monitor_gen.generate_plan("D", "Monitor channel_unique", device_context=context)
        self.assertNotIn(omitted, model.call_args_list[0].args[0])

    def test_rag_checks_cloud_existence_without_reading_all_embeddings_twice(self):
        import datastore
        collection = Mock()
        collection.select.return_value.limit.return_value.stream.return_value = iter([Mock()])
        with patch("datastore.is_firestore", return_value=True), \
                patch("datastore._fs_collection", return_value=collection), \
                patch("datastore.list_docs") as listing:
            self.assertTrue(datastore.has_docs("device/chunks"))
        collection.select.assert_called_once_with([])
        collection.select.return_value.limit.assert_called_once_with(1)
        listing.assert_not_called()

    def test_new_and_legacy_unconfigured_devices_do_not_enable_demo(self):
        device = devices.create_device(name="Unknown", type="CUSTOM")
        self.assertEqual(device_data.mode(device["id"]), "unconfigured")
        import datastore
        datastore.set_doc("devices", "old", {"id": "old"})
        self.assertEqual(device_data.mode("old"), "unconfigured")

    def test_demo_switch_does_not_delete_saved_readings(self):
        device = devices.create_device(name="D", type="CUSTOM")
        data = small_dataset()
        device_data.set_dataset(device["id"], data)
        device_data.set_synthetic(device["id"])
        self.assertEqual(device_data.get_dataset(device["id"]).model_dump_json(), data.model_dump_json())
        device_data.set_dataset(device["id"], data)
        self.assertFalse(device_data.uses_demo_profile(device["id"]))

    def test_selector_requires_explicit_choice(self):
        device = devices.create_device(name="Demo", type="CUSTOM")
        app = AppTest.from_string("from ui_common import device_selector\ndevice_selector()\n").run()
        self.assertIsNone(app.selectbox[0].value)
        app.selectbox[0].set_value(device["id"]).run()
        self.assertEqual(app.session_state["active_device"], device["id"])

    def test_unconfigured_analysis_does_not_create_power_engine(self):
        device = devices.create_device(name="New", type="CUSTOM")
        app = AppTest.from_string(
            "import streamlit as st\nfrom views.analysis import render\nrender()\n")
        app.session_state["analysis_device"] = device["id"]
        with patch("views.analysis.Engine") as engine:
            app.run()
        engine.assert_not_called()
        self.assertFalse(app.exception)
        self.assertTrue(any("No telemetry is configured" in item.value for item in app.info))

    def test_device_analysis_renders_and_steps_without_simulator(self):
        device = devices.create_device(name="Source-only", type="CUSTOM")
        device_data.set_dataset(device["id"], small_dataset())
        app = AppTest.from_string("from views.analysis import render\nrender()\n", default_timeout=30)
        app.session_state["analysis_device"] = device["id"]
        app.session_state["user"] = {"email": "engineer@example.com", "role": "engineer"}
        app.run()
        self.assertFalse(app.exception)
        next(button for button in app.button if button.label == "Analyze next reading").click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.session_state["engine"].demo_profile)
        self.assertEqual(app.session_state["engine"].current_idx, 0)
        next(button for button in app.button if button.label == "Analyze next reading").click().run()
        self.assertFalse(app.exception)
        next(button for button in app.button if button.label == "Research public guidance for this case").click().run()
        self.assertEqual(app.session_state["research_case_context"]["device_id"], device["id"])
        from prism_extensions import _audit_rows
        self.assertEqual(_audit_rows(app.session_state["engine"])[0]["root_cause"], "UNCLASSIFIED")

    def test_generic_engine_never_loads_bundled_sop_or_playbook(self):
        data = small_dataset()
        original = data.model_dump_json()
        with patch("prism_cloud.storage.load_text") as bundled, patch("rag.retrieve") as retrieve:
            engine = Engine(dataset=data, offline=True)
            engine.run_all()
        bundled.assert_not_called()
        retrieve.assert_not_called()
        self.assertTrue(engine.cases)
        self.assertTrue(all(c.fault_type == "UNCLASSIFIED" and c.decision == "ESCALATE" for c in engine.cases))
        self.assertEqual(engine.summary()["cost_avoided_usd"], 0)
        self.assertEqual(engine.summary()["downtime_avoided_min"], 0)
        self.assertIsNone(engine.summary()["diagnosis_accuracy"])
        self.assertEqual(data.model_dump_json(), original)
        self.assertTrue(all(detail["errors"] for detail in engine.cloud_cases.values()))
        self.assertNotIn("lookup_playbook", {tool.tool for tool in engine.tool_calls})

    def test_model_diagnosis_retrieves_only_selected_device_sources(self):
        result = {"confidence": .2, "_citation_check": "validated"}
        with patch("rag.retrieve", return_value=[{"text": "Custom channel guidance.", "source_name": "user-note"}]) as retrieve, \
                patch("prism_cloud.diagnose", return_value=result) as diagnose:
            engine = Engine(dataset=small_dataset(), device_id="selected", offline=False)
            engine.tick()
            engine.tick()
        self.assertEqual(retrieve.call_args.args[0], "selected")
        self.assertEqual(diagnose.call_args.kwargs, {"device_only": True})
        self.assertIn("Custom channel guidance.", diagnose.call_args.args[1])
        self.assertNotIn("SOP-106", diagnose.call_args.args[1])

    def test_independent_generator_uses_supplied_frequency_range_not_50hz(self):
        spec = dataset_gen.DatasetSpec(assets=[dataset_gen.AssetSpec(
            id="A", metrics=[dataset_gen.MetricSpec(name="frequency_hz", unit="Hz", normal_min=700, normal_max=710)])],
            sample_interval_min=60, duration_days=1)
        data = dataset_gen.build_dataset(spec)
        self.assertTrue(all(700 <= r.metrics["frequency_hz"] <= 710 for r in data.history + data.telemetry))
        self.assertFalse(data.scenarios)

    def test_generic_dataset_retains_recovery_and_escalation_mix(self):
        spec = dataset_gen.DatasetSpec(assets=[dataset_gen.AssetSpec(
            id="A", criticality="low",
            metrics=[dataset_gen.MetricSpec(name="arbitrary_channel", normal_min=10, normal_max=11)])],
            sample_interval_min=60, duration_days=3)
        spec.anomalies = dataset_gen.ensure_anomaly_mix(spec)
        self.assertTrue(any(a.recovers for a in spec.anomalies))
        self.assertTrue(any(not a.recovers for a in spec.anomalies))
        engine = Engine(dataset=dataset_gen.build_dataset(spec), offline=True)
        engine.run_all()
        self.assertGreaterEqual(engine.summary()["auto_resolved"], 1)
        self.assertGreaterEqual(engine.summary()["escalated"], 1)
        self.assertEqual(engine.summary()["cost_avoided_usd"], 0)

    def test_ungrounded_generated_spec_is_rejected(self):
        raw = {"assets": [{"id": "A", "metrics": [{"name": "invented", "unit": "%", "normal_min": 1, "normal_max": 2}]}]}
        with patch("dataset_gen.is_live", return_value=True), \
                patch("dataset_gen.generate_json", return_value={"spec": raw}):
            with self.assertRaisesRegex(ValueError, "evidence"):
                dataset_gen.generate_spec("", "Build a dataset")

    def test_missing_dataset_details_are_questions_not_defaults(self):
        with patch("dataset_gen.is_live", return_value=True), \
                patch("dataset_gen.generate_json", return_value={"missing_details": ["Which units and ranges?"]}):
            with self.assertRaisesRegex(ValueError, "Which units"):
                dataset_gen.generate_spec("", "Build a dataset")

    def test_literal_fields_must_occur_in_exact_supplied_quote(self):
        verify_quote("channel_x unit mm range 1 to 5", "channel_x unit mm range 1 to 5", ["channel_x", "mm", 1., 5.])
        for quote, values in (("invented text", []), ("channel_x unit mm range 1 to 5", [100.])):
            with self.assertRaises(ValueError):
                verify_quote(quote, "channel_x unit mm range 1 to 5", values)

    def test_missing_monitor_plan_details_are_rejected(self):
        with patch("monitor_gen.is_live", return_value=True), patch("monitor_gen.rag.retrieve", return_value=[]), \
                patch("monitor_gen.generate_json", return_value={"params": [], "missing_details": ["Which parameter ID?"]}):
            with self.assertRaisesRegex(ValueError, "parameter ID"):
                monitor_gen.generate_plan("D", "monitor")

    def test_unknown_encoding_is_not_replaced_by_float(self):
        config = monitor_gen.plan_to_config({"poll_interval_s": 30, "params": []}, "https://device.example/api")
        self.assertIsNone(config["hex_as_float"])
        response = Mock()
        response.json.return_value = {"value": "0x41F00000"}
        with patch("live.rest_adapter.requests.get", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "explicit encoding"):
                poll("rest", config)
            config["hex_as_float"] = False
            self.assertEqual(poll("rest", config)["metrics"]["value"], int("41F00000", 16))
            config["hex_as_float"] = True
            self.assertEqual(poll("rest", config)["metrics"]["value"], 30.)

    def test_real_monitor_does_not_assign_demo_code_meanings(self):
        monitor = LiveMonitor()
        monitor.observe({"metrics": {"raw": 10}, "codes": ["BREAKER_OPEN"]})
        self.assertFalse(monitor.issues)

    def test_live_baseline_requires_confirmation(self):
        with self.assertRaisesRegex(ValueError, "Confirm"):
            LiveFleetEngine(asset_id="A")
        engine = LiveFleetEngine(asset_id="A", asset_type="CUSTOM", device_id="D",
                                 baseline_confirmed=True, offline=True)
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for i in range(21):
            engine.observe({"timestamp": (base + timedelta(minutes=i)).isoformat(),
                            "metrics": {"raw": 10 + i % 2}, "codes": ["DEVICE_READY"]})
        self.assertEqual(engine.engine.topo.get("A").type, "CUSTOM")
        self.assertFalse(engine.engine.demo_profile)
        self.assertEqual(engine.engine.dataset.history[0].event_codes, ["DEVICE_READY"])
        self.assertEqual(engine.engine.current_idx, 0)
        engine.observe({"timestamp": (base + timedelta(minutes=21)).isoformat(),
                        "metrics": {"raw": 77}, "codes": ["RAW_EVENT"]})
        self.assertEqual(engine.engine.current_idx, 1)
        self.assertEqual(engine.engine.history[1]["A"]["metrics"], {"raw": 77})
        with self.assertRaisesRegex(ValueError, "missing"):
            engine.observe({"metrics": {}})

    def test_field_mapping_outage_does_not_substitute_heuristics(self):
        with patch("live.inference.is_live", return_value=True), \
                patch("live.inference.rag.retrieve", return_value=[]), \
                patch("live.inference.generate_json", side_effect=ModelUnavailable("down")):
            with self.assertRaises(ModelUnavailable):
                suggest_field_map("D", '{"temperature": 5}')

    def test_field_mapping_rejects_invented_path_or_metric(self):
        with patch("live.inference.is_live", return_value=True), \
                patch("live.inference.rag.retrieve", return_value=[]), \
                patch("live.inference.generate_json", return_value={"field_map": {"temperature_c": "raw"}}):
            with self.assertRaisesRegex(ValueError, "metric name"):
                suggest_field_map("D", '{"raw":5}')

    def test_numeric_array_mapping_accepts_actual_nonfirst_element(self):
        with patch("live.inference.is_live", return_value=True), \
                patch("live.inference.rag.retrieve", return_value=[]), \
                patch("live.inference.generate_json", return_value={"field_map": {"raw": "readings.1.raw"}}):
            self.assertEqual(suggest_field_map("D", '{"readings":[{"raw":1},{"raw":2}]}')["field_map"],
                             {"raw": "readings.1.raw"})

    def test_device_live_page_does_not_default_to_simulation_or_hex_decoding(self):
        device = devices.create_device(name="Unknown", type="CUSTOM")
        app = AppTest.from_string(
            "import streamlit as st\nfrom views.live_monitoring import render\n"
            "st.session_state['user']={'email':'engineer@example.com','role':'engineer'}\nrender()\n")
        app.session_state["live_device"] = device["id"]
        app.run()
        self.assertFalse(app.exception)
        self.assertFalse(app.toggle[0].value)
        self.assertEqual(next(s for s in app.selectbox if s.label == "Hex value format").value,
                         "Unknown / numeric values only")
        self.assertTrue(next(b for b in app.button if b.label == "▶ Start").disabled)

    def test_raw_parameter_ids_do_not_match_longer_identifiers(self):
        with self.assertRaises(ValueError):
            verify_quote("ID 7000", "ID 7000", ["700"])

    def test_absent_event_codes_are_not_fabricated_as_ok(self):
        row = Reading(timestamp=datetime.now(timezone.utc), asset_id="A", metrics={"raw": 10})
        self.assertEqual(row.event_codes, [])


if __name__ == "__main__":
    unittest.main()
