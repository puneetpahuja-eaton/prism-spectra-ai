"""Cloud adapters around the unchanged PRISM models, simulator and agent engine."""
from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
import uuid

REFERENCE = Path(__file__).resolve().parent / "prism_reference"
if str(REFERENCE) not in sys.path:
    sys.path.insert(0, str(REFERENCE))

from prism import config
from prism.agent_runtime import Blackboard
from prism.agents.actuator import ActuatorAgent
from prism.agents.decider import DeciderAgent
from observation_stats import CodeMembership, MetricStatistics
from prism.case import Case, Observation
from prism.orchestrator import Engine as ReferenceEngine
from prism.signals import SignalSource
from prism.topology import Asset, Topology

from actuator import act
from diagnostician import diagnose
from fleet_data import FleetDataset, load_bundle, load_csv
from nebius_client import is_live
from orchestrator import MODEL_FAILURES
import storage
import jobs
import workflow


def load_topology(text: str) -> Topology:
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    required = {"asset_id", "name", "type", "parent_id", "site", "rated_kw", "criticality"}
    if set(reader.fieldnames or []) != required:
        raise ValueError("PRISM asset file must contain the seven original topology fields.")
    assets = {}
    for row in reader:
        asset_id = row["asset_id"]
        if not asset_id or asset_id in assets:
            raise ValueError("Topology IDs must be nonempty and unique.")
        if row["criticality"] not in {"LOW", "MEDIUM", "HIGH", "CRITICAL", "UNKNOWN"}:
            raise ValueError(f"Invalid criticality for {asset_id}.")
        rated_kw = float(row["rated_kw"])
        if not 0 < rated_kw <= 1_000_000:
            raise ValueError("rated_kw must be positive and finite.")
        assets[asset_id] = Asset(asset_id, row["name"], row["type"], row["parent_id"] or None,
                                 row["site"], rated_kw, row["criticality"])
    if not 1 <= len(assets) <= 50:
        raise ValueError("Topology needs 1-50 assets.")
    for asset in assets.values():
        visited = {asset.asset_id}
        parent = asset.parent_id
        while parent:
            if parent not in assets or parent in visited:
                raise ValueError("Topology has an unknown parent or a cycle.")
            visited.add(parent)
            parent = assets[parent].parent_id
        if asset.parent_id:
            assets[asset.parent_id].children.append(asset.asset_id)
    return Topology(assets)


def assess_device(device_id: str | None, evidence: dict) -> dict:
    import rag

    with workflow.Workflow("diagnosis") as run:
        chunks = rag.retrieve(device_id, json.dumps(evidence), top_k=10) if device_id else []
        detail = {"evidence": evidence, "diagnosis": None, "work_order": None, "errors": [],
                  "sources": [{"name": c.get("source_name", "source"), "text": c.get("text", "")} for c in chunks]}
        if not chunks:
            detail["errors"].append("Insufficient supplied evidence. Add this device's sources; no diagnosis was invented.")
        else:
            sop = "\n\n".join(f"## SOP-{index}\nSource: {c.get('source_name', 'source')}\n{c.get('text', '')}"
                             for index, c in enumerate(chunks, 1))
            detail["diagnosis"] = workflow.run_step("diagnosis", lambda: diagnose(evidence, sop, device_only=True))
    detail["workflow"] = run.report
    return detail


class CloudBlackboard(Blackboard):
    def __init__(self, topo, metric_model, code_model, engine):
        super().__init__(topo, metric_model, code_model)
        self.engine = engine

    def record_tool(self, agent: str, tool: str, detail: str, result: str) -> None:
        super().record_tool(agent, tool, detail, result)
        event = {"timestamp": datetime.now(timezone.utc).isoformat(), "step": self.step,
                 "agent": agent, "tool": tool, "detail": detail, "result": str(result)}
        self.engine.cloud_events.append(event)
        if self.engine.on_event:
            self.engine.on_event(event)


class CloudDecider(DeciderAgent):
    def __init__(self, engine):
        super().__init__(engine.topo)
        self.engine = engine

    def decide(self, case, bb) -> None:
        # Preserve PRISM's exact deterministic confidence/reversibility policy.
        super().decide(case, bb)
        # Real uploaded field data has no measured intervention, so it stays
        # observation-only. A LABELED generated dataset recovers in its own
        # telemetry, so the engine may auto-remediate and the Verifier confirms.
        if self.engine.source == "uploaded" and not self.engine.labeled:
            case.decision = "ESCALATE"
            case.decision_reason = "Uploaded telemetry is observation-only; simulated control is disabled. " + case.decision_reason
        self.engine.enrich(case, bb)


class CloudActuator(ActuatorAgent):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine

    def act(self, case, step, fault, bb) -> str:
        if self.engine.source == "uploaded" and not self.engine.labeled and case.decision == "AUTO_REMEDIATE":
            raise RuntimeError("Uploaded measurements cannot receive simulated control.")
        result = super().act(case, step, fault, bb)
        detail = self.engine.cloud_cases.get(case.case_id)
        if detail:
            detail["simulator_only"] = self.engine.source == "synthetic"
            detail["equipment_commands_sent"] = False
        return result


class ReplaySource:
    def __init__(self, data: FleetDataset, *, demo_profile: bool = False):
        self.frames = {}
        for index, timestamp in enumerate(sorted({row.timestamp for row in data.telemetry})):
            self.frames[index] = {
                row.asset_id: {"metrics": dict(row.metrics), "codes": list(row.event_codes),
                               "primary_code": (next((code for code in row.event_codes if code in config.CODE_TO_FAULT),
                                                     row.event_codes[0] if row.event_codes else "OK")
                                                if demo_profile else next(iter(row.event_codes), ""))}
                for row in data.telemetry if row.timestamp == timestamp
            }

    def append_frame(self, frame: dict) -> int:
        """Append a new live frame; returns its index. Enables streaming replay."""
        index = len(self.frames)
        self.frames[index] = frame
        return index

    def sample_fleet(self, abs_idx, live_idx, active):
        # Each caller receives an independent snapshot; no simulator mutations.
        import copy
        return copy.deepcopy(self.frames[live_idx])


class Engine(ReferenceEngine):
    def __init__(self, *, offline: bool | None = None, dataset: FleetDataset | None = None,
                 device_id: str | None = None, demo_profile: bool = False, defer_ai: bool = False):
        self.offline = not is_live() if offline is None else offline
        self.defer_ai = defer_ai
        self.source = "uploaded" if dataset is not None else "synthetic"
        self.device_id = device_id
        self.demo_profile = dataset is None or demo_profile
        self.dataset = dataset.model_copy(deep=True) if dataset is not None else None
        self.labeled = True  # synthetic or ground-truth-generated data can score accuracy/value
        self.credit_value = False  # live monitoring credits estimated early-detection value
        self.on_event = None
        self.sop = storage.load_text("sop.md") if self.demo_profile else ""
        if self.dataset is None:
            self.topo = load_topology(storage.load_prism_text("assets.csv"))
            self.signal = SignalSource(self.topo)
            self._scenarios = json.loads(storage.load_prism_text("live_scenarios.json"))["scenarios"]
            for scenario in self._scenarios:
                if scenario["asset_id"] not in self.topo.assets or scenario["fault_type"] not in config.PLAYBOOK:
                    raise ValueError("Invalid configured PRISM scenario.")
            self.total_samples = config.LIVE_SAMPLES
        else:
            # Revalidate at the boundary, including callers other than the UI.
            self.dataset = FleetDataset.model_validate(self.dataset.model_dump())
            self.dataset.source = "uploaded"
            assets = {asset.id: Asset(asset.id, asset.name or asset.id, asset.type.upper(), asset.parent_id,
                                      asset.site or "UPLOADED", asset.rated_kw, asset.criticality.upper()) for asset in self.dataset.assets}
            for asset in assets.values():
                if asset.parent_id:
                    assets[asset.parent_id].children.append(asset.asset_id)
            self.topo = Topology(assets)
            self.signal = ReplaySource(self.dataset, demo_profile=self.demo_profile)
            # Ground-truth scenarios make a GENERATED dataset labeled (accuracy + value);
            # real uploaded field data has none and stays observation-only.
            self._scenarios = [s for s in (self.dataset.scenarios or []) if self.demo_profile
                               if s.get("asset_id") in self.topo.assets and s.get("fault_type") in config.PLAYBOOK]
            self.labeled = bool(self._scenarios)
            self.total_samples = len(self.signal.frames)
        self.metric_model = MetricStatistics()
        self.code_model = CodeMembership()
        self.train_report = {}
        self.train()
        self.reset()

    @staticmethod
    def _detection_features(rows, all_features):
        """Drop monotonic cumulative columns (e.g. energy registers) from anomaly
        detection. They only ever increase, so during replay they sit far above the
        healthy-history median and would saturate the robust-Z detector on every
        sample — making the asset look permanently faulted and blocking auto-resolve.
        They remain in the dataset for display; they are just not deviation signals."""
        if len(rows) < 4:
            return all_features
        kept = []
        for feature in all_features:
            series = [float(r.metrics.get(feature, 0.0) or 0.0) for r in rows]
            deltas = [b - a for a, b in zip(series, series[1:])]
            non_decreasing = sum(1 for d in deltas if d >= -1e-9) / len(deltas)
            growth = series[-1] - series[0]
            cumulative = non_decreasing >= 0.95 and growth > abs(series[0]) * 0.05 + 1e-6
            if not cumulative:
                kept.append(feature)
        return kept or all_features

    def build_history(self):
        if self.dataset is None:
            return super().build_history()
        history, codes = {}, {}
        for asset_id in self.topo.all_ids():
            rows = [row for row in self.dataset.history if row.asset_id == asset_id]
            features = (self._detection_features(rows, sorted(rows[0].metrics))
                        if self.demo_profile else sorted(rows[0].metrics))
            history[asset_id] = {"X": [[row.metrics[key] for key in features] for row in rows], "features": features}
            codes[asset_id] = [row.event_codes[0] if row.event_codes else ("OK" if self.demo_profile else "")
                               for row in rows]
        return history, codes

    def train(self) -> None:
        super().train()
        self.train_report.update(
            metric_algorithms=["Deterministic median/MAD deviation (not a learned model)"],
            code_algorithm="Baseline code membership; no learned transition model",
            code_transitions_learned=0)
        if self.dataset is not None:
            counts = [sum(row.asset_id == asset.id for row in self.dataset.history) for asset in self.dataset.assets]
            timestamps = sorted({row.timestamp for row in self.dataset.history})
            interval = (timestamps[1] - timestamps[0]).total_seconds() / 60 if len(timestamps) > 1 else 0
            self.train_report.update(history_samples=min(counts), history_days=round(
                (timestamps[-1] - timestamps[0]).total_seconds() / 86400, 3), sample_interval_min=interval)
        if not self.demo_profile:
            self.train_report["code_algorithm"] = "Unseen codes relative to user-declared baseline; no fault semantics"
            self.train_report["code_vocabulary"] = len({c for row in self.dataset.history for c in row.event_codes})

    def reset(self) -> None:
        for future in getattr(self, "_pending_ai", {}).values():
            future.cancel()
        self._pending_ai = {}
        super().reset()
        self.run_id = str(uuid.uuid4())
        self.cloud_cases = {}
        self.cloud_events = []
        self.audit_revisions = {}
        self.audit_error = None
        self.bb = CloudBlackboard(self.topo, self.metric_model, self.code_model, self)
        self.decider = CloudDecider(self)
        self.actuator = CloudActuator(self)
        self._known_codes = ({aid: {code for row in self.dataset.history if row.asset_id == aid for code in row.event_codes}
                              for aid in self.topo.all_ids()} if self.dataset is not None else {})

    @property
    def at_end(self) -> bool:
        return self.current_idx + 1 >= self.total_samples

    def timestamp(self, live_idx):
        if self.dataset is None:
            return super().timestamp(live_idx)
        if getattr(self, "_live_timestamps", None):
            return self._live_timestamps[live_idx]
        timestamps = sorted({row.timestamp for row in self.dataset.telemetry})
        return timestamps[live_idx]

    def append_live(self, frame: dict, timestamp) -> int:
        """Append a streamed live frame and extend the replay window by one tick.

        `frame` is {asset_id: {metrics, codes, primary_code}}. Used by live
        monitoring so the SAME PRISM engine analyzes a growing stream.
        """
        if not hasattr(self, "_live_timestamps"):
            # Seed from the initial telemetry timestamps so indices stay aligned.
            self._live_timestamps = sorted({row.timestamp for row in self.dataset.telemetry})
        index = self.signal.append_frame(frame)
        self._live_timestamps.append(timestamp)
        self.total_samples = len(self.signal.frames)
        return index

    def enrich(self, case, bb) -> None:
        asset = self.topo.get(case.root_asset_id)
        metrics = bb.sample[case.root_asset_id]["metrics"]
        state = self.metric_model.assets[case.root_asset_id]
        evidence = {
            "asset_id": asset.asset_id, "asset_type": asset.type, "asset_criticality": asset.criticality.lower(),
            "event_codes": bb.sample[asset.asset_id]["codes"],
            "signals": [{"metric": key, "latest": metrics[key], "unit": "",
                         "robust_z": round(float((metrics[key] - state["median"][index]) / state["scale"][index]), 2),
                         "anomalous": abs(float((metrics[key] - state["median"][index]) / state["scale"][index])) >= config.ROBUST_Z_THRESHOLD}
                        for index, key in enumerate(state["features"])],
            "has_anomaly": True,
            "fleet_context": {"category": {"VOLTAGE_SAG": "electrical_supply", "BREAKER_TRIP": "electrical_supply",
                                          "THERMAL_RUNAWAY": "thermal", "OVERLOAD": "overload",
                                          "HARMONIC_DISTORTION": "harmonics", "BATTERY_DEGRADATION": "battery"}[case.fault_type],
                              "fault_type": case.fault_type, "symptom_asset_ids": case.symptom_asset_ids,
                              "ancestors": self.topo.ancestors(asset.asset_id), "source": self.source,
                              "heuristic_confidence": case.confidence, "deterministic_decision": case.decision},
        }
        evidence["anomalous_metrics"] = [signal["metric"] for signal in evidence["signals"] if signal["anomalous"]]
        detail = {"evidence": evidence, "errors": [], "work_order": None}
        self.cloud_cases[case.case_id] = detail
        bb.record_tool("Diagnostician", "nebius_grounded_diagnosis", asset.asset_id,
                       "Started explicit offline explanation" if self.offline else "Started configured model request; waiting for validated output")
        try:
            diagnosis = diagnose(evidence, self.sop, offline=self.offline)
            detail["diagnosis"] = diagnosis
            bb.record_tool("Diagnostician", "validate_sop_citations", asset.asset_id,
                           f"{diagnosis['_source']} | {diagnosis['_citation_check']}")
            bb.log("Diagnostician", case.case_id, f"{diagnosis['_source']}: {diagnosis['root_cause']}")
            # Advisory cloud reasoning does not alter the canonical simulator's
            # classification/confidence. Missing grounding is not accepted as success.
            if diagnosis["insufficient_evidence"]:
                case.decision = "ESCALATE"
                case.decision_reason = "Source assessment reports insufficient evidence; engineer review required."
            if case.decision == "ESCALATE":
                bb.record_tool("Actuator", "nebius_work_order", case.case_id, "Preparing review-only work-order draft")
                output = act(evidence, diagnosis, {"action": "escalate"}, self.sop, offline=self.offline)
                detail["work_order"] = output["work_order"]
                bb.record_tool("Actuator", "nebius_work_order", case.case_id, output["work_order"]["_source"])
        except MODEL_FAILURES as error:
            detail["errors"].append(f"{type(error).__name__}: {error}")
            case.decision = "ESCALATE"
            case.decision_reason = "Cloud analysis failed; no offline result substituted. Engineer review required."
            bb.record_tool("Diagnostician", "cloud_analysis_failed", asset.asset_id, str(error))
            bb.log("Diagnostician", case.case_id, case.decision_reason, "escalate")

    def tick(self, on_event=None) -> None:
        self.drain_ai()
        if self.at_end:
            return
        self.on_event = on_event
        try:
            if not self.demo_profile:
                self._tick_device()
                return
            super().tick()
            if self.source == "uploaded" and not self.labeled and not self.credit_value:
                # Real uploaded field data has no measured intervention/counterfactual.
                for case in self.cases:
                    case.downtime_avoided_min = 0
                    case.cost_avoided_usd = 0
        finally:
            self.on_event = None

    def _tick_device(self) -> None:
        """Observation-only workflow; no reference-device classifications or controls."""
        self.current_idx += 1
        sample = self.signal.sample_fleet(0, self.current_idx, [])
        self.history[self.current_idx] = sample
        self.bb.begin_tick(self.current_idx, sample)
        for aid, reading in sample.items():
            score = self.metric_model.score(aid, reading["metrics"])
            self.score_hist[aid].append(score["score"])
            self.surprise_hist[aid].append(0)
            self.bb.record_tool("Monitor", "score_metric_model", aid, json.dumps(score))
            new_codes = set(reading["codes"]) - self._known_codes[aid]
            for code in reading["codes"]:
                self.events.append({"idx": self.current_idx, "asset": aid, "code": code, "codes": reading["codes"]})
            metric = score["top_metric"] if score["is_anomaly"] else "event_codes"
            anomalous = score["is_anomaly"] or bool(new_codes)
            for (asset_id, observed_metric), existing in list(self.case_index.items()):
                still_active = (bool(new_codes) if observed_metric == "event_codes" else
                                score["is_anomaly"] and abs(score["per_metric_z"].get(observed_metric, 0)) >= config.ROBUST_Z_THRESHOLD)
                if asset_id == aid and not still_active and existing.status == "ESCALATED":
                    existing.status = "RESOLVED"
                    existing.resolved_step = self.current_idx
                    self.bb.record_tool("Verifier", "observe_recovery", existing.case_id,
                                        "Returned to supplied baseline; no intervention or repair inferred")
            if not anomalous:
                continue
            key = (aid, metric)
            if key in self.case_index and self.case_index[key].status != "RESOLVED":
                continue
            observation = Observation(aid, metric, reading["metrics"].get(metric, 0), score["score"],
                                      "unclassified", model_score=score["score"], zmax=score["zmax"],
                                      codes=list(reading["codes"]), source="metric" if score["is_anomaly"] else "code")
            case = Case(self._next_case_id(), self.current_idx, aid, "UNCLASSIFIED", 0,
                        observations=[observation], decision="ESCALATE",
                        decision_reason="Observation only; source-grounded interpretation requires engineer review.",
                        action="Review device evidence and observed deviation; no equipment command.",
                        reversibility="NONE", status="ESCALATED")
            self.cases.append(case)
            self.case_index[key] = case
            self._enrich_device(case, reading, score)
            self.bb.record_tool("Decider", "require_engineer_review", case.case_id, case.decision_reason)
            self.bb.record_tool("Actuator", "review_only_handoff", case.case_id,
                                "Draft retained locally; no command or ticket sent")

    def _enrich_device(self, case, reading: dict, score: dict) -> None:
        evidence = {"asset_id": case.root_asset_id, "metrics": reading["metrics"],
                    "event_codes": reading["codes"], "statistical_score": score,
                    "source": "selected-device telemetry", "equipment_commands_sent": False}
        detail = {"evidence": evidence, "diagnosis": None, "work_order": None, "errors": []}
        self.cloud_cases[case.case_id] = detail
        self.bb.record_tool("Diagnostician", "source_only_interpretation", case.root_asset_id,
                            "No bundled SOP, manufacturer knowledge or demo playbook")
        if self.offline:
            detail["errors"].append("Model interpretation not enabled; device observations remain unclassified.")
            return
        try:
            if self.defer_ai:
                if len(self._pending_ai) >= 4:
                    raise RuntimeError("This device already has four pending interpretations. Observations continue; review this case manually.")
                self._pending_ai[case.case_id] = jobs.submit(lambda: assess_device(self.device_id, evidence))
                detail["ai_status"] = "pending"
                return
            self._accept_assessment(case, assess_device(self.device_id, evidence))
        except MODEL_FAILURES as error:
            detail["errors"].append(f"{type(error).__name__}: {error}")
            self.bb.record_tool("Diagnostician", "source_analysis_failed", case.case_id, str(error))

    def _accept_assessment(self, case, detail: dict) -> None:
        self.cloud_cases[case.case_id].update(detail)
        self.cloud_cases[case.case_id]["ai_status"] = "complete" if detail.get("diagnosis") else "unavailable"
        if detail.get("diagnosis"):
            case.confidence = detail["diagnosis"]["confidence"]
            self.bb.record_tool("Diagnostician", "validate_source_citations", case.case_id,
                                detail["diagnosis"]["_citation_check"])

    def drain_ai(self) -> None:
        for case_id, future in list(getattr(self, "_pending_ai", {}).items()):
            if not future.done():
                continue
            self._pending_ai.pop(case_id)
            case = next((c for c in self.cases if c.case_id == case_id), None)
            if case is None:
                continue
            try:
                self._accept_assessment(case, future.result())
            except MODEL_FAILURES as error:
                self.cloud_cases[case_id]["ai_status"] = "failed"
                self.cloud_cases[case_id]["errors"].append(f"AI interpretation failed: {error}")
                self.bb.record_tool("Diagnostician", "source_analysis_failed", case_id, type(error).__name__)

    def summary(self) -> dict:
        result = super().summary()
        if self.source == "uploaded" and not self.labeled:
            result["diagnosis_accuracy"] = None
        elif not any(case.ground_truth_root for case in self.cases):
            result["diagnosis_accuracy"] = None
        return result

    def audit(self) -> dict:
        return {"run_id": self.run_id, "source": self.source, "offline": self.offline,
                "summary": self.summary(), "training": self.train_report,
                "cases": [asdict(case) for case in self.cases], "cloud_cases": self.cloud_cases,
                "log": [asdict(entry) for entry in self.log], "tools": [asdict(call) for call in self.tool_calls],
                "events": self.cloud_events, "equipment_commands_sent": False, "realized_savings_usd": 0,
                "demo_profile": self.demo_profile, "device_id": self.device_id}


def replacement(text: str, filename: str) -> FleetDataset:
    return load_csv(text) if filename.lower().endswith(".csv") else load_bundle(text)
