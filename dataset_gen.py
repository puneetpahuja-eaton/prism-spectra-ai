"""Generate an Analysis-ready dataset from a natural-language request via the configured model.

Flow:
  1. generate_spec(): the model returns a compact JSON spec (assets, metrics with
     normal ranges, anomalies) grounded in the device's docs / a referenced file.
     Missing facts raise ClarificationNeeded so the conversation can resume.
  2. build_dataset(): deterministic synthesis expands the spec into a full
     FleetDataset (healthy history + replay with injected anomalies) that the
     Analysis engine loads, updating its asset/metric selectors.

Generating a compact spec (not thousands of rows) avoids model output limits and
keeps the data reproducible.
"""
from __future__ import annotations

import hashlib
import math
from datetime import datetime, timedelta, timezone
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from device_grounding import ClarificationNeeded, SOURCE_ONLY, generate_grounded, verify_passages, verify_quote
from fleet_data import FleetAsset, FleetDataset, Reading
from gemini_client import generate_json, is_live

_ANOMALY_CODE = {"spike": "OVERLOAD_ALARM", "sag": "VOLT_SAG", "drift": "THD_HIGH", "dropout": "LOAD_LOST"}
# Map an anomaly kind to the PRISM ground-truth fault type (for accuracy scoring).
_ANOMALY_FAULT = {"spike": "OVERLOAD", "sag": "VOLTAGE_SAG", "drift": "HARMONIC_DISTORTION",
                  "dropout": "BREAKER_TRIP"}


class MetricSpec(BaseModel):
    name: str = Field(description="Exact metric name from supplied sources or explicit user guidance")
    unit: str = Field(default="", description="Source-defined unit; empty only for a documented dimensionless metric")
    normal_min: float = Field(description="Typical minimum healthy value")
    normal_max: float = Field(description="Typical maximum healthy value")
    evidence: str = Field(default="", description="Copy exact words from a user's reply or supplied source; user guidance itself is evidence, without invented prefixes")
    evidence_passages: list[str] = Field(default_factory=list, description="Additional exact passages when name, unit and range occur separately")

    @model_validator(mode="after")
    def valid_range(self):
        if not math.isfinite(self.normal_min) or not math.isfinite(self.normal_max) or self.normal_min >= self.normal_max:
            raise ValueError("Supply finite metric ranges with normal_min < normal_max.")
        return self


class AssetSpec(BaseModel):
    id: str = Field(description="Asset/device identifier")
    type: str = Field(default="GENERIC")
    criticality: str = Field(default="unknown")
    metrics: list[MetricSpec] = Field(min_length=1, max_length=120)

    @model_validator(mode="after")
    def unique_metrics(self):
        if len({m.name for m in self.metrics}) != len(self.metrics):
            raise ValueError("Metric names must be unique within an asset.")
        return self


class AnomalySpec(BaseModel):
    asset_id: str
    metric: str
    start_fraction: float = Field(default=0.5, ge=0, le=1, description="Where in the replay it begins (0..1)")
    kind: Literal["spike", "sag", "drift", "dropout"] = "spike"
    severity: float = Field(default=0.6, ge=0, le=1)
    recovers: bool = Field(default=True, description="If true, the fault clears after its window so the "
                                                     "engine auto-resolves it; if false, it persists and is escalated.")
    duration_fraction: float = Field(default=0.25, ge=0.05, le=1.0,
                                     description="How much of the replay the fault lasts before it clears (when recovers).")
    description: str = ""


class DatasetSpec(BaseModel):
    profile: Literal["independent", "power-demo"] = "independent"
    assets: list[AssetSpec] = Field(min_length=1, max_length=10, json_schema_extra={"uniqueItems": True})
    sample_interval_min: int = Field(default=5, ge=1, le=1440)
    duration_days: int = Field(default=3, ge=1, le=14)
    history_days: int = Field(default=1, ge=1, le=14)
    anomalies: list[AnomalySpec] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def unique_assets(self):
        if len({a.id for a in self.assets}) != len(self.assets):
            raise ValueError("Asset IDs must be unique.")
        return self


class DatasetDraft(BaseModel):
    spec: DatasetSpec | None = None
    missing_details: list[str] = Field(default_factory=list, description="Only missing device facts; do not ask for document quotes when explicit user guidance supplies the values")
    answer: str = Field(default="", description="Brief useful response in plain language, without schema or validation jargon")


_SCHEMA_HINT = (
    "Return a DatasetDraft with spec set to the proposed DatasetSpec, or spec=null and "
    "missing_details containing specific questions. Use profile=independent. "
    "Explicit user guidance/answers are authoritative supplied sources for this task, just like docs/code. "
    "No separate document is required to reconfirm facts the user supplied. Copy the user's words as evidence. "
    "Each metric requires exact supplied evidence for its original name, unit and numeric range. "
    "Use evidence_passages when these facts are in different documents or user replies; they need not "
    "occur in one sentence. Copy the passages yourself; never ask the user to format citations. "
    "Empty units require explicit dimensionless guidance. "
    "Use asset type GENERIC and criticality unknown unless explicitly documented. "
    "If necessary details are missing, answer with what is already known and ask at most 3 concise, "
    "plain-language questions about the next missing facts. Never use words like DatasetSpec, normal_min, "
    "normal_max, spec or profile in the user-facing answer or questions. Do not list schema field names, profile names "
    "or demands for exact passages. Do not re-ask settings already selected in the UI or answered in conversation. "
    "If ranges are unavailable, ask the user to supply illustrative synthetic ranges or attach a source. "
    "Do not infer physical relationships or translate unknown event codes into fault meanings."
    " Do not repeat assets or metrics; match the user's requested asset count exactly."
)


def generate_spec(device_context: str, request: str, reference: str = "", answers: str = "") -> dict:
    if not is_live():
        raise RuntimeError("A configured live AI provider is required to generate a dataset spec.")
    prompt = (
        f"{SOURCE_ONLY}\n"
        "Design a COMPACT dataset specification (not the data itself) for a synthetic dataset of the "
        "monitored device. Select only supplied metrics, ranges and units, grounded "
        "in the context and referenced parameter list. Include only synthetic anomalies requested by "
        "the user. Ask for missing metrics, units, ranges, assets or sampling details. "
        "Explicit user-selected synthetic sampling governs this dataset only; it does not change "
        "the actual device's live polling configuration.\n\n"
        f"{_SCHEMA_HINT}\n\n"
        f"Device context:\n{device_context[:60000]}\n\n"
        f"Referenced parameter file (optional, authoritative for names/IDs/units):\n{reference[:120000]}\n\n"
        f"User request: {request}\n\n"
        f"User answers to clarifying questions (if any): {answers}"
    )
    supplied = "\n".join((device_context, reference, request, answers))
    return generate_grounded(generate_json, prompt, DatasetDraft, lambda raw: _validate_draft(raw, supplied))


def _validate_draft(raw: dict, supplied: str) -> dict:
    draft = DatasetDraft.model_validate(raw)
    if draft.missing_details or draft.spec is None:
        raise ClarificationNeeded(draft.missing_details or [
            "Which metrics should I include, and what units and illustrative numeric ranges should I use?"
        ], answer=draft.answer)
    spec = draft.spec
    if spec.profile != "independent":
        raise ValueError("The power-demo profile is reserved for explicitly selected demonstration fixtures.")
    questions = []
    for asset in spec.assets:
        for metric in asset.metrics:
            try:
                verify_passages(metric.evidence, metric.evidence_passages, supplied,
                                [metric.name, metric.unit, metric.normal_min, metric.normal_max], unit=metric.unit)
            except ValueError:
                questions.append(f"I couldn't verify the supplied evidence for **{metric.name}**. "
                                 "What unit and illustrative minimum/maximum should I use, or which device source documents them?")
        if asset.type != "GENERIC":
            try:
                verify_quote(asset.type, supplied, [])
            except ValueError:
                questions.append(f"What documented device type should I use for **{asset.id}**?")
        if asset.criticality != "unknown":
            try:
                verify_quote(asset.criticality, supplied, [])
            except ValueError:
                questions.append(f"What criticality is confirmed for **{asset.id}**, or should I leave it unknown?")
    if questions:
        raise ClarificationNeeded(questions, answer=draft.answer, repairable=True)
    return spec.model_dump()


# --------------------------------------------------------------------------- #
# Deterministic, physically-consistent synthesis
# --------------------------------------------------------------------------- #
def _rand01(*keys) -> float:
    digest = hashlib.md5("|".join(str(k) for k in keys).encode()).hexdigest()[:8]
    return int(digest, 16) / 0xFFFFFFFF


def _diurnal(abs_idx: int, interval_min: int, seed: str) -> float:
    minutes = abs_idx * interval_min
    tod = (minutes % 1440) / 1440.0
    phase = _rand01(seed, "phase") * 0.12
    return math.sin(2 * math.pi * (tod - 0.25 - phase))


def _phase_of(name: str) -> str | None:
    low = name.lower()
    for key, tag in (("_a_", "a"), ("_b_", "b"), ("_c_", "c"), ("_a", "a"), ("_b", "b"), ("_c", "c"),
                     ("phase_a", "a"), ("phase_b", "b"), ("phase_c", "c"), ("_1", "a"), ("_2", "b"), ("_3", "c")):
        if key in low or low.endswith(key):
            return tag
    return None


def _role_of(name: str) -> str:
    low = name.lower()
    has = lambda *s: any(x in low for x in s)  # noqa: E731
    if has("power_factor", "powerfactor") or low in {"pf", "pf_total"} or low.endswith("_pf"):
        return "power_factor"
    if has("kwh") or (has("energy") and has("active")) or (has("wh") and has("active")):
        return "active_energy"
    if has("kvarh") or (has("energy") and has("reactive")):
        return "reactive_energy"
    if has("kvah") or (has("energy") and has("apparent")) or (has("energy") and has("va")):
        return "apparent_energy"
    if has("energy"):
        return "active_energy"
    if has("reactive") or (has("var") and not has("variance")):
        return "reactive_power"
    if has("apparent") or (("kva" in low or low.endswith("_va") or "_va_" in low) and "kvar" not in low):
        return "apparent_power"
    if has("active_power") or (has("power") and has("active")) or has("kw") or (has("power") and not has("factor")):
        return "active_power"
    if has("frequency", "freq") or low.endswith("_hz") or "_hz_" in low:
        return "frequency"
    if has("thd", "distortion"):
        return "thd"
    if has("temperature", "temp"):
        return "temperature"
    if has("voltage", "volt") or low.endswith("_v") or "_v_" in low:
        return "voltage"
    if has("current", "amp") or low.endswith("_a") or "_a_" in low or has("_i_"):
        return "current"
    if has("demand"):
        return "active_power"
    return "generic"


def _is_kilo(name: str) -> bool:
    low = name.lower()
    return any(u in low for u in ("kwh", "kvarh", "kvah", "kw", "kvar", "kva", "_k"))


def _generic_value(metric: MetricSpec, seed: str, abs_idx: int, interval_min: int) -> float:
    span = metric.normal_max - metric.normal_min
    midpoint = (metric.normal_max + metric.normal_min) / 2.0
    swing = 0.25 * span * _diurnal(abs_idx, interval_min, seed + metric.name)
    noise = (span or 1.0) * 0.03 * (_rand01(seed, metric.name, abs_idx) * 2 - 1)
    return midpoint + swing + noise


def _range_for(metrics: list[MetricSpec], role: str, default_min: float, default_max: float) -> tuple[float, float]:
    for metric in metrics:
        if _role_of(metric.name) == role:
            return metric.normal_min, metric.normal_max
    return default_min, default_max


def ensure_anomaly_mix(spec: DatasetSpec) -> list[AnomalySpec]:
    """Guarantee a size-scaled mix of auto-resolving and escalating faults.

    Model-authored specs are inconsistent (often zero or only-recovering faults),
    so for a predictable Analysis demo we synthesize a balanced anomaly set from the
    spec's own assets/metrics:
      * ~2/3 transient faults (recovers=True)  -> AUTO-RESOLVED
      * ~1/3 sustained faults (recovers=False)  -> ESCALATED
    The count scales with the replay length and asset count. Auto faults run early
    (and clear), escalations run late (and persist), so they never overlap.
    Distinct fault types per asset (sag / spike / dropout) keep the cases separate.
    """
    slots: list[list[tuple[str, str, str]]] = []
    for asset in spec.assets:
        if spec.profile == "independent":
            slots.append([(asset.id, metric.name, kind)
                          for metric in asset.metrics for kind in ("spike", "sag", "dropout")])
            continue
        volt = next((m.name for m in asset.metrics if _role_of(m.name) == "voltage"), None)
        curr = next((m.name for m in asset.metrics
                     if _role_of(m.name) == "current" and "neutral" not in m.name.lower()), None)
        asset_slots: list[tuple[str, str, str]] = []
        if volt:
            asset_slots.append((asset.id, volt, "sag"))       # VOLTAGE_SAG
        if curr:
            asset_slots.append((asset.id, curr, "spike"))     # OVERLOAD
        if curr:
            asset_slots.append((asset.id, curr, "dropout"))   # BREAKER_TRIP
        slots.append(asset_slots)
    # Interleave across assets so different assets are used before reusing one.
    ordered: list[tuple[str, str, str]] = []
    for depth in range(max((len(s) for s in slots), default=0)):
        for asset_slots in slots:
            if depth < len(asset_slots):
                ordered.append(asset_slots[depth])
    if not ordered:
        return list(spec.anomalies)  # no voltage/current metrics to perturb

    per_day = max(1, (24 * 60) // spec.sample_interval_min)
    replay_n = min(max(2, spec.duration_days * per_day), max(2, 10_000 // max(1, len(spec.assets))))
    if spec.profile == "independent" and replay_n < 8:
        raise ValueError("At least 8 replay readings per asset are needed for distinct recovery and escalation events. "
                         "Increase the duration or reduce the sampling interval.")
    n_total = max(min(3, len(ordered)), min(round(replay_n / 24), len(ordered)))
    n_escalate = max(1, n_total // 3) if n_total >= 2 else 0
    n_auto = n_total - n_escalate
    chosen = ordered[:n_total]

    anomalies: list[AnomalySpec] = []
    for i in range(n_auto):
        asset_id, metric, kind = chosen[i]
        frac = round(0.25 + 0.40 * (i / max(1, n_auto - 1)), 3) if n_auto > 1 else 0.4
        anomalies.append(AnomalySpec(asset_id=asset_id, metric=metric, kind=kind,
                                     start_fraction=min(0.65, frac), severity=0.8, recovers=True,
                                     duration_fraction=0.05,
                                     description=f"Transient {kind} on {metric} — auto-resolved."))
    for j in range(n_escalate):
        asset_id, metric, kind = chosen[n_auto + j]
        frac = round(0.80 + 0.08 * (j / max(1, n_escalate - 1)), 3) if n_escalate > 1 else 0.82
        anomalies.append(AnomalySpec(asset_id=asset_id, metric=metric, kind=kind,
                                     start_fraction=min(0.9, frac), severity=0.9, recovers=False,
                                     description=f"Sustained {kind} on {metric} — escalated to an engineer."))
    return anomalies


def _build_independent(spec: DatasetSpec) -> FleetDataset:
    """Illustrative independent signals; no implicit equipment physics or fault meanings."""
    interval = spec.sample_interval_min
    per_day = max(1, 1440 // interval)
    history_n = min(max(20, spec.history_days * per_day), 20_000 // len(spec.assets))
    replay_n = min(max(2, spec.duration_days * per_day), 10_000 // len(spec.assets))
    start = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(minutes=(history_n + replay_n) * interval)
    assets = [FleetAsset(id=a.id, name=a.id, type=a.type, criticality=a.criticality, site="SYNTHETIC")
              for a in spec.assets]
    metrics_by_asset = {a.id: {m.name for m in a.metrics} for a in spec.assets}
    for anomaly in spec.anomalies:
        if anomaly.metric not in metrics_by_asset.get(anomaly.asset_id, set()):
            raise ValueError("Synthetic anomaly references an unknown asset or metric.")
        if anomaly.kind not in _ANOMALY_CODE:
            raise ValueError("Synthetic anomaly kind must be spike, sag, drift or dropout.")
    history, replay = [], []
    for asset in spec.assets:
        for index in range(history_n + replay_n):
            values = {}
            codes = []
            for metric in asset.metrics:
                span = metric.normal_max - metric.normal_min
                value = (metric.normal_min + metric.normal_max) / 2 + span * 0.35 * (
                    2 * _rand01(asset.id, metric.name, index) - 1)
                for anomaly in spec.anomalies:
                    if (index < history_n or anomaly.asset_id != asset.id or anomaly.metric != metric.name):
                        continue
                    tick = index - history_n
                    first = int(anomaly.start_fraction * (replay_n - 1))
                    last = first + max(1, int(anomaly.duration_fraction * replay_n))
                    if tick < first or (anomaly.recovers and tick >= last):
                        continue
                    value += span * (1 + anomaly.severity) * (-1 if anomaly.kind in ("sag", "dropout") else 1)
                    codes.append(f"SYNTHETIC_{anomaly.kind.upper()}")
                values[metric.name] = round(value, 6)
            row = Reading(timestamp=start + timedelta(minutes=index * interval), asset_id=asset.id,
                          metrics=values, event_codes=sorted(set(codes)) or ["OK"])
            (history if index < history_n else replay).append(row)
    return FleetDataset(source="uploaded", assets=assets, history=history, telemetry=replay)


def build_dataset(spec: DatasetSpec) -> FleetDataset:
    if spec.profile == "independent":
        return _build_independent(spec)
    return _build_power_demo(spec)


def _build_power_demo(spec: DatasetSpec) -> FleetDataset:
    interval = spec.sample_interval_min
    per_day = max(1, (24 * 60) // interval)
    n_assets = len(spec.assets)
    history_n = min(max(20, spec.history_days * per_day), max(20, 20_000 // n_assets))
    replay_n = min(max(2, spec.duration_days * per_day), max(2, 10_000 // n_assets))
    dt_hours = interval / 60.0

    assets = [FleetAsset(id=a.id, name=a.id, type=a.type.upper(), criticality=a.criticality.lower(),
                         site="SYNTHETIC", rated_kw=100.0) for a in spec.assets]
    start = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(
        minutes=(history_n + replay_n) * interval)

    anomalies_by_asset: dict[str, list[AnomalySpec]] = {}
    for anomaly in spec.anomalies:
        anomalies_by_asset.setdefault(anomaly.asset_id, []).append(anomaly)

    history: list[Reading] = []
    telemetry: list[Reading] = []
    for asset_spec in spec.assets:
        metrics = asset_spec.metrics
        roles = {m.name: _role_of(m.name) for m in metrics}
        v_nom_min, v_nom_max = _range_for(metrics, "voltage", 228.0, 232.0)
        i_min, i_max = _range_for(metrics, "current", 10.0, 90.0)
        pf_min, pf_max = _range_for(metrics, "power_factor", 0.88, 0.96)
        v_nom = (v_nom_min + v_nom_max) / 2.0
        # Monotonic energy accumulators (Wh), started at a realistic baseline.
        energy = {"active_energy": _rand01(asset_spec.id, "e0") * 5e5,
                  "reactive_energy": _rand01(asset_spec.id, "e1") * 2e5,
                  "apparent_energy": _rand01(asset_spec.id, "e2") * 5e5}

        def emit(abs_idx: int, phase: str) -> Reading:
            active = anomalies_by_asset.get(asset_spec.id, [])
            live_idx = abs_idx - history_n
            progress = (live_idx / max(1, replay_n - 1)) if phase == "replay" else -1.0

            def anomaly_factor(target_role: str, target_phase: str | None):
                """Return (v_mult, i_mult, code) adjustments for active anomalies."""
                v_mult, i_mult, codes = 1.0, 1.0, []
                for a in active:
                    if phase != "replay":
                        continue
                    start_idx = a.start_fraction * max(1, replay_n - 1)
                    if live_idx < start_idx:
                        continue
                    # Transient faults are a short burst (a few samples) then CLEAR,
                    # so the engine can Act and the Verifier confirms recovery within
                    # its grace window -> AUTO-RESOLVED. A non-recovering fault persists
                    # to the end -> escalated to a human.
                    if getattr(a, "recovers", True):
                        hold = max(1, round(getattr(a, "duration_fraction", 0.25) * replay_n))
                        hold = min(hold, 1)  # clear within the Verifier grace window so it auto-resolves
                        if live_idx > start_idx + hold:
                            continue  # recovered: asset is healthy again
                        strength = a.severity
                    else:
                        local = (progress - a.start_fraction) / max(1e-6, 1 - a.start_fraction)
                        strength = a.severity * (0.4 + 0.6 * local)
                    a_role = _role_of(a.metric)
                    a_phase = _phase_of(a.metric)
                    if a_phase and target_phase and a_phase != target_phase:
                        continue
                    if a_role in ("voltage",) and target_role == "voltage":
                        v_mult *= (1 - 0.18 * strength) if a.kind == "sag" else (1 + 0.12 * strength)
                        codes.append("VOLT_SAG" if a.kind == "sag" else "VOLT_SWELL")
                    if a_role in ("current", "active_power", "apparent_power") and target_role == "current":
                        if a.kind == "spike":
                            i_mult *= (1 + 1.2 * strength)
                            codes.append("OVERLOAD_ALARM")
                        elif a.kind == "dropout":
                            i_mult *= max(0.0, 1 - strength)
                            codes.append("LOAD_LOST")
                    if a_role == "reactive_power" and target_role == "current" and a.kind != "dropout":
                        codes.append("THD_HIGH")
                return v_mult, i_mult, codes

            # Base per-phase voltages and currents (with anomalies applied at the source).
            volts, amps, codes = {}, {}, []
            for ph in ("a", "b", "c"):
                vbase = v_nom + 1.5 * _diurnal(abs_idx, interval, asset_spec.id + "v" + ph) \
                    + (v_nom * 0.006) * (_rand01(asset_spec.id, "v", ph, abs_idx) * 2 - 1)
                vmult, _, vcodes = anomaly_factor("voltage", ph)
                volts[ph] = vbase * vmult
                load = 0.5 + 0.35 * (_diurnal(abs_idx, interval, asset_spec.id + "load") + 1) / 2.0
                ibase = (i_min + (i_max - i_min) * load) \
                    + (i_max - i_min) * 0.03 * (_rand01(asset_spec.id, "i", ph, abs_idx) * 2 - 1)
                _, imult, icodes = anomaly_factor("current", ph)
                amps[ph] = max(0.0, ibase * imult)
                codes.extend(vcodes + icodes)

            pf = min(0.999, max(0.3, (pf_min + pf_max) / 2.0
                                + 0.02 * _diurnal(abs_idx, interval, asset_spec.id + "pf")))
            # Derive power per phase: S=V*I, P=S*pf, Q=sqrt(S^2-P^2). Totals are consistent.
            s_ph = {ph: volts[ph] * amps[ph] for ph in ("a", "b", "c")}
            s_tot = sum(s_ph.values())
            p_tot = s_tot * pf
            q_tot = math.sqrt(max(0.0, s_tot ** 2 - p_tot ** 2))
            # Energy registers accumulate continuously (history AND replay) so they
            # are genuinely monotonic — the engine then excludes them from anomaly
            # detection instead of mistaking their growth for a fault every sample.
            energy["active_energy"] += p_tot * dt_hours
            energy["reactive_energy"] += q_tot * dt_hours
            energy["apparent_energy"] += s_tot * dt_hours

            out: dict[str, float] = {}
            for m in metrics:
                role = roles[m.name]
                mph = _phase_of(m.name)
                kilo = _is_kilo(m.name)
                if role == "voltage":
                    val = volts.get(mph, sum(volts.values()) / 3)
                elif role == "current":
                    if "neutral" in m.name.lower():
                        # Neutral current ~ small residual imbalance of the three phases.
                        val = abs(amps["a"] - amps["b"]) * 0.3 + abs(amps["b"] - amps["c"]) * 0.3 \
                            + (sum(amps.values()) / 3) * 0.05
                    else:
                        val = amps.get(mph, sum(amps.values()) / 3)
                elif role == "power_factor":
                    val = pf
                elif role == "active_power":
                    val = (s_ph.get(mph, s_tot / 3) * pf if mph else p_tot) / (1000 if kilo else 1)
                elif role == "reactive_power":
                    base = s_ph.get(mph, s_tot / 3) if mph else s_tot
                    val = (base * math.sqrt(max(0.0, 1 - pf ** 2)) if mph else q_tot) / (1000 if kilo else 1)
                elif role == "apparent_power":
                    val = (s_ph.get(mph, s_tot / 3) if mph else s_tot) / (1000 if kilo else 1)
                elif role == "active_energy":
                    val = energy["active_energy"] / (1000 if kilo else 1)
                elif role == "reactive_energy":
                    val = energy["reactive_energy"] / (1000 if kilo else 1)
                elif role == "apparent_energy":
                    val = energy["apparent_energy"] / (1000 if kilo else 1)
                elif role == "frequency":
                    val = 50.0 + 0.02 * (_rand01(asset_spec.id, "f", abs_idx) * 2 - 1)
                elif role == "thd":
                    thd = 2.5 + 0.5 * _diurnal(abs_idx, interval, asset_spec.id + "thd")
                    val = thd + (6.0 if "THD_HIGH" in codes else 0.0)
                else:
                    val = _generic_value(m, asset_spec.id, abs_idx, interval)
                out[m.name] = round(val, 4)
            return Reading(timestamp=start + timedelta(minutes=abs_idx * interval),
                           asset_id=asset_spec.id, event_codes=sorted(set(codes)) or ["OK"], metrics=out)

        for index in range(history_n):
            history.append(emit(index, "history"))
        for index in range(replay_n):
            telemetry.append(emit(history_n + index, "replay"))

    # Ground-truth scenarios from the injected anomalies (enables accuracy + value).
    scenarios: list[dict] = []
    for i, anomaly in enumerate(spec.anomalies):
        fault_type = _ANOMALY_FAULT.get(anomaly.kind, "VOLTAGE_SAG")
        scenarios.append({
            "scenario_id": f"S{i + 1}", "asset_id": anomaly.asset_id, "fault_type": fault_type,
            "start_sample": int(anomaly.start_fraction * replay_n),
            "description": anomaly.description or f"Injected {anomaly.kind} on {anomaly.metric}",
            "ground_truth_root_cause": anomaly.asset_id})
    return FleetDataset(source="uploaded", assets=assets, history=history, telemetry=telemetry,
                        scenarios=scenarios)
