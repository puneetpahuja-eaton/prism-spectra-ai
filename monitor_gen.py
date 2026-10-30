"""Generate a live-monitoring plan from a natural-language instruction via the selected model.

The agent reads the device's ingested docs (RAG) to resolve parameter IDs, a
polling interval, and an event-log parameter plus a decode spec, so the existing
REST poller can monitor them. Event logs rarely decode trivially, so the flow
asks clarifying questions about the event structure first.

Flow mirrors the dataset builder:
  1. clarifying_questions(): the model asks about params, interval, event-log layout.
  2. generate_plan(): the model returns a compact JSON plan grounded in the docs.
The plan maps straight onto the Live Monitoring config (monitored params + an
optional event-log decode spec).
"""
from __future__ import annotations

from pydantic import BaseModel, Field

import rag
from gemini_client import generate_json, generate_text, is_live
from device_grounding import SOURCE_ONLY, verify_quote, verify_unit
from live.rest_adapter import _PARAM_URL


class MonitorParam(BaseModel):
    pid: str = Field(description="Exact parameter ID from supplied device context")
    label: str = Field(default="", description="Source-defined parameter label")
    unit: str = Field(default="")
    lower: float | None = Field(default=None, description="Lower alarm limit, if known")
    upper: float | None = Field(default=None, description="Upper alarm limit, if known")
    path: str = Field(default="", description="JSON path to the value if not the default 'value'")
    evidence: str = Field(default="", description="Exact supplied passage containing this ID, unit and any limits")


class EventLogSpec(BaseModel):
    pid: str = Field(default="", description="Parameter ID / endpoint that returns the event log")
    format: str = Field(default="", description="How the event log is encoded (free-text understanding)")
    decode: str = Field(default="", description="Step-by-step decode instructions, for the engineer")
    word_index: int = Field(default=-1, description="0-based index of the hex word holding the event code (-1=unknown)")
    code_map: dict[str, str] = Field(default_factory=dict,
                                     description="Map raw code value (decimal or 0x hex string) -> event code name")
    evidence: str = Field(default="", description="Exact supplied passage documenting the layout and code mappings")


class MonitorPlan(BaseModel):
    poll_interval_s: int | None = Field(default=None, ge=1, le=86400)
    hex_as_float: bool | None = None
    encoding_evidence: str = Field(default="", description="Exact supplied passage specifying hex encoding, if configured")
    params: list[MonitorParam] = Field(default_factory=list)
    event_log: EventLogSpec | None = Field(default=None)
    notes: str = Field(default="")
    missing_details: list[str] = Field(default_factory=list)


_SCHEMA_HINT = (
    "Return a MonitorPlan. Use null for unknown limits and encoding. Every parameter needs an exact evidence passage "
    "containing its ID, units and any limits. Populate missing_details when necessary facts are missing; "
    "do not invent them. Leave event_log null unless the supplied material specifies the full layout."
)


def _context(device_id: str) -> str:
    chunks = rag.retrieve(device_id, "parameter identifiers register map event log units thresholds "
                                     "encoding REST interface", top_k=20)
    return "\n\n".join(c.get("text", "")[:900] for c in chunks)


def clarifying_questions(device_id: str, request: str, reference: str = "") -> str:
    prompt = (
        f"{SOURCE_ONLY}\n"
        "A user wants to set up LIVE monitoring of a device over REST. Before building the plan, ask 3-6 "
        "concise clarifying questions that materially change it: which exact parameters/IDs to poll, the "
        "polling interval, alarm thresholds, and—critically—the EVENT LOG structure (which parameter/"
        "endpoint holds it, how entries are encoded, where the event code lives, how to decode it). Ask "
        "only questions; do not build the plan yet. Use the context to avoid asking what is already known.\n\n"
        f"Device context:\n{_context(device_id)[:8000]}\n\n"
        f"Referenced parameter file (optional):\n{reference[:8000]}\n\n"
        f"User request: {request}"
    )
    return generate_text(prompt, temperature=0.3)


def generate_plan(device_id: str, request: str, reference: str = "", answers: str = "") -> dict:
    if not is_live():
        raise RuntimeError("A configured live AI provider is required to generate a monitoring plan.")
    context = _context(device_id)
    supplied = "\n".join((context, reference, request, answers))
    prompt = (
        f"{SOURCE_ONLY}\n"
        "Design a COMPACT live-monitoring plan for this device over REST, grounded in the context and the "
        "referenced parameter list. Resolve real parameter IDs and units. Include alarm thresholds where the "
        "docs imply them. If the user described an event log, fill event_log with a concrete decode spec; "
        "otherwise set it to null. Never assume IDs, limits, units, encoding or polling interval.\n\n"
        f"{_SCHEMA_HINT}\n\n"
        f"Device context:\n{context[:8000]}\n\n"
        f"Referenced parameter file (optional, authoritative for IDs/units):\n{reference[:12000]}\n\n"
        f"User request: {request}\n\n"
        f"User answers to clarifying questions (if any): {answers}"
    )
    plan = MonitorPlan.model_validate(generate_json(prompt, MonitorPlan, max_output_tokens=16384))
    if plan.missing_details or not plan.params or plan.poll_interval_s is None:
        raise ValueError("Provide missing monitoring details: " + "; ".join(
            plan.missing_details or ["parameter IDs with evidence and the polling interval"]))
    for param in plan.params:
        verify_unit(param.unit, param.evidence)
        verify_quote(param.evidence, supplied,
                     [param.pid, param.unit] + ([param.label] if param.label else []) +
                     ([param.path] if param.path else []) +
                     [v for v in (param.lower, param.upper) if v is not None])
    if plan.hex_as_float is not None:
        verify_quote(plan.encoding_evidence, supplied, [])
    if plan.event_log is not None:
        event = plan.event_log
        if not event.pid or event.word_index < 0:
            raise ValueError("Provide the event-log parameter and documented word index, or leave event_log unset.")
        verify_quote(event.evidence, supplied,
                     [event.pid, event.word_index, *event.code_map.keys(), *event.code_map.values()])
    return plan.model_dump()


def plan_to_config(plan: dict, base_endpoint: str) -> dict:
    """Translate a generated plan into the Live Monitoring config structure."""
    if not base_endpoint.strip() or plan.get("poll_interval_s") is None:
        raise ValueError("Supply the device endpoint and polling interval before applying a plan.")
    monitored = [{"pid": str(p["pid"]).strip(), "lower": p.get("lower"), "upper": p.get("upper"),
                  "label": p.get("label", ""), "path": p.get("path", "")}
                 for p in plan.get("params", []) if str(p.get("pid", "")).strip()]
    event_log = plan.get("event_log") or None
    field_map = {}
    if not _PARAM_URL.search(base_endpoint) and monitored:
        if any(not p["path"] for p in monitored):
            raise ValueError("A generic endpoint requires a source-defined JSON value path for every parameter.")
        field_map = {f"p{p['pid']}": p["path"] for p in monitored}
    return {
        "endpoint": base_endpoint,
        "interval": int(plan["poll_interval_s"]),
        "hex_as_float": plan.get("hex_as_float"),
        "monitored": monitored,
        "param_ids": [m["pid"] for m in monitored],
        "event_log": event_log,
        "field_map": field_map,
    }
