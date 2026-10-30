"""Generate a live-monitoring plan from a natural-language instruction via the selected model.

The agent reads the device's ingested docs (RAG) to resolve parameter IDs, a
polling interval, and an event-log parameter plus a decode spec, so the existing
REST poller can monitor them. Event logs rarely decode trivially, so the flow
asks clarifying questions about the event structure first.

generate_plan returns a compact plan or raises ClarificationNeeded for the
conversation to collect missing facts and resume the same request.
The plan maps straight onto the Live Monitoring config (monitored params + an
optional event-log decode spec).
"""
from __future__ import annotations

from pydantic import BaseModel, Field

import rag
from gemini_client import generate_json, is_live
from device_grounding import ClarificationNeeded, SOURCE_ONLY, generate_grounded, verify_passages, verify_quote
from live.rest_adapter import _PARAM_URL


class MonitorParam(BaseModel):
    pid: str = Field(description="Exact parameter ID from supplied sources OR explicit user guidance, including the current reply")
    label: str = Field(default="", description="Source-defined parameter label")
    unit: str = Field(default="")
    lower: float | None = Field(default=None, description="Lower alarm limit, if known")
    upper: float | None = Field(default=None, description="Upper alarm limit, if known")
    path: str = Field(default="", description="JSON path to the value if not the default 'value'")
    evidence: str = Field(default="", description="Copy exact text from the user's reply or supplied source, without prefixes or paraphrasing; user replies themselves are evidence")
    evidence_passages: list[str] = Field(default_factory=list)


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
    missing_details: list[str] = Field(default_factory=list, description="Only missing device facts, never requests for document quotes when the user has supplied the facts explicitly")
    answer: str = Field(default="", description="Brief response in plain language; no schema or validation jargon")


_SCHEMA_HINT = (
    "Return a MonitorPlan. Explicit user guidance and answers are authoritative supplied sources for this task, "
    "just like device docs/code. Copy the user's actual words into evidence or evidence_passages. "
    "A separate document is NOT required to reconfirm facts explicitly supplied by the user. "
    "Never ask for exact document passages or citations from the user; select the quotes yourself. "
    "Set missing_details=[] when IDs, units, value paths and polling are already supplied. "
    "Leave optional limits/event decoding/hex encoding unset when unknown or intentionally excluded; don't ask for them. "
    "Use null for unknown limits and encoding. Every parameter needs an exact evidence passage "
    "for its ID, units and any limits. Use evidence_passages when facts occur in different supplied "
    "passages. Copy citations yourself; do not ask users to format source quotes. Populate missing_details "
    "with at most 3 focused, plain-language questions when facts are missing, and answer with what is known. "
    "Do not re-ask answered questions or demand schema names. Never invent device facts. "
    "Leave event_log null unless the supplied material specifies the full layout."
)


def _context(device_id: str) -> str:
    chunks = rag.retrieve(device_id, "parameter identifiers register map event log units thresholds "
                                     "encoding REST interface", top_k=20)
    return rag._format_context(chunks)[:60000]


def generate_plan(device_id: str, request: str, reference: str = "", answers: str = "") -> dict:
    if not is_live():
        raise RuntimeError("A configured live AI provider is required to generate a monitoring plan.")
    context = _context(device_id)
    supplied = "\n".join((context, reference, request, answers))
    prompt = (
        f"{SOURCE_ONLY}\n"
        "Design a COMPACT live-monitoring plan for this device over REST, grounded in the context and the "
        "referenced parameter list OR explicit user answers. Include only explicitly supplied alarm thresholds. "
        "If the user described an event log, fill event_log with a concrete decode spec; "
        "otherwise set it to null. Never assume IDs, limits, units, encoding or polling interval.\n\n"
        f"{_SCHEMA_HINT}\n\n"
        f"Device context:\n{context[:60000]}\n\n"
        f"Referenced parameter file (optional, authoritative for IDs/units):\n{reference[:120000]}\n\n"
        f"User request: {request}\n\n"
        f"User answers to clarifying questions (if any): {answers}"
    )
    return generate_grounded(generate_json, prompt, MonitorPlan, lambda raw: _validate_plan(raw, supplied))


def _validate_plan(raw: dict, supplied: str) -> dict:
    plan = MonitorPlan.model_validate(raw)
    if plan.missing_details or not plan.params or plan.poll_interval_s is None:
        raise ClarificationNeeded(plan.missing_details or [
            "Which documented parameters should I monitor, and how often should I poll them?"
        ], answer=plan.answer)
    for param in plan.params:
        try:
            verify_passages(param.evidence, param.evidence_passages, supplied,
                            [param.pid, param.unit] + ([param.label] if param.label else []) +
                            ([param.path] if param.path else []) +
                            [v for v in (param.lower, param.upper) if v is not None], unit=param.unit)
        except ValueError as error:
            raise ClarificationNeeded([
                f"I couldn't verify the supplied evidence for parameter **{param.pid}**. "
                "Can you confirm its unit, value path and any limits, or attach the relevant source?"
            ], repairable=True) from error
    if plan.hex_as_float is not None:
        try:
            verify_quote(plan.encoding_evidence, supplied, [])
        except ValueError as error:
            raise ClarificationNeeded(["How does this device encode hexadecimal values? Please supply its decoding rule."], repairable=True) from error
    if plan.event_log is not None:
        event = plan.event_log
        if not event.pid or event.word_index < 0:
            raise ClarificationNeeded(["Which parameter contains the event log, and which word holds its event code?"], repairable=True)
        try:
            verify_quote(event.evidence, supplied,
                         [event.pid, event.word_index, *event.code_map.keys(), *event.code_map.values()])
        except ValueError as error:
            raise ClarificationNeeded(["Can you supply the documented event-log layout and code meanings, or should I leave event decoding disabled?"], repairable=True) from error
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
