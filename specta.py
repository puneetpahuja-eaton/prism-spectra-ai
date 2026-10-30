"""Device-scoped conversations and resumable, read-only generation tasks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal
from uuid import uuid4

import requests

import auth
import dataset_gen
import datastore
import device_data
import devices
import ingest
import knowledge
import monitor_gen
import rag
from device_grounding import ClarificationNeeded

TaskKind = Literal["dataset", "monitor"]


@dataclass
class Task:
    kind: TaskKind
    request: str
    reference_path: str = ""
    reference_text: str = ""
    prior_guidance: str = ""
    answers: list[str] = field(default_factory=list)
    days: int | None = None
    interval_min: int | None = None
    asset_count: int = 1
    id: str = field(default_factory=lambda: uuid4().hex)


@dataclass
class Conversation:
    device_id: str
    messages: list[dict] = field(default_factory=list)
    task: Task | None = None
    attachments: list[dict[str, str]] = field(default_factory=list)
    mode: str = "agent"


def begin_task(conversation: Conversation, kind: TaskKind, request: str, *,
               reference_path: str = "", days: int | None = None,
               interval_min: int | None = None, asset_count: int = 1) -> None:
    if kind not in ("dataset", "monitor"):
        raise ValueError("Specta supports dataset and monitoring-plan generation tasks.")
    conversation.task = Task(kind, request, reference_path=reference_path.strip(),
                             days=days, interval_min=interval_min, asset_count=asset_count,
                             prior_guidance=_history_guidance(conversation))


def add_attachments(conversation: Conversation, files: list[tuple[str, bytes]]) -> None:
    sources = []
    for name, data in files:
        if len(data) > 5 * 1024 * 1024:
            raise ValueError(f"{name} exceeds the 5 MB chat attachment limit.")
        text = ingest.extract_text(name, data)
        if not text.strip():
            raise ValueError(f"No readable text found in {name}. Attach a text-readable source.")
        sources.append({"name": name, "text": text})
    if sum(len(s["text"]) for s in conversation.attachments + sources) > 120000:
        raise ValueError("Attachments exceed 120,000 extracted characters. Start a new chat or attach a smaller relevant excerpt.")
    conversation.attachments.extend(sources)


def load_dataset(device_id: str, message: dict, *, role: str, confirmed: bool) -> None:
    if not auth.can_configure(role):
        raise PermissionError("Only an admin can replace a device's Analysis dataset.")
    if not confirmed:
        raise ValueError("Confirm that you want to replace this device's current dataset.")
    if not devices.get_device(device_id):
        raise ValueError("The selected device no longer exists.")
    device_data.set_dataset(device_id, message["dataset"])


def apply_monitor_plan(device_id: str, message: dict, endpoint: str, *,
                       role: str, confirmed: bool, running: bool = False) -> None:
    if not auth.can_operate(role):
        raise PermissionError("Only an engineer or admin can configure live monitoring.")
    if not confirmed:
        raise ValueError("Verify the plan against this device's sources first.")
    if running:
        raise ValueError("Stop live monitoring before applying a different plan.")
    if not devices.get_device(device_id):
        raise ValueError("The selected device no longer exists.")
    config = datastore.get_doc(devices.live_path(device_id), "config") or {}
    config.update(monitor_gen.plan_to_config(message["plan"], endpoint))
    config["simulated"] = False
    datastore.set_doc(devices.live_path(device_id), "config", config)


def clarification_message(error: ClarificationNeeded) -> str:
    introduction = error.answer.strip() or "I can help with that. I need a little more information before I can finish."
    questions = "\n".join(f"{index}. {question}" for index, question in enumerate(error.questions[:3], 1))
    if len(error.questions) > 3:
        questions += "\n\nI'll check the remaining fields after these answers."
    return (f"{introduction}\n\n{questions}\n\n"
            "Reply here with the missing details or attach the relevant file. "
            "I'll continue the same request using your answer; you don't need to start again.")


def _device_context(device_id: str, query: str) -> str:
    device = devices.get_device(device_id)
    if not device:
        raise ValueError("The selected device no longer exists. Select another device.")
    identity = (f"Selected registered device: ID {device['id']}; name {device['name']}; "
                f"type {device['type']}; criticality {device['criticality']}. "
                "Unless the user explicitly requests multiple assets, generate one asset representing "
                "this selected device, using its registered ID. Do not ask for its asset count again.")
    chunks = rag.retrieve(device_id, query, top_k=20)
    return identity + "\n\n" + rag._format_context(chunks)[:60000]


def _history_guidance(conversation: Conversation) -> str:
    return "\n\n".join(m["content"] for m in conversation.messages[-24:] if m["role"] == "user")


def _run_task(conversation: Conversation, notify: Callable[[str], None]) -> dict:
    task = conversation.task
    if task is None:
        raise ValueError("No Specta task is active.")
    notify("Reading the selected device's supplied sources")
    context = _device_context(conversation.device_id, task.request + "\n" + "\n".join(task.answers))
    if task.prior_guidance:
        context += ("\n\nEarlier user guidance for this selected device (not assistant assumptions). "
                    "Use relevant facts; the latest explicit request/replies supersede earlier guidance. "
                    "Ask about unresolved conflicts:\n" + task.prior_guidance)
    if task.reference_path and not task.reference_text:
        notify("Reading the referenced repository file")
        task.reference_text = knowledge.fetch_repo_file_text(conversation.device_id, task.reference_path)["text"]
    attachments = "\n\n".join(f"Attached source {a['name']}:\n{a['text']}" for a in conversation.attachments)
    reference = task.reference_text + "\n\n" + attachments
    answers = "\n\n".join(task.answers)
    if task.kind == "dataset":
        notify("Checking dataset requirements and source-supported values")
        request = task.request
        if task.days is not None and task.interval_min is not None:
            request += (f"\nAuthoritative UI choices: synthetic replay duration {task.days} days; "
                        f"sample interval {task.interval_min} minutes. These choices do not require reconfirmation.")
        spec = dataset_gen.generate_spec(context, request, reference, answers, asset_count=task.asset_count)
        if task.days is not None:
            spec["duration_days"] = task.days
        if task.interval_min is not None:
            spec["sample_interval_min"] = task.interval_min
        spec_obj = dataset_gen.DatasetSpec.model_validate(spec)
        try:
            spec_obj.anomalies = dataset_gen.ensure_anomaly_mix(spec_obj)
        except ValueError as error:
            raise ClarificationNeeded([str(error)]) from error
        notify("Building and validating the synthetic readings")
        data = dataset_gen.build_dataset(spec_obj)
        ticks = len({row.timestamp for row in data.telemetry})
        return {
            "content": (f"Your **synthetic dataset is ready**: {len(data.assets)} asset(s), "
                        f"{len(data.history):,} baseline rows and {len(data.telemetry):,} replay rows "
                        f"({ticks:,} samples per asset).\n\n"
                        "Download it below, or review and load it into Analysis if you have admin access. "
                        "These are illustrative readings, not measurements or manufacturer specifications."),
            "dataset": data, "spec": spec_obj.model_dump(), "status": "complete",
        }
    notify("Checking monitoring IDs, units, limits and decoding against supplied sources")
    plan = monitor_gen.generate_plan(conversation.device_id, task.request, context + "\n\n" + reference, answers)
    return {
        "content": (f"Your **monitoring plan is ready**: {len(plan['params'])} parameter(s), "
                    f"polled every {plan['poll_interval_s']} seconds.\n\n"
                    "Review its source evidence and connection details below before applying it. "
                    "No monitoring has been started and no equipment commands have been sent."),
        "plan": plan, "status": "complete",
    }


def respond(conversation: Conversation, message: str, *, mode: str = "agent",
            page_context: str = "", notify: Callable[[str], None] | None = None,
            start_task: bool = False, role: str = "user") -> dict:
    """Always retain a user-visible reply, including clarifications and failures."""
    notify = notify or (lambda _: None)
    message = message.strip()
    if not message:
        raise ValueError("Enter a message for Specta.")
    conversation.messages.append({"role": "user", "content": message})
    conversation.mode = mode
    try:
        if conversation.task is not None:
            if not start_task:
                conversation.task.answers.append(message)
            if mode != "agent":
                result = {"content": "This generation task is paused. Select **Agent** to continue, or cancel the task "
                          "to ask a separate question.", "status": "paused"}
            else:
                result = _run_task(conversation, notify)
                conversation.task = None
        else:
            notify("Retrieving device context and preparing a response")
            attached = "\n\n".join(f"Attached source {a['name']}:\n{a['text']}" for a in conversation.attachments)
            response = rag.converse(
                conversation.device_id, message, mode=mode, screen_context=page_context,
                role=role, history=conversation.messages[:-1], attached_context=attached)
            task_kind = response.get("task", "chat")
            if task_kind in ("dataset", "monitor") and mode == "agent" and not response.get("needs_clarification"):
                begin_task(conversation, task_kind, message, asset_count=response.get("asset_count") or 1)
                result = _run_task(conversation, notify)
                conversation.task = None
            else:
                result = {"content": response["answer"], "status": (
                    "clarification" if response.get("needs_clarification") else
                    "confirmation" if response.get("awaiting_confirmation") else
                    "error" if response.get("backend") in ("unavailable", "offline") else "complete"),
                    "citations": response.get("citations", []), "artifacts": response.get("artifacts", [])}
    except ClarificationNeeded as error:
        result = {"content": clarification_message(error), "status": "clarification"}
    except (ValueError, RuntimeError, OSError, requests.RequestException) as error:
        result = {
            "content": ("I couldn't complete this request.\n\n"
                        f"**{type(error).__name__}:** {error}\n\n"
                        "Your conversation and task are retained. Correct the source/reference or configuration "
                        "if needed, then send **retry**. No dataset, applied plan or equipment action is being claimed."),
            "status": "error",
        }
    result["role"] = "assistant"
    conversation.messages.append(result)
    return result
