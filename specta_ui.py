"""Reusable Specta chat surface: full-page workspace and floating cross-page dock."""
from __future__ import annotations

import json
import zipfile

import streamlit as st

import devices
import specta
import ui_common
import research_ui
import workflow
from nebius_client import backend as model_backend

_STYLE = """
<style>
.st-key-specta_launcher {
  position: fixed; right: 24px; bottom: 24px; width: auto; z-index: 999;
}
.st-key-specta_launcher button {
  border-radius: 28px; min-height: 48px; padding: 0 22px;
  box-shadow: 0 6px 24px #0004; font-weight: 600;
}
.st-key-specta_dock {
  position: fixed; right: 24px; bottom: 88px;
  width: min(460px, calc(100vw - 32px)); max-height: calc(100dvh - 116px);
  z-index: 998; overflow-y: auto; gap: 8px;
  background: SPECTA_BACKGROUND;
  box-shadow: 0 12px 48px #0004; border-radius: 14px;
}
.st-key-specta_dock [data-testid="stChatMessage"] {
  padding: 10px 8px; border-radius: 8px;
}
.st-key-specta_dock:has(.st-key-specta_dock_thread) {
  height: min(740px, calc(100dvh - 116px)); overflow: hidden;
}
.st-key-specta_dock > div:has(> .st-key-specta_dock_controls) {
  flex: 0 1 auto; min-height: 110px; max-height: 230px; overflow-y: auto;
}
.st-key-specta_dock > div:has(> .st-key-specta_dock_thread) {
  flex: 1 1 0%; height: auto; min-height: 80px;
}
.st-key-specta_dock_thread, .st-key-specta_full_thread {
  height: 100%; min-height: 0;
}
.st-key-specta_workspace > div:has(> .st-key-specta_full_thread) {
  height: clamp(240px, calc(100dvh - 430px), 520px); flex-basis: auto;
}
.st-key-specta_dock_controls {
  gap: 8px;
}
.st-key-specta_dock [data-testid="stChatInput"],
.st-key-specta_workspace [data-testid="stChatInput"] {
  border-radius: 10px;
}
@media (max-width: 640px) {
  .st-key-specta_launcher { right: 16px; bottom: 16px; }
  .st-key-specta_dock {
    right: 8px; bottom: 78px; width: calc(100vw - 16px);
    max-height: calc(100dvh - 94px);
  }
  .st-key-specta_dock [data-testid="stHorizontalBlock"] { flex-wrap: nowrap; }
  .st-key-specta_dock [data-testid="stColumn"] { min-width: 0; }
}
@media (max-height: 700px) {
  .st-key-specta_dock > div:has(> .st-key-specta_dock_controls) {
    min-height: 64px; max-height: 150px;
  }
  .st-key-specta_dock > div:has(> .st-key-specta_dock_thread) { min-height: 40px; }
}
</style>
"""
_MODES = {"Ask": "ask", "Plan": "plan", "Agent": "agent"}
_FILE_TYPES = ["txt", "md", "json", "csv", "yaml", "yml", "py", "c", "h", "cpp", "js", "ts",
               "pdf", "docx", "xlsx", "xlsm", "log"]


def conversation(device_id: str) -> specta.Conversation:
    key = f"specta_conversation_{device_id}"
    if key not in st.session_state:
        st.session_state[key] = specta.Conversation(device_id)
    return st.session_state[key]


def _remember_device(key: str) -> None:
    st.session_state["specta_selected_device"] = st.session_state[key]


def _remember_mode(key: str, device_id: str) -> None:
    conversation(device_id).mode = _MODES[st.session_state[key]]


def _toggle_dock() -> None:
    st.session_state["specta_open"] = not st.session_state.get("specta_open", False)


def _remember_task_field(key: str, device_id: str, field: str) -> None:
    task = conversation(device_id).task
    if task is not None:
        setattr(task, field, st.session_state[key])
        if field == "reference_path":
            task.reference_text = ""


def _device_picker(surface: str) -> dict | None:
    records = devices.list_devices()
    labels = {d["id"]: f"{d['name']} ({d['type']})" for d in records}
    chosen = st.session_state.get("specta_selected_device")
    key = f"specta_device_{surface}"
    # Widget keys differ between surfaces; the selected device is shared.
    st.session_state[key] = chosen if chosen in labels else None
    device_id = st.selectbox("Device context", list(labels), index=None, key=key,
                            format_func=lambda value: labels[value], on_change=_remember_device,
                            args=(key,), placeholder="Select a device")
    return next((d for d in records if d["id"] == device_id), None)


def run_message(conv: specta.Conversation, text: str, *, start_task: bool = False) -> None:
    with st.status("Specta is working...", expanded=True) as progress:
        result = specta.respond(
            conv, text, mode=conv.mode, role=ui_common.current_role(),
            page_context=f"Current application page: {st.session_state.get('specta_page', 'Chat')}.",
            notify=progress.write, start_task=start_task)
        label = {"clarification": "Waiting for your answer", "error": "Request needs attention",
                 "confirmation": "Plan ready for review", "paused": "Task paused"}.get(result["status"], "Done")
        progress.update(label=label, state="error" if result["status"] == "error" else "complete", expanded=False)


def start_generation(device_id: str, kind: specta.TaskKind, request: str, *,
                     reference_path: str = "", days: int | None = None,
                     interval_min: int | None = None, asset_count: int = 1) -> None:
    conv = conversation(device_id)
    conv.mode = "agent"
    specta.begin_task(conv, kind, request, reference_path=reference_path,
                      days=days, interval_min=interval_min, asset_count=asset_count)
    st.session_state["specta_selected_device"] = device_id
    if st.session_state.get("specta_page") != "Ask the device":
        st.session_state["specta_open"] = True
    run_message(conv, request, start_task=True)
    st.rerun()


def _dataset_actions(conv: specta.Conversation, message: dict, key: str) -> None:
    data = message["dataset"]
    st.download_button("Download dataset JSON", data.model_dump_json(indent=2),
                       "analysis-dataset.json", "application/json", key=f"{key}_dataset", on_click="ignore",
                       icon=":material/download:")
    with st.expander("Review dataset and load into Analysis"):
        st.caption("Assets: " + ", ".join(a.id for a in data.assets))
        st.json(message["spec"], expanded=False)
        if ui_common.current_role() != "admin":
            st.caption("An admin can load this dataset. You can download it above.")
            return
        confirmed = st.checkbox("Replace this device's current Analysis dataset", key=f"{key}_replace")
        if st.button("Load into Analysis", key=f"{key}_load", disabled=not confirmed):
            try:
                specta.load_dataset(conv.device_id, message, role=ui_common.current_role(), confirmed=confirmed)
            except (ValueError, RuntimeError, OSError) as error:
                st.error(f"Dataset was not loaded: {error}")
            else:
                st.session_state.pop("prism_device", None)
                st.success("Loaded. Open Analysis and press Play to run it.")


def _monitor_actions(conv: specta.Conversation, message: dict, key: str) -> None:
    st.download_button("Download monitoring plan", json.dumps(message["plan"], indent=2),
                       "monitoring-plan.json", "application/json", key=f"{key}_plan", on_click="ignore",
                       icon=":material/download:")
    with st.expander("Review and apply monitoring plan"):
        st.json(message["plan"])
        if ui_common.current_role() not in ("admin", "engineer"):
            st.caption("An engineer or admin can apply this plan.")
            return
        endpoint = st.text_input("Base REST endpoint for these params", key=f"{key}_endpoint")
        reviewed = st.checkbox("I verified the IDs, limits, units and encoding against this device's sources",
                               key=f"{key}_review")
        if st.button("Apply to Live Monitoring", key=f"{key}_apply", disabled=not reviewed or not endpoint.strip()):
            try:
                specta.apply_monitor_plan(
                    conv.device_id, message, endpoint, role=ui_common.current_role(), confirmed=reviewed,
                    running=st.session_state.get(f"live_on_{conv.device_id}", False))
            except (ValueError, RuntimeError, OSError) as error:
                st.error(f"Plan was not applied: {error}")
            else:
                st.session_state.pop(f"params_table_{conv.device_id}", None)
                message["applied"] = True
                st.rerun()
        if message.get("applied"):
            st.success("Applied without starting monitoring. Review connection/auth/TLS in Live Monitoring, then Start.")


def _render_message(conv: specta.Conversation, message: dict, key: str) -> None:
    with st.chat_message(message["role"], avatar=":material/smart_toy:" if message["role"] == "assistant" else None):
        if message["role"] == "assistant":
            st.caption("Specta" + (" - awaiting your answer" if message.get("status") == "clarification" else ""))
        st.markdown(message["content"])
        if message.get("citations"):
            st.caption("Retrieved sources: " + ", ".join(message["citations"]))
        for index, artifact in enumerate(message.get("artifacts", [])):
            st.download_button(f"Download {artifact['name']}", artifact["content"], artifact["name"],
                               artifact["mime"], key=f"{key}_artifact_{index}", on_click="ignore")
            if artifact.get("truncated"):
                st.warning("This file reached the model's output limit. Ask Specta to split or shorten it.")
        if message.get("dataset"):
            _dataset_actions(conv, message, key)
        if message.get("plan"):
            _monitor_actions(conv, message, key)
        if message["role"] == "assistant":
            if message.get("workflow"):
                workflow.show_report(message["workflow"], key=f"{key}_workflow")
            st.download_button("Save response", message["content"], "specta-response.md", "text/markdown",
                               key=f"{key}_response", on_click="ignore", type="tertiary",
                               icon=":material/content_copy:", help="Save this response; use code-block Copy for code.")


def _chat_controls(surface: str) -> specta.Conversation | None:
    device = _device_picker(surface)
    if not device:
        st.caption("Select a device to start. Specta never mixes device sources or selects the demo for you.")
        return None
    conv = conversation(device["id"])
    cols = st.columns([3, 1])
    mode_key = f"specta_mode_{surface}_{device['id']}"
    st.session_state[mode_key] = next(label for label, value in _MODES.items() if value == conv.mode)
    selected_mode = cols[0].selectbox("Mode", list(_MODES), key=mode_key, label_visibility="collapsed",
                                     on_change=_remember_mode, args=(mode_key, device["id"]),
                                     help="Ask: answers only. Plan: propose, don't execute. Agent: generate artifacts; never control equipment.")
    conv.mode = _MODES[selected_mode]
    if cols[1].button("New" if surface == "dock" else "New chat", key=f"specta_new_{surface}", icon=":material/add:",
                      help="Clear this device's session conversation, pending task and attachments. Stored device data stays intact."):
        st.session_state[f"specta_conversation_{device['id']}"] = specta.Conversation(device["id"], mode=conv.mode)
        st.rerun()
    if conv.task:
        active = "dataset" if conv.task.kind == "dataset" else "monitoring plan"
        row = st.columns([4, 1])
        row[0].caption(f"Active task: {active}. Reply below to continue.")
        if row[1].button("Cancel", key=f"specta_cancel_{surface}"):
            conv.task = None
            st.rerun()
        with st.expander("Task settings"):
            ref_key = f"specta_ref_{surface}_{conv.task.id}"
            st.session_state[ref_key] = conv.task.reference_path
            st.text_input("Repository reference path (optional)", key=ref_key,
                          on_change=_remember_task_field, args=(ref_key, conv.device_id, "reference_path"))
            if conv.task.kind == "dataset":
                count_key = f"specta_count_{surface}_{conv.task.id}"
                st.session_state[count_key] = conv.task.asset_count
                st.number_input("Synthetic asset count", min_value=1, max_value=10, key=count_key,
                                on_change=_remember_task_field, args=(count_key, conv.device_id, "asset_count"))
            if conv.task.days is not None and conv.task.interval_min is not None:
                timing = st.columns(2)
                for col, field, label, maximum in (
                    (timing[0], "days", "Replay days", 14),
                    (timing[1], "interval_min", "Sample minutes", 1440),
                ):
                    key = f"specta_{field}_{surface}_{conv.task.id}"
                    st.session_state[key] = getattr(conv.task, field)
                    col.number_input(label, min_value=1, max_value=maximum, key=key,
                                     on_change=_remember_task_field, args=(key, conv.device_id, field))
            st.caption("After correcting a setting, reply with the missing facts or send retry.")
    return conv


def chat_surface(surface: str) -> None:
    with st.container(key=f"specta_{surface}_controls"):
        conv = _chat_controls(surface)
    if conv is None:
        return
    thread = st.container(height=330 if surface == "dock" else 520, border=False,
                          key=f"specta_{surface}_thread", autoscroll=True)
    with thread:
        if not conv.messages:
            st.markdown("**Hi, I'm Specta.**\n\nAsk about this device, plan a workflow, or create a synthetic "
                        "dataset or monitoring plan. If facts are missing, I'll ask here and continue from your reply.")
        for index, message in enumerate(conv.messages):
            _render_message(conv, message, f"specta_{surface}_{conv.device_id}_{index}")
        def remember_public(packet: dict) -> None:
            conv.public_evidence = packet
        evidence = [{"text": a["text"]} for a in conv.attachments]
        # Research gets its own approval and never contributes facts to generation tools.
        research_ui.render(conv.device_id, f"chat:{conv.id}", surface=surface,
                           device_evidence=evidence, on_result=remember_public)
    st.caption("Device sources + your guidance; explicitly approved public evidence is supplementary to chat only. "
               "Attach authorized files; they are sent to Nebius. No automatic public web search.")
    if conv.attachments:
        st.caption("Attached: " + ", ".join(a["name"] for a in conv.attachments))
    value = st.chat_input("Reply to continue..." if conv.task else "Ask Specta or describe a task...",
                          key=f"specta_input_{surface}_{conv.device_id}", accept_file="multiple",
                          file_type=_FILE_TYPES, max_upload_size=5)
    if value:
        text = value if isinstance(value, str) else value.text
        files = [] if isinstance(value, str) else value.files
        if files:
            try:
                specta.add_attachments(conv, [(file.name, file.getvalue()) for file in files])
            except (ValueError, RuntimeError, OSError, zipfile.BadZipFile) as error:
                conv.messages.extend([
                    {"role": "user", "content": text or "Read my attached sources."},
                    {"role": "assistant", "status": "error",
                     "content": f"I couldn't read the attachment: {error}\n\nAttach a supported readable source and resend."},
                ])
                st.rerun()
        prompt = text.strip() or "Use the attached sources to continue my request."
        with thread:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant", avatar=":material/smart_toy:"):
                run_message(conv, prompt)
        st.rerun()


@st.fragment
def mount_dock() -> None:
    background = "#0e1117" if st.context.theme.type == "dark" else "#ffffff"
    st.html(_STYLE.replace("SPECTA_BACKGROUND", background))
    opened = st.session_state.get("specta_open", False)
    with st.container(key="specta_launcher"):
        st.button("Close Specta" if opened else "Specta", key="specta_toggle",
                      icon=":material/close:" if opened else ":material/chat_bubble:",
                      type="primary", help="Your device assistant, available on every page", on_click=_toggle_dock)
    if opened:
        with st.container(key="specta_dock", border=True):
            st.markdown("**Specta**")
            if not ui_common.current_user():
                st.markdown("Sign in to ask device questions and generate artifacts. Your device sources and "
                            "conversations are available only after authentication. Contact the deployment "
                            "owner for an account; judges should use the supplied testing instructions.")
                return
            st.caption(f"{model_backend()} / {st.session_state.get('specta_page', 'Home')}")
            chat_surface("dock")


def dataset_builder(device_id: str) -> None:
    with st.expander("Build a dataset for Analysis"):
        st.caption("Specta will generate it or ask for missing details in this same chat. "
                   "Your selected duration and sampling rate are retained.")
        request = st.text_area("What dataset do you want?", key=f"ds_req_{device_id}", height=90,
                               placeholder="Describe the metrics, supplied units/ranges and desired synthetic scenarios.")
        path = st.text_input("Reference a repo file for parameters (optional path)", key=f"ds_ref_{device_id}",
                             placeholder="docs/parameters.csv")
        count = st.number_input("Synthetic asset count", min_value=1, max_value=10, value=1,
                                step=1, key=f"ds_count_{device_id}")
        cols = st.columns(2)
        days = cols[0].number_input("How many days of data?", min_value=1, max_value=14, value=3,
                                    step=1, key=f"ds_days_{device_id}")
        interval = cols[1].number_input("Sample every (minutes)?", min_value=1, max_value=1440,
                                        value=5, step=1, key=f"ds_interval_{device_id}")
        rows = int(days) * max(1, 1440 // int(interval))
        st.caption(f"{rows:,} replay samples per asset before the 10,000-row fleet limit. "
                   "At least 8 are needed for both recovering and persistent anomaly scenarios.")
        if st.button("Generate dataset", key=f"ds_gen_{device_id}", type="primary", disabled=not request.strip()):
            start_generation(device_id, "dataset", request, reference_path=path,
                             days=int(days), interval_min=int(interval), asset_count=int(count))


def render_workspace() -> None:
    st.title("Specta")
    st.caption("Your device copilot - Ask, Plan, Agent. The same conversation follows you across every screen.")
    st.caption(f"Configured model: {model_backend()}. No terminal, editor execution or equipment-control tools.")
    with st.container(key="specta_workspace"):
        selected = st.session_state.get("specta_selected_device")
        if selected and devices.get_device(selected):
            dataset_builder(selected)
        chat_surface("full")
