"""Shared: per-device analysis — renders the PRISM dashboard for the selected device."""
from __future__ import annotations

from pathlib import Path
import json
import runpy

import streamlit as st

import device_data
from nebius_client import is_live
from prism_cloud import Engine
import prism_extensions as cloud
import ui_common

_DASHBOARD = Path(__file__).resolve().parent.parent / "prism_reference" / "dashboard" / "app.py"


def _prepare_engine_state(device_id: str) -> None:
    """Point the embedded PRISM dashboard at the selected device's telemetry."""
    if st.session_state.get("prism_device") != device_id:
        # Device changed: rebuild the engine for this device.
        st.session_state["prism_device"] = device_id
        st.session_state["prism_demo_profile"] = device_data.uses_demo_profile(device_id)
        st.session_state.pop("engine", None)
        st.session_state["playing"] = False
        for key in ("live_asset", "live_metric", "cf_metric", "cf_asset"):
            st.session_state.pop(key, None)
        if device_data.mode(device_id) == "dataset":
            dataset = device_data.get_dataset(device_id)
            st.session_state["prism_pending_dataset"] = dataset
        else:
            st.session_state.pop("prism_pending_dataset", None)


def render() -> None:
    st.title("📊 Analysis")
    device = ui_common.device_selector("analysis_device")
    if not device:
        return
    device_id = device["id"]

    if device_data.mode(device_id) not in ("dataset", "synthetic"):
        st.info("No telemetry is configured. Upload device readings or build a source-grounded synthetic dataset. "
                "The built-in power demonstration runs only when explicitly selected in Simulate / Upload.")
        return
    if device_data.mode(device_id) == "dataset" and device_data.get_dataset(device_id) is None:
        session_demo = (device_data.uses_demo_profile(device_id) and
                        st.session_state.get("prism_device") == device_id and
                        st.session_state.get("prism_pending_dataset") is not None)
        if not session_demo:
            st.warning("This device's telemetry file is unavailable on this instance. Existing stored records "
                       "have not been changed. Ask an admin to provide persistent telemetry storage or upload the data.")
            if not device_data.uses_demo_profile(device_id):
                return
            if not st.button("Load synthetic demo fixture for this session"):
                return
            from seed_demo import build_demo_dataset
            st.session_state["prism_device"] = device_id
            st.session_state["prism_demo_profile"] = True
            st.session_state["prism_pending_dataset"] = build_demo_dataset()
            st.session_state.pop("engine", None)
            st.session_state["playing"] = False

    st.caption(f"Device: **{device['name']}** · mode: **{device_data.mode(device_id)}**")
    if device_data.uses_demo_profile(device_id):
        st.warning("Explicit synthetic power demo: fixture specifications and playbooks are not evidence about your device.")
        st.info("The preserved reference dashboard's tutorial mentions historical learned models. "
                "This runtime instead uses deterministic median/MAD deviation and baseline code membership, "
                "with zero learned transitions. Detection scores and simulated outcomes may differ from the reference.")
    else:
        st.info("Source-only, observation-only analysis. Unknown event codes remain uninterpreted. "
                "A statistical anomaly is not a physical diagnosis; no equipment actions or assumed savings.")
    _prepare_engine_state(device_id)
    if not device_data.uses_demo_profile(device_id):
        _render_device_analysis(device_id)
        return

    # The PRISM dashboard calls st.set_page_config itself; suppress the duplicate call.
    original = st.set_page_config
    st.set_page_config = lambda *args, **kwargs: None
    try:
        runpy.run_path(str(_DASHBOARD), run_name="__main__")
    finally:
        st.set_page_config = original


def _render_device_analysis(device_id: str) -> None:
    use_model = st.checkbox("Authorize the configured model to interpret this device's sources and readings",
                            disabled=not is_live(), key=f"analysis_model_{device_id}")
    engine = st.session_state.get("engine")
    if engine is None or engine.offline != (not use_model):
        try:
            engine = Engine(dataset=st.session_state["prism_pending_dataset"], device_id=device_id,
                            offline=not use_model)
        except (ValueError, RuntimeError) as error:
            st.error(f"Cannot prepare device analysis: {error}")
            return
        st.session_state["engine"] = engine
    st.caption("Training history is the baseline you supplied, not a certified healthy operating range. "
               "Recovery means observed return to that baseline, not a repair or proven fault resolution.")
    play, pause, reset = st.columns(3)
    if play.button("Play device analysis", disabled=engine.at_end):
        st.session_state["playing"] = True
    if pause.button("Pause device analysis"):
        st.session_state["playing"] = False
    if reset.button("Reset device analysis"):
        engine.reset()
        st.session_state["playing"] = False

    @st.fragment(run_every=0.5 if st.session_state.get("playing") else None)
    def panel():
        if st.session_state.get("playing") and not engine.at_end:
            engine.tick()
        if engine.at_end:
            st.session_state["playing"] = False
        if st.button("Analyze next reading", disabled=engine.at_end):
            engine.tick()
        summary = engine.summary()
        cols = st.columns(3)
        cols[0].metric("Observations", summary["total_cases"])
        cols[1].metric("Engineer review", summary["escalated"])
        cols[2].metric("Returned to baseline", summary["auto_resolved"])
        st.progress((engine.current_idx + 1) / engine.total_samples)
        telemetry, cases, trace, training = st.tabs(["Telemetry", "Cases and source evidence", "Agent trace", "Training"])
        with telemetry:
            st.dataframe([{"timestamp": r.timestamp, "asset": r.asset_id, **r.metrics,
                           "raw event codes": ", ".join(r.event_codes)} for r in engine.dataset.telemetry],
                         hide_index=True)
        with cases:
            audit = engine.audit()
            st.dataframe(audit["cases"], hide_index=True)
            for case in engine.cases:
                with st.expander(case.case_id):
                    cloud.case_panel(engine, case)
                    st.json(engine.cloud_cases[case.case_id])
            st.download_button("Download device audit", json.dumps(audit, default=str, indent=2),
                               "prism-spectra-ai-device-audit.json", "application/json")
        with trace:
            st.dataframe(engine.cloud_events, hide_index=True)
        with training:
            st.json(engine.train_report)
            st.caption("Algorithm thresholds are statistical settings, not manufacturer limits. "
                       "No bundled device specifications, playbooks or economic assumptions are applied.")
        cloud.audit_panel(engine)
    panel()
