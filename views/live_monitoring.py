"""Engineer/Admin: live REST monitoring with real-time agentic analysis."""
from __future__ import annotations

import ipaddress
import os
from urllib.parse import urlparse

import pandas as pd
import streamlit as st

import datastore
import devices
import rag
import secretmgr
import specta_ui
import ui_common
import research_ui
from nebius_client import is_live
from live import rest_adapter, sim_feed
from live.inference import suggest_field_map
from live.monitor import LiveMonitor


def _is_private_host(host: str) -> bool:
    if not host:
        return False
    if host == "localhost" or host.endswith(".local"):
        return True
    try:
        return ipaddress.ip_address(host).is_private
    except ValueError:
        return False


def _poll_failure_hint(error: Exception, config: dict) -> str | None:
    """Return an actionable hint for connection/timeout failures, else None."""
    text = str(error).lower()
    connection = any(s in text for s in (
        "timed out", "timeout", "max retries", "connection", "failed to establish",
        "name or service", "unreachable", "refused", "no route"))
    if not connection:
        return None
    host = ""
    try:
        host = urlparse(config.get("endpoint") or "").hostname or ""
    except ValueError:
        host = ""
    on_cloud = bool(os.getenv("K_SERVICE"))  # Cloud Run sets K_SERVICE.
    if _is_private_host(host) and on_cloud:
        return (f"**{host}** is a private network address, which this cloud deployment cannot reach. "
                "For a live demo, enable **Use built-in simulated feed** in *Configure connection* above. "
                "To poll this real device, run Prism-Spectra-AI on a machine on the same network as the device "
                "(for example locally), or expose the device through a public endpoint or VPN.")
    if _is_private_host(host):
        return (f"Couldn't reach **{host}**. Make sure this machine is on the same network as the device "
                "and the endpoint, TLS and credentials are correct. For a no-device demo, enable the "
                "built-in simulated feed in *Configure connection*.")
    return ("Couldn't connect to the device endpoint. Check the URL, network access and TLS/auth. "
            "For a no-device demo, enable **Use built-in simulated feed** in *Configure connection*.")


def _plan_builder(device: dict) -> None:
    """Chat-style generator that turns an instruction into a monitoring config."""
    device_id = device["id"]
    with st.expander("🧰 Build monitoring from an instruction (agent)"):
        st.caption("Describe what to monitor in plain language. The agent reads this device's docs to "
                   "resolve parameter IDs, polling interval, thresholds and the event-log decode. It asks "
                   "clarifying questions first — especially about how the event log is structured.")
        request = st.text_area("What should I monitor?", key=f"mp_req_{device_id}", height=90,
                               placeholder="Specify documented parameters, polling interval, limits and event-log layout.")
        ref_path = st.text_input("Reference a repo file for parameters (optional path)",
                                 key=f"mp_ref_{device_id}",
                                 placeholder="docs/interface.json")
        st.caption("Questions, replies, downloads and reviewed application all happen in Specta at the lower right.")
        if st.button("Generate plan", type="primary", key=f"mp_gen_{device_id}", disabled=not request.strip()):
            specta_ui.start_generation(device_id, "monitor", request, reference_path=ref_path)


def _render_prism_live(device_id: str, prism_key: str) -> None:
    """Render the live PRISM five-agent engine results (same engine as Analysis)."""
    if st.session_state.get(f"live_prism_err_{device_id}"):
        st.error("Deep analysis stopped: " + st.session_state[f"live_prism_err_{device_id}"])
        return
    prism = st.session_state.get(prism_key)
    if prism is None:
        return
    st.divider()
    st.subheader("🧠 PRISM five-agent analysis (live)")
    if prism.status == "warming":
        collected = len(prism._warmup_readings)
        needed = prism.warmup + 1
        st.progress(min(collected / needed, 1.0),
                    f"Learning the healthy baseline… {collected}/{needed} readings")
        st.caption("Once the baseline is established, each poll streams through the PRISM agents.")
        return
    summary = prism.summary() or {}
    cols = st.columns(5)
    cols[0].metric("Cases", summary.get("total_cases", 0))
    cols[1].metric("Auto-resolved", summary.get("auto_resolved", 0))
    cols[2].metric("Escalated", summary.get("escalated", 0))
    acc = summary.get("diagnosis_accuracy")
    cols[3].metric("Diagnosis acc.", f"{acc * 100:.0f}%" if acc is not None else "n/a")
    cols[4].metric("Cost avoided (est.)", f"${summary.get('cost_avoided_usd', 0):,.0f}")
    st.caption(f"Downtime avoided (estimated): {summary.get('downtime_avoided_min', 0)} min · "
               "live field data has no ground truth; no repair or economic benefit is inferred.")
    audit = prism.audit() or {}
    cases = audit.get("cases", [])
    if cases:
        st.dataframe([{"case": c.get("case_id"), "asset": c.get("root_asset_id"),
                       "fault": c.get("fault_type"), "decision": c.get("decision"),
                       "status": c.get("status"), "confidence": round(c.get("confidence", 0), 2)}
                      for c in reversed(cases)], hide_index=True, width="stretch")
    trace = audit.get("log", [])[-8:]
    if trace:
        with st.expander("Agent reasoning trace (live)"):
            st.dataframe([{"agent": e.get("agent"), "case": e.get("case_id"), "detail": e.get("message")}
                          for e in reversed(trace)], hide_index=True, width="stretch")




def _config_key(device_id: str) -> str:
    return f"live_cfg_{device_id}"


def _load_config(device_id: str) -> dict:
    stored = datastore.get_doc(devices.live_path(device_id), "config") or {}
    return stored


def _save_config(device_id: str, config: dict) -> None:
    datastore.set_doc(devices.live_path(device_id), "config", config)


def _monitored_list(config: dict) -> list[dict]:
    """Return the list of monitored params, migrating any legacy param_ids/thresholds."""
    monitored = [dict(item) for item in (config.get("monitored") or [])]
    if monitored:
        return monitored
    thresholds = config.get("thresholds") or {}
    for pid in config.get("param_ids") or []:
        limits = thresholds.get(f"p{pid}", {}) or {}
        monitored.append({"pid": str(pid), "lower": limits.get("lower"),
                          "upper": limits.get("upper"), "label": limits.get("label", "")})
    return monitored


def _persist_monitored(device_id: str, monitored: list[dict]) -> None:
    config = _load_config(device_id)
    config["monitored"] = monitored
    config["param_ids"] = [m["pid"] for m in monitored]
    config["thresholds"] = {
        f"p{m['pid']}": {k: v for k, v in
                         {"lower": m.get("lower"), "upper": m.get("upper"), "label": m.get("label")}.items()
                         if v not in (None, "")}
        for m in monitored if m.get("lower") is not None or m.get("upper") is not None or m.get("label")}
    _save_config(device_id, config)

def _render_config(device: dict) -> dict:
    device_id = device["id"]
    config = _load_config(device_id)
    st.subheader("Connection")
    if device["protocol"] != "rest":
        st.warning(f"This device's protocol is `{device['protocol']}`. Modbus and DLMS are **disabled** "
                   "in this edition — only REST polling is available.")

    simulated = st.toggle("Use built-in simulated feed (no device needed)",
                          value=config.get("simulated", False),
                          help="Explicit synthetic power-demo opt-in. It is not evidence about the selected device.")
    endpoint = st.text_input("REST endpoint (base — the param ID in the path is the template)",
                             value=config.get("endpoint", ""), disabled=simulated,
                             placeholder="https://device.example/your-documented-api")
    encoding = st.selectbox(
        "Hex value format", ["Unknown / numeric values only", "IEEE-754 float (32-bit)", "Unsigned integer"],
        index={None: 0, True: 1, False: 2}[config.get("hex_as_float")], disabled=simulated,
        help="Select only from your device code/docs or explicit guidance. Unknown hex data is rejected, not guessed.")
    hex_as_float = {"IEEE-754 float (32-bit)": True, "Unsigned integer": False}.get(encoding)

    # Monitored parameters: a clean inline table (add rows, edit, delete) — no cluttered buttons.
    monitored = _monitored_list(config)
    if not simulated:
        st.markdown("**Monitored parameters**")
        st.caption("Add a row per parameter: ID, optional lower/upper limits, a label, and (optional) the "
                   "JSON path to its value if the response structure differs. A value below the lower or "
                   "above the upper limit raises an issue with a grounded description. Use the + row to add "
                   "and the 🗑 to remove.")
        table = pd.DataFrame(
            [{"Parameter ID": m["pid"], "Lower": m.get("lower"), "Upper": m.get("upper"),
              "Label": m.get("label", ""), "Value path": m.get("path", "")} for m in monitored],
            columns=["Parameter ID", "Lower", "Upper", "Label", "Value path"])
        edited = st.data_editor(
            table, num_rows="dynamic", hide_index=True, width="stretch", key=f"params_table_{device_id}",
            column_config={
                "Parameter ID": st.column_config.TextColumn("Parameter ID", required=True, width="small"),
                "Lower": st.column_config.NumberColumn("Lower", format="%.4f", width="small"),
                "Upper": st.column_config.NumberColumn("Upper", format="%.4f", width="small"),
                "Label": st.column_config.TextColumn("Label", width="medium"),
                "Value path": st.column_config.TextColumn("Value path (optional)", width="small",
                                                          help="Default: value. Override if a param's "
                                                               "response nests the value differently."),
            })
        new_monitored: list[dict] = []
        seen: set[str] = set()
        for _, record in edited.iterrows():
            pid = str(record["Parameter ID"]).strip() if pd.notna(record["Parameter ID"]) else ""
            if not pid or pid in seen:
                continue
            seen.add(pid)
            new_monitored.append({
                "pid": pid,
                "lower": float(record["Lower"]) if pd.notna(record["Lower"]) else None,
                "upper": float(record["Upper"]) if pd.notna(record["Upper"]) else None,
                "label": str(record["Label"]).strip() if pd.notna(record["Label"]) else "",
                "path": str(record["Value path"]).strip() if pd.notna(record["Value path"]) else "",
            })
        if new_monitored != monitored:
            _persist_monitored(device_id, new_monitored)
        monitored = new_monitored
        if monitored:
            st.caption(f"Monitoring **{len(monitored)}** parameter(s) — each tracked and charted separately.")
        else:
            st.info("No parameters added yet. Add at least one row above.")

    param_ids = [m["pid"] for m in monitored]
    thresholds = {f"p{m['pid']}": {k: v for k, v in
                                   {"lower": m.get("lower"), "upper": m.get("upper"),
                                    "label": m.get("label")}.items() if v not in (None, "")}
                  for m in monitored if m.get("lower") is not None or m.get("upper") is not None or m.get("label")}

    col1, col2, col3 = st.columns(3)
    auth_modes = ["none", "basic", "digest", "bearer"]
    auth_mode = col1.selectbox("Auth", auth_modes,
                               index=auth_modes.index(config.get("auth_mode", "none"))
                               if config.get("auth_mode", "none") in auth_modes else 0,
                               disabled=simulated,
                               help="Select the authentication scheme documented for this device.")
    username = col2.text_input("Username", disabled=simulated or auth_mode not in ("basic", "digest"))
    password = col3.text_input("Password / token", type="password",
                               disabled=simulated or auth_mode == "none")
    interval = st.number_input("Poll interval (seconds)", min_value=1, max_value=86400,
                               value=int(config.get("interval", 10)), step=1, disabled=simulated,
                               help="How often to poll each parameter. Any value from 1 second upward.")

    with st.expander("🔒 TLS / certificate (for HTTPS field devices)"):
        st.caption("On-prem devices often use a self-signed or private-CA certificate. Prefer pinning the "
                   "device's CA certificate. Skipping verification is an explicit opt-in for a trusted "
                   "private-network device only.")
        ca_pem = st.text_area("Device CA certificate (PEM, optional)", value="", height=100,
                              disabled=simulated, placeholder="-----BEGIN CERTIFICATE-----\n...")
        verify_tls = not st.checkbox(
            "Skip TLS verification (insecure — private-network device only)",
            value=not config.get("verify_tls", True), disabled=simulated,
            help="Equivalent to curl -k. Only use for a device you trust on an internal network.")
        if not verify_tls:
            st.warning("TLS verification is OFF for this device. Use only on a trusted private network.")

    field_map = config.get("field_map", {})
    if not rest_adapter._PARAM_URL.search(endpoint) and monitored and all(m.get("path") for m in monitored):
        field_map = {f"p{m['pid']}": m["path"] for m in monitored}
    with st.expander("🤖 Advanced: agent field mapping (for non-/rs/param endpoints)"):
        st.caption("Optional. For a generic endpoint, paste a sample response and let the agent map fields "
                   "from the datasheet. Not needed for /rs/param devices — their values are auto-extracted.")
        sample = st.text_area("Sample response JSON", value=config.get("sample", ""), height=100,
                              disabled=simulated)
        if st.button("Ask agent to map fields", disabled=not sample.strip()):
            with st.spinner("Agent inferring field mapping from the datasheet..."):
                try:
                    suggestion = suggest_field_map(device_id, sample)
                except Exception as error:  # noqa: BLE001
                    st.error(str(error))
                else:
                    field_map = suggestion.get("field_map", {})
                    config["event_path"] = suggestion.get("event_path", "")
                    st.success("Agent proposed a field mapping.")
                    if suggestion.get("rationale"):
                        st.caption(suggestion["rationale"])
        if field_map:
            st.write("Field map (metric → JSON path):", field_map)

    if st.button("💾 Save connection"):
        secret_ref = config.get("secret_ref") or f"live_{device_id}"
        cred_saved = False
        if not simulated and auth_mode != "none" and password:
            value = f"{username}:{password}" if auth_mode in ("basic", "digest") else password
            secretmgr.put_secret(secret_ref, value)
            cred_saved = True
        ca_ref = config.get("ca_cert_ref") or f"live_ca_{device_id}"
        if not simulated and ca_pem.strip():
            secretmgr.put_secret(ca_ref, ca_pem.strip())
        config.update({"simulated": simulated, "endpoint": endpoint, "auth_mode": auth_mode,
                       "interval": interval, "field_map": field_map, "sample": sample,
                       "secret_ref": secret_ref, "verify_tls": verify_tls, "param_ids": param_ids,
                       "hex_as_float": hex_as_float, "thresholds": thresholds, "monitored": monitored,
                       "ca_cert_ref": ca_ref if (ca_pem.strip() or config.get("ca_cert_ref")) else None})
        _save_config(device_id, config)
        if not simulated and auth_mode != "none" and not cred_saved and not secretmgr.get_secret(secret_ref):
            st.warning("Saved, but no credentials are stored yet. Enter username/password and save again.")
        else:
            st.success("Saved connection configuration.")
    config["interval"] = interval
    config["simulated"] = simulated
    config["verify_tls"] = verify_tls
    config["param_ids"] = param_ids
    config["endpoint"] = endpoint
    config["hex_as_float"] = hex_as_float
    config["thresholds"] = thresholds
    config["monitored"] = monitored
    config["field_map"] = field_map
    return config


def _poll(device: dict, config: dict) -> dict:
    if config.get("simulated", False):
        state = st.session_state
        step_key = f"live_step_{device['id']}"
        state[step_key] = state.get(step_key, 0) + 1
        step = state[step_key]
        # Inject an anomaly burst periodically so the demo shows detection + resolution.
        return sim_feed.simulated_reading(step, inject_anomaly=(step % 12 in (0, 1)))
    return rest_adapter.poll("rest", config)


def render() -> None:
    ui_common.require_role("engineer")
    st.title("📡 Live Monitoring")
    st.caption("Agent polls the device over REST at a configurable interval and analyzes in real time.")

    device = ui_common.device_selector("live_device")
    if not device:
        return
    device_id = device["id"]

    _plan_builder(device)

    with st.expander("⚙️ Configure connection", expanded=not _load_config(device_id)):
        config = _render_config(device)

    mon_key = f"live_monitor_{device_id}"
    hist_key = f"live_hist_{device_id}"
    on_key = f"live_on_{device_id}"
    st.session_state.setdefault(on_key, False)
    prism_key = f"live_prism_{device_id}"
    deep = st.checkbox("🧠 Deep analysis (authorize Nebius interpretation of incident evidence)",
                       value=st.session_state.get(f"live_deep_{device_id}", False),
                       key=f"live_deep_{device_id}",
                       help="Uses the baseline you confirm and sends selected-device sources and incident readings "
                            "to Nebius in background jobs. Use only authorized, non-confidential data. "
                            "No Tavily request runs without separate public-only consent.")
    baseline_confirmed = False
    if deep and not config.get("simulated"):
        baseline_confirmed = st.checkbox(
            "I confirm the initial readings are an acceptable baseline for this device",
            key=f"live_baseline_{device_id}")
        st.caption("Warmup is not an automatic health certification. Unknown event codes remain raw.")
    if config.get("simulated"):
        st.warning("Synthetic demo feed selected. Its values, event meanings and behavior are not device evidence.")
    if (st.session_state[on_key] and
            st.session_state.get(f"live_run_simulated_{device_id}") != bool(config.get("simulated"))):
        st.session_state[on_key] = False
        st.session_state.pop(prism_key, None)
        st.session_state.pop(mon_key, None)
        st.info("The data source changed. Start again to avoid mixing synthetic and device readings.")

    col1, col2, col3 = st.columns(3)
    ready = bool(config.get("simulated") or config.get("endpoint", "").strip())
    if (not config.get("simulated") and config.get("monitored") and
            not rest_adapter._PARAM_URL.search(config.get("endpoint", "")) and
            any(not m.get("path") for m in config["monitored"])):
        ready = False
        st.warning("Generic endpoints require a documented JSON value path for every monitored parameter.")
    if not ready:
        st.info("Enter your device endpoint or explicitly enable the synthetic demo feed.")
    if col1.button("▶ Start", type="primary", disabled=st.session_state[on_key] or not ready or
                   (deep and not config.get("simulated") and not baseline_confirmed)):
        st.session_state.pop(f"live_prism_err_{device_id}", None)
        st.session_state[mon_key] = LiveMonitor(criticality=device["criticality"],
                                                thresholds={} if config.get("simulated") else config.get("thresholds") or {},
                                                demo_profile=bool(config.get("simulated")))
        if deep:
            from live_engine import LiveFleetEngine
            st.session_state[prism_key] = LiveFleetEngine(
                asset_id=device["name"] or device_id, criticality=device["criticality"],
                offline=not is_live(),
                device_id=device_id, asset_type=device["type"], demo_profile=bool(config.get("simulated")),
                baseline_confirmed=baseline_confirmed)
        else:
            st.session_state.pop(prism_key, None)
        st.session_state[hist_key] = []
        st.session_state[f"live_step_{device_id}"] = 0
        st.session_state[on_key] = True
        st.session_state[f"live_run_simulated_{device_id}"] = bool(config.get("simulated"))
        st.rerun()
    if col2.button("⏸ Stop", disabled=not st.session_state[on_key]):
        st.session_state[on_key] = False
        st.rerun()
    if col3.button("🔄 Reset"):
        for key in (mon_key, hist_key, on_key, prism_key, f"live_prism_err_{device_id}"):
            st.session_state.pop(key, None)
        st.rerun()

    interval = int(config.get("interval", 3))
    placeholder = st.container()

    @st.fragment(run_every=interval if st.session_state[on_key] else None)
    def live_panel() -> None:
        monitor: LiveMonitor = st.session_state.get(mon_key)
        if monitor is None:
            return
        if st.session_state.get(on_key):
            try:
                reading = _poll(device, config)
            except Exception as error:  # noqa: BLE001
                st.error(f"Poll failed: {error}")
                hint = _poll_failure_hint(error, config)
                if hint:
                    st.info(hint)
                    if not config.get("simulated", False) and st.button(
                            "▶ Switch to simulated feed and restart", key=f"live_sim_switch_{device_id}"):
                        cfg = _load_config(device_id)
                        cfg["simulated"] = True
                        _save_config(device_id, cfg)
                        st.session_state[mon_key] = LiveMonitor(
                            criticality=device["criticality"], demo_profile=True)
                        st.session_state.pop(prism_key, None)
                        st.session_state[hist_key] = []
                        st.session_state[on_key] = True
                        st.session_state[f"live_run_simulated_{device_id}"] = True
                        st.rerun()
                st.session_state[on_key] = False
                return
            new_issues = monitor.observe(reading)
            # Interpretation is explicit and off the polling path.
            for issue in new_issues:
                issue["description"] = ("Synthetic observation. " if config.get("simulated") else "") + \
                    "Review the incident evidence below. AI interpretation never changes polling limits or controls."
            st.session_state[hist_key].append({"timestamp": reading["timestamp"], **reading["metrics"]})
            st.session_state[hist_key] = st.session_state[hist_key][-120:]
            st.session_state[f"live_lastraw_{device_id}"] = reading.get("raw")
            st.session_state[f"live_lastmetrics_{device_id}"] = reading.get("metrics")
            prism = st.session_state.get(prism_key)
            if prism is not None:
                try:
                    prism.observe(reading)
                except Exception as error:  # noqa: BLE001
                    st.session_state[f"live_prism_err_{device_id}"] = str(error)
                    st.session_state.pop(prism_key, None)

        summary = monitor.summary()
        metric_cols = st.columns(5)
        metric_cols[0].metric("Readings", summary["readings"])
        metric_cols[1].metric("Issues", summary["issues"])
        metric_cols[2].metric("Auto-resolved", summary["auto_resolved"])
        metric_cols[3].metric("Escalated", summary["escalated"])
        metric_cols[4].metric("Cost avoided (est.)", f"${summary['cost_avoided_usd']:,.0f}")
        st.caption(f"Downtime avoided (estimated): {summary['downtime_avoided_min']} min · "
                   "no repair or avoided-loss value is inferred for real-device observations.")

        _render_prism_live(device_id, prism_key)

        history = st.session_state.get(hist_key, [])
        last_metrics = st.session_state.get(f"live_lastmetrics_{device_id}")
        if last_metrics is not None and not config.get("simulated", False) and not last_metrics:
            st.warning("The device responded but no numeric value was found. Expand the raw response "
                       "below, or set the value path for that parameter.")
        if history:
            frame = pd.DataFrame(history).set_index("timestamp")
            columns = [c for c in frame.columns if c != "timestamp"]
            if config.get("simulated", False):
                if columns:
                    st.line_chart(frame[columns])
            else:
                # One chart per parameter so different PIDs are never clubbed together.
                labels = {f"p{m['pid']}": (m.get("label") or f"p{m['pid']}")
                          for m in (config.get("monitored") or [])}
                groups: dict[str, list[str]] = {}
                for col in columns:
                    base = col.split(".", 1)[0]  # p<id> (multi-word words grouped with their param)
                    groups.setdefault(base, []).append(col)
                for base, cols in groups.items():
                    title = labels.get(base, base)
                    st.markdown(f"**{title}** (`{base}`)")
                    series = frame[cols].dropna(how="all")
                    latest = series[cols[0]].dropna()
                    if not latest.empty:
                        st.caption(f"latest: {latest.iloc[-1]:g}")
                    if len(series):
                        st.line_chart(series)
        if monitor.issues:
            st.subheader("Detected issues")
            _badge = {"ESCALATED": "🔴", "MONITORING": "🟠", "RESOLVED": "🟢"}
            for issue in reversed(monitor.issues):
                badge = _badge.get(issue["status"], "⚪")
                with st.container(border=True):
                    top = st.columns([3, 2, 2, 2])
                    top[0].markdown(f"{badge} **{issue.get('label', issue['metric'])}**")
                    top[1].write(f"value: {issue['value']}")
                    top[2].write(f"{issue['status']}")
                    top[3].write(f"{issue['decision']}")
                    st.caption(f"{issue.get('reason', '')}"
                               + (f" · est. cost avoided ${issue['cost_avoided_usd']:,.0f}"
                                  if issue.get("cost_avoided_usd") else ""))
                    if issue.get("description"):
                        st.write(issue["description"])
        raw = st.session_state.get(f"live_lastraw_{device_id}")
        if raw is not None and not config.get("simulated", False):
            with st.expander("🔎 Latest raw device response"):
                st.json(raw)

    with placeholder:
        live_panel()
    _incident_review(device_id, prism_key, mon_key)


def _incident_review(device_id: str, prism_key: str, mon_key: str) -> None:
    import json
    from prism_cloud import assess_device
    import jobs

    @st.fragment(run_every=2)
    def review():
        prism = st.session_state.get(prism_key)
        engine = prism.engine if prism else None
        if engine:
            engine.drain_ai()
            st.subheader("Incident evidence and AI review")
            for case in engine.cases[-6:]:
                with st.expander(case.case_id):
                    from prism_extensions import case_panel
                    case_panel(engine, case)
            return
        monitor = st.session_state.get(mon_key)
        if not monitor or not monitor.issues:
            return
        st.subheader("Selected incident review")
        # Object identity is session-local and changes on reset.
        index = st.selectbox("Incident", list(range(len(monitor.issues))),
                             format_func=lambda i: f"{i + 1}: {monitor.issues[i]['metric']}",
                             key=f"incident_{device_id}_{id(monitor)}")
        issue = monitor.issues[index]
        future = issue.get("_ai_future")
        if future and future.done():
            issue.pop("_ai_future")
            try:
                issue["_assessment"] = future.result()
            except (ValueError, RuntimeError, OSError) as error:
                issue["_ai_error"] = f"Interpretation failed: {error}"
        if st.button("Interpret selected incident with Nebius", disabled=not is_live() or bool(issue.get("_ai_future")),
                     key=f"interpret_{device_id}_{id(monitor)}_{index}"):
            evidence = {k: v for k, v in issue.items() if not k.startswith("_")}
            try:
                issue["_ai_future"] = jobs.submit(lambda: assess_device(device_id, evidence))
            except RuntimeError as error:
                issue["_ai_error"] = str(error)
        if issue.get("_ai_future"):
            st.info("Interpreting in the background; polling continues.")
        if issue.get("_ai_error"):
            st.error(issue["_ai_error"])
        assessment = issue.get("_assessment", {})
        if assessment:
            st.json({k: v for k, v in assessment.items() if k != "workflow"}, expanded=False)
            if assessment.get("workflow"):
                import workflow
                workflow.show_report(assessment["workflow"], key=f"live_trace_{device_id}_{id(monitor)}_{index}")
        research_ui.render(device_id, f"live:{id(monitor)}:{index}", surface="live",
                           device_evidence=[{"text": json.dumps({k: v for k, v in issue.items() if not k.startswith("_")})}]
                                           + assessment.get("sources", []),
                           on_result=lambda packet: issue.update(_public_research=packet))
    review()
