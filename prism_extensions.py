"""Additive cloud, input and audit panels; PRISM's six original tabs stay intact."""
from __future__ import annotations

from dataclasses import asdict
from datetime import timezone
import json

import streamlit as st
from google.api_core.exceptions import GoogleAPICallError
from google.auth.exceptions import GoogleAuthError

import bq
import auth
from fleet_data import FleetAsset, FleetDataset, Reading, csv_template
from nebius_client import backend, is_live
from prism_cloud import Engine, replacement
import storage
import ui_common
import research_ui
import workflow

CLOUD_ERRORS = (GoogleAPICallError, GoogleAuthError, RuntimeError, ValueError, OSError, TimeoutError)


def _mode_changed() -> None:
    # Do not discard uploaded data or implicitly authorize a new cloud mode.
    engine = st.session_state.get("engine")
    if engine and engine.source == "uploaded" and not st.session_state["prism_offline"] and not st.session_state.get("prism_consent"):
        st.session_state["prism_offline"] = True
        st.session_state["prism_notice"] = f"Authorize processing by {backend()} before enabling uploaded telemetry."
        return
    st.session_state.pop("engine", None)
    st.session_state["playing"] = False
    if engine:
        st.session_state["prism_pending_dataset"] = engine.dataset


def create_engine() -> Engine:
    dataset = st.session_state.get("prism_pending_dataset")
    return Engine(offline=st.session_state.get("prism_offline", not is_live()), dataset=dataset,
                  device_id=st.session_state.get("prism_device"),
                  demo_profile=st.session_state.get("prism_demo_profile", False))


def tick(engine: Engine) -> None:
    live = st.empty()
    def update(event: dict) -> None:
        live.caption(f"Analyzing sample {event['step'] + 1} | {event['agent']} | "
                     f"{event['tool']} | {event['detail']} | {event['result'][:180]}")
    engine.tick(on_event=update)
    live.empty()


def sidebar(engine: Engine) -> None:
    with st.sidebar:
        with st.expander("Hosting & data source", expanded=False, key="prism_cloud_panel"):
            st.toggle("Explicit offline simulation", value=engine.offline, key="prism_offline", on_change=_mode_changed)
            st.write("Reasoning:", "offline rules (not a model call)" if engine.offline else backend())
            st.write("Documents & scenarios:", storage.backend())
            st.write("Audit:", "BigQuery" if bq.is_enabled() else "session only")
            st.write("Telemetry:", engine.source)
            st.caption("Automatic actions are synthetic simulator state changes only. No equipment commands or ticket dispatch.")
            if "prism_notice" in st.session_state:
                st.warning(st.session_state["prism_notice"])


def template(engine: Engine) -> FleetDataset:
    if engine.dataset is not None:
        return engine.dataset.model_copy(deep=True)
    assets = [FleetAsset(id=asset.asset_id, name=asset.name, site=asset.site, rated_kw=asset.rated_kw,
                         type=asset.type, parent_id=asset.parent_id, criticality=asset.criticality.lower())
              for asset in engine.topo.assets.values()]
    history, telemetry = [], []
    # Forty clean samples plus the original replay keep the template below 2 MB.
    for index in range(40):
        fleet = engine.signal.sample_fleet(index, 0, [])
        for asset_id, sample in fleet.items():
            from prism import signals
            history.append(Reading(timestamp=signals.timestamp_for(index).replace(tzinfo=timezone.utc),
                                   asset_id=asset_id, metrics={key: value for key, value in sample["metrics"].items()
                                                             if isinstance(value, (int, float))},
                                   event_codes=sample["codes"]))
    for index in range(engine.total_samples):
        fleet = engine.baseline_sample(index)
        for asset_id, sample in fleet.items():
            telemetry.append(Reading(timestamp=engine.timestamp(index).replace(tzinfo=timezone.utc),
                                     asset_id=asset_id, metrics={key: value for key, value in sample["metrics"].items()
                                                               if isinstance(value, (int, float))},
                                     event_codes=sample["codes"]))
    return FleetDataset(source="synthetic", assets=assets, history=history, telemetry=telemetry)


def data_panel(engine: Engine) -> None:
    with st.expander("Replace telemetry / download original fleet fields", expanded=False, key="prism_data_panel"):
        st.caption("Original fields remain available: asset_id, name, type, parent_id, site, rated_kw, criticality. "
                   "Use the JSON/CSV template to replace the whole fleet. Timestamps require a timezone; "
                   "at least 20 known-healthy readings per asset, complete replay snapshots, max 2 MB. "
                   "Uploads are observation-only; they never trigger synthetic or physical control.")
        if st.button("Prepare PRISM telemetry templates"):
            st.session_state["prism_template"] = template(engine)
        prepared = st.session_state.get("prism_template")
        if prepared is not None:
            st.download_button("Download PRISM fleet JSON", prepared.model_dump_json(indent=2), "prism-fleet.json", "application/json",
                               on_click="ignore")
            st.download_button("Download PRISM fleet CSV", csv_template(prepared), "prism-fleet.csv", "text/csv",
                               on_click="ignore")
        upload = st.file_uploader("Replacement telemetry CSV / JSON (authorized non-confidential data only)", type=["csv", "json"],
                                  key="prism_upload")
        st.checkbox(f"I authorize sending this uploaded telemetry to {backend()}.", key="prism_consent")
        if st.button("Replace entire fleet", disabled=upload is None):
            try:
                if upload.size > 2_000_000:
                    raise ValueError("Upload must be under 2 MB.")
                if not engine.offline and not st.session_state.get("prism_consent"):
                    raise ValueError(f"Authorize processing by {backend()} or choose explicit offline simulation.")
                data = replacement(upload.getvalue().decode("utf-8-sig"), upload.name)
                with st.spinner("Validating replacement and training per-asset PRISM models..."):
                    replacement_engine = Engine(offline=engine.offline, dataset=data,
                                                device_id=st.session_state.get("prism_device"))
            except CLOUD_ERRORS as error:
                st.error(f"Replacement rejected; existing fleet is unchanged: {error}")
            else:
                st.session_state["engine"] = replacement_engine
                st.session_state["prism_pending_dataset"] = data
                st.session_state.pop("prism_template", None)
                st.session_state["playing"] = False
                for key in ("live_asset", "live_metric", "cf_metric", "cf_asset"):
                    st.session_state.pop(key, None)
                st.rerun()
        if st.button("Restore original PRISM simulator"):
            try:
                with st.spinner("Restoring original PRISM fleet and scenarios..."):
                    replacement_engine = Engine(offline=engine.offline)
            except CLOUD_ERRORS as error:
                st.error(f"Restore failed; existing fleet is unchanged: {error}")
            else:
                st.session_state["engine"] = replacement_engine
                st.session_state.pop("prism_pending_dataset", None)
                st.session_state.pop("prism_template", None)
                st.session_state["playing"] = False
                for key in ("live_asset", "live_metric", "cf_metric", "cf_asset"):
                    st.session_state.pop(key, None)
                st.rerun()
    with st.expander("Original topology and scenario fields", key="prism_topology_fields"):
        st.dataframe([{key: value for key, value in asdict(asset).items() if key != "children"}
                      for asset in engine.topo.assets.values()], hide_index=True, width="stretch")
        st.dataframe(engine._scenarios, hide_index=True, width="stretch")


def case_panel(engine: Engine, case) -> None:
    detail = engine.cloud_cases.get(case.case_id, {})
    for error in detail.get("errors", []):
        st.error(f"Analysis failed; this is not a successful model result: {error}")
    diagnosis = detail.get("diagnosis")
    if diagnosis:
        st.markdown(f"**Model / offline hypothesis:** {diagnosis['root_cause']}")
        st.caption(f"Source: {diagnosis['_source']} | Model self-confidence: {diagnosis['confidence']:.2f} "
                   "(separate from PRISM's original heuristic Confidence field)")
        st.write(diagnosis["reasoning"])
        for identifier, quote in diagnosis["citation_quotes"].items():
            st.info(f'{identifier}: "{quote}"')
    if detail.get("work_order"):
        st.markdown("**Review-only work-order draft (not dispatched):**")
        st.json(detail["work_order"], expanded=False)
    if detail.get("ai_status") == "pending":
        st.info("Nebius interpretation is pending; observations continue.")
    if detail.get("workflow"):
        workflow.show_report(detail["workflow"], key=f"case_workflow_{engine.run_id}_{case.case_id}")
    if auth.can_operate(ui_common.current_role()):
        if st.button("Research public guidance for this case", key=f"public_research_{case.case_id}"):
            st.session_state["research_case_context"] = {
                "device_id": st.session_state.get("prism_device"), "run_id": engine.run_id,
                "case_id": case.case_id,
            }
            st.info("Open Public Research in the sidebar. Enter and review public product/fault terms there; "
                    "this case's private details are not automatically sent.")
        if engine.device_id:
            evidence = [{"text": json.dumps(detail.get("evidence", {}))}] + detail.get("sources", [])
            research_ui.render(engine.device_id, f"{engine.run_id}:{case.case_id}", surface="case",
                               device_evidence=evidence,
                               on_result=lambda packet: engine.cloud_cases[case.case_id].update(public_research=packet))
    if detail.get("public_research"):
        packet = detail["public_research"]
        st.caption(f"Public research attached: {packet['id']} ({packet['status']}). "
                   "Supplementary evidence only; it did not alter this decision.")
        st.download_button("Download attached public evidence", json.dumps(packet, indent=2),
                           "case-public-evidence.json", "application/json",
                           key=f"public_packet_{case.case_id}", on_click="ignore")
    st.caption("The planned action describes a synthetic playbook, not an equipment instruction." if engine.demo_profile
               else "Observation-only handoff. No device action, ticket dispatch or repair has been performed.")


def _audit_rows(engine: Engine) -> list[dict]:
    result = []
    for case in engine.cases:
        detail = engine.cloud_cases.get(case.case_id, {})
        diagnosis = detail.get("diagnosis") or {}
        result.append({
            "ts": datetime_now(), "case_id": f"{engine.run_id}:{case.case_id}", "revision": engine.current_idx + 1,
            "asset_id": case.root_asset_id, "root_cause": diagnosis.get("root_cause", case.fault_type),
            "cited_sop_ids": ",".join(diagnosis.get("cited_sop_ids", [])), "decision": case.decision,
            "verification": case.status, "confidence": case.confidence,
            "source": diagnosis.get("_source", "unavailable"), "mode": f"PRISM / {engine.source}",
            "elapsed_seconds": 0, "approved": False, "reviewer": None,
            "potential_usd": int(case.cost_avoided_usd), "errors": "; ".join(detail.get("errors", [])),
        })
    return result


def datetime_now() -> str:
    from datetime import datetime
    return datetime.now(timezone.utc).isoformat()


def audit_panel(engine: Engine) -> None:
    with st.expander("Cloud audit & full agent trace", expanded=False, key="prism_audit_panel"):
        st.download_button("Download PRISM audit JSON", json.dumps(engine.audit(), indent=2, default=str),
                           "prism-audit.json", "application/json", on_click="ignore")
        st.caption("BigQuery stores case-summary revisions; the full tool trace is available in the download.")
        consent = engine.source == "synthetic" or st.checkbox("I authorize storing uploaded case summaries in BigQuery.", key="prism_bq_consent")
        if st.button("Save PRISM audit to BigQuery", disabled=not bq.is_enabled() or not consent):
            rows = _audit_rows(engine)
            try:
                if not rows:
                    raise ValueError("No cases to save yet.")
                errors = bq.client().insert_rows_json(bq.table_id(), rows,
                                                     row_ids=[f"{row['case_id']}:{row['revision']}" for row in rows])
                if errors:
                    raise RuntimeError(f"BigQuery rejected audit rows: {errors}")
            except CLOUD_ERRORS as error:
                st.session_state["prism_audit_notice"] = (engine.run_id, False, f"Audit save failed: {error}")
            else:
                st.session_state["prism_audit_notice"] = (engine.run_id, True, f"Saved {len(rows)} PRISM case summaries to BigQuery.")
        notice = st.session_state.get("prism_audit_notice")
        if notice and notice[0] == engine.run_id:
            (st.success if notice[1] else st.error)(notice[2])
        if st.button("Refresh PRISM audit history", disabled=not bq.is_enabled()):
            try:
                st.session_state["prism_audit_history"] = bq.recent_history()
            except CLOUD_ERRORS as error:
                st.error(f"History query failed: {error}")
        if "prism_audit_history" in st.session_state:
            st.dataframe(st.session_state["prism_audit_history"], hide_index=True, width="stretch")
