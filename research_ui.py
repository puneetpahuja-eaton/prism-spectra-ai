"""Shared session-scoped research controls, separate from the Public Research page."""
from __future__ import annotations

from concurrent.futures import Future
import hashlib
import json

import streamlit as st

import inline_research
from integrations import tavily
import jobs
import public_research
import ui_common
import workflow


def show(packet: dict, key: str) -> None:
    st.caption(f"Public evidence: {packet['status']} | retrieved {packet['created_at']}. "
               "Supplementary only; no decision, threshold or equipment action changed.")
    for error in packet.get("errors", []):
        st.error(error)
    comparison = packet.get("comparison")
    if comparison:
        st.markdown("**Nemotron evidence comparison**")
        st.caption("Bounded review: up to six supplied and six public excerpts, 1,200 characters each. "
                   "The download retains the public research packet; this is not an exhaustive source review.")
        for finding in comparison["findings"]:
            st.write(finding["statement"])
            for passage in finding["evidence"]:
                st.caption(f"{passage['id']}: {passage['kind']}")
                st.text(passage["text"])
                if passage.get("url"):
                    st.link_button("Open cited public source", passage["url"])
        for question in comparison["uncertainties"]:
            st.write("- " + question)
    for source in packet.get("sources", []):
        st.link_button(f"{source['id']}: {source['title']}", source["url"])
    st.dataframe(packet.get("trace", []), hide_index=True)
    st.download_button("Download research and comparison", json.dumps(packet, indent=2),
                       "public-evidence-comparison.json", "application/json", key=f"{key}_download", on_click="ignore")
    if packet.get("workflow"):
        workflow.show_report(packet["workflow"], key=f"{key}_workflow")


def render(device_id: str, target: str, *, surface: str, device_evidence: list[dict],
           on_result=None) -> None:
    if not ui_common.current_role() in ("admin", "engineer"):
        return
    digest = hashlib.sha256(f"{device_id}:{target}".encode()).hexdigest()[:20]
    storage_key = f"inline_research_{digest}"
    widget = f"{storage_key}_{surface}"
    state = st.session_state.setdefault(storage_key, {})

    @st.fragment(run_every=2 if state.get("future") else None)
    def panel():
        with st.expander("Enrich with approved public evidence"):
            st.caption("Tavily Search / optional Map / Extract, then Nemotron comparison. "
                       "Enter public terms yourself; private readings, code and chat are never copied into queries.")
            product = st.text_input("Public product name", key=f"{widget}_product")
            symptom = st.text_input("Public symptom", key=f"{widget}_symptom")
            domains = st.text_input("Approved public domains (comma-separated)", key=f"{widget}_domains")
            root = st.text_input("Optional public documentation URL for Map", key=f"{widget}_map")
            request = None
            fingerprint = "empty"
            if product.strip() and symptom.strip():
                try:
                    request = inline_research.PublicRequest(product=product, symptom=symptom,
                                                           domains_text=domains, map_url=root)
                    plan = public_research.research_plan(**request.model_dump())
                except ValueError as error:
                    st.error(str(error))
                    request = None
                else:
                    st.markdown("**Exact outbound request preview**")
                    st.json(plan, expanded=False)
                    fingerprint = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:16]
            approved = st.checkbox("I reviewed these public-only fields and authorize Tavily and Nebius briefing",
                                   key=f"{widget}_approve_{fingerprint}")
            future: Future | None = state.get("future")
            if future and future.done():
                state.pop("future")
                try:
                    packet = future.result()
                except (ValueError, RuntimeError, OSError) as error:
                    state["error"] = f"Research failed: {error}"
                else:
                    state["packet"] = packet
                    state["request"] = state["pending_request"]
                    if on_result:
                        on_result(packet)
                state.pop("pending_request", None)
                st.rerun()
            cached = fingerprint == state.get("request") and state.get("packet") is not None
            if st.button("Run approved research", key=f"{widget}_run",
                         disabled=not request or not approved or bool(future) or cached or not tavily.configured()):
                # Only approved fields reach the tool; model output cannot grant consent.
                role = ui_common.current_role()
                evidence = [dict(item) for item in device_evidence]
                state.pop("error", None)
                try:
                    state["future"] = jobs.submit(lambda: inline_research.run(
                        device_id, request, role=role, consent=True, device_evidence=evidence))
                    state["pending_request"] = fingerprint
                except RuntimeError as error:
                    state["error"] = str(error)
                st.rerun()
            if state.get("future"):
                st.info("Research is running in the background. Monitoring does not wait for the model.")
            if not tavily.configured():
                st.warning("Set TAVILY_API_KEY to enable research.")
            if cached:
                st.caption("This exact request already ran for this target. Reusing the session result avoids duplicate charges.")
                if st.button("Allow a fresh paid run", key=f"{widget}_refresh"):
                    state.pop("request", None)
                    st.rerun()
            if state.get("error"):
                st.error(state["error"])
            if state.get("packet"):
                st.caption("Last completed research packet for this target:")
                show(state["packet"], widget)
    panel()
