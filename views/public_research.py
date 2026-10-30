"""Engineer-operated public evidence research with explicit outbound consent."""
from __future__ import annotations

import json

import streamlit as st

from integrations import tavily
import public_research
import ui_common


def _show_packet(packet: dict) -> None:
    st.subheader("Public evidence packet")
    st.write(f"Status: **{packet['status']}** · Retrieved at: {packet['created_at']}")
    st.caption("Session-only result, not a background monitor. Run again to refresh; each run makes new API calls.")
    for error in packet["errors"]:
        st.error(error)
    brief = packet.get("brief")
    by_id = {source["id"]: source for source in packet["sources"]}
    if brief:
        st.caption(f"Brief model: {brief.get('_source', 'unknown')}. Quotes checked against retrieved text.")
        for finding in brief["findings"]:
            st.write(finding["statement"])
            for citation in finding["evidence"]:
                source = by_id[citation["source_id"]]
                st.text(f"{source['id']}: {citation['quote']}")
                st.link_button(f"Open {source['id']} source", source["url"])
        if not brief["findings"]:
            st.warning("The model found no supported findings. Do not treat this as an all-clear.")
        st.markdown("**Unresolved applicability and coverage**")
        for question in brief["open_questions"]:
            st.write("- " + question)
        st.markdown("**Questions for the qualified engineer (not repair instructions)**")
        for question in brief["engineer_review_questions"]:
            st.write("- " + question)
    else:
        st.warning("No validated model brief is available. The evidence and runtime trace remain visible below.")
    st.subheader("Evidence and coverage")
    st.dataframe(packet["searches"], hide_index=True, width="stretch")
    for source in packet["sources"]:
        with st.expander(f"{source['id']} | {source['title']} | {source['kind']}"):
            st.link_button("Open public source", source["url"])
            st.caption("Found for: " + ", ".join(source["intents"]))
            st.text(source["text"])
    if packet["discovered_urls"]:
        with st.expander("Tavily Map: public documentation discovered"):
            for url in packet["discovered_urls"]:
                st.link_button(url, url)
    elif packet["plan"]["map_url"]:
        st.warning("Map found no approved documentation links beyond the starting page. "
                   "Try a public support/documentation index; discovery coverage is unknown.")
    st.subheader("Runtime API trace")
    st.caption("Real Tavily HTTP requests from this run only. Credits are shown only if reported by Tavily. "
               "Trace metadata supports debugging; it is not independent proof of award eligibility.")
    st.dataframe(packet["trace"], hide_index=True, width="stretch")
    for limitation in packet["limitations"]:
        st.caption(limitation)
    st.download_button("Download evidence packet and runtime trace",
                       json.dumps(packet, indent=2), "prism-spectra-ai-public-evidence.json", "application/json",
                       key="research_download", on_click="ignore")


def render() -> None:
    ui_common.require_role("engineer")
    st.title("Public Research")
    st.caption("Tavily Search + Map + Extract -> Nemotron evidence brief -> engineer-reviewed case attachment")
    st.info("Private device knowledge and public web research are separate. No device documents, telemetry, "
            "site, IP, chat history or case details are automatically sent to Tavily. Review the public fields "
            "below before every run. Model briefing sends the public excerpts to the selected AI provider.")
    if not tavily.configured():
        st.warning("Tavily is not configured. Set TAVILY_API_KEY in this project's .env and restart, "
                   "or ask an admin to save it in Integrations. Do not paste API keys into search fields.")
    device = ui_common.device_selector("research_device")
    if not device:
        return
    device_id = device["id"]
    st.caption("The device selection associates the packet locally; it does not populate outbound fields.")
    product = st.text_input("Public manufacturer and model", placeholder="Eaton 5PX UPS", key="research_product")
    symptom = st.text_input("Public symptom or fault category", placeholder="battery alarm", key="research_symptom")
    domains = st.text_input("Approved public source domains (comma-separated)",
                           value="eaton.com", key="research_domains")
    discover = st.checkbox("Discover official documentation using Tavily Map", key="research_map_enabled")
    root = st.text_input("Public documentation starting URL (HTTPS, no query parameters)",
                         placeholder="https://www.eaton.com/us/en-us/support.html",
                         key="research_map_url") if discover else ""
    plan = None
    if product.strip() and symptom.strip():
        try:
            plan = public_research.research_plan(product, symptom, domains, root)
            if discover and not root.strip():
                raise ValueError("Provide a public starting URL to enable Map.")
        except ValueError as error:
            st.error(str(error))
            plan = None
        else:
            st.markdown("**Exact outbound request preview**")
            st.json(plan, expanded=True)
    st.caption("Bounded usage: 3 advanced searches, at most 1 Map request (5 links), "
               "and 1 Extract request (up to 4 pages). No silent retries or periodic polling. "
               "An API request can consume credits even if evidence is not useful.")
    consent = st.checkbox("I reviewed these public-only terms, domains and URLs, authorize the Tavily calls "
                          "and public-evidence model briefing, and confirm no confidential information is included.",
                          key="research_consent")
    if st.button("Research public evidence", type="primary",
                 disabled=not plan or not consent or not tavily.configured()):
        with st.spinner("Searching public evidence, discovering docs, extracting passages and validating citations..."):
            try:
                packet = public_research.run_research(product=product, symptom=symptom, domains_text=domains,
                                                       map_url=root, consent=consent)
            except ValueError as error:
                st.error(str(error))
            else:
                st.session_state[f"research_packet_{device_id}"] = packet
    packet = st.session_state.get(f"research_packet_{device_id}")
    if not packet:
        return
    _show_packet(packet)
    context = st.session_state.get("research_case_context")
    engine = st.session_state.get("engine")
    if context and context["device_id"] == device_id:
        st.subheader("Attach to selected analysis case")
        st.caption(f"Local case reference: {context['case_id']}. Case details were not used as search terms.")
        current = (engine is not None and st.session_state.get("prism_device") == device_id
                   and engine.run_id == context["run_id"])
        if not current:
            st.warning("The analysis run changed. Select the case again in Analysis before attaching.")
        elif st.button("Attach evidence packet (no action or policy changes)",
                       disabled=not packet["sources"] or packet["status"] == "failed"):
            try:
                public_research.attach_to_case(engine, context["case_id"], packet)
            except ValueError as error:
                st.error(str(error))
            else:
                st.success("Attached to the current case's audit evidence. No decision, action or approval changed.")
