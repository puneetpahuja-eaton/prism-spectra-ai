"""Bounded LangChain stages and explicit metadata-only LangSmith export."""
from __future__ import annotations

from contextvars import ContextVar
from datetime import datetime, timezone
import logging
import os
import time
from typing import Callable, TypeVar
from uuid import uuid4

from langchain_core.runnables import RunnableLambda
from langsmith import Client
from langsmith.run_helpers import tracing_context
from langsmith.utils import LangSmithError
import requests

import netsetup

T = TypeVar("T")
_active: ContextVar["Workflow | None"] = ContextVar("workflow", default=None)
_STAGES = {"specta", "retrieve", "dataset", "monitor", "nebius_chat", "nebius_embeddings",
           "diagnosis", "public_research", "evidence_comparison", "evaluation"}
_USAGE = {"prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens", "credits"}


def tracing_enabled() -> bool:
    return os.getenv("LANGSMITH_TRACING", "false").lower() == "true" and os.getenv("FORCE_MOCK") != "1"


def trace_client() -> Client:
    endpoint = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    if endpoint not in {"https://api.smith.langchain.com", "https://eu.api.smith.langchain.com"}:
        raise ValueError("Use an official US/EU LangSmith API endpoint.")
    if not os.getenv("LANGSMITH_API_KEY"):
        raise ValueError("LANGSMITH_API_KEY is missing.")
    netsetup.configure_tls()
    return Client(api_url=endpoint, api_key=os.environ["LANGSMITH_API_KEY"],
                  workspace_id=os.getenv("LANGSMITH_WORKSPACE_ID") or None,
                  auto_batch_tracing=False, hide_inputs=True, hide_outputs=True,
                  omit_traced_runtime_info=True, timeout_ms=3000)


class Workflow:
    def __init__(self, name: str):
        if name not in _STAGES:
            raise ValueError("Unknown workflow stage.")
        self.report = {"id": str(uuid4()), "workflow": name, "orchestrator": "LangChain Core",
                       "stages": [], "tracing": "disabled", "privacy": "metadata only"}
        self._token = None

    def __enter__(self) -> "Workflow":
        self.started_at = datetime.now(timezone.utc)
        self._token = _active.set(self)
        return self

    def __exit__(self, error_type, error, traceback) -> None:
        _active.reset(self._token)
        failed = bool(error_type) or any(s["status"] == "failed" for s in self.report["stages"])
        partial = any(s["status"] == "partial" for s in self.report["stages"])
        self.report["status"] = "failed" if failed else "partial" if partial else "complete"
        if not tracing_enabled():
            return
        # Never pass original exceptions, model inputs, outputs or arbitrary metadata.
        now = datetime.now(timezone.utc)
        try:
            client = trace_client()
            client.create_run(
                name=self.report["workflow"], inputs={}, run_type="chain",
                id=self.report["id"], start_time=self.started_at, end_time=now,
                outputs={}, error="workflow_failed" if failed else None,
                project_name=os.getenv("LANGSMITH_PROJECT", "prism-spectra-ai"),
                extra={"metadata": {"orchestrator": "LangChain Core",
                                    "privacy": "metadata only", "stages": self.report["stages"]}})
            self.report["tracing"] = "submitted"
        except (LangSmithError, requests.RequestException, ValueError, OSError):
            self.report["tracing"] = "failed"
            logging.getLogger(__name__).warning("LangSmith metadata export failed; application result retained.")


def run_step(name: str, function: Callable[[], T]) -> T:
    if name not in _STAGES:
        raise ValueError("Unknown workflow stage.")
    current = _active.get()
    if current is None:
        with Workflow(name):
            return run_step(name, function)
    event = {"stage": name, "status": "running"}
    current.report["stages"].append(event)
    start = time.monotonic()
    try:
        # Explicit tracing only. No LangChain/global callback may capture device text.
        with tracing_context(enabled=False):
            result = RunnableLambda(lambda _: function(), name=name).invoke(
                None, config={"callbacks": [], "recursion_limit": 8})
        event["status"] = "succeeded"
        if isinstance(result, dict):
            if result.get("status") in ("error", "failed", "no_evidence"):
                event["status"] = "failed"
            elif result.get("status") == "partial":
                event["status"] = "partial"
            usage = result.get("usage")
            if isinstance(usage, dict):
                event["usage"] = {key: value for key, value in usage.items()
                                  if key in _USAGE and type(value) in (int, float)}
            trace = result.get("trace")
            if isinstance(trace, list):
                reported = [item["credits"] for item in trace if isinstance(item, dict)
                            and type(item.get("credits")) in (int, float)]
                if reported:
                    event["usage"] = {"credits": sum(reported)}
        return result
    finally:
        if event["status"] == "running":
            event["status"] = "failed"
        event["seconds"] = round(time.monotonic() - start, 3)


def model_usage(data: dict, model: str) -> None:
    current = _active.get()
    if current and current.report["stages"]:
        # Only known model IDs, never a free-form environment value.
        if model.startswith(("nvidia/", "Qwen/")) and all(c.isalnum() or c in "/._-" for c in model):
            current.report["stages"][-1]["model"] = model
        usage = data.get("usage")
        if isinstance(usage, dict):
            current.report["stages"][-1]["usage"] = {
                key: value for key, value in usage.items() if key in _USAGE and type(value) in (int, float)}
            details = usage.get("completion_tokens_details")
            if isinstance(details, dict) and type(details.get("reasoning_tokens")) in (int, float):
                current.report["stages"][-1]["usage"]["reasoning_tokens"] = details["reasoning_tokens"]


def recovered_output_budget() -> None:
    current = _active.get()
    if current:
        for event in reversed(current.report["stages"]):
            if event["stage"] == "nebius_chat" and event["status"] == "failed":
                event.update(status="recovered", recovery="larger_output_budget")
                return


def show_report(report: dict, *, key: str) -> None:
    import json
    import streamlit as st

    with st.expander("Workflow provenance and service usage"):
        st.caption("LangChain orchestration. LangSmith receives metadata only, never prompts or device text.")
        st.caption(f"LangSmith: {report.get('tracing', 'disabled')}. Submission is not proof of dashboard ingestion.")
        if report.get("tracing") == "failed":
            st.warning("LangSmith export failed. The application result is retained; check tracing credentials.")
        st.dataframe(report.get("stages", []), hide_index=True)
        st.download_button("Download workflow metadata", json.dumps(report, indent=2),
                           "workflow-metadata.json", "application/json", key=key, on_click="ignore")
