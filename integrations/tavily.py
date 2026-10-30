"""Bounded Tavily runtime requests; never log API keys or response bodies."""
from __future__ import annotations

from datetime import datetime, timezone
import os
import time

import requests

import netsetup
import secretmgr


class TavilyUnavailable(RuntimeError):
    pass


def configured() -> bool:
    return bool(os.getenv("TAVILY_API_KEY") or secretmgr.get_secret("tavily_api_key"))


class Client:
    def __init__(self) -> None:
        self.trace: list[dict] = []

    def _post(self, endpoint: str, payload: dict) -> dict:
        key = os.getenv("TAVILY_API_KEY") or secretmgr.get_secret("tavily_api_key")
        if not key:
            raise TavilyUnavailable("Set TAVILY_API_KEY locally or save a Tavily key in Integrations.")
        if len(self.trace) >= 5:
            raise TavilyUnavailable("This research run reached its five-request Tavily limit.")
        netsetup.configure_tls()
        event = {
            "provider": "tavily", "endpoint": endpoint, "status": "started",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        self.trace.append(event)
        start = time.monotonic()
        try:
            response = requests.post(
                f"https://api.tavily.com/{endpoint}",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={**payload, "include_usage": True},
                timeout=(10, 90),
                allow_redirects=False,
            )
        except requests.RequestException as error:
            event.update(status="failed", seconds=round(time.monotonic() - start, 2))
            raise TavilyUnavailable(f"Tavily {endpoint} connection failed; check connectivity and retry.") from error
        event.update(http_status=response.status_code, seconds=round(time.monotonic() - start, 2))
        if response.status_code != 200:
            event["status"] = "failed"
            hint = {
                401: "Check that the Tavily API key is valid.",
                403: "Check account permissions or an organization network restriction.",
                429: "Rate or credit limit reached; check your Tavily account.",
                432: "Plan credit limit reached; check your Tavily account.",
                433: "Pay-as-you-go limit reached; check your Tavily account.",
            }.get(response.status_code, "Retry later or check your Tavily account.")
            raise TavilyUnavailable(f"Tavily {endpoint} returned HTTP {response.status_code}. {hint}")
        try:
            data = response.json()
        except ValueError as error:
            event["status"] = "failed"
            raise TavilyUnavailable(f"Tavily {endpoint} returned invalid JSON.") from error
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            event["status"] = "failed"
            raise TavilyUnavailable(f"Tavily {endpoint} returned an invalid results response.")
        event.update(status="succeeded", results=len(data["results"]))
        request_id = data.get("request_id")
        if isinstance(request_id, str):
            event["request_id"] = request_id[:200]
        usage = data.get("usage")
        credits = usage.get("credits") if isinstance(usage, dict) else None
        if type(credits) in (int, float):
            event["credits"] = credits
        return data

    def search(self, query: str, domains: list[str], *, recent: bool = False) -> dict:
        payload = {
            "query": query, "topic": "general", "search_depth": "advanced",
            "max_results": 3, "include_domains": domains,
            "include_answer": False, "include_raw_content": False,
        }
        if recent:
            payload["time_range"] = "year"
        return self._post("search", payload)

    def extract(self, urls: list[str], query: str) -> dict:
        if not 1 <= len(urls) <= 4:
            raise ValueError("Extract requires between one and four approved public URLs.")
        return self._post("extract", {
            "urls": urls, "extract_depth": "advanced", "format": "text",
            "query": query, "chunks_per_source": 3, "timeout": 60,
        })

    def map(self, url: str, product: str) -> dict:
        return self._post("map", {
            "url": url, "max_depth": 1, "max_breadth": 5, "limit": 5,
            "allow_external": False,
            "instructions": f"Find public documentation, manuals, support and safety advisories for {product}.",
            "timeout": 60,
        })
