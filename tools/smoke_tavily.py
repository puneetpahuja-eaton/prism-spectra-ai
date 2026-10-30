"""Verify Tavily + Nemotron using public product terms, never private device data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from public_research import run_research


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", action="store_true", help="Also test bounded public documentation discovery.")
    parser.add_argument("--output", type=Path, help="Optional local evidence-packet path (contains public excerpts).")
    args = parser.parse_args()
    packet = run_research(
        product="Eaton 5PX UPS", symptom="battery alarm", domains_text="eaton.com",
        map_url="https://www.eaton.com/us/en-us/support.html"
        if args.map else "",
        consent=True,
    )
    print(json.dumps({
        "status": packet["status"], "sources": len(packet["sources"]),
        "brief_source": (packet["brief"] or {}).get("_source"),
        "runtime_trace": packet["trace"], "errors": packet["errors"],
    }, indent=2))
    if args.output:
        args.output.write_text(json.dumps(packet, indent=2), encoding="utf-8")
    expected = {"search", "extract", "map"} if args.map else {"search", "extract"}
    successful = {event["endpoint"] for event in packet["trace"] if event["status"] == "succeeded"}
    if packet["status"] != "complete" or not expected.issubset(successful) or not packet["sources"]:
        raise RuntimeError("Live Tavily workflow verification failed; inspect the explicit status/errors.")
    if args.map and not packet["discovered_urls"]:
        raise RuntimeError("Map returned no usable discovery links; try another approved public starting page.")
    if (not packet["brief"] or not packet["brief"]["findings"]
            or not packet["brief"].get("_source", "").startswith("nebius:")):
        raise RuntimeError("Tavily calls succeeded but the cited Nebius brief was not verified.")


if __name__ == "__main__":
    main()
