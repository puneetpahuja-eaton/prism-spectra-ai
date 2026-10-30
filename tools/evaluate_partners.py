"""Reproducible synthetic integration evaluation; no external API calls."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.update(FORCE_MOCK="1", USE_FIRESTORE="0", USE_SECRET_MANAGER="0",
                  AI_PROVIDER="nebius", EMBEDDING_PROVIDER="nebius", LANGSMITH_TRACING="false",
                  GCS_BUCKET="", BQ_DATASET="")

if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromNames([
        "tests.test_partner_workflow", "tests.test_nebius_client",
        "tests.test_embeddings", "tests.test_specta", "tests.test_device_grounding",
        "tests.test_public_research",
    ])
    started = time.monotonic()
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    failures = len(result.failures) + len(result.errors)
    assessed = result.testsRun - len(result.skipped)
    print(json.dumps({"evaluation": "synthetic integration and guardrails, not field accuracy",
                      "tests": result.testsRun, "failed": failures, "skipped": len(result.skipped),
                      "passed": assessed - failures, "pass_rate": (assessed - failures) / assessed if assessed else None,
                      "seconds": round(time.monotonic() - started, 2)}, indent=2))
    sys.exit(0 if result.wasSuccessful() else 1)
