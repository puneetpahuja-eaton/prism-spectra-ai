# Kairos AI | Evidence-to-action industrial reliability copilot

## Problem

Industrial teams receive alarms and measurements, but interpreting a fault
still requires finding the right device documentation, comparing readings, and
preparing a useful engineer handoff. This prototype focuses on that triage
bottleneck for power and industrial equipment.

## Product

Kairos AI is a role-based, multi-device workspace. It ingests device manuals,
procedures, notes, and optional source links; supports Ask / Plan / Agent modes;
and analyzes synthetic, uploaded, historical, or REST-polled telemetry. Its
five-stage workflow connects deterministic anomaly evidence to a grounded model
hypothesis, policy gate, human review output, and later-sample verification.

## Nebius and NVIDIA use

The default model path makes a runtime call to Nebius Token Factory's
OpenAI-compatible inference API with `nvidia/Nemotron-3_5-Lightning`. The model
supports structured diagnosis and assistant-generation features. Responses
must validate against application schemas, and diagnosis citations are checked
against supplied SOP passages. The app records the actual provider/model source.
If the Nebius request fails or output is invalid, the result is unavailable and
requires review; the app does not silently substitute another model.

## Safety and execution

Telemetry anomaly detection and remediation decisions are not delegated to the
language model. Deterministic rules enforce confidence, reversibility, and
criticality safeguards. The actuator prepares review-only work orders or
human-approved sandbox simulations. No real equipment command or external
work-order dispatch is implemented.

## Infrastructure and data

The app is a Streamlit service. The hackathon copy retains optional Google Cloud
Run hosting and Google persistence services (Firestore, Cloud Storage, BigQuery,
Secret Manager). Qwen3-Embedding-8B on Token Factory supplies document/query
embeddings; NVIDIA Nemotron supplies reasoning. All demo inputs are synthetic; uploaded
real data should be authorized and non-confidential.

## Impact hypothesis and limits

Device-specific citations and a prepared incident brief could reduce the time
spent collecting context and improve handoff completeness. This is a hypothesis
to test in a pilot, not a measured result. The demo reports no production
savings; counterfactual downtime/cost values are assumption-based estimates.
The prototype is not validated predictive maintenance.

## Existing-project work

Kairos AI existed before the event. The Nebius-specific updates made during the
submission period are described in
[`prior-work-disclosure.md`](prior-work-disclosure.md).
