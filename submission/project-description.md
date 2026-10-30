# Kairos AI — Evidence-to-action industrial reliability copilot

Industrial alarms say what changed, but engineers still have to connect the
telemetry to the right device documentation, decide what is safe to do, and
prepare a useful handoff. Kairos AI brings those steps into one reliability
workspace for plant operators and reliability engineers.

An operator selects a device, loads its manuals and procedures, and streams
synthetic, uploaded, or REST-polled telemetry. The analysis workflow builds
deterministic evidence, asks **NVIDIA Nemotron-3.5-Lightning through Nebius
Token Factory** for a grounded diagnosis, validates the response and SOP
citations, applies deterministic safety rules, and then verifies later readings.
It can prepare a review-only work-order draft or demonstrate a human-approved
sandbox simulation; it never sends commands to real equipment.

The product also includes an Ask / Plan / Agent assistant, device-specific
retrieval, monitoring-plan and test-case generation, and historic dataset
creation. The live-monitoring demo uses generated data. In a real deployment,
the existing read-only REST adapter can poll an authorized endpoint over
approved network connectivity.

Token Factory is the default reasoning provider and is called at runtime through
its OpenAI-compatible inference API. Nebius/NVIDIA model calls power diagnosis
and assistant-generation features; deterministic code remains responsible for
anomaly detection, policy gates, and simulation. The app displays model
provenance and does not silently substitute another model after a failed
Nebius call.

Document and query embeddings use Qwen3-Embedding-8B on Token Factory, while
NVIDIA Nemotron handles reasoning. Stored vectors carry provider/model
provenance so indexes from different models are not mixed.
The application is hosted as a Streamlit service, with optional Google Cloud
services for hosting, persistence, and audit records. All demo telemetry and
procedures are synthetic.
Downtime and cost values are assumption-based estimates, not measured savings.

### Tavily: current public evidence, not decorative search

The engineer-triggered Public Research workflow uses Tavily at runtime to
search manufacturer fault guidance, documentation, and recent safety-advisory
coverage. Bounded Map discovery finds documentation links that are fed into
Extract; extracted passages and clearly labeled search excerpts become a
source-linked evidence packet. Nemotron then produces a structured brief with
validated passage references, unresolved applicability questions, and questions
for a qualified engineer, not physical repair instructions. The app attaches
quoted passages directly from retrieved evidence rather than asking the model
to reproduce their wording.

Every run exposes its actual Search/Map/Extract HTTP trace, source URLs,
retrieval time, request IDs and credit usage when returned by Tavily. Queries
are public-only, editable, and previewed before consent; private telemetry and
repository content are never automatically included. An engineer can attach
the packet to an analysis case without changing any safety decision.
Missing results are reported as unknown coverage, never proof that no advisory
exists. This complements device-specific RAG with current external evidence.
An October 28 public-only verification completed all five runtime HTTP requests,
retained eight sources, extracted a mapped documentation link, and produced
three Nemotron findings whose quotes passed source-membership checks. This is
a working integration test, not proof of clinical/industrial accuracy or award
selection.

### Tool feedback from implementation

Tavily's request IDs and reported credit usage made runtime evidence easy to
inspect. Map returned no links from our first product landing page, but a public
support index yielded links that could feed Extract. We therefore expose empty
discovery and incomplete extraction rather than infer an all-clear. A useful
improvement would be clearer guidance on choosing crawlable documentation entry
points.

Token Factory's OpenAI-compatible API let us reuse the orchestration while
switching the model provider. Nemotron reasoning needed room beyond the final
JSON: a 4,096-token call exhausted its budget before emitting text. Increasing
the research budget to 16,384 enabled validated briefs. JSON-object mode still
occasionally omitted required sections, so we switched to the supported
JSON-schema response format and verified it on live research and diagnosis.
Passage-ID citations avoid model transcription errors without weakening
source-membership checks. Exposing reasoning-token usage, finish reasons and
schema-constrained examples prominently would help builders diagnose these
cases faster.

**Track:** Best Apps and Agents
