# Prism-Spectra-AI — Industrial Reliability Copilot

Prism-Spectra-AI helps reliability engineers investigate industrial power and equipment
incidents. It combines telemetry-derived evidence, device-specific documents,
and a five-stage analysis workflow: Monitor, Diagnose, Decide, Act, Verify.

- **Hackathon track:** Best Apps and Agents
- **Required AI path:** NVIDIA Nemotron-3.5-Lightning through Nebius Token Factory
- **Demo data:** Synthetic; the app does not control equipment or dispatch real work orders.

This is a separate adaptation of an existing industrial reliability project for the
[Nebius x NVIDIA Global AI Hackathon](https://nebiusglobalaihackathon.devpost.com).
The official submission period ends October 30, 2026 at 10:00 a.m. Pacific.

## Published demo

- **Working application:** https://prism-spectra-ai-498775210896.asia-south1.run.app
- **Public source:** https://github.com/puneetpahuja-eaton/prism-spectra-ai
- **Judge walkthrough:** [testing instructions](submission/judge-guide.md).
- **License:** [MIT](LICENSE).

The hosted demo uses an isolated synthetic Firestore database and a dedicated
runtime identity. Nebius and Tavily keys are injected from hosting secrets,
not shipped with the source. Judges receive an engineer login in the Devpost
testing instructions; the local default admin login is not valid on this demo.
No payment or personal API key is required from judges.

## What the project does

- Onboards multiple devices and organizes their manuals, notes, telemetry, and
  optional GitHub/Confluence sources by device.
- Provides a grounded Ask / Plan / Agent assistant for device questions,
  monitoring plans, test cases, and synthetic datasets.
- Analyzes synthetic, uploaded, historical, or REST-polled telemetry.
- Uses statistical detection and source-grounded Nemotron interpretation for
  user devices. Source citations are checked against supplied passages.
- Keeps the original electrical SOP, fault mappings, physics and remediation
  policy inside explicitly selected synthetic power demonstrations only.
- Creates review-only work-order drafts or sandbox simulations; real equipment
  commands and external engineer notifications are not implemented.
- Shows the selected model/provider and retains model provenance in case records.
- Adds a separate public-evidence research desk using Tavily Search, Map and
  Extract, followed by a citation-checked Nemotron brief for engineer review.

The five labels describe an orchestrated workflow, not five independently
trained or separately hosted AI models. Monitor, Decide, Act, and Verify use
deterministic logic and safety gates; model reasoning is used for supported
diagnosis and assistant-generation tasks. Downtime/cost figures are estimates
based on disclosed demo assumptions, not measured production savings. User-device
analysis does not apply the demo's fault meanings, control actions or assumed savings.

## Device knowledge: supplied sources only

New devices start **unconfigured**, not attached to the power simulator. Every
page requires an explicit device selection. Real-device live monitoring defaults
to **off** for simulation, with no endpoint or hex encoding preselected.

1. Register your device with its supplied identity. Leave unknown facts unknown.
2. In **Data Sources**, upload authorized device code/docs or use **Paste text /
   explicit device guidance** to save your own facts for that device. Optional
   GitHub/Confluence ingestion remains device-scoped.
3. Supply the actual parameter names/IDs, units, numeric ranges, alarm limits,
   response paths and decoding rules. The assistant asks about missing details
   instead of choosing industry defaults. An unanswered clarification does not
   authorize invented specifications.
4. Generated monitoring parameters and dataset metrics must cite exact passages
   from the supplied material containing their literal field values. Separate
   passages can support a field's name, unit and limits. Missing or unverifiable
   evidence becomes a conversational clarification. Review a monitoring plan before applying
   it. Source membership checks do not prove semantic correctness.
5. Generated user-device datasets use independent illustrative signals within
   supplied ranges. They do not assume three phases, electrical relationships or
   nominal frequency. Injected codes are labeled `SYNTHETIC_*`, not manufacturer
   fault meanings. The recovery/escalation showcase requires at least eight replay
   samples per asset.
6. User-device Analysis uses supplied history as the user-declared baseline,
   not a health certification. Unknown codes remain uninterpreted. Cases are
   review-only; a return to baseline is observed recovery, not an automatic repair.
   No downtime, cost savings or diagnosis accuracy is invented.
7. For live deep analysis, explicitly confirm that the initial readings are an
   acceptable baseline. Missing/changed fields and unknown hex encodings are
   reported, not filled with guessed values. Simulated feed changes require restart.

The AI is **not retrained** on your device. It retrieves the selected device's
sources at runtime (RAG). Specta shares device-scoped conversation guidance across
pages in the current authenticated browser session; save guidance in Data Sources
when it should be reused in future sessions.
These safeguards reduce unsupported assumptions but do not guarantee that a
language model cannot hallucinate; engineer review remains necessary.

Existing uploaded knowledge, device records, credentials and stored telemetry
are not purged or rewritten by this change. Switching to the simulator retains
saved telemetry. Existing demonstration records remain available but are never
automatically selected. If a cloud instance lacks a locally stored demo file,
Analysis offers an explicit **Load synthetic demo fixture for this session**
button without replacing stored records. Local files are not durable across
Cloud Run instances; configure a private GCS bucket for persistent uploaded datasets.

Tavily stays optional and separate: public product, symptom and approved domains
start blank, require explicit input and consent, and are not promoted into
private device truth or control instructions.

## Specta: conversational assistant on every screen

Open **Specta** from the lower-right chat button on any page, or use the full
**Ask the device** workspace. The responsive dock uses a Copilot-inspired layout:
device context, Ask / Plan / Agent selector, scrollable conversation, fenced code
with Copy, downloadable responses/artifacts and a file-enabled message composer.
It is not the VS Code extension and does not have terminal or editor access.

- Select a device explicitly. The dock and workspace share its conversation,
  active task and attachments; switching devices restores separate conversations.
- **Ask** answers questions. **Plan** proposes work without executing it.
  **Agent** generates datasets, monitoring plans and other artifacts.
- **Build a dataset for Analysis** starts generation immediately. If units, ranges
  or other necessary facts are missing, Specta asks up to three focused questions
  in the conversation. Reply in either chat surface to continue the same task.
  There is no separate clarification button or answers form. Selected duration,
  sampling, synthetic asset count and repository reference are retained. The model's
  output schema is constrained to that exact asset count. Active **Task settings** let you
  correct the reference or sampling; then send `retry`.
- The monitoring builder follows the same flow. Downloads, source evidence and
  reviewed application appear inside the conversation. Generation never starts
  polling. Engineers/admins must verify the plan and supply the endpoint; stop
  any existing live run before applying a different plan.
- Authorized chat attachments (5 MB per file; 120,000 extracted characters total)
  are sent to the configured AI provider as context, not silently added to the
  permanent corpus or sent to Tavily. Do not upload secrets or employer-private data.
- Provider, invalid-output and file failures remain explicit assistant responses,
  not fake successes. The pending task is retained for correction/retry. Missing
  device facts are never replaced with guessed specifications. Generation can
  correct schema errors once and unsupported model-written citations once before
  reporting a failure; these bounded corrections never relax validation or
  convert the model's previous claims into source evidence.
- **New chat** clears the current device's conversation, attachments and pending
  task, without deleting stored devices, documents or datasets. Conversation state
  is session-local, not durable across browser reconnection/server restarts.
- Loading a dataset requires admin access and explicit replacement confirmation.
  Monitoring application requires engineer/admin access and source review.
  Specta cannot control physical equipment or invoke public research automatically.

## Nebius Token Factory

The default provider is Nebius, using its OpenAI-compatible Chat Completions API:

- Endpoint: `https://api.tokenfactory.nebius.com/v1/chat/completions`
- Model: `nvidia/Nemotron-3_5-Lightning`
- Embeddings: `Qwen/Qwen3-Embedding-8B` through
  `https://api.tokenfactory.nebius.com/v1/embeddings` using the same API key.
- Source: Nebius' [Nemotron model guide](https://github.com/nebius/token-factory-cookbook/blob/main/models/nemotron/README.md)
  and [Token Factory cookbook](https://github.com/nebius/token-factory-cookbook).

Diagnosis, grounded chat, dataset generation, monitoring-plan generation, and
review-only work-order drafts use the selected text model. The Nebius path is
explicit: if Token Factory is unavailable or returns invalid output, the app
reports the failure and escalates rather than silently switching providers.
`AI_PROVIDER=google` is available only as an explicit local comparison setting;
use `AI_PROVIDER=nebius` for the hackathon demo.

The Nebius API key is required for a live Nebius call. It is not stored in source
control, and an offline run does not count as evidence that the Nebius requirement
is met. The case trace and sidebar display the actual configured provider.

## Architecture

- **Streamlit** application with role-based access and multiple pages.
- **Nebius Token Factory** for NVIDIA Nemotron text generation and Qwen3
  document/query embeddings (4096 dimensions).
- Optional **Google Cloud Run**, **Firestore**, **Cloud Storage**, **BigQuery**,
  and **Secret Manager** for hosting and persistence.
- Embedding selection is explicit: `EMBEDDING_PROVIDER=nebius` for Token
  Factory, `local` for deterministic keyword vectors, or `google` for an
  explicitly configured Google embedding model. In Nebius mode, inherited
  Google credentials never select Google embeddings. A remote embedding failure
  is reported; it is not silently replaced by local or another provider's vectors.
- Synthetic or authorized telemetry only. The built-in live demo feed is
  simulated; real REST monitoring is read-only.

## Run locally (Windows PowerShell)

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

The launcher creates `.venv` and installs `requirements.txt` on its first run.
If you prefer, you can create the virtual environment and install requirements
manually before starting it.

In the local `.env`, configure a Token Factory key:

```text
AI_PROVIDER=nebius
NEBIUS_API_KEY=your-token-factory-key
NEBIUS_MODEL=nvidia/Nemotron-3_5-Lightning
EMBEDDING_PROVIDER=nebius
NEBIUS_EMBED_MODEL=Qwen/Qwen3-Embedding-8B
```

Get the key from the [Token Factory API keys page](https://tokenfactory.nebius.com/).
Do not paste it into chat, commit it, or include it in a video. Then start the app:

```powershell
.\run_local.ps1
```

The launcher explicitly selects Nebius for both text and embeddings and disables
mock mode. The sidebar displays both selected backends.

### Hackathon credits and provider errors

Creating an API key does not redeem promotional credits. The official
[hackathon resources](https://nebiusglobalaihackathon.devpost.com/resources)
list a $25 Token Factory offer via the
[promo form](https://nebius.com/promo-code?utm_promo_event_code=2026-devpost-global-ai-hack&utm_promo_code_type=Token_Factory&utm_promo_activation_code=NEBIUS-DEVPOST-GLOBAL26),
using activation code `NEBIUS-DEVPOST-GLOBAL26`. The free
[Nebius Builders Program](https://dev.nebius.com/builders) also lists $25 Token
Factory credit and $25 Tavily credit, subject to eligibility, availability and
partner terms. Complete the claims and confirm activation/remaining balances
in the respective provider accounts. Use the account associated with the app's
configured key; the app login is separate from provider accounts.

A connection failure is not proof of exhausted credit. Specta distinguishes
connection/read timeouts, TLS/proxy failures, API-key authentication, HTTP 402
payment/credit responses and HTTP 429 rate/quota responses. Sanitized transport
exception types are logged without request contents or keys. Transport failures
are not automatically retried: a timed-out generation may still be processing.
The conversation and active task remain available for a deliberate `retry`.
`NEBIUS_TIMEOUT_SECONDS` sets the read deadline (default 120 seconds); the
connection deadline is 10 seconds. Adjust the deadline only after confirming
that slow responses, rather than connectivity or account errors, are the cause.

The seeded demo login is `admin@kairos.ai` / `admin12345`; change the password
before exposing a deployment. Without a model key, the app can run in local
offline mode for development and tests, but it does not make a qualifying live
Nebius inference call. To index documents without a key, explicitly select
`EMBEDDING_PROVIDER=local` and start Streamlit directly instead of using the
Nebius-only launcher.

### Existing knowledge after switching embedding models

Every newly indexed chunk records its provider/model. Search only compares
vectors with the same provider, model, and dimension. Old or untracked vectors
are reported as stale, even if they happen to have the same dimension.
In **Data Sources**, use **Re-embed stale chunks with current model** to migrate
stored text, or **Force full re-ingest** on the GitHub integration to rebuild it.
Migration sends the selected stored text to the configured embedding service.
It is never performed automatically. GitHub browser access and API access use
different authentication; the integration uses your configured token, not your
browser login.

## Tavily public research

The **Public Research** page is available to engineers and admins. It adds
current public-source investigation without copying private device documents,
telemetry, site/IP information, case details, or chat history into web queries.
It is a user-triggered research tool, not an always-on advisory monitor.

1. Add `TAVILY_API_KEY=` to the local `.env`, paste your key after `=`, save, and
   restart the app. Alternatively, an admin can save it under **Integrations →
   API keys**. Environment configuration takes precedence over a saved key.
2. Select a device for local association, then enter public manufacturer/model
   and generic fault terms. Specify approved public manufacturer/regulator domains.
3. Optionally enable **Map** and enter a public documentation starting URL.
   For the public Eaton example, use `https://www.eaton.com/us/en-us/support.html`.
   A product landing page can return zero links; this is shown as unknown
   discovery coverage, not a successful documentation search.
4. Review the exact outbound query/URL preview and confirm public-only consent.
5. Run research. The workflow makes three advanced **Search** calls (fault
   guidance, manuals, and last-year advisory search), at most one bounded
   **Map** call, and one **Extract** call for up to four pages. Map discoveries
   are included in Extract rather than being decorative link lists.
6. Nemotron builds a schema-constrained brief from the returned excerpts. Every
   finding must cite known passage IDs. The app attaches each selected passage
   directly from the retrieved source, with whitespace normalized; the model
   cannot rewrite the quoted text. Unknown/fabricated references reject the
   entire model brief; raw evidence remains visible with explicit errors.
   The research call reserves 16,384 output tokens for reasoning plus the concise
   final JSON. An exhausted reasoning budget or invalid response is an explicit
   failure, never a fabricated brief.
7. Download the evidence packet with timestamps, source URLs, coverage, model
   provenance, and real runtime request IDs/credits when Tavily reports them.

In an Analysis case, **Research public guidance for this case** records only a
local case reference. Open **Public Research** to enter public terms. Explicitly
attach the packet afterwards if useful; it appears in the case/audit download.
Attachment never changes a diagnosis, decision, action, approval or safety policy.
Case links are invalidated when the analysis run changes.

No evidence does **not** mean no recall, no risk, or safe operation. Retrieved-at
time is not publication time. Allowlisting a domain does not establish that a
page applies to a particular hardware revision; a qualified engineer must check.
Exact quotes prove membership in the retrieved text, not factual truth or
semantic entailment. Public research is not mixed into private RAG or used for
automatic equipment control. Results live in the current app session unless
explicitly downloaded/attached. No automatic retries or polling spend credits.

For a reproducible **live** public-only smoke test (up to five Tavily calls plus
one model call):

```powershell
.\.venv\Scripts\python.exe -m tools.smoke_tavily --map
```

The example queries public Eaton 5PX documentation. It prints request metadata,
not API keys or private data. Unit tests mock Tavily; passing unit tests alone
does not demonstrate a functional runtime Tavily call.

## Optional Google Cloud services

Google Cloud is retained for data persistence and optional hosting; it is not
the default reasoning or embedding provider. To enable Google embeddings,
explicitly select `EMBEDDING_PROVIDER=google`, configure a currently supported
`EMBED_MODEL` and Google credentials. For optional Firestore, Cloud Storage, or
BigQuery, configure the corresponding variables in `.env` and use
`gcloud auth application-default login`. Only use public or non-confidential
demo documents; both retrieval embeddings and grounded model prompts may be
sent to external AI services.

For a separate Cloud Run deployment:

1. Set `GOOGLE_CLOUD_PROJECT` to a project where you have deployment
   permissions. Optionally set `GCS_BUCKET` to a globally unique bucket name.
2. Run `python -m cloud.provision` to provision this copy's separate storage,
   dataset, and service identities.
3. In Google Secret Manager, create a secret named `nebius-api-key` and add the
   Token Factory key as a version. Create a separate `prism-spectra-admin-password`
   secret with a strong owner password.
4. Create a named Firestore database, for example `prism-spectra-demo`, and
   grant the runtime service account access. Do not reuse a database containing
   private devices or existing integration credentials. Seed an engineer account
   for judges and supply its credentials privately in Devpost testing instructions.
5. Run `.\cloud\deploy.ps1 -AdminPasswordSecret prism-spectra-admin-password`.
   It deploys to the separate service
   `prism-spectra-ai`; it does not update the original project's service.
   The database defaults to `prism-spectra-demo`; override with
   `-FirestoreDatabase`. The script rejects `(default)` to avoid accidental
   reuse. Instance count defaults to two with scale-to-zero enabled.
   To enable public research in the hosted demo, create a `tavily-api-key`
   Secret Manager secret, grant the runtime identity access, and pass
   `-TavilySecret tavily-api-key`. Local `.env` files are not deployed.
   The container and source-upload exclusions also omit virtual environments,
   local device/secret stores, cached files and archived recordings.

Cloud infrastructure and model calls may incur charges. Configure budget
alerts. Never commit `.env`, API keys, service-account credentials, or
deployment-specific `cloud/deployment.json`.

## Tests and verification

```powershell
python -m unittest discover -s tests -v
```

The Token Factory client tests mock HTTP responses and validate request
formatting, schema validation, provider attribution, and failure behavior. To
verify an actual live diagnosis with your configured key, run:

```powershell
python -m cloud.smoke --embeddings
```

That live test uses synthetic scenarios and synthetic embedding inputs; it
checks the configured embedding provider and reports vector dimensions.
Do not describe the live model call as
verified until it has been run successfully with your own Token Factory access.

## Existing-project disclosure

The underlying project existed before the hackathon submission period under
the name Kairos AI. This edition is now named Prism-Spectra-AI. The adaptation and
significant changes made during the period are documented in
[`submission/prior-work-disclosure.md`](submission/prior-work-disclosure.md).
The older Google Cloud AI Builder Cup pitch decks and videos are local archive
artifacts, not materials for this submission.

## Submission materials

- [Project description](submission/project-description.md)
- [Prior-work disclosure](submission/prior-work-disclosure.md)
- [Video script (under three minutes)](submission/video-script.md)
- [Submission checklist](submission/checklist.md)

This adaptation is published in its own public repository; the original project
and its history remain separate. GitHub detects the MIT license in
[`LICENSE`](LICENSE).

## Branding compatibility

The current product name is **Prism-Spectra-AI**, including UI titles, generated
audit/evidence filenames and submission materials. Existing `KAIROS_ADMIN_EMAIL`
and `KAIROS_ADMIN_PASSWORD` configuration keys, local development login and the
stored demo marker are retained for compatibility; the rename does not reset
accounts or reseed saved devices. Historical disclosure, original copyright
attribution and ignored archive filenames preserve their original names.
