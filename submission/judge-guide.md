# Prism-Spectra-AI | Judge testing guide

- **Track:** Best Apps and Agents
- **Bonus consideration:** Best Use of Tavily
- **Application:** https://prism-spectra-ai-498775210896.asia-south1.run.app
- **Source:** https://github.com/puneetpahuja-eaton/prism-spectra-ai
- **License:** [MIT](../LICENSE)

Use the engineer credentials supplied in Devpost's testing instructions. The
hosted demo does not use the local development admin password. Judges do not
need their own API keys or a paid account. Scale-to-zero cold starts and model
inference may take time; wait for the page to finish before clicking again.

## Suggested walkthrough

1. **Home:** sign in and confirm the synthetic Power Distribution Panel is
   listed. The sidebar shows Nebius Token Factory and Qwen embeddings.
2. **Ask the device:** select the demo device, choose Ask, and ask:
   "Which assets are included and which injected fault needs engineer
   escalation?" A small, explicitly synthetic knowledge corpus is pre-indexed.
   The answer should include source references. Plan and Agent modes use the
   same device-scoped assistant for plans and synthetic artifact generation.
   The lower-right **Specta** launcher opens the same conversation on every
   screen. Dataset/monitoring generators ask missing questions directly in chat;
   reply there to continue rather than restarting or clicking a clarification
   button. Switch devices to verify conversation isolation. New chat clears only
   the current device's session conversation, not stored data.
   If a builder conflicts with an explicit dataset/monitoring-plan request,
   Specta asks which artifact to create before any source retrieval or inference.
   Reply `dataset` or `monitoring plan` to resume. Large sources are selected
   within a shared input budget, not reviewed exhaustively. An explicit provider
   output-budget exhaustion permits one larger-budget retry (32,768-token cap);
   transport timeouts do not. Metadata retains usage for failed/recovered calls.
3. **Live Monitoring:** explicitly select the demo device, enable **Use built-in
   simulated feed**, then click Start. Observe changing readings, then click Stop. This exercises the live
   polling engine, not a physical device. Authorized REST sources can be
   configured separately; there is no real equipment control.
4. **Analysis:** explicitly select the demo device. If its local demo file is
   unavailable on this cloud instance, click **Load synthetic demo fixture for
   this session**; stored device data is not replaced. Use the synthetic replay controls to
   inspect Monitor, Diagnose, Decide, Act and Verify. Read model provenance,
   escalation state, evidence and assumptions rather than interpreting demo
   outcomes as measured industrial accuracy.
5. **Public Research:** select the demo device. The public fields have no
   manufacturer/domain defaults. For this optional public-only example, enter product `Eaton 5PX UPS`, symptom
   `battery alarm`, and approved domain `eaton.com`. Enable Map with starting
   URL `https://www.eaton.com/us/en-us/support.html`. Review the outbound
   preview, consent, and click Research public evidence.
6. Review findings, source-linked passages, unresolved applicability questions,
   discovered documentation and the real Search/Map/Extract HTTP trace.
   Download the evidence packet. A complete run uses three searches, at most
   one Map and one Extract call; actual credit usage is shown when reported.
   If a provider fails, evidence is retained with explicit partial/failure
   status, not replaced with a fabricated brief.
7. **Inline evidence:** in an Analysis case, selected live incident or either
   Specta surface, expand **Enrich with approved public evidence**. Supply public
   terms/domains, inspect the preview and approve. It uses the same bounded
   Tavily pipeline, without sending private observations as queries. Review the
   Nemotron comparison: D passages are supplied evidence, P passages public
   supplementary evidence. It does not modify device limits or controls.
8. **Workflow provenance:** expand the result's metadata panel for LangChain
   stages, elapsed time, validation outcome and provider-reported usage.
   LangSmith tracing is optional and metadata-only. Confirm the displayed
   tracing status; no dashboard ingestion or independent evaluation is claimed
   solely from the presence of a key.

## Verified hosted behavior

On October 30, 2026, the deployed application passed engineer login and role
checks, produced a source-referenced device answer, and completed public
research with six sources, two mapped links, three Nemotron findings, five
Tavily HTTP 200 responses and a downloaded JSON evidence packet. The simulated
live feed was started and stopped successfully. These are integration checks,
not production efficacy or award claims.

## Boundaries and disclosure

- Use only synthetic or public data; do not upload private employer material.
- Private device records are not automatically used as public web queries.
- New devices have no simulator, device-specific defaults or assumed event-code
  meanings. Supply their own authorized code/docs and explicit guidance. User-device
  Analysis is source-only and review-only, with no assumed economic benefit.
- Source passage membership does not establish truth or model applicability.
- Automatic equipment recovery is simulated. Real equipment commands and
  external engineer notifications are not implemented.
- Downtime and cost figures are assumption-based, not measured savings.
- The five stages form an orchestrated workflow, not five separately trained
  models.
- Active analysis retains deterministic baseline arithmetic, not local learned
  models. Scores can differ from the historical PRISM reference. Inference and
  embeddings are Nebius-only; provider failure never selects Gemini or local AI.
- Public research/incident interpretation jobs are session-local and bounded.
  Their completion does not mean a diagnosis is proven or a repair occurred.
- This is an existing-project adaptation. See the
  [significant-update disclosure](prior-work-disclosure.md) and
  [tool feedback](project-description.md#tool-feedback-from-implementation).
