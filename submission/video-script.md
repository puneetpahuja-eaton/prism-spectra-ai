# Prism-Spectra-AI — hackathon demo video (target: 2:35; must be under 3:00)

**Record only after a live Token Factory call succeeds.** Show the app's model
label or case provenance as evidence. All readings shown must be labeled
synthetic. Use only cleared visuals/audio; no unlicensed music or third-party
marks.

| Time | Show | Say |
|---|---|---|
| 0:00–0:22 | On camera, then a device alarm | “A machine alarm tells an engineer that something changed. It rarely tells them why, which procedure applies, or what evidence to bring to the repair. While they search manuals and call the site, a small fault can become a long interruption.” |
| 0:22–0:43 | Prism-Spectra-AI device page and knowledge sources | “Prism-Spectra-AI is an industrial reliability copilot. It learns from each device’s own documents, then connects that knowledge to telemetry. An engineer can ask a question, plan a task, or let Agent mode prepare the next step.” |
| 0:43–1:02 | Live Monitoring: start the built-in feed; show changing readings | “Live monitoring is the headline: watch device parameters stream into the same analysis engine used for historical data. This demo uses generated readings. In production, an authorized device would connect through an approved REST endpoint and secure network path. Monitoring is read-only.” |
| 1:02–1:20 | Public Research: approved queries, Search/Map/Extract trace, cited source | “A manual can go stale. Tavily searches public manufacturer guidance, maps documentation links, and extracts evidence. Nemotron builds a cited brief for engineer review. We show exactly what was found—and what remains unknown—without sending private device records to web search.” |
| 1:20–2:08 | Analysis trace; highlight each stage while a voltage-sag case runs | “Here is a voltage-sag example. **Monitor** detects an abnormal change against the device’s learned healthy baseline. **Diagnose** sends the evidence and relevant procedure to NVIDIA Nemotron through Nebius Token Factory; the app checks the structured answer and verifies its procedure citations. **Decide** is a separate deterministic safety gate: critical or uncertain cases go to human review. **Act** prepares a review-only work order or a human-approved sandbox scenario—never a real equipment command. **Verify** checks later readings and reports whether the simulated condition recovered or must remain escalated.” |
| 2:08–2:28 | Model provenance, architecture labels, then case summary | “The reasoning call is made at runtime through Nebius Token Factory using Nemotron-3.5-Lightning. We retain Cloud Run and optional Google Cloud persistence, but the diagnosis path shown here is Nebius—not a screenshot or a silent fallback. Any downtime or cost avoided is an assumption-based estimate, not measured savings.” |
| 2:28–2:38 | Closing title | “Prism-Spectra-AI turns scattered alarms into evidence an engineer can act on. Better context, faster triage, and people still in control.” |

## Capture notes

- Keep the dashboard model label visible during diagnosis and open the case
  trace so the actual `nebius:` provenance is visible.
- Use only synthetic/demo telemetry. Do not claim a production repair, external
  notification, measured savings, or validated fault-prediction accuracy.
- The final public YouTube upload must be shorter than 180 seconds.
- Run Tavily live before recording. Show at least one successful runtime
  request and its source evidence. Avoid claiming absence of recalls or
  automatic live advisory monitoring.
