# Nebius x NVIDIA submission checklist

## Ready in this copy

- [x] Separate project copy; original Kairos project remains unchanged.
- [x] Default Nebius Token Factory provider and NVIDIA Nemotron model configured in code.
- [x] Model provenance is retained and displayed; failed Nebius calls do not
      silently switch to Google AI or offline rules.
- [x] Root MIT license added after the project owner confirmed licensing authority.
- [x] README, project description, prior-work disclosure, and sub-three-minute
      script prepared.
- [x] Cloud Run deployment uses a distinct service name from the original app.

## Required before submitting

- [x] Obtain a Nebius Token Factory API key and configure it locally or in
      Secret Manager. Never paste the key into chat or commit it.
- [x] Run `python -m cloud.smoke` with the key and confirm the case provenance
      begins with `nebius:`. Record only after this succeeds.
- [x] Run `python -m unittest discover -s tests -v` (168 tests passed October 30,
      with `FORCE_MOCK=1` for the regression run).
- [x] Create a **new public GitHub repository** for this adaptation; keep the
      original repository unchanged. Confirm GitHub detects the root MIT license.
      Published as https://github.com/puneetpahuja-eaton/prism-spectra-ai;
      anonymous access and GitHub's MIT detection verified October 30.
- [x] If deploying, set the intended Google Cloud project, provision the
      separate resources, add the Token Factory key to Secret Manager as
      `nebius-api-key`, and deploy `kairos-ai-nebius`. Verify the public URL and
      provide judge test access through the end of judging.
      Published October 30 as `prism-spectra-ai`, using a separate named
      synthetic database, dedicated runtime identity and per-secret access.
      Keep this service and its API credits available through December 15.
- [ ] Record and upload a new public YouTube video under 180 seconds. Show a
      real Nebius inference call and model provenance; no old Google hackathon
      decks or recordings.
- [ ] Complete the Devpost fields: Best Apps and Agents track, description,
      demo URL, public repository URL, video URL, feedback, and existing-project
      disclosure.
- [ ] Ensure all submitted images, text, music, and code are owned or properly
      licensed; do not include confidential employer or personal data.

## Best Use of Tavily bonus award

- [x] Configure `TAVILY_API_KEY` locally, or save it in Integrations.
- [x] Run `python -m tools.smoke_tavily --map` and review actual runtime statuses.
      October 28: complete, five HTTP 200 calls, eight sources, mapped page
      extracted, and three citation-validated Nemotron findings.
- [x] Verify the browser research flow after schema-constrained output and
      source-bound passage citations: October 29, complete with Nemotron
      provenance, sources and runtime trace, no visible application alerts.
- [ ] In the working demo, run Public Research with public product/fault terms.
      Show Search, Map discovery feeding Extract, source-linked quotes,
      Nemotron provenance, coverage limits and the real runtime trace.
- [ ] Use a public/synthetic demo device; do not expose private employer
      repositories, credentials, site identifiers or device addresses in recording.
- [ ] Attach the packet to a case without changing its safety decision, or
      download the packet to demonstrate a useful engineering handoff.
- [ ] Keep the final video below three minutes total, including the Tavily segment.
- [x] Provide the hosted runtime with its own Tavily secret/credits; a key
      on the developer laptop does not configure the hosted demo.
      Hosted research and JSON export passed October 30 with five HTTP 200
      responses and live Nemotron provenance.
- [ ] Describe why external research changes the engineer's evidence gathering,
      not unsupported numerical claims about savings or guaranteed awards.

Official rules: a functional runtime Tavily call is the bonus eligibility
threshold, not a guarantee of winning. One overall OR track award plus one
bonus award is permitted. Verify final rules and judge access requirements.

## Truthful demo guardrails

- Demo telemetry and procedures are synthetic.
- Live polling is REST read-only; there is no real device control or external
  engineer notification.
- Any avoided downtime/cost is assumption-based and not measured.
- The five named stages are an orchestrated workflow; not five separately
  trained models.
