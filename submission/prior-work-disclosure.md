# Existing project and significant updates

Prism-Spectra-AI builds on the team's project previously named Kairos AI, which
existed before the Nebius x NVIDIA Global AI Hackathon submission period.
The entry is an adaptation of the team's existing industrial reliability
assistant, not a claim that the entire product originated during this event.
The prior project included device onboarding, document retrieval, Ask / Plan /
Agent workflows, generated datasets, telemetry monitoring, and the PRISM-based
analysis workflow. The underlying team-authored reliability method is also
pre-existing and is not presented as newly invented.

During the Nebius hackathon submission period, this separate project copy was
updated to:

- Add a Nebius Token Factory client using the OpenAI-compatible Chat Completions
  API and an NVIDIA Nemotron model.
- Route diagnosis, grounded chat, dataset generation, monitoring-plan generation,
  and review-only work-order drafting through the selected provider, with
  Nebius as the default.
- Validate structured model responses and keep deterministic citation checks,
  safety policies, human review, and synthetic-only actions outside model
  control.
- Show provider/model provenance and fail explicitly rather than silently
  falling back to Google AI or offline rules after a Nebius error.
- Add Qwen3-Embedding-8B document/query embeddings on Token Factory, explicit
  embedding-provider selection, and provider/model-aware index migration.
  Update local setup, isolated Cloud Run deployment configuration, and
  hackathon submission materials.
- Add the Tavily Public Research workflow: public-only Search, bounded Map
  discovery, Extract, a citation-checked Nemotron brief, runtime API trace and
  explicit evidence-only attachment to analysis cases.
- Rename this edition to Prism-Spectra-AI across the application, exports and
  submission materials. Existing account and persisted-data identifiers remain
  compatible; rebranding itself is not claimed as a technical innovation.

The submission demo must use a configured live Token Factory key so the judges
can observe the actual Nebius/NVIDIA inference path. Synthetic data, safety
limitations, and estimate-based impact metrics are disclosed in the project
description and video.
