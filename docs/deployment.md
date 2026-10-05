# Deployment boundary and next stage

This refactor organizes the application for maintenance. It does not turn the current loopback service into a publicly approved deployment or implement a production authentication layer.

## What works now

The local server serves the UI, API and an installed Ollama model through the existing read-only corpus. The service intentionally rejects public/LAN bind settings and non-loopback browser origins. There are no accounts, user permissions, multi-user inference queue, persistent chat database or emergency monitoring.

Do not change the host to `0.0.0.0`, publish a tunnel, disable origin checks or open a firewall port as a shortcut to hosting. Localhost is also not an authentication mechanism: other programs on the laptop can call the service.

## Requested public architecture

The requested target is a public website, including access by adults and minors, with a server-hosted model. Vercel was proposed for the frontend. Uploading the frontend alone does not relocate Ollama or make the corpus/API available to public visitors.

The existing Python process expects an always-on backend environment, an installed model, and the corpus files. The current project is not a Vercel serverless conversion. No hosting plan, server, domain, billing account or public deployment is provisioned by this refactor. Recheck provider capability, plan eligibility and pricing when choosing hosting; do not assume an indefinitely free deployment.

A separately designed public deployment needs:

- A private inference service and read-only corpus volume; never expose Ollama directly to browsers.
- An authenticated application service or gateway, HTTPS, explicit access permissions and narrowly configured origins.
- Bounded per-user requests, quotas, abuse protection and a measured inference queue/concurrency strategy.
- Health checks, operational monitoring without collecting chat content by default, failure handling, patching and recovery procedures.
- A defined user/privacy model: what is collected, consent, access, retention, deletion and whether any external provider receives conversation text.
- Signed-off content and safeguarding review for the intended ages and languages, plus an accountable campus contact and escalation policy.

These are outstanding engineering/review requirements, not implemented features or a legal compliance certification. A named administrator can coordinate the reviews; naming an owner alone does not mean they are complete.

## Release decision

The preserved actual-model evaluation contains a source-faithfulness failure. Public/student release is not approved in the recorded artifacts. A cleaner directory structure, removed warning or passing HTTP test cannot clear that failure.

Before a launch decision, record the exact code, corpus, model and configuration; rerun real-model scenarios and qualified review; verify security, operational capacity and the selected provider's current constraints. Use [safety and evaluation](safety-and-evaluation.md) as the project checklist and retain both failed and successful runs.
