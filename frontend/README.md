# SOL browser interface

This directory is the editable source for the existing SOL chat interface. It uses
native browser modules and same-origin HTTP requests; no framework, build step,
Node runtime, CDN, browser storage or analytics is required to run it.

## Files and served routes

| Source | Backend route | Responsibility |
| --- | --- | --- |
| `index.html` | `/chat` | Semantic page structure, accessible labels and visible limitations |
| `src/app.js` | `/assets/app.js` | DOM rendering, in-memory conversation state and backend requests |
| `src/client-core.js` | `/assets/client-core.js` | Pure validation, response acceptance and safe source-link rules |
| `styles/main.css` | `/assets/styles.css` | Design tokens, components, responsive layouts and readability overrides |

The backend maps these exact routes to these files. Do not open `index.html` as a
`file:` URL: it requires the backend's same-origin routes. Start the application
using the root project instructions, then visit `/chat`.

## Maintenance rules

- Keep the HTML element IDs synchronized with the `elements` map in `src/app.js`.
- Keep validation in `client-core.js` independent of the DOM so it can be tested
  without a browser or model. Unicode limits must remain aligned with Python.
- Render model output and source text through `textContent`, never `innerHTML`.
  Validate source URLs before creating links.
- Only accepted, complete user/assistant pairs become model context. Do not
  silently shorten, drop or persist earlier messages. Retain explicit reset and
  stale-response guards when changing the request flow.
- Preserve the response-envelope checks. An HTTP error may contain a fixed safe
  fallback, but that is not an accepted generated answer.
- Keep requests same-origin. Adding a hosted endpoint, cookies, accounts, storage
  or analytics changes the privacy/security contract and needs a separate review.
- Keep CSS in its existing cascade order. Later readability overrides intentionally
  take precedence over earlier component and breakpoint typography. This refactor
  expands the original rules; it does not redesign the interface.
- Do not remove answer-quality or safeguarding notices merely because the source
  code has been reorganized. Code structure is not student-release approval.

## Checks after editing

Run the project's Python backend tests and JavaScript client-rule tests using the
root test instructions. Browser-test desktop and mobile layouts, a cited reply,
withheld/error replies, long-history blocking, explicit reset and cancellation.
Use synthetic examples; browser requests can invoke the local model. Reset and
cancel must never allow a late reply to reappear in a cleared conversation.
