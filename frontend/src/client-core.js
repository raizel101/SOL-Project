/** Pure client rules. Keep Python Unicode limits and response acceptance explicit. */
export const LIMITS = Object.freeze({ message: 1200, historyMessages: 8, historyEach: 1000, historyTotal: 2000 });

const ACCEPTED_MODES = new Set([
  "local_rag", "urgent_support", "treatment_boundary", "human_support",
  "insufficient_sources", "supportive_abstention",
]);
const WITHHELD_MODES = new Set([
  "citation_check_failed", "source_claim_withheld", "support_response_withheld", "incomplete_response",
]);

/** Count Unicode code points, matching Python len rather than JavaScript UTF-16 units. */
export function countCharacters(value) {
  return typeof value === "string" ? Array.from(value).length : 0;
}

function hasInvalidText(value) {
  return /[\u0000-\u0008\u000b\u000c\u000e-\u001f]/.test(value) || /[\ud800-\udfff]/u.test(value);
}

export function validateMessage(message) {
  const count = countCharacters(message);
  if (typeof message !== "string" || !message.trim()) {
    return { valid: false, count, reason: "Write a question or reflection first." };
  }
  if (count > LIMITS.message) {
    return { valid: false, count, reason: `This message has ${count.toLocaleString()} characters. Please keep it within 1,200; nothing has been shortened.` };
  }
  if (hasInvalidText(message)) {
    return { valid: false, count, reason: "This message contains unsupported control characters or invalid Unicode. Please remove them; the interface will not rewrite your text." };
  }
  return { valid: true, count, reason: "" };
}

/** Validate complete alternating pairs without discarding or shortening any message. */
export function validateHistory(history) {
  if (!Array.isArray(history) || history.length % 2 !== 0 || history.length > LIMITS.historyMessages) {
    return { valid: false, reason: "This conversation has reached its context limit. Start a new conversation to continue; visible messages have not been removed.", total: 0 };
  }
  let total = 0;
  for (let index = 0; index < history.length; index += 1) {
    const entry = history[index];
    if (!entry || typeof entry !== "object" || Array.isArray(entry) || Object.keys(entry).length !== 2 ||
        !Object.hasOwn(entry, "role") || !Object.hasOwn(entry, "content") ||
        entry.role !== (index % 2 ? "assistant" : "user") ||
        typeof entry.content !== "string" || !entry.content.trim() || hasInvalidText(entry.content)) {
      return { valid: false, reason: "The conversation context is incomplete. Start a new conversation; the interface will not silently discard context.", total };
    }
    const count = countCharacters(entry.content);
    if (count > LIMITS.historyEach) {
      return { valid: false, reason: "A completed message exceeds the backend’s 1,000-character history limit. Read the full reply here, then start a new conversation; it will not be silently shortened.", total: total + count };
    }
    total += count;
  }
  if (total > LIMITS.historyTotal) {
    return { valid: false, reason: "This conversation exceeds the backend’s 2,000-character history budget. Start a new conversation to continue; the full visible exchange is retained until you clear it.", total };
  }
  return { valid: true, reason: "", total };
}

export function historyAfterAccepted(history, user, answer) {
  if (!Array.isArray(history) || typeof user !== "string" || typeof answer !== "string") {
    throw new TypeError("Only complete plain-text user/assistant pairs can become history.");
  }
  const next = history.map(({ role, content }) => ({ role, content }));
  next.push({ role: "user", content: user }, { role: "assistant", content: answer });
  return { history: next, validation: validateHistory(next) };
}

/** Only the documented successful envelope becomes history; known fallbacks remain withheld. */
export function classifyChatResponse(status, body) {
  if (!Number.isInteger(status) || !body || typeof body !== "object" || Array.isArray(body)) {
    return { kind: "error", mode: "error", answer: "", errorMessage: "The backend returned an unreadable response.", retryable: false };
  }
  const mode = typeof body.mode === "string" ? body.mode : "error";
  const answer = typeof body.answer === "string" ? body.answer : "";
  if (status === 200 && body.ok === true && body.error === null && ACCEPTED_MODES.has(mode) && answer.trim()) {
    return { kind: "accepted", mode, answer, errorMessage: "", retryable: false };
  }
  const error = body.error && typeof body.error === "object" && !Array.isArray(body.error) ? body.error : null;
  if (status >= 400 && body.ok === false && error && WITHHELD_MODES.has(mode) && answer.trim()) {
    return { kind: "withheld", mode, answer,
      errorMessage: typeof error.message === "string" ? error.message : "A generated answer was withheld.", retryable: error.retryable === true };
  }
  return { kind: "error", mode, answer: "",
    errorMessage: error && typeof error.message === "string" ? error.message : "The backend did not return an accepted answer.",
    retryable: error?.retryable === true };
}

/** External source links must be public HTTP(S) URLs without embedded credentials. */
export function safeSourceUrl(value) {
  if (typeof value !== "string" || value.length > 4096 || value !== value.trim() ||
      !/^https?:\/\//i.test(value) || /[\u0000-\u0020\u007f]/.test(value) || value.split("/")[2]?.includes("@")) return null;
  try {
    const url = new URL(value);
    return ["http:", "https:"].includes(url.protocol) && url.hostname && !url.username && !url.password ? url.href : null;
  } catch {
    return null;
  }
}

export function modeLabel(mode) {
  return {
    local_rag: "Source-backed reflection",
    urgent_support: "Human help first · fixed support",
    treatment_boundary: "Care boundary · fixed support",
    human_support: "Human support · fixed response",
    insufficient_sources: "Evidence boundary",
    supportive_abstention: "Gentle reflection · no suitable evidence",
    citation_check_failed: "Answer withheld · citation check",
    source_claim_withheld: "Answer withheld · source mismatch",
    support_response_withheld: "Answer withheld · support check",
    incomplete_response: "Answer withheld · incomplete response",
  }[mode] || "Response not accepted";
}
