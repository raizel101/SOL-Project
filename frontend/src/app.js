import {
  countCharacters, validateMessage, validateHistory, historyAfterAccepted,
  classifyChatResponse, safeSourceUrl, modeLabel,
} from "./client-core.js";

/** Browser controller: no storage, HTML interpolation, or external model fallback. */

const elements = Object.fromEntries([
  "chat-form", "message", "send", "send-label", "character-count", "composer-error",
  "message-list", "welcome", "conversation-scroll", "connection-status", "connection-label",
  "connection-notice", "context-notice", "check-connection", "new-conversation",
  "new-conversation-mobile", "reset-dialog", "cancel-response", "show-help", "help-panel",
].map((id) => [id, document.getElementById(id)]));

// Accepted complete pairs stay in page memory only. Never silently trim or rewrite context.
let history = [];
let blockedReason = "";
let pending = false;
let engineReachable = null;
// Version guards stop late responses from changing a conversation after an explicit reset.
let conversationVersion = 0;
let activeRequest = null;
let connectionVersion = 0;
let connectionRequest = null;
let composing = false;
let visibleMessageCount = 0;

// textContent keeps model replies and source excerpts inert, even when they contain markup.
function node(tag, className = "", text = "") {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text) element.textContent = text;
  return element;
}

function scrollToLatest() {
  const scroll = elements["conversation-scroll"];
  scroll.scrollTo({ top: scroll.scrollHeight, behavior: "auto" });
}

function updateComposer() {
  const validation = validateMessage(elements.message.value);
  elements["character-count"].textContent = `${validation.count.toLocaleString()} / 1,200`;
  elements["character-count"].classList.toggle("over-limit", validation.count > 1200);
  elements["composer-error"].textContent = elements.message.value && !validation.valid ? validation.reason : "";
  elements.message.setAttribute("aria-invalid", String(Boolean(elements.message.value && !validation.valid)));
  elements.send.disabled = pending || !validation.valid || Boolean(blockedReason) || engineReachable !== true;
  elements.message.disabled = pending;
  elements["send-label"].textContent = pending ? "Waiting…" : "Send";
  elements["cancel-response"].hidden = !pending;
  elements["context-notice"].hidden = !blockedReason;
  if (blockedReason) {
    elements["context-notice"].replaceChildren(node("span", "", blockedReason));
    const reset = node("button", "", "Start a new conversation");
    reset.type = "button";
    reset.addEventListener("click", askToReset);
    elements["context-notice"].append(reset);
  }
}

function updateConnection(kind, detail = "") {
  const labels = { checking: "Checking local engine", ready: "Local engine ready", partial: "Local engine limited", offline: "Engine disconnected" };
  elements["connection-status"].className = `connection-status ${kind === "partial" ? "offline" : kind}`;
  elements["connection-label"].textContent = labels[kind];
  elements["connection-notice"].hidden = kind === "ready" || kind === "checking";
  if (kind === "partial" || kind === "offline") {
    elements["connection-notice"].replaceChildren(node("span", "", detail));
    const retry = node("button", "", "Recheck connection");
    retry.type = "button";
    retry.addEventListener("click", checkConnection);
    elements["connection-notice"].append(retry);
  }
  updateComposer();
}

// A reachable-but-unready backend can still return its fixed support responses.
async function checkConnection() {
  const version = ++connectionVersion;
  connectionRequest?.abort();
  const controller = new AbortController();
  connectionRequest = controller;
  elements["check-connection"].disabled = true;
  updateConnection("checking");
  const timer = setTimeout(() => controller.abort(), 10000);
  try {
    const response = await fetch("/health/ready", { signal: controller.signal, cache: "no-store", credentials: "omit" });
    const body = await response.json();
    if (version !== connectionVersion) return;
    if (!body || typeof body !== "object" || typeof body.ready !== "boolean" || ![200, 503].includes(response.status)) {
      throw new Error("Unexpected readiness response.");
    }
    engineReachable = true;
    if (response.status === 200 && body.ready === true && body.error === null) {
      updateConnection("ready");
    } else {
      updateConnection("partial", "The local backend is reachable, but its model or corpus is not ready. Fixed support can still be returned; source-backed generation may fail. Open Ollama and check the backend documentation.");
    }
  } catch {
    if (version !== connectionVersion) return;
    engineReachable = false;
    updateConnection("offline", "The local engine could not be checked. Keep the backend terminal and Ollama running, then recheck. This page will not send your question to another service.");
  } finally {
    clearTimeout(timer);
    if (version === connectionVersion) {
      connectionRequest = null;
      elements["check-connection"].disabled = false;
    }
  }
}

function addUserMessage(message) {
  elements.welcome.hidden = true;
  const article = node("article", "message user");
  const content = node("div", "message-content");
  const meta = node("div", "message-meta");
  meta.append(node("strong", "", "YOU"));
  content.append(meta, node("div", "message-text", message));
  article.append(content);
  elements["message-list"].append(article);
  visibleMessageCount += 1;
  return article;
}

function assistantMessage(className = "", label = "") {
  const article = node("article", `message assistant ${className}`.trim());
  const avatar = node("span", "assistant-avatar", "S");
  avatar.setAttribute("aria-hidden", "true");
  const content = node("div", "message-content");
  const meta = node("div", "message-meta");
  meta.append(node("strong", "", "SOL"));
  if (label) meta.append(node("span", "response-label", label));
  content.append(meta);
  article.append(avatar, content);
  elements["message-list"].append(article);
  return { article, content };
}

function addPendingMessage() {
  const message = assistantMessage("pending", "Preparing a reflection");
  const line = node("div", "message-text");
  const dots = node("span", "loading-dots");
  dots.setAttribute("aria-hidden", "true");
  dots.append(node("i"), node("i"), node("i"));
  line.append(dots, node("span", "", "Reading passages and checking a reply…"));
  message.content.append(line, node("p", "pending-note", "The local model may take up to two minutes. No automatic retries."));
  return message.article;
}

// Combine exact passages and citation-only metadata without losing either representation.
function sourceEntries(body) {
  const passages = Array.isArray(body.source_passages) ? body.source_passages.filter((source) => source && typeof source === "object" && !Array.isArray(source)) : [];
  const citations = Array.isArray(body.citations) ? body.citations.filter((source) => source && typeof source === "object" && !Array.isArray(source)) : [];
  const result = passages.map((source) => ({ ...source }));
  for (const citation of citations) {
    if (!result.some((passage) => passage.citation && passage.citation === citation.citation)) result.push({ ...citation });
  }
  return result;
}

function appendSources(content, body) {
  const sources = sourceEntries(body);
  if (!sources.length) return;
  const details = node("details", "source-list");
  details.append(node("summary", "", `Read the source passages · ${sources.length}`));
  details.append(node("p", "source-note", "Passages supplied to this reply; not all may be cited. These references do not certify the answer’s interpretation."));
  for (const source of sources) {
    const card = node("section", "source-card");
    const heading = node("div", "source-heading");
    if (typeof source.citation === "string") heading.append(node("span", "citation-label", source.citation));
    heading.append(node("span", "source-title", typeof source.source_title === "string" && source.source_title ? source.source_title : "Extracted reference"));
    card.append(heading);
    if (source.source_locator !== null && source.source_locator !== undefined) {
      let locator;
      try { locator = typeof source.source_locator === "string" ? source.source_locator : JSON.stringify(source.source_locator); }
      catch { locator = "Locator could not be displayed."; }
      if (locator) card.append(node("div", "source-location", `Location: ${locator}`));
    }
    if (typeof source.source_role === "string") card.append(node("div", "source-role", `Source role: ${source.source_role.replaceAll("_", " ")} · routing hint, not authorship verification`));
    const offsets = source.text_offsets;
    if (offsets && Number.isInteger(offsets.start) && Number.isInteger(offsets.end_exclusive) && offsets.start >= 0 && offsets.end_exclusive >= offsets.start) {
      card.append(node("div", "source-offsets", `Canonical text characters ${offsets.start}–${offsets.end_exclusive} (end exclusive)`));
    }
    if (typeof source.text === "string" && source.text) {
      card.append(node("blockquote", "source-excerpt", source.text));
    } else {
      card.append(node("p", "source-no-excerpt", "Citation metadata only; no exact excerpt was returned for this reference."));
    }
    const url = safeSourceUrl(source.source_url);
    if (url) {
      const link = node("a", "source-link", "Open original source ↗");
      link.href = url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.referrerPolicy = "no-referrer";
      card.append(link);
    } else if (typeof source.source_url === "string" && source.source_url) {
      card.append(node("p", "source-no-excerpt", "The source URL is not a valid public HTTP(S) link and was not made clickable."));
    }
    details.append(card);
  }
  content.append(details);
}

function appendWarnings(content, body) {
  const warnings = Array.isArray(body.warnings) ? body.warnings.filter((warning) => typeof warning === "string" && warning) : [];
  if (!warnings.length) return;
  const details = node("details", "warnings");
  details.append(node("summary", "", `Important notes · ${warnings.length}`));
  const list = node("ul");
  warnings.forEach((warning) => list.append(node("li", "", warning)));
  details.append(list);
  if (["urgent_support", "treatment_boundary", "human_support"].includes(body.mode)) details.open = true;
  content.append(details);
}

function renderResponse(classification, body) {
  const { kind, mode } = classification;
  const message = assistantMessage(kind === "withheld" ? "withheld" : kind === "error" ? "failed" : "", modeLabel(mode));
  if (kind === "error") {
    message.content.append(node("div", "message-text", "No accepted reply was returned. Your question remains visible, but this exchange will not be silently dropped from context. Start a new conversation before trying again."));
    message.content.append(node("p", "error-detail", classification.errorMessage));
  } else {
    message.content.append(node("div", "message-text", classification.answer));
    if (kind === "withheld") message.content.append(node("p", "error-detail", "This is a fixed fallback, not an accepted generated answer. The rejected model text is not displayed."));
    appendWarnings(message.content, body);
    appendSources(message.content, body);
    if (body.llm_called === false) message.content.append(node("div", "request-id", "No model generation was used for this response."));
  }
  if (typeof body.request_id === "string") message.content.append(node("div", "request-id", `Request ${body.request_id}`));
  visibleMessageCount += 1;
}

function showTransportFailure(detail) {
  const message = assistantMessage("failed", "Reply not received");
  message.content.append(node("div", "message-text", detail));
  message.content.append(node("p", "error-detail", "The previous question stays visible. Start a new conversation to continue; failed context will not be silently removed."));
  visibleMessageCount += 1;
}

// The backend decides answer acceptance. Cancellation stops browser waiting, not inference.
async function sendMessage(event) {
  event.preventDefault();
  if (pending || blockedReason || engineReachable !== true) return;
  const question = elements.message.value;
  const validation = validateMessage(question);
  const historyValidation = validateHistory(history);
  if (!validation.valid) { updateComposer(); return; }
  if (!historyValidation.valid) {
    blockedReason = historyValidation.reason;
    updateComposer();
    return;
  }
  const version = conversationVersion;
  const controller = new AbortController();
  const request = { controller, version, cancelledByUser: false, timedOut: false };
  activeRequest = request;
  pending = true;
  addUserMessage(question);
  const pendingArticle = addPendingMessage();
  elements.message.value = "";
  updateComposer();
  scrollToLatest();
  const timer = setTimeout(() => { request.timedOut = true; controller.abort(); }, 150000);
  try {
    const response = await fetch("/api/v1/chat", {
      method: "POST", headers: { "Content-Type": "application/json" }, credentials: "omit", cache: "no-store",
      signal: controller.signal, body: JSON.stringify({ message: question, history }),
    });
    // A 503 can intentionally contain a safe fixed fallback. Always parse JSON.
    const body = await response.json();
    if (version !== conversationVersion || activeRequest !== request) return;
    if (controller.signal.aborted) throw new Error("The browser request was cancelled.");
    pendingArticle.remove();
    const classification = classifyChatResponse(response.status, body);
    renderResponse(classification, body && typeof body === "object" ? body : {});
    if (classification.kind === "accepted") {
      const next = historyAfterAccepted(history, question, classification.answer);
      history = next.history;
      if (!next.validation.valid) blockedReason = next.validation.reason;
    } else {
      blockedReason = "This exchange did not produce an accepted answer. Continuing would omit its question from the backend context. Please explicitly start a new conversation.";
    }
  } catch {
    if (version !== conversationVersion || activeRequest !== request) return;
    pendingArticle.remove();
    const detail = request.cancelledByUser
      ? "You stopped waiting. This cancelled the browser request, but the model may still be generating. No unreceived answer has been accepted."
      : request.timedOut
        ? "The browser stopped waiting after 150 seconds. The model may still be generating; no complete answer was received or accepted."
        : "The local backend could not return a readable reply. This page did not switch to a remote model or send your question elsewhere.";
    showTransportFailure(detail);
    blockedReason = "This exchange is incomplete. Start a new conversation before another message so the interface does not silently lose your question or safety context.";
  } finally {
    clearTimeout(timer);
    if (version === conversationVersion && activeRequest === request) {
      activeRequest = null;
      pending = false;
      updateComposer();
      scrollToLatest();
      if (!blockedReason) elements.message.focus();
    }
  }
}

function resetConversation() {
  conversationVersion += 1;
  activeRequest?.controller.abort();
  activeRequest = null;
  pending = false;
  history = [];
  blockedReason = "";
  visibleMessageCount = 0;
  elements["message-list"].replaceChildren();
  elements.message.value = "";
  elements.welcome.hidden = false;
  updateComposer();
  elements["conversation-scroll"].scrollTo({ top: 0, behavior: "auto" });
  elements.message.focus();
}

function askToReset() {
  if (!visibleMessageCount && !pending && !elements.message.value) {
    resetConversation();
    return;
  }
  elements["reset-dialog"].returnValue = "cancel";
  if (!elements["reset-dialog"].open) elements["reset-dialog"].showModal();
}

// Bind the UI once. Suggested prompts fill the composer; they never send automatically.
elements["chat-form"].addEventListener("submit", sendMessage);
elements.message.addEventListener("input", updateComposer);
elements.message.addEventListener("compositionstart", () => { composing = true; });
elements.message.addEventListener("compositionend", () => { composing = false; });
elements.message.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing && !composing && event.keyCode !== 229) {
    event.preventDefault();
    if (!elements.send.disabled) elements["chat-form"].requestSubmit();
  }
});
elements["cancel-response"].addEventListener("click", () => {
  if (!activeRequest) return;
  activeRequest.cancelledByUser = true;
  activeRequest.controller.abort();
});
elements["check-connection"].addEventListener("click", checkConnection);
elements["new-conversation"].addEventListener("click", askToReset);
elements["new-conversation-mobile"].addEventListener("click", askToReset);
elements["reset-dialog"].addEventListener("close", () => {
  if (elements["reset-dialog"].returnValue === "reset") resetConversation();
});
// Explicitly close the dialog: no navigation or form submission is required.
for (const button of elements["reset-dialog"].querySelectorAll("button[value]")) {
  button.type = "button";
  button.addEventListener("click", () => elements["reset-dialog"].close(button.value));
}
elements["show-help"].addEventListener("click", () => {
  const hidden = !elements["help-panel"].hidden;
  elements["help-panel"].hidden = hidden;
  elements["show-help"].setAttribute("aria-expanded", String(!hidden));
  elements["show-help"].textContent = hidden ? "Know the limits" : "Close details";
});
for (const suggestion of document.querySelectorAll("button[data-prompt]")) {
  suggestion.addEventListener("click", () => {
    if (pending) return;
    elements.message.value = suggestion.dataset.prompt;
    updateComposer();
    elements.message.focus();
  });
}

updateComposer();
checkConnection();
