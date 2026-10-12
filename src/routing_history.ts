// Router-only history compaction shared by production routing and GRPO
// training. Provider-bound requests keep the complete conversation.

// Characters of conversation the anonymous policy selectors see.
export const ROUTER_HISTORY_CHAR_BUDGET = 16_000;
// A session has no single task: the user's intent is spread across their
// messages, so compaction keeps every one, each shortened to this size.
export const ROUTER_USER_MESSAGE_CHAR_CAP = 2_000;

type RoutingHistoryMessage = {
  role?: unknown;
  content?: unknown;
  [key: string]: unknown;
};

// Keeps the leading instructions verbatim, every user message (capped, newest
// first, within half the budget), and the most recent agent turns in what is
// left, all in their original order.
export function compactRoutingHistory<T extends RoutingHistoryMessage>(
  messages: T[],
  historyCharBudget: number | undefined,
): T[] {
  if (historyCharBudget === undefined || messages.length <= 1) return messages;
  if (!Number.isSafeInteger(historyCharBudget) || historyCharBudget < 512) {
    throw new Error("GRPO routing history character budget must be an integer of at least 512");
  }
  const leadRole = messages[0]?.role;
  const lead = leadRole === "system" || leadRole === "developer" ? messages.slice(0, 1) : [];
  const history = messages.slice(lead.length);
  const sizes = history.map((message) => JSON.stringify(message).length);
  const totalChars = sizes.reduce((total, size) => total + size, 0);
  if (totalChars <= historyCharBudget) return messages;

  // The final marker embeds omission counts, so budget against a
  // worst-case-width placeholder to keep the result within the char budget.
  const markerContent = (omittedMessages: number, omittedChars: number): string =>
    `[Router-only compaction omitted ${omittedMessages} earlier messages (~${omittedChars} chars); ` +
    "kept the user's messages and the most recent context.]";
  const markerReserve = JSON.stringify({
    role: "system",
    content: markerContent(99_999_999, 999_999_999),
  }).length;
  let remainingChars = historyCharBudget - markerReserve;
  const kept = new Map<number, T>();

  let userChars = Math.floor(remainingChars / 2);
  for (let index = history.length - 1; index >= 0; index -= 1) {
    const message = history[index] as T;
    if (message.role !== "user") continue;
    const limit = Math.min(ROUTER_USER_MESSAGE_CHAR_CAP, userChars);
    const capped = sizes[index]! <= limit ? message : compactRoutingMessage(message, limit);
    if (capped === undefined) break;
    const cappedChars = JSON.stringify(capped).length;
    kept.set(index, capped);
    userChars -= cappedChars;
    remainingChars -= cappedChars;
  }

  for (let index = history.length - 1; index >= 0 && remainingChars > 0; index -= 1) {
    const message = history[index] as T;
    if (message.role === "user") continue;
    if (sizes[index]! <= remainingChars) {
      kept.set(index, message);
      remainingChars -= sizes[index]!;
      continue;
    }
    const compacted = compactRoutingMessage(message, remainingChars);
    if (compacted !== undefined) kept.set(index, compacted);
    break;
  }

  // Providers reject tool results whose originating assistant tool_calls
  // message was compacted away, so a tool result stays only beside it.
  const retained: T[] = [];
  for (const index of [...kept.keys()].sort((left, right) => left - right)) {
    if (history[index]?.role === "tool") {
      let owner = index - 1;
      while (owner >= 0 && history[owner]?.role === "tool") owner -= 1;
      if (owner < 0 || !kept.has(owner)) continue;
    }
    retained.push(kept.get(index)!);
  }
  const retainedChars = retained.reduce(
    (total, message) => total + JSON.stringify(message).length,
    0,
  );
  const marker = {
    role: "system",
    content: markerContent(
      history.length - retained.length,
      Math.max(totalChars - retainedChars, 0),
    ),
  } as T;
  return [...lead, marker, ...retained];
}

function compactRoutingMessage<T extends RoutingHistoryMessage>(
  message: T,
  maxChars: number,
): T | undefined {
  const content = message.content;
  if (Array.isArray(content)) {
    return compactRoutingMessageParts(message, content, maxChars);
  }
  const text = contentText(content);
  const shell = { ...message, content: "" };
  // Leave room for JSON escaping introduced by the truncation marker.
  const availableChars = maxChars - JSON.stringify(shell).length - 32;
  if (availableChars < 128 || text.length === 0) return undefined;
  const compacted = { ...message, content: elideMiddle(text, availableChars) };
  return JSON.stringify(compacted).length <= maxChars ? compacted : undefined;
}

function compactRoutingMessageParts<T extends RoutingHistoryMessage>(
  message: T,
  content: unknown[],
  maxChars: number,
): T | undefined {
  const text = contentText(content);
  const nonTextContent = content.filter((part) => !isTextPart(part));
  const shell = { ...message, content: nonTextContent };
  const shellChars = JSON.stringify(shell).length;
  if (shellChars > maxChars) return undefined;
  if (text.length === 0) return shell;

  const availableChars = maxChars - shellChars - 32;
  if (availableChars < 128) {
    return nonTextContent.length > 0 ? shell : undefined;
  }
  const textPart = { type: "text", text: elideMiddle(text, availableChars) };
  const compacted = {
    ...message,
    content: compactContentParts(content, textPart),
  };
  if (JSON.stringify(compacted).length <= maxChars) return compacted;
  return nonTextContent.length > 0 ? shell : undefined;
}

function compactContentParts(
  content: unknown[],
  textPart: Record<string, unknown>,
): unknown[] {
  let insertedText = false;
  const compacted: unknown[] = [];
  for (const part of content) {
    if (isTextPart(part)) {
      if (!insertedText) compacted.push(textPart);
      insertedText = true;
      continue;
    }
    compacted.push(part);
  }
  if (!insertedText) compacted.push(textPart);
  return compacted;
}

function isTextPart(value: unknown): value is Record<string, unknown> {
  return (
    typeof value === "object" &&
    value !== null &&
    !Array.isArray(value) &&
    (value as Record<string, unknown>).type === "text" &&
    typeof (value as Record<string, unknown>).text === "string"
  );
}

function contentText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .flatMap((block) => {
      if (typeof block !== "object" || block === null || Array.isArray(block)) return [];
      const text = (block as Record<string, unknown>).text;
      return typeof text === "string" ? [text] : [];
    })
    .join("\n");
}

function elideMiddle(value: string, maxChars: number): string {
  if (value.length <= maxChars) return value;
  const marker = "\n...[router history truncated]...\n";
  if (maxChars <= marker.length) return value.slice(-maxChars);
  const retainedChars = maxChars - marker.length;
  const prefixChars = Math.floor(retainedChars / 2);
  return `${value.slice(0, prefixChars)}${marker}${value.slice(-(retainedChars - prefixChars))}`;
}
