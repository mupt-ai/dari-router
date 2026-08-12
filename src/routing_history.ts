// Router-only history compaction shared by production routing and GRPO
// training. Provider-bound requests keep the complete conversation.

type RoutingHistoryMessage = {
  role?: unknown;
  content?: unknown;
  [key: string]: unknown;
};

export function compactRoutingHistory<T extends RoutingHistoryMessage>(
  messages: T[],
  historyCharBudget: number | undefined,
): T[] {
  if (historyCharBudget === undefined || messages.length <= 1) return messages;
  if (!Number.isSafeInteger(historyCharBudget) || historyCharBudget < 512) {
    throw new Error("GRPO routing history character budget must be an integer of at least 512");
  }
  const first = messages[0] as T;
  const history = messages.slice(1);
  const totalChars = history.reduce(
    (total, message) => total + JSON.stringify(message).length,
    0,
  );
  if (totalChars <= historyCharBudget) return messages;

  // The final marker embeds omission counts, so budget the tail against a
  // worst-case-width placeholder to keep the result within the char budget.
  const markerContent = (omittedMessages: number, omittedChars: number): string =>
    `[Router-only compaction omitted ${omittedMessages} earlier messages (~${omittedChars} chars); ` +
    "retained the task and most recent context.]";
  const markerReserve = JSON.stringify({
    role: "system",
    content: markerContent(99_999_999, 999_999_999),
  }).length;
  let remainingChars = historyCharBudget - markerReserve;
  const tail: T[] = [];
  for (let index = history.length - 1; index >= 0 && remainingChars > 0; index -= 1) {
    const message = history[index] as T;
    const serializedChars = JSON.stringify(message).length;
    if (serializedChars <= remainingChars) {
      tail.unshift(message);
      remainingChars -= serializedChars;
      continue;
    }
    const compacted = compactRoutingMessage(message, remainingChars);
    if (compacted !== undefined) tail.unshift(compacted);
    break;
  }
  // Providers reject tool results whose originating assistant tool_calls
  // message was compacted away, so the retained tail must never lead with one.
  while (tail.length > 0 && tail[0]?.role === "tool") {
    tail.shift();
  }
  const retainedChars = tail.reduce(
    (total, message) => total + JSON.stringify(message).length,
    0,
  );
  const marker = {
    role: "system",
    content: markerContent(
      history.length - tail.length,
      Math.max(totalChars - retainedChars, 0),
    ),
  } as T;
  return [first, marker, ...tail];
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
