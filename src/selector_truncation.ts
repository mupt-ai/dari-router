// Selector prompt sizing. The selector can see a front-trimmed view of the
// conversation; the eventual provider call still receives the full request.

import { isRecord, type ChatMessage } from "./types.js";

const SELECTOR_TRUNCATION_MARKER = "<earlier messages truncated for routing>";
const TASK_TRUNCATION_MARKER = "\n...[original task truncated for routing]...\n";
const TASK_OMITTED_PLACEHOLDER = "<original task omitted for routing>";
const MARKER_MESSAGE: ChatMessage = { role: "user", content: SELECTOR_TRUNCATION_MARKER };

// Maximum removable history while keeping the latest message intact. Sizing
// uses this before compacting a separately retained task or the latest turn.
export function olderMessageChars(messages: ChatMessage[]): number {
  if (messages.length <= 1) return 0;
  return Math.max(
    0,
    JSON.stringify(messages).length - JSON.stringify([MARKER_MESSAGE, messages.at(-1)]).length,
  );
}

// Drop oldest messages first — in multi-turn conversations the latest
// messages carry the routing signal.
export function trimMessagesFromFront(messages: ChatMessage[], dropChars: number): ChatMessage[] {
  const targetLength = Math.max(0, JSON.stringify(messages).length - Math.max(0, dropChars));
  const kept = [...messages];
  while (kept.length > 1 && JSON.stringify([MARKER_MESSAGE, ...kept]).length > targetLength) {
    kept.shift();
  }

  const marked = [MARKER_MESSAGE, ...kept];
  if (JSON.stringify(marked).length <= targetLength) return marked;

  if (kept.length === 1) {
    const first = kept[0]!;
    const trimmableContent = trimSourceContent(first);
    if (trimmableContent === null) return [MARKER_MESSAGE];
    const prefix = `${SELECTOR_TRUNCATION_MARKER}\n`;
    let content = `${prefix}${trimmableContent}`;
    while (content.length > prefix.length) {
      const excess = JSON.stringify([{ ...first, content }]).length - targetLength;
      if (excess <= 0) break;
      content = `${prefix}${content.slice(prefix.length + excess)}`;
    }
    return [{ ...first, content }];
  }

  return [MARKER_MESSAGE];
}

// The original task normally stays intact even when conversation history is
// compacted. If that retained task alone exceeds the selector window, preserve
// both ends so the request and its trailing constraints remain visible.
export function trimTaskMessage(task: ChatMessage, dropChars: number): ChatMessage {
  const targetLength = Math.max(0, JSON.stringify(task).length - Math.max(0, dropChars));
  if (JSON.stringify(task).length <= targetLength) return task;

  const source = trimSourceContent(task);
  if (source === null) return { role: task.role, content: TASK_OMITTED_PLACEHOLDER };

  let contentChars = source.length;
  while (contentChars > 0) {
    const candidate: ChatMessage = {
      role: task.role,
      content: elideTaskMiddle(source, contentChars),
    };
    const excess = JSON.stringify(candidate).length - targetLength;
    if (excess <= 0) return candidate;
    contentChars = Math.max(0, contentChars - excess);
  }
  return { role: task.role, content: TASK_OMITTED_PLACEHOLDER };
}

function elideTaskMiddle(value: string, maxChars: number): string {
  if (value.length <= maxChars) return value;
  if (maxChars <= TASK_TRUNCATION_MARKER.length) {
    return TASK_TRUNCATION_MARKER.slice(0, maxChars);
  }
  const retainedChars = maxChars - TASK_TRUNCATION_MARKER.length;
  const prefixChars = Math.ceil(retainedChars / 2);
  return `${value.slice(0, prefixChars)}${TASK_TRUNCATION_MARKER}${value.slice(-(retainedChars - prefixChars))}`;
}

function trimSourceContent(message: ChatMessage): string | null {
  if (typeof message.content === "string") return message.content;
  if (!Array.isArray(message.content)) return null;
  return message.content.map(formatContentPartForTrim).join("\n");
}

function formatContentPartForTrim(part: Record<string, unknown>): string {
  if (isRecord(part) && part.type === "text" && typeof part.text === "string") return part.text;
  if (isRecord(part) && part.type === "image_url") return "<image omitted>";
  return JSON.stringify(part);
}
