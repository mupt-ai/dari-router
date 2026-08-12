import { expect, test } from "bun:test";

import { buildSizedSelectorRequest } from "../src/selector_request.js";
import { selectorSafeMessages } from "../src/selector_input.js";
import type { ChatMessage, RoutingCandidate } from "../src/types.js";

const CANDIDATES: RoutingCandidate[] = [
  { model: "openai/gpt-4.1-mini", reasoningEffort: "medium" },
];

function build(messages: ChatMessage[], contextWindowChars: number) {
  return buildSizedSelectorRequest({
    candidates: CANDIDATES,
    selectorModel: "selector/model",
    contextWindowChars,
    messages,
  });
}

test("small conversations pass through untrimmed", () => {
  const messages: ChatMessage[] = [{ role: "user", content: "hello" }];
  const built = build(messages, 100_000);
  expect(built.selectorInput.messages).toEqual(messages);
  expect(built.selectorRequest.messages?.[0]?.role).toBe("system");
  expect(built.selectorRequest.response_format?.type).toBe("json_schema");
});

test("oversized conversations trim oldest messages first", () => {
  const built = build(
    [
      { role: "user", content: "x".repeat(24_000) },
      { role: "assistant", content: "old reply" },
      { role: "user", content: "the question that matters" },
    ],
    4_000,
  );
  const sent = built.selectorInput.messages;
  expect(String(sent[0]?.content)).toBe("<earlier messages truncated for routing>");
  expect(sent.some((message) => String(message.content).includes("the question that matters"))).toBe(true);
  expect(sent.some((message) => String(message.content).includes("x".repeat(16_000)))).toBe(false);
});

test("throws a configuration error when trimming cannot converge", () => {
  expect(() => build([{ role: "user", content: "hello" }], 10)).toThrow(
    "Selector prompt cannot fit configured context window for selector/model.",
  );
});

test("image bytes never reach the selector serialization", () => {
  const dataUrl = `data:image/png;base64,${"A".repeat(5000)}`;
  const messages = selectorSafeMessages([
    {
      role: "user",
      content: [
        { type: "text", text: "what is in this image?" },
        { type: "image_url", image_url: { url: dataUrl } },
      ],
    },
  ]);
  const built = build(messages, 100_000);
  const serialized = JSON.stringify(built.selectorInput.messages);
  expect(serialized).toContain("<image omitted>");
  expect(serialized).not.toContain("A".repeat(100));
});
