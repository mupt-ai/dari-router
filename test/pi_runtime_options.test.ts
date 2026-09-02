import { expect, test } from "bun:test";
import type {
  Api,
  AssistantMessage,
  AssistantMessageEvent,
  Context,
  Model,
  SimpleStreamOptions,
} from "@mupt-ai/pi-ai";

import {
  createPiRuntime,
  createRouter,
  type PiCredentialInput,
  type PiModelRegistry,
  type PiRuntime,
  type RouterCompletion,
  type RouterRequest,
} from "../src/index.js";
import { createDariRoutingPolicy } from "../src/policy-engine.js";
import { encodeProviderContinuationState } from "../src/protocols.js";
import { piContext } from "../src/pi_context.js";
import { piOptions } from "../src/pi_options.js";
import {
  decodeOpenAIReasoningSignature,
  encodeOpenAIReasoningSignature,
} from "../src/pi_reasoning_signature.js";
import {
  ANTHROPIC_ID,
  ANTHROPIC_MODEL,
  assistant,
  asyncEvents,
  AZURE_MODEL,
  BEDROCK_MODEL,
  COMPLETIONS_MODEL,
  fakeRegistry,
  GOOGLE_MODEL,
  jsonRequest,
  OPENAI_ID,
  OPENAI_MODEL,
  piExecution,
  piModel,
  routerRequest,
  SELECTOR_ID,
  SELECTOR_MODEL,
  singleModelRouter,
  sseJsonFrames,
  VERTEX_MODEL,
} from "./pi_runtime.fixtures.js";

test("piOptions rejects options the fork hard-throws for, per API", async () => {
  const gated: Array<[Model<Api>, Partial<RouterRequest>]> = [
    [OPENAI_MODEL, { generation: { stop: ["END"] } }],
    [AZURE_MODEL, { generation: { stop: "END" } }],
    [ANTHROPIC_MODEL, { generation: { topP: 0.5 } }],
    [GOOGLE_MODEL, { toolChoice: { type: "tool", name: "lookup" } }],
    [VERTEX_MODEL, { toolChoice: { type: "tool", name: "lookup" } }],
    [BEDROCK_MODEL, { responseFormat: { type: "json_object" } }],
  ];
  for (const [model, overrides] of gated) {
    expect(
      piOptions({ apiKey: "test-key" }, piExecution(routerRequest(overrides), model), model),
    ).rejects.toMatchObject({
      name: "RouterFrameworkError",
      kind: "invalid_request",
      code: "pi_option_unsupported",
      message: expect.stringContaining(`'${model.api}'`),
    });
  }

  // The same combinations pass on APIs that support them natively.
  const stopOptions = await piOptions(
    { apiKey: "test-key" },
    piExecution(routerRequest({ generation: { stop: ["END"], topP: 0.5 } }), COMPLETIONS_MODEL),
    COMPLETIONS_MODEL,
  );
  expect(stopOptions).toMatchObject({ stop: ["END"], topP: 0.5 });
});

test("Bedrock reasoning is delegated to Pi options", async () => {
  const model = piModel({
    id: "global.openai.gpt-5.6-sol",
    provider: "amazon-bedrock",
    api: "bedrock-converse-stream",
    reasoning: true,
  });

  for (const level of ["off", "low", "medium", "high", "xhigh", "max"] as const) {
    const request = routerRequest({ reasoning: { effort: level } });
    const execution = { ...piExecution(request, model), reasoningEffort: level };
    const options = await piOptions({ apiKey: "test-key" }, execution, model);

    expect(options.reasoning).toBe(level === "off" ? undefined : level);
    expect(options.onPayload).toBeUndefined();
  }
});

test("unsupported Pi options surface as 400s through the router", async () => {
  const registry = fakeRegistry([ANTHROPIC_MODEL], {});
  const runtime = await createPiRuntime({ registry, apiKey: "test-key" });
  const router = singleModelRouter(runtime, ANTHROPIC_ID);

  const response = await router.fetch(jsonRequest("/v1/messages", {
    model: "my-router",
    max_tokens: 64,
    messages: [{ role: "user", content: "hello" }],
    top_p: 0.5,
  }));
  expect(response.status).toBe(400);
  expect(await response.json()).toMatchObject({
    type: "error",
    error: { type: "invalid_request_error" },
  });
});

test("json_object degrades to a permissive schema on anthropic without payload patching", async () => {
  const options = await piOptions(
    { apiKey: "test-key" },
    piExecution(routerRequest({ responseFormat: { type: "json_object" } }), ANTHROPIC_MODEL),
    ANTHROPIC_MODEL,
  );
  expect(options.responseFormat).toEqual({
    type: "json_schema",
    jsonSchema: { name: "response", schema: { type: "object" } },
  });
  expect(options.onPayload).toBeUndefined();

  const schemaOptions = await piOptions(
    { apiKey: "test-key" },
    piExecution(
      routerRequest({
        responseFormat: { type: "json_schema", name: "answer", schema: { type: "object" } },
      }),
      ANTHROPIC_MODEL,
    ),
    ANTHROPIC_MODEL,
  );
  expect(schemaOptions.responseFormat).toEqual({
    type: "json_schema",
    jsonSchema: { name: "answer", schema: { type: "object" } },
  });
  expect(schemaOptions.onPayload).toBeUndefined();
});

test("started-but-unended thinking blocks reconcile from the terminal message", async () => {
  const terminal = assistant(ANTHROPIC_MODEL, {
    content: [
      { type: "thinking", thinking: "Let me think.", thinkingSignature: "anthropic-sig" },
      { type: "text", text: "Answer." },
    ],
    stopReason: "stop",
  });
  const registry = fakeRegistry([ANTHROPIC_MODEL], {
    stream: (model) => asyncEvents([
      { type: "start", partial: assistant(model) },
      { type: "thinking_start", contentIndex: 0, partial: terminal },
      { type: "thinking_delta", contentIndex: 0, delta: "Let me ", partial: terminal },
      // No thinking_end and no streamed text: the terminal message must
      // complete both blocks in order.
      { type: "done", reason: "stop", message: terminal },
    ]),
  });
  const runtime = await createPiRuntime({ registry, apiKey: "test-key" });
  const router = singleModelRouter(runtime, ANTHROPIC_ID);

  const response = await router.fetch(jsonRequest("/v1/messages", {
    model: "my-router",
    max_tokens: 128,
    messages: [{ role: "user", content: "think and answer" }],
    stream: true,
  }));
  const body = await response.text();

  expect(response.status).toBe(200);
  expect(body).not.toContain("event: error");
  expect(body).toContain('"type":"thinking_delta","thinking":"Let me "');
  expect(body).toContain('"type":"thinking_delta","thinking":"think."');
  expect(body).toContain('"type":"signature_delta","signature":"dari-pcs-v1.');
  expect(body).toContain('"type":"text_delta","text":"Answer."');
  expect(body).toContain("event: message_stop");
});

test("invalid internally-built selector requests blame configuration, not the client", async () => {
  const registry = fakeRegistry([SELECTOR_MODEL], {});
  const runtime = await createPiRuntime({ registry, apiKey: "test-key" });

  expect(
    runtime.select({ model: SELECTOR_ID, messages: [] }),
  ).rejects.toMatchObject({
    name: "RouterFrameworkError",
    kind: "configuration",
    code: "pi_selector_request_invalid",
  });
});

test("replayed tool calls with empty-string arguments become empty objects", async () => {
  let captured: Context | undefined;
  const registry = fakeRegistry([OPENAI_MODEL], {
    complete: async (model, nextContext) => {
      captured = nextContext;
      return assistant(model, { content: [{ type: "text", text: "ok" }], stopReason: "stop" });
    },
  });
  const runtime = await createPiRuntime({ registry, apiKey: "test-key" });
  const router = singleModelRouter(runtime, OPENAI_ID);

  const response = await router.fetch(jsonRequest("/v1/chat/completions", {
    model: "my-router",
    messages: [
      { role: "user", content: "look it up" },
      {
        role: "assistant",
        content: null,
        tool_calls: [{ id: "call_1", type: "function", function: { name: "lookup", arguments: "" } }],
      },
      { role: "tool", tool_call_id: "call_1", content: "result" },
    ],
    tools: [{ type: "function", function: { name: "lookup", parameters: { type: "object" } } }],
  }));

  expect(response.status).toBe(200);
  const assistantMessage = captured!.messages.find((message) => message.role === "assistant") as
    | { content: Array<Record<string, unknown>> }
    | undefined;
  expect(assistantMessage?.content).toContainEqual({
    type: "toolCall",
    id: "call_1",
    name: "lookup",
    arguments: {},
  });
});

test("hosted tool history fails closed on non-Responses APIs and forwards on Responses", () => {
  const payload = { type: "web_search_call", id: "ws_1", status: "completed" };
  const request = routerRequest({
    items: [
      { type: "message", role: "user", content: [{ type: "text", text: "search please" }] },
      {
        type: "hosted_tool_call",
        id: "ws_1",
        tool: "web_search",
        providerType: "web_search_call",
        status: "completed",
        payload,
      },
      { type: "message", role: "user", content: [{ type: "text", text: "and then?" }] },
    ],
  });

  // Anthropic Messages cannot represent the hosted call: dropping it would
  // silently lose a conversation turn, so the input path fails closed like
  // the Anthropic response serialization does.
  let error: unknown;
  try {
    piContext(request, ANTHROPIC_MODEL);
  } catch (caught) {
    error = caught;
  }
  expect(error).toMatchObject({
    name: "RouterFrameworkError",
    kind: "configuration",
    code: "hosted_tool_call_unrepresentable",
  });

  // OpenAI Responses replays it as a provider item.
  const context = piContext(request, OPENAI_MODEL);
  const assistantMessage = context.messages.find((message) => message.role === "assistant") as
    | { content: Array<Record<string, unknown>> }
    | undefined;
  expect(assistantMessage?.content).toContainEqual({ type: "providerItem", item: payload });
});

test("hosted web_search executions round-trip from Pi output through the OpenAI wire", async () => {
  const payload = {
    type: "web_search_call",
    id: "ws_1",
    status: "completed",
    action: { type: "search", query: "dari router" },
  };
  const contexts: Context[] = [];
  const registry = fakeRegistry([OPENAI_MODEL], {
    complete: async (model, nextContext) => {
      contexts.push(nextContext);
      return assistant(model, {
        content: [
          { type: "text", text: "Searching." },
          { type: "providerItem", item: payload },
        ] as unknown as AssistantMessage["content"],
        stopReason: "stop",
      });
    },
  });
  const runtime = await createPiRuntime({ registry, apiKey: "test-key" });
  const router = singleModelRouter(runtime, OPENAI_ID);

  const first = await router.fetch(jsonRequest("/v1/chat/completions", {
    model: "my-router",
    messages: [{ role: "user", content: "search please" }],
  }));
  expect(first.status).toBe(200);
  const body = await first.json() as { choices: Array<{ message: Record<string, unknown> }> };
  const message = body.choices[0]!.message;
  expect(message.content).toBe("Searching.");
  const calls = message.tool_calls as Array<{
    id: string;
    type: string;
    function: { name: string; arguments: string };
  }>;
  expect(calls).toHaveLength(1);
  expect(calls[0]).toMatchObject({
    id: "ws_1",
    type: "function",
    function: { name: "web_search" },
  });
  expect(JSON.parse(calls[0]!.function.arguments)).toEqual(payload);

  const replay = await router.fetch(jsonRequest("/v1/chat/completions", {
    model: "my-router",
    messages: [
      { role: "user", content: "search please" },
      { role: "assistant", content: message.content, tool_calls: calls },
      { role: "user", content: "and then?" },
    ],
  }));
  expect(replay.status).toBe(200);
  const replayedAssistant = contexts[1]!.messages.find((item) => item.role === "assistant") as
    | { content: Array<Record<string, unknown>> }
    | undefined;
  expect(replayedAssistant?.content).toEqual([
    { type: "text", text: "Searching." },
    { type: "providerItem", item: payload },
  ]);
});

test("streamed hosted web_search executions surface as complete OpenAI tool-call chunks", async () => {
  const payload = { type: "web_search_call", id: "ws_2", status: "completed" };
  const registry = fakeRegistry([OPENAI_MODEL], {
    stream: (model) => {
      const message = assistant(model, {
        content: [
          { type: "text", text: "Looking." },
          { type: "providerItem", item: payload },
        ] as unknown as AssistantMessage["content"],
        stopReason: "stop",
      });
      return asyncEvents([
        { type: "start", partial: assistant(model) },
        { type: "text_start", contentIndex: 0, partial: message },
        { type: "text_delta", contentIndex: 0, delta: "Looking.", partial: message },
        { type: "text_end", contentIndex: 0, content: "Looking.", partial: message },
        { type: "done", reason: "stop", message },
      ]);
    },
  });
  const runtime = await createPiRuntime({ registry, apiKey: "test-key" });
  const router = singleModelRouter(runtime, OPENAI_ID);

  const response = await router.fetch(jsonRequest("/v1/chat/completions", {
    model: "my-router",
    messages: [{ role: "user", content: "search please" }],
    stream: true,
  }));
  expect(response.status).toBe(200);
  const chunks = sseJsonFrames(await response.text());
  const toolDeltas = chunks.flatMap((chunk) => {
    const choices = chunk.choices as Array<{ delta?: { tool_calls?: unknown[] } }> | undefined;
    return choices?.[0]?.delta?.tool_calls ?? [];
  });
  expect(toolDeltas).toEqual([{
    index: 0,
    id: "ws_2",
    type: "function",
    function: { name: "web_search", arguments: JSON.stringify(payload) },
  }]);
});

test("streamed hosted web_search executions fail closed on the Anthropic protocol", async () => {
  const registry = fakeRegistry([OPENAI_MODEL], {
    stream: (model) => {
      const message = assistant(model, {
        content: [
          { type: "providerItem", item: { type: "web_search_call", id: "ws_3", status: "completed" } },
        ] as unknown as AssistantMessage["content"],
        stopReason: "stop",
      });
      return asyncEvents([
        { type: "start", partial: assistant(model) },
        { type: "done", reason: "stop", message },
      ]);
    },
  });
  const runtime = await createPiRuntime({ registry, apiKey: "test-key" });
  const router = singleModelRouter(runtime, OPENAI_ID);

  const response = await router.fetch(jsonRequest("/v1/messages", {
    model: "my-router",
    max_tokens: 128,
    messages: [{ role: "user", content: "search please" }],
    stream: true,
  }));
  const body = await response.text();
  expect(body).toContain("event: error");
  expect(body).toContain("not representable on the anthropic_messages protocol");
});

test("unsupported Pi provider items fail closed instead of being dropped", async () => {
  const registry = fakeRegistry([OPENAI_MODEL], {
    complete: async (model) => assistant(model, {
      content: [
        { type: "providerItem", item: { type: "file_search_call", id: "fs_1" } },
      ] as unknown as AssistantMessage["content"],
      stopReason: "stop",
    }),
  });
  const runtime = await createPiRuntime({ registry, apiKey: "test-key" });
  const router = singleModelRouter(runtime, OPENAI_ID);

  const response = await router.fetch(jsonRequest("/v1/chat/completions", {
    model: "my-router",
    messages: [{ role: "user", content: "search please" }],
  }));
  expect(response.status).toBe(502);
  expect(await response.json()).toMatchObject({
    error: {
      code: "pi_hosted_tool_unsupported",
      message: expect.stringContaining("file_search_call"),
    },
  });
});

test("the openai reasoning signature codec round-trips", () => {
  const encoded = encodeOpenAIReasoningSignature({
    id: "rs_1",
    summary: ["First pass."],
    content: ["Deeper detail."],
    encryptedContent: "enc-blob",
  });
  expect(JSON.parse(encoded)).toEqual({
    type: "reasoning",
    id: "rs_1",
    summary: [{ type: "summary_text", text: "First pass." }],
    content: [{ type: "reasoning_text", text: "Deeper detail." }],
    encrypted_content: "enc-blob",
  });
  expect(decodeOpenAIReasoningSignature(encoded)).toEqual({
    id: "rs_1",
    encryptedContent: "enc-blob",
  });
  expect(decodeOpenAIReasoningSignature("not json")).toBeNull();
  expect(decodeOpenAIReasoningSignature('{"type":"other"}')).toBeNull();
});
