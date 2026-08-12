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

test("Pi runtime executes a declared model end to end without a custom executor", async () => {
  let context: Context | undefined;
  let streamOptions: SimpleStreamOptions | undefined;
  let patchedPayload: unknown;
  const credentials: PiCredentialInput[] = [];
  const registry = fakeRegistry([OPENAI_MODEL], {
    complete: async (model, nextContext, options) => {
      context = nextContext;
      streamOptions = options;
      patchedPayload = await options?.onPayload?.({
        input: [{ type: "input_image", image_url: "data:image/png;base64,YQ==" }],
        tools: [{ name: "lookup" }],
        text: {},
      }, model);
      return assistant(model, {
        content: [{ type: "text", text: "done" }],
        stopReason: "stop",
      });
    },
  });
  const runtime = await createPiRuntime({
    registry,
    apiKey: (input) => {
      credentials.push(input);
      return "test-key";
    },
    maxRetries: 0,
  });
  const declared = runtime.model(OPENAI_ID);
  const router = createRouter({
    models: [declared],
        executor: runtime,
    policy: ({ candidates }) => ({
      model: candidates[0]!.id,
      reasoningEffort: "high",
      reason: "Use the configured model.",
    }),
  });

  const response = await router.fetch(jsonRequest("/v1/chat/completions", {
    model: "my-router",
    messages: [
      { role: "system", content: "Be concise." },
      {
        role: "user",
        content: [
          { type: "text", text: "Look this up." },
          {
            type: "image_url",
            image_url: { url: "data:image/png;base64,YQ==", detail: "low" },
          },
        ],
      },
    ],
    tools: [{
      type: "function",
      function: {
        name: "lookup",
        description: "Look something up",
        parameters: { type: "object" },
        strict: true,
      },
    }],
    parallel_tool_calls: false,
    response_format: {
      type: "json_schema",
      json_schema: {
        name: "answer",
        schema: { type: "object" },
        strict: true,
      },
    },
    reasoning_effort: "high",
    prompt_cache_key: "conversation-1",
    user: "user-1",
  }));

  expect(response.status).toBe(200);
  expect(await response.json()).toMatchObject({
    model: OPENAI_ID,
    choices: [{ message: { content: "done" } }],
  });
  expect(declared).toMatchObject({
    provider: "openai",
    api: "openai-responses",
    reasoningEfforts: ["off", "minimal", "low", "medium", "high"],
    capabilities: {
      imageInput: true,
      toolUse: true,
      structuredOutput: true,
      streaming: true,
    },
  });
  expect(context).toMatchObject({
    systemPrompt: "Be concise.",
    messages: [{
      role: "user",
      content: [
        { type: "text", text: "Look this up." },
        { type: "image", data: "YQ==", mimeType: "image/png" },
      ],
    }],
    tools: [{ name: "lookup", description: "Look something up" }],
  });
  expect(streamOptions).toMatchObject({
    apiKey: "test-key",
    maxRetries: 0,
    reasoning: "high",
    sessionId: "conversation-1",
    responseFormat: {
      type: "json_schema",
      jsonSchema: { name: "answer", schema: { type: "object" }, strict: true },
    },
  });
  // No fork API other than anthropic-messages reads metadata, so none is
  // forwarded here; responseFormat maps natively instead of via onPayload.
  expect(streamOptions?.metadata).toBeUndefined();
  expect(patchedPayload).toEqual({
    input: [{
      type: "input_image",
      image_url: "data:image/png;base64,YQ==",
      detail: "low",
    }],
    tools: [{ name: "lookup", strict: true }],
    text: {},
    parallel_tool_calls: false,
  });
  expect(credentials).toEqual([{
    provider: "openai",
    model: OPENAI_ID,
    api: "openai-responses",
    purpose: "execution",
  }]);
});

test("Pi runtime translates provider streams into Anthropic SSE", async () => {
  let streamOptions: SimpleStreamOptions | undefined;
  const registry = fakeRegistry([ANTHROPIC_MODEL], {
    stream: (model, _context, options) => {
      streamOptions = options;
      const text = assistant(model, {
        content: [{ type: "text", text: "Checking" }],
        stopReason: "stop",
      });
      const withTool = assistant(model, {
        content: [
          { type: "text", text: "Checking" },
          { type: "toolCall", id: "call_1", name: "lookup", arguments: {} },
        ],
        stopReason: "toolUse",
      });
      return asyncEvents([
        { type: "start", partial: assistant(model) },
        { type: "text_start", contentIndex: 0, partial: text },
        { type: "text_delta", contentIndex: 0, delta: "Checking", partial: text },
        { type: "text_end", contentIndex: 0, content: "Checking", partial: text },
        { type: "toolcall_start", contentIndex: 1, partial: withTool },
        { type: "toolcall_delta", contentIndex: 1, delta: "{\"q\":", partial: withTool },
        { type: "toolcall_delta", contentIndex: 1, delta: "\"x\"}", partial: withTool },
        {
          type: "toolcall_end",
          contentIndex: 1,
          toolCall: { type: "toolCall", id: "call_1", name: "lookup", arguments: { q: "x" } },
          partial: withTool,
        },
        { type: "done", reason: "toolUse", message: withTool },
      ]);
    },
  });
  const runtime = await createPiRuntime({ registry, apiKey: "anthropic-key" });
  const router = singleModelRouter(runtime, ANTHROPIC_ID);

  const response = await router.fetch(jsonRequest("/v1/messages", {
    model: "my-router",
    max_tokens: 128,
    messages: [{ role: "user", content: "Find it." }],
    tools: [{ name: "lookup", input_schema: { type: "object" } }],
    metadata: { user_id: "user-9" },
    stream: true,
  }));
  const body = await response.text();

  expect(response.status).toBe(200);
  expect(streamOptions).toMatchObject({
    apiKey: "anthropic-key",
    maxTokens: 128,
    metadata: { user_id: "user-9" },
  });
  expect(body).toContain('event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Checking"}}');
  expect(body).toContain('"type":"tool_use","id":"call_1","name":"lookup","input":{}');
  expect(body).toContain('"type":"input_json_delta","partial_json":"{\\"q\\":"');
  expect(body).toContain('event: message_stop\ndata: {"type":"message_stop"}');
});

test("Pi runtime reconciles output present only in a terminal stream message", async () => {
  const registry = fakeRegistry([OPENAI_MODEL], {
    stream: (model) => {
      const terminal = assistant(model, {
        content: [
          { type: "text", text: "terminal text" },
          { type: "toolCall", id: "call_terminal", name: "lookup", arguments: { q: "x" } },
        ],
        stopReason: "toolUse",
      });
      return asyncEvents([
        { type: "start", partial: assistant(model) },
        { type: "done", reason: "toolUse", message: terminal },
      ]);
    },
  });
  const runtime = await createPiRuntime({ registry, apiKey: "test-key" });
  const router = singleModelRouter(runtime, OPENAI_ID);

  const response = await router.fetch(jsonRequest("/v1/chat/completions", {
    model: "my-router",
    messages: [{ role: "user", content: "hello" }],
    stream: true,
  }));
  const body = await response.text();

  expect(response.status).toBe(200);
  expect(body).toContain('"content":"terminal text"');
  expect(body).toContain('"id":"call_terminal"');
  expect(body).toContain("data: [DONE]");
});

test("Pi streams reasoning as OpenAI reasoning_content and reasoning_details", async () => {
  const thinkingPartial = assistant(OPENAI_MODEL, {
    content: [
      { type: "thinking", thinking: "Let me think.", thinkingSignature: JSON.stringify({ type: "reasoning", id: "rs_1", summary: [{ type: "summary_text", text: "Let me think." }], encrypted_content: "enc-openai" }) },
    ],
    stopReason: "stop",
  });
  const terminal = assistant(OPENAI_MODEL, {
    content: [
      { type: "thinking", thinking: "Let me think.", thinkingSignature: JSON.stringify({ type: "reasoning", id: "rs_1", summary: [{ type: "summary_text", text: "Let me think." }], encrypted_content: "enc-openai" }) },
      { type: "text", text: "Answer." },
    ],
    stopReason: "stop",
  });
  const registry = fakeRegistry([OPENAI_MODEL], {
    stream: (model) => asyncEvents([
      { type: "start", partial: assistant(model) },
      { type: "thinking_start", contentIndex: 0, partial: thinkingPartial },
      { type: "thinking_delta", contentIndex: 0, delta: "Let me think.", partial: thinkingPartial },
      { type: "thinking_end", contentIndex: 0, content: "Let me think.", partial: thinkingPartial },
      { type: "text_start", contentIndex: 1, partial: terminal },
      { type: "text_delta", contentIndex: 1, delta: "Answer.", partial: terminal },
      { type: "text_end", contentIndex: 1, content: "Answer.", partial: terminal },
      { type: "done", reason: "stop", message: terminal },
    ]),
  });
  const runtime = await createPiRuntime({ registry, apiKey: "test-key" });
  const router = createRouter({
    models: [runtime.model(OPENAI_ID)],
        executor: runtime,
    policy: ({ candidates }) => ({ model: candidates[0]!.id, reasoningEffort: "high" }),
  });

  const response = await router.fetch(jsonRequest("/v1/chat/completions", {
    model: "my-router",
    messages: [{ role: "user", content: "think and answer" }],
    stream: true,
  }));
  const body = await response.text();

  expect(response.status).toBe(200);
  expect(body).toContain('"reasoning_content":"Let me think."');
  expect(body).toContain('"type":"reasoning.encrypted"');
  expect(body).toContain('dari-pcs-v1.');
  expect(body).toContain('"content":"Answer."');
  expect(body).toContain("data: [DONE]");
});

test("Pi streams reasoning as Anthropic thinking and signature_delta", async () => {
  const thinkingPartial = assistant(ANTHROPIC_MODEL, {
    content: [
      { type: "thinking", thinking: "Let me think.", thinkingSignature: "anthropic-sig" },
    ],
    stopReason: "stop",
  });
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
      { type: "thinking_start", contentIndex: 0, partial: thinkingPartial },
      { type: "thinking_delta", contentIndex: 0, delta: "Let me think.", partial: thinkingPartial },
      { type: "thinking_end", contentIndex: 0, content: "Let me think.", partial: thinkingPartial },
      { type: "text_start", contentIndex: 1, partial: terminal },
      { type: "text_delta", contentIndex: 1, delta: "Answer.", partial: terminal },
      { type: "text_end", contentIndex: 1, content: "Answer.", partial: terminal },
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
  expect(body).toContain('"type":"thinking","thinking":"","signature":""');
  expect(body).toContain('"type":"thinking_delta","thinking":"Let me think."');
  expect(body).toContain('"type":"signature_delta"');
  expect(body).toContain('"type":"text_delta","text":"Answer."');
  expect(body).toContain('event: message_stop');
});

test("Pi streams redacted thinking as redacted_thinking content blocks", async () => {
  const redactedPartial = assistant(ANTHROPIC_MODEL, {
    content: [
      { type: "thinking", thinking: "", thinkingSignature: "anthropic-redacted-sig", redacted: true },
    ],
    stopReason: "stop",
  });
  const terminal = assistant(ANTHROPIC_MODEL, {
    content: [
      { type: "thinking", thinking: "", thinkingSignature: "anthropic-redacted-sig", redacted: true },
      { type: "text", text: "Answer." },
    ],
    stopReason: "stop",
  });
  const registry = fakeRegistry([ANTHROPIC_MODEL], {
    stream: (model) => asyncEvents([
      { type: "start", partial: assistant(model) },
      { type: "thinking_start", contentIndex: 0, partial: redactedPartial },
      { type: "thinking_end", contentIndex: 0, content: "", partial: redactedPartial },
      { type: "text_start", contentIndex: 1, partial: terminal },
      { type: "text_delta", contentIndex: 1, delta: "Answer.", partial: terminal },
      { type: "text_end", contentIndex: 1, content: "Answer.", partial: terminal },
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
  expect(body).toContain('"type":"redacted_thinking"');
  expect(body).toContain('"type":"text_delta","text":"Answer."');
  expect(body).toContain('event: message_stop');
});

test("stream onStreamClose hook fires on successful stream completion", async () => {
  const registry = fakeRegistry([OPENAI_MODEL], {
    stream: (model) => asyncEvents([
      { type: "start", partial: assistant(model) },
      { type: "text_start", contentIndex: 0, partial: assistant(model, { content: [{ type: "text", text: "hi" }] }) },
      { type: "text_delta", contentIndex: 0, delta: "hi", partial: assistant(model, { content: [{ type: "text", text: "hi" }] }) },
      { type: "text_end", contentIndex: 0, content: "hi", partial: assistant(model, { content: [{ type: "text", text: "hi" }] }) },
      { type: "done", reason: "stop", message: assistant(model, { content: [{ type: "text", text: "hi" }] }) },
    ]),
  });
  const runtime = await createPiRuntime({ registry, apiKey: "test-key" });
  let streamCloseCalled = false;
  let streamCloseCompletion: RouterCompletion | null = null;
  let streamCloseError: unknown = null;
  const router = createRouter({
    models: [runtime.model(OPENAI_ID)],
        executor: runtime,
    policy: ({ candidates }) => ({ model: candidates[0]!.id }),
    hooks: {
      onStreamClose: (completion, _selection, error) => {
        streamCloseCalled = true;
        streamCloseCompletion = completion;
        streamCloseError = error;
      },
    },
  });

  const response = await router.fetch(jsonRequest("/v1/chat/completions", {
    model: "my-router",
    messages: [{ role: "user", content: "hello" }],
    stream: true,
  }));
  await response.text(); // consume the stream

  expect(streamCloseCalled).toBe(true);
  expect(streamCloseError).toBeNull();
  expect(streamCloseCompletion).toMatchObject({ finishReason: "stop" });
});

test("Pi stream aborts return cancellation errors instead of successful SSE", async () => {
  const aborted = assistant(OPENAI_MODEL, {
    stopReason: "aborted",
    errorMessage: "provider request aborted",
  });
  const eventSequences = [
    [{ type: "error", reason: "aborted", error: aborted }],
    [{ type: "done", reason: "aborted", message: aborted }],
  ] as unknown as AssistantMessageEvent[][];

  for (const events of eventSequences) {
    const registry = fakeRegistry([OPENAI_MODEL], {
      stream: () => asyncEvents(events),
    });
    const runtime = await createPiRuntime({ registry, apiKey: "test-key" });
    const router = singleModelRouter(runtime, OPENAI_ID);

    const response = await router.fetch(jsonRequest("/v1/chat/completions", {
      model: "my-router",
      messages: [{ role: "user", content: "hello" }],
      stream: true,
    }));

    expect(response.status).toBe(499);
    expect(await response.json()).toMatchObject({
      error: { code: "request_cancelled" },
    });
  }
});

test("Dari policy uses the same Pi runtime for selector and model execution", async () => {
  const credentialPurposes: PiCredentialInput["purpose"][] = [];
  const calledModels: string[] = [];
  const registry = fakeRegistry([SELECTOR_MODEL, ANTHROPIC_MODEL], {
    complete: async (model) => {
      calledModels.push(model.id);
      if (model.id === "selector-test") {
        return assistant(model, {
          content: [{
            type: "text",
            text: JSON.stringify({
              selected_model: ANTHROPIC_ID,
              reasoning_effort: "off",
              reason: "The only eligible model.",
            }),
          }],
          stopReason: "stop",
        });
      }
      return assistant(model, {
        content: [{ type: "text", text: "final answer" }],
        stopReason: "stop",
      });
    },
  });
  const runtime = await createPiRuntime({
    registry,
    apiKey: (input) => {
      credentialPurposes.push(input.purpose);
      return `${input.provider}-key`;
    },
  });
  const policy = createDariRoutingPolicy({
    runtime,
    selectorModel: SELECTOR_ID,
    selectorContextWindowChars: 100_000,
    pricing: () => ({ input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 }),
    averageOutputTokensByModel: { [ANTHROPIC_ID]: { off: 800 } },
  });
  const router = createRouter({
    models: [runtime.model(ANTHROPIC_ID)],
    policy,
    executor: runtime,
  });

  const response = await router.fetch(jsonRequest("/v1/chat/completions", {
    model: "my-router",
    messages: [{ role: "user", content: "hello" }],
  }));

  expect(response.status).toBe(200);
  expect(await response.json()).toMatchObject({
    model: ANTHROPIC_ID,
    choices: [{ message: { content: "final answer" } }],
    dari_routing: {
      requested_model: "my-router",
      selected_model: ANTHROPIC_ID,
      reasoning_effort: "off",
      reason: "The only eligible model.",
    },
  });
  expect(calledModels).toEqual(["selector-test", "claude-test"]);
  expect(credentialPurposes).toEqual(["selector", "execution"]);
});

test("Pi replays same-provider continuation and degrades cross-provider reasoning to text", async () => {
  let captured: Context | undefined;
  const continuation = {
    kind: "openai_reasoning" as const,
    source: { provider: "openai", api: "openai-responses", model: "gpt-test" },
    encryptedContent: "encrypted-openai-state",
    providerItemId: "rs_1",
  };
  const registry = fakeRegistry([OPENAI_MODEL, ANTHROPIC_MODEL], {
    complete: async (model, nextContext) => {
      captured = nextContext;
      return assistant(model, { content: [{ type: "text", text: "ok" }], stopReason: "stop" });
    },
  });
  const runtime = await createPiRuntime({
    registry,
    apiKey: () => "test-key",
    maxRetries: 0,
  });
  const openaiRouter = createRouter({
    models: [runtime.model(OPENAI_ID)],
    executor: runtime,
    policy: ({ candidates }) => ({ model: candidates[0]!.id, reasoningEffort: "high", reason: "openai" }),
  });
  const anthropicRouter = createRouter({
    models: [runtime.model(ANTHROPIC_ID)],
    executor: runtime,
    policy: ({ candidates }) => ({ model: candidates[0]!.id, reasoningEffort: "off", reason: "anthropic" }),
  });
  const history = {
    model: "my-router",
    messages: [{
      role: "assistant",
      content: null,
      reasoning_content: "Inspect first.",
      reasoning_details: [{ type: "reasoning.encrypted", id: "rs_1", data: encodeProviderContinuationState(continuation) }],
    }],
  };

  // Same-provider: encrypted continuation is replayed as a thinking block.
  await openaiRouter.fetch(jsonRequest("/v1/chat/completions", history));
  const sameProviderAssistant = captured!.messages.find((message) => message.role === "assistant") as
    | { content: Array<Record<string, unknown>> }
    | undefined;
  expect(sameProviderAssistant?.content.some((part) => part.type === "thinking" && typeof part.thinkingSignature === "string" && part.thinkingSignature.includes("encrypted-openai-state"))).toBe(true);

  // Cross-provider: route to Anthropic; encrypted OpenAI state is not replayed.
  captured = undefined;
  await anthropicRouter.fetch(jsonRequest("/v1/chat/completions", history));
  const crossAssistant = captured!.messages.find((message) => message.role === "assistant") as
    | { content: Array<Record<string, unknown>> }
    | undefined;
  expect(crossAssistant?.content.some((part) => part.type === "thinking")).toBe(false);
  expect(crossAssistant?.content.some((part) => part.type === "text" && part.text === "Inspect first.")).toBe(true);
});
