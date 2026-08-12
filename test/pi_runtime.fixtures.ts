import type {
  Api,
  AssistantMessage,
  AssistantMessageEvent,
  Model,
} from "@mupt-ai/pi-ai";

import {
  createRouter,
  type PiModelRegistry,
  type PiRuntime,
  type RouterRequest,
} from "../src/index.js";

export const OPENAI_ID = "openai/gpt-test";
export const ANTHROPIC_ID = "anthropic/claude-test";
export const SELECTOR_ID = "openai/selector-test";

export const OPENAI_MODEL = piModel({
  id: "gpt-test",
  provider: "openai",
  api: "openai-responses",
  reasoning: true,
  input: ["text", "image"],
});

export const ANTHROPIC_MODEL = piModel({
  id: "claude-test",
  provider: "anthropic",
  api: "anthropic-messages",
  reasoning: false,
  input: ["text"],
});

export const SELECTOR_MODEL = piModel({
  id: "selector-test",
  provider: "openai",
  api: "openai-responses",
  reasoning: false,
  input: ["text"],
});

export const GOOGLE_MODEL = piModel({
  id: "gemini-test",
  provider: "google",
  api: "google-generative-ai",
});

export const VERTEX_MODEL = piModel({
  id: "vertex-test",
  provider: "google",
  api: "google-vertex",
});

export const AZURE_MODEL = piModel({
  id: "azure-test",
  provider: "azure",
  api: "azure-openai-responses",
});

export const BEDROCK_MODEL = piModel({
  id: "bedrock-test",
  provider: "amazon",
  api: "bedrock-converse-stream",
});

export const COMPLETIONS_MODEL = piModel({
  id: "completions-test",
  provider: "openai",
  api: "openai-completions",
});

export function singleModelRouter(runtime: PiRuntime, modelId: string) {
  return createRouter({
    models: [runtime.model(modelId)],
    executor: runtime,
    policy: ({ candidates }) => ({ model: candidates[0]!.id }),
  });
}

export function fakeRegistry(
  models: Model<Api>[],
  implementations: {
    complete?: PiModelRegistry["completeSimple"];
    stream?: PiModelRegistry["streamSimple"];
  },
): PiModelRegistry {
  return {
    getModel: (provider, id) =>
      models.find(
        (model) => model.provider === provider && model.id === id
      ),
    completeSimple:
      implementations.complete ?? (async (model) => assistant(model)),
    streamSimple:
      implementations.stream ??
      ((model) =>
        asyncEvents([
          { type: "start", partial: assistant(model) },
          { type: "done", reason: "stop", message: assistant(model) },
        ])),
  };
}

export function piModel(
  overrides: Partial<Model<Api>> &
    Pick<Model<Api>, "id" | "provider" | "api">
): Model<Api> {
  return {
    name: overrides.id,
    baseUrl: "https://provider.example/v1",
    reasoning: false,
    input: ["text"],
    cost: { input: 1, output: 1, cacheRead: 1, cacheWrite: 1 },
    contextWindow: 128_000,
    maxTokens: 16_384,
    ...overrides,
  };
}

export function assistant(
  model: Model<Api>,
  overrides: Partial<AssistantMessage> = {}
): AssistantMessage {
  return {
    role: "assistant",
    content: [],
    api: model.api,
    provider: model.provider,
    model: model.id,
    usage: {
      input: 10,
      output: 5,
      cacheRead: 2,
      cacheWrite: 1,
      totalTokens: 15,
      cost: {
        input: 0,
        output: 0,
        cacheRead: 0,
        cacheWrite: 0,
        total: 0,
      },
    },
    stopReason: "stop",
    timestamp: 1_785_283_200_000,
    ...overrides,
  };
}

export function asyncEvents(
  events: AssistantMessageEvent[]
): AsyncIterable<AssistantMessageEvent> {
  return {
    async *[Symbol.asyncIterator]() {
      yield* events;
    },
  };
}

export function sseJsonFrames(
  body: string
): Array<Record<string, unknown>> {
  return body
    .split("\n\n")
    .map((frame) => frame.replace(/^data: /, ""))
    .filter((data) => data.startsWith("{"))
    .map((data) => JSON.parse(data) as Record<string, unknown>);
}

export function jsonRequest(path: string, body: unknown): Request {
  return new Request(`https://router.example${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function routerRequest(
  overrides: Partial<RouterRequest> = {}
): RouterRequest {
  return {
    protocol: "openai_chat_completions",
    requestedModel: "my-router",
    items: [
      {
        type: "message",
        role: "user",
        content: [{ type: "text", text: "hi" }],
      },
    ],
    tools: [],
    generation: {},
    stream: false,
    ...overrides,
  };
}

export function piExecution(request: RouterRequest, model: Model<Api>) {
  return {
    request,
    candidate: {
      id: `${model.provider}/${model.id}`,
      provider: model.provider,
      api: model.api,
    },
    reasoningEffort: "off" as const,
    signal: new AbortController().signal,
    purpose: "execution" as const,
  };
}
