import type { SimpleStreamOptions } from "@mupt-ai/pi-ai";

import { isOpenAIResponsesApi } from "./continuation_state.js";
import { RouterFrameworkError } from "./framework_error.js";
import { isRecord } from "./types.js";
import type { RouterContent, RouterRequest, RouterToolChoice } from "./framework_types.js";
import type {
  CreatePiRuntimeOptions,
  PiExecution,
  PiModel,
} from "./pi_types.js";

const PAYLOAD_PATCH_APIS = new Set([
  "anthropic-messages",
  "azure-openai-responses",
  "openai-completions",
  "openai-responses",
]);

export async function piOptions(
  runtime: CreatePiRuntimeOptions,
  execution: PiExecution,
  model: PiModel,
): Promise<SimpleStreamOptions> {
  rejectUnsupportedOptions(execution.request, model);
  const apiKey = typeof runtime.apiKey === "string"
    ? runtime.apiKey
    : await runtime.apiKey({
        provider: model.provider,
        model: execution.candidate.id,
        api: model.api,
        purpose: execution.purpose,
      });
  if (typeof apiKey !== "string" || !apiKey.trim()) {
    throw new RouterFrameworkError(
      "configuration",
      `No API key was provided for Pi model '${execution.candidate.id}'.`,
      "pi_api_key_missing",
    );
  }

  const request = execution.request;
  const options: SimpleStreamOptions = {
    apiKey,
    signal: execution.signal,
    ...(runtime.timeoutMs === undefined ? {} : { timeoutMs: runtime.timeoutMs }),
    ...(runtime.maxRetries === undefined ? {} : { maxRetries: runtime.maxRetries }),
    ...(request.generation.temperature === undefined
      ? {}
      : { temperature: request.generation.temperature }),
    ...(request.generation.topP === undefined ? {} : { topP: request.generation.topP }),
    ...(request.generation.maxOutputTokens === undefined
      ? {}
      : { maxTokens: request.generation.maxOutputTokens }),
    ...(request.generation.stop === undefined ? {} : { stop: request.generation.stop }),
    ...(request.cacheKey === undefined ? {} : { sessionId: request.cacheKey }),
    // Only anthropic-messages reads metadata (and only user_id); no other
    // fork API consumes metadata fields.
    ...(request.user !== undefined && model.api === "anthropic-messages"
      ? { metadata: { user_id: request.user } }
      : {}),
    ...(execution.reasoningEffort === "off" ? {} : { reasoning: execution.reasoningEffort }),
    ...(request.responseFormat === undefined
      ? {}
      : { responseFormat: piResponseFormat(request.responseFormat, model.api) }),
    ...(request.toolChoice === undefined ? {} : { toolChoice: piToolChoice(request.toolChoice) }),
  };

  const budget = request.reasoning?.budgetTokens;
  if (
    budget !== undefined &&
    execution.reasoningEffort !== "off" &&
    execution.reasoningEffort !== "xhigh" &&
    execution.reasoningEffort !== "max"
  ) {
    options.thinkingBudgets = { [execution.reasoningEffort]: budget };
  }

  const patch = payloadPatch(request, model);
  if (patch !== undefined) options.onPayload = patch;
  return options;
}

// The fork hard-throws for these (api, option) pairs inside streamSimple;
// reject them up front as client errors instead of surfacing 502s.
function rejectUnsupportedOptions(request: RouterRequest, model: PiModel): void {
  if (request.generation.stop !== undefined && isOpenAIResponsesApi(model.api)) {
    throw unsupportedPiOption("stop sequences", model.api);
  }
  if (request.generation.topP !== undefined && model.api === "anthropic-messages") {
    throw unsupportedPiOption("top_p", model.api);
  }
  if (
    typeof request.toolChoice === "object" &&
    (model.api === "google-generative-ai" || model.api === "google-vertex")
  ) {
    throw unsupportedPiOption("forcing a specific tool with tool_choice", model.api);
  }
  if (request.responseFormat?.type === "json_object" && model.api === "bedrock-converse-stream") {
    throw unsupportedPiOption("response_format json_object", model.api);
  }
}

function unsupportedPiOption(option: string, api: string): RouterFrameworkError {
  return new RouterFrameworkError(
    "invalid_request",
    `Pi API '${api}' does not support ${option}.`,
    "pi_option_unsupported",
  );
}

function piResponseFormat(
  format: NonNullable<RouterRequest["responseFormat"]>,
  api: string,
): NonNullable<SimpleStreamOptions["responseFormat"]> {
  if (format.type === "json_object" && api === "anthropic-messages") {
    // Anthropic has no json_object mode; degrade to a permissive object
    // schema so the request still produces JSON output.
    return {
      type: "json_schema",
      jsonSchema: { name: "response", schema: { type: "object" } },
    };
  }
  if (format.type !== "json_schema") return format;
  return {
    type: "json_schema",
    jsonSchema: {
      name: format.name,
      schema: format.schema,
      ...(format.description === undefined ? {} : { description: format.description }),
      ...(format.strict === undefined ? {} : { strict: format.strict }),
    },
  };
}

function piToolChoice(choice: RouterToolChoice): NonNullable<SimpleStreamOptions["toolChoice"]> {
  if (choice === "required") return "any";
  if (typeof choice === "string") return choice;
  return { type: "function", function: { name: choice.name } };
}

function payloadPatch(
  request: RouterRequest,
  model: PiModel,
): SimpleStreamOptions["onPayload"] | undefined {
  const strictTools = request.tools.some((tool) => tool.strict !== undefined);
  const parallel = request.parallelToolCalls;
  const imageDetails = imageDetailQueues(request);
  const thinkingPatch = model.api === "anthropic-messages" && (
    request.reasoning?.enabled === false || request.reasoning?.budgetTokens !== undefined
  );
  const imageDetailPatch = imageDetails.size > 0 &&
    (isOpenAIResponsesApi(model.api) || model.api === "openai-completions");
  if (
    imageDetails.size > 0 &&
    !imageDetailPatch &&
    [...imageDetails.values()].some((queue) => queue.some((detail) => detail !== "auto"))
  ) {
    throw new RouterFrameworkError(
      "invalid_request",
      `Pi API '${model.api}' cannot preserve image detail controls.`,
      "pi_image_detail_unsupported",
    );
  }
  if (!strictTools && parallel === undefined && !thinkingPatch && !imageDetailPatch) {
    return undefined;
  }
  if ((strictTools || parallel !== undefined) && !PAYLOAD_PATCH_APIS.has(model.api)) {
    throw new RouterFrameworkError(
      "invalid_request",
      `Pi API '${model.api}' cannot preserve strict or parallel tool controls.`,
      "pi_tool_control_unsupported",
    );
  }

  return (value) => {
    if (!isRecord(value)) return undefined;
    let payload = value;
    if (strictTools) payload = patchStrictTools(payload, request, model.api);
    if (parallel !== undefined) payload = patchParallelTools(payload, parallel, model.api);
    if (imageDetailPatch) payload = patchImageDetails(payload, imageDetails);
    if (thinkingPatch) payload = patchAnthropicThinking(payload, request);
    return payload;
  };
}

function patchStrictTools(
  payload: Record<string, unknown>,
  request: RouterRequest,
  api: string,
): Record<string, unknown> {
  if (!Array.isArray(payload.tools)) return payload;
  const tools = new Map(request.tools.map((tool) => [tool.name, tool]));
  return {
    ...payload,
    tools: payload.tools.map((entry) => {
      if (!isRecord(entry)) return entry;
      if (api === "anthropic-messages") {
        const tool = typeof entry.name === "string" ? tools.get(entry.name) : undefined;
        return tool?.strict === undefined ? entry : { ...entry, strict: tool.strict };
      }
      const definition = api === "openai-completions" && isRecord(entry.function)
        ? entry.function
        : entry;
      const tool = typeof definition.name === "string" ? tools.get(definition.name) : undefined;
      if (tool?.strict === undefined) return entry;
      return api === "openai-completions"
        ? { ...entry, function: { ...definition, strict: tool.strict } }
        : { ...entry, strict: tool.strict };
    }),
  };
}

function patchParallelTools(
  payload: Record<string, unknown>,
  enabled: boolean,
  api: string,
): Record<string, unknown> {
  if (api !== "anthropic-messages") return { ...payload, parallel_tool_calls: enabled };
  if (enabled) return payload;
  const current = isRecord(payload.tool_choice) ? payload.tool_choice : {};
  return {
    ...payload,
    tool_choice: {
      type: typeof current.type === "string" ? current.type : "auto",
      ...current,
      disable_parallel_tool_use: true,
    },
  };
}

function patchImageDetails(
  payload: Record<string, unknown>,
  details: Map<string, Array<"auto" | "low" | "high">>,
): Record<string, unknown> {
  const queues = new Map([...details].map(([url, queue]) => [url, [...queue]]));
  const patch = (value: unknown): unknown => {
    if (Array.isArray(value)) return value.map(patch);
    if (!isRecord(value)) return value;
    const mapped = Object.fromEntries(
      Object.entries(value).map(([key, entry]) => [key, patch(entry)]),
    );
    const responseUrl = mapped.type === "input_image" && typeof mapped.image_url === "string"
      ? mapped.image_url
      : undefined;
    const chatImage = mapped.type === "image_url" && isRecord(mapped.image_url)
      ? mapped.image_url
      : undefined;
    const chatUrl = typeof chatImage?.url === "string" ? chatImage.url : undefined;
    const detail = queues.get(responseUrl ?? chatUrl ?? "")?.shift();
    if (detail === undefined) return mapped;
    return responseUrl === undefined
      ? { ...mapped, image_url: { ...chatImage, detail } }
      : { ...mapped, detail };
  };
  const patched = patch(payload);
  return isRecord(patched) ? patched : payload;
}

function imageDetailQueues(
  request: RouterRequest,
): Map<string, Array<"auto" | "low" | "high">> {
  const details = new Map<string, Array<"auto" | "low" | "high">>();
  const add = (content: readonly RouterContent[]) => {
    for (const part of content) {
      if (part.type !== "image" || part.detail === undefined) continue;
      const queue = details.get(part.url) ?? [];
      queue.push(part.detail);
      details.set(part.url, queue);
    }
  };
  for (const item of request.items) {
    if (item.type === "message" || item.type === "tool_result") add(item.content);
  }
  return details;
}

function patchAnthropicThinking(
  payload: Record<string, unknown>,
  request: RouterRequest,
): Record<string, unknown> {
  if (request.reasoning?.enabled === false) {
    return { ...payload, thinking: { type: "disabled" } };
  }
  const budget = request.reasoning?.budgetTokens;
  if (budget === undefined) return payload;
  return {
    ...payload,
    thinking: {
      ...(isRecord(payload.thinking) ? payload.thinking : {}),
      type: "enabled",
      budget_tokens: budget,
    },
  };
}
