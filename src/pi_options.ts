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
    ...(request.generation.maxOutputTokens === undefined
      ? {}
      : { maxTokens: request.generation.maxOutputTokens }),
    ...(request.cacheKey === undefined ? {} : { sessionId: request.cacheKey }),
    // Only anthropic-messages reads metadata (and only user_id); no other
    // fork API consumes metadata fields.
    ...(request.user !== undefined && model.api === "anthropic-messages"
      ? { metadata: { user_id: request.user } }
      : {}),
    ...(execution.reasoningEffort === "off" ? {} : { reasoning: execution.reasoningEffort }),
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

  const patch = payloadPatch(request, model, execution);
  if (patch !== undefined) options.onPayload = patch;
  return options;
}

// Reject controls the provider API cannot represent before starting a request.
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
  if (request.toolChoice === "none" && model.api === "bedrock-converse-stream") {
    throw unsupportedPiOption("disabling tools with tool_choice", model.api);
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
): Record<string, unknown> {
  if (api === "anthropic-messages") {
    // Anthropic has no json_object mode; use a permissive object schema.
    return {
      type: "json_schema",
      schema: format.type === "json_schema" ? format.schema : { type: "object" },
    };
  }
  if (format.type !== "json_schema") return format;
  const { type: _, ...schema } = format;
  return isOpenAIResponsesApi(api)
    ? { type: "json_schema", ...schema }
    : { type: "json_schema", json_schema: schema };
}

function piToolChoice(choice: RouterToolChoice, api: string): unknown {
  if (api === "anthropic-messages") {
    return typeof choice === "string"
      ? { type: choice === "required" ? "any" : choice }
      : { type: "tool", name: choice.name };
  }
  if (typeof choice === "string") return choice;
  return isOpenAIResponsesApi(api)
    ? { type: "function", name: choice.name }
    : { type: "function", function: { name: choice.name } };
}

function payloadPatch(
  request: RouterRequest,
  model: PiModel,
  execution: PiExecution,
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
  if (
    !strictTools &&
    parallel === undefined &&
    !thinkingPatch &&
    !imageDetailPatch &&
    request.responseFormat === undefined &&
    request.toolChoice === undefined &&
    request.generation.topP === undefined &&
    request.generation.stop === undefined
  ) {
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
    const api = model.api;
    const google = api === "google-generative-ai" || api === "google-vertex";
    const config = isRecord(payload.config) ? payload.config : {};
    if (request.generation.topP !== undefined || request.generation.stop !== undefined) {
      const stop = request.generation.stop;
      const sequences = typeof stop === "string" ? [stop] : stop;
      const controls = {
        ...(request.generation.topP === undefined ? {} : { [google || api === "bedrock-converse-stream" ? "topP" : "top_p"]: request.generation.topP }),
        ...(stop === undefined ? {} : { [google || api === "bedrock-converse-stream" ? "stopSequences" : api === "anthropic-messages" ? "stop_sequences" : "stop"]: api === "openai-completions" ? stop : sequences }),
      };
      payload = google ? { ...payload, config: { ...config, ...controls } }
        : api === "bedrock-converse-stream" ? { ...payload, inferenceConfig: { ...(isRecord(payload.inferenceConfig) ? payload.inferenceConfig : {}), ...controls } }
        : { ...payload, ...controls };
    }
    if (
      request.responseFormat !== undefined &&
      (request.responseFormat.type !== "text" || isOpenAIResponsesApi(api) || api === "openai-completions")
    ) {
      const format = piResponseFormat(request.responseFormat, api);
      if (google) {
        payload = { ...payload, config: { ...(isRecord(payload.config) ? payload.config : {}), responseMimeType: "application/json", ...(request.responseFormat.type === "json_schema" ? { responseJsonSchema: request.responseFormat.schema } : {}) } };
      } else if (api === "bedrock-converse-stream") {
        const response = request.responseFormat;
        if (response.type === "json_schema") payload = { ...payload, outputConfig: { textFormat: { type: "json_schema", structure: { jsonSchema: { name: response.name, schema: JSON.stringify(response.schema), ...(response.description === undefined ? {} : { description: response.description }) } } } } };
      } else if (isOpenAIResponsesApi(api)) {
        payload = { ...payload, text: { ...(isRecord(payload.text) ? payload.text : {}), format } };
      } else if (api === "anthropic-messages") {
        payload = { ...payload, output_config: { ...(isRecord(payload.output_config) ? payload.output_config : {}), format } };
      } else {
        payload = { ...payload, response_format: format };
      }
    }
    if (request.toolChoice !== undefined) {
      const choice = request.toolChoice;
      if (google) {
        const mode = choice === "none" ? "NONE" : choice === "required" ? "ANY" : "AUTO";
        payload = { ...payload, config: { ...(isRecord(payload.config) ? payload.config : {}), toolConfig: { functionCallingConfig: { mode } } } };
      } else if (api === "bedrock-converse-stream") {
        payload = { ...payload, toolConfig: { ...(isRecord(payload.toolConfig) ? payload.toolConfig : {}), toolChoice: typeof choice === "string" ? choice === "required" ? { any: {} } : { auto: {} } : { tool: { name: choice.name } } } };
      } else if (Array.isArray(payload.tools) && payload.tools.length > 0) {
        payload = { ...payload, tool_choice: piToolChoice(choice, api) };
      } else {
        const { tool_choice: _, ...withoutToolChoice } = payload;
        payload = withoutToolChoice;
      }
    }
    if (strictTools) payload = patchStrictTools(payload, request, model.api);
    if (parallel !== undefined) payload = patchParallelTools(payload, parallel, model.api);
    if (imageDetailPatch) payload = patchImageDetails(payload, imageDetails);
    if (thinkingPatch) payload = patchAnthropicThinking(payload, request);
    if (
      api !== "bedrock-converse-stream" &&
      !google &&
      (!Array.isArray(payload.tools) || payload.tools.length === 0)
    ) {
      const { tool_choice: _, ...withoutToolChoice } = payload;
      payload = withoutToolChoice;
    }
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
