import { RouterFrameworkError } from "./framework_error.js";
import { nativeModelId, providerForModel } from "./model_ids.js";
import { piContext } from "./pi_context.js";
import { piOptions } from "./pi_options.js";
import { routerCompletion, routerEvents } from "./pi_output.js";
import { openAIChatRequest } from "./protocol_openai_chat.js";
import type { ReasoningEffort } from "./types.js";
import type {
  RouterExecutorResult,
  RouterModel,
  RouterRequest,
} from "./framework_types.js";
import type {
  CreatePiRuntimeOptions,
  PiExecution,
  PiModel,
  PiModelRegistry,
  PiRouterModelOptions,
  PiRuntime,
} from "./pi_types.js";

export type {
  CreatePiRuntimeOptions,
  PiApiKey,
  PiCredentialInput,
  PiModelRegistry,
  PiRouterModelOptions,
  PiRuntime,
} from "./pi_types.js";

type PiRuntimeDependencies = {
  registry: PiModelRegistry;
  supportedThinkingLevels(model: PiModel): ReasoningEffort[];
};

const STRUCTURED_OUTPUT_APIS = new Set([
  "anthropic-messages",
  "azure-openai-responses",
  "google-generative-ai",
  "google-vertex",
  "mistral-conversations",
  "openai-completions",
  "openai-responses",
]);

export async function createPiRuntime(
  options: CreatePiRuntimeOptions,
): Promise<PiRuntime> {
  validateRuntimeOptions(options);
  const dependencies = await piDependencies(options.registry);

  const model = <Metadata = unknown>(
    id: string,
    overrides: Partial<PiRouterModelOptions<Metadata>> = {},
  ): RouterModel<Metadata> => {
    const provider = modelProvider(id);
    const registered = registeredModel(dependencies.registry, provider, id);
    const reasoningEfforts = overrides.reasoningEfforts === undefined
      ? dependencies.supportedThinkingLevels(registered)
      : [...overrides.reasoningEfforts];
    const defaultReasoningEffort = overrides.defaultReasoningEffort
      ?? (reasoningEfforts.includes("off") ? "off" : reasoningEfforts[0]);
    return {
      id,
      provider,
      api: registered.api,
      reasoningEfforts,
      ...(defaultReasoningEffort === undefined ? {} : { defaultReasoningEffort }),
      capabilities: {
        imageInput: registered.input.includes("image"),
        toolUse: true,
        structuredOutput: STRUCTURED_OUTPUT_APIS.has(registered.api),
        streaming: true,
        ...overrides.capabilities,
      },
      ...(overrides.metadata === undefined ? {} : { metadata: overrides.metadata }),
    };
  };

  const run = async (execution: PiExecution): Promise<RouterExecutorResult> => {
    const registered = registeredModel(
      dependencies.registry,
      execution.candidate.provider,
      execution.candidate.id,
    );
    if (registered.api !== execution.candidate.api) {
      throw new RouterFrameworkError(
        "configuration",
        `Model '${execution.candidate.id}' declares API '${execution.candidate.api}' but Pi uses '${registered.api}'.`,
        "pi_model_api_mismatch",
      );
    }
    const context = piContext(execution.request, registered);
    const streamOptions = await piOptions(options, execution, registered);
    if (execution.request.stream) {
      const events = dependencies.registry.streamSimple(registered, context, streamOptions);
      return { type: "stream", events: routerEvents(events, execution.signal) };
    }
    const message = await dependencies.registry.completeSimple(registered, context, streamOptions);
    return { type: "complete", output: routerCompletion(message, execution.signal) };
  };

  return {
    model,
    execute: async (input) => run({
      request: input.request,
      candidate: input.model,
      reasoningEffort: input.decision.reasoningEffort,
      signal: input.signal,
      purpose: "execution",
    }),
    select: async (request, signal = new AbortController().signal) => {
      let normalized: RouterRequest;
      try {
        normalized = openAIChatRequest(request);
      } catch (error) {
        // The selector request is constructed internally; an invalid request
        // here is a router bug, not the end user's fault.
        if (error instanceof RouterFrameworkError && error.kind === "invalid_request") {
          throw new RouterFrameworkError(
            "configuration",
            error.message,
            "pi_selector_request_invalid",
            error.param,
            { cause: error },
          );
        }
        throw error;
      }
      const selectorModel = requiredSelectorModel(request.model);
      const candidate = model(selectorModel);
      const result = await run({
        request: { ...normalized, stream: false },
        candidate: {
          id: candidate.id,
          provider: candidate.provider!,
          api: candidate.api!,
        },
        reasoningEffort: normalized.reasoning?.effort ?? candidate.defaultReasoningEffort ?? "off",
        signal,
        purpose: "selector",
      });
      if (result.type !== "complete") {
        throw new RouterFrameworkError(
          "configuration",
          "Pi returned a stream for a non-streaming selector request.",
          "pi_selector_mode_mismatch",
        );
      }
      return result.output.content
        .filter((item) => item.type === "text")
        .map((item) => item.text)
        .join("");
    },
  };
}

async function piDependencies(registry: PiModelRegistry | undefined): Promise<PiRuntimeDependencies> {
  const pi = await import("@mupt-ai/pi-ai");
  const resolvedRegistry = registry ?? (await import("@mupt-ai/pi-ai/providers/all")).builtinModels();
  return {
    registry: resolvedRegistry,
    supportedThinkingLevels: (model) => pi.getSupportedThinkingLevels(model) as ReasoningEffort[],
  };
}

function validateRuntimeOptions(options: CreatePiRuntimeOptions): void {
  if (!options || (typeof options.apiKey !== "string" && typeof options.apiKey !== "function")) {
    throw new RouterFrameworkError(
      "configuration",
      "createPiRuntime requires an API key or API-key resolver.",
      "pi_api_key_missing",
    );
  }
  if (typeof options.apiKey === "string" && !options.apiKey.trim()) {
    throw new RouterFrameworkError(
      "configuration",
      "createPiRuntime API keys must be non-empty strings.",
      "pi_api_key_missing",
    );
  }
  for (const [name, value] of [
    ["timeoutMs", options.timeoutMs],
    ["maxRetries", options.maxRetries],
  ] as const) {
    if (value !== undefined && (!Number.isSafeInteger(value) || value < 0)) {
      throw new RouterFrameworkError(
        "configuration",
        `createPiRuntime ${name} must be a non-negative integer.`,
        "pi_options_invalid",
      );
    }
  }
}

function modelProvider(id: string): string {
  try {
    return providerForModel(id);
  } catch (error) {
    throw new RouterFrameworkError(
      "configuration",
      `Pi model '${id}' must include a provider prefix.`,
      "pi_model_provider_missing",
      undefined,
      { cause: error },
    );
  }
}

function registeredModel(registry: PiModelRegistry, provider: string, id: string): PiModel {
  let nativeId: string;
  try {
    nativeId = nativeModelId(id);
  } catch {
    nativeId = id;
  }
  const registered = registry.getModel(provider, nativeId) ?? registry.getModel(provider, id);
  if (registered === undefined) {
    throw new RouterFrameworkError(
      "configuration",
      `Pi does not know model '${id}' for provider '${provider}'.`,
      "pi_model_not_found",
    );
  }
  return registered;
}

function requiredSelectorModel(value: unknown): string {
  if (typeof value !== "string" || !value.trim()) {
    throw new RouterFrameworkError(
      "configuration",
      "The Pi selector request requires a model.",
      "pi_selector_model_missing",
    );
  }
  return value;
}
