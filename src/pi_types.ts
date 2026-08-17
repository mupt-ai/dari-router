import type {
  Api,
  AssistantMessage,
  AssistantMessageEvent,
  Context,
  Model,
  SimpleStreamOptions,
} from "@mupt-ai/pi-ai";

import type { ChatCompletionRequest, ReasoningEffort } from "./types.js";
import type {
  RouterCandidate,
  RouterExecutorInput,
  RouterExecutorResult,
  RouterModel,
  RouterRequest,
} from "./framework_types.js";

export type PiModel = Model<Api>;

export type PiCredentialInput = {
  provider: string;
  model: string;
  api: string;
  purpose: "execution" | "selector";
};

export type PiApiKey = string | ((input: PiCredentialInput) => string | Promise<string>);

export type PiModelRegistry = {
  getModel(provider: string, id: string): PiModel | undefined;
  completeSimple(
    model: PiModel,
    context: Context,
    options?: SimpleStreamOptions,
  ): Promise<AssistantMessage>;
  streamSimple(
    model: PiModel,
    context: Context,
    options?: SimpleStreamOptions,
  ): AsyncIterable<AssistantMessageEvent>;
};

export type CreatePiRuntimeOptions = {
  apiKey: PiApiKey;
  registry?: PiModelRegistry;
  timeoutMs?: number;
  maxRetries?: number;
};

export type PiRouterModelOptions<Metadata = unknown> = Omit<
  RouterModel<Metadata>,
  "id" | "executor" | "api"
>;

export type PiRuntime = {
  model<Metadata = unknown>(
    id: string,
    options?: Partial<PiRouterModelOptions<Metadata>>,
  ): RouterModel<Metadata>;
  execute<Metadata>(input: RouterExecutorInput<Metadata>): Promise<RouterExecutorResult>;
  select(request: ChatCompletionRequest, signal?: AbortSignal): Promise<string>;
};

export type PiExecution = {
  request: RouterRequest;
  candidate: Pick<RouterCandidate, "id" | "provider" | "providerModelId" | "api">;
  reasoningEffort: ReasoningEffort;
  signal: AbortSignal;
  purpose: PiCredentialInput["purpose"];
};
