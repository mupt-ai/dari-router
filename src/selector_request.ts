import { RouterCoreError } from "./errors.js";
import { CUSTOM_SELECTOR_SYSTEM_PROMPT, SELECTOR_SYSTEM_PROMPT } from "./prompts.js";
import {
  buildSelectorInput,
  type CustomSelectorInput,
  type SelectorInput,
} from "./selector_input.js";
import { trimMessagesFromFront } from "./selector_truncation.js";
import { REASONING_EFFORTS } from "./types.js";
import type {
  CandidateCostEstimate,
  ChatCompletionRequest,
  ChatMessage,
  CustomRouterRule,
  PreviousDecision,
  ReasoningEffort,
  RouterEval,
  RoutingCandidate,
} from "./types.js";

export type BuiltSelectorRequest = {
  selectorInput: SelectorInput | CustomSelectorInput;
  selectorRequest: ChatCompletionRequest;
};

export type SelectorRequestArgs = {
  candidates: RoutingCandidate[];
  evals?: RouterEval[];
  previousDecision?: PreviousDecision;
  costEstimates?: CandidateCostEstimate[];
  customRules?: CustomRouterRule[];
  defaultTarget?: {
    model: string;
    thinkingLevel: ReasoningEffort | null;
  } | null;
  selectorModel: string;
  messages: ChatMessage[];
  modelFallbackEnabled?: boolean;
  fallbackRequiresDifferentProvider?: boolean;
  imputeEvalScores?: boolean;
  imputationReferenceEvals?: RouterEval[];
};

// Builds the selector request, front-trimming conversation messages until the
// serialized prompt fits the selector's context window. The window size is an
// explicit input (in characters) so the core stays free of model-registry
// lookups.
export function buildSizedSelectorRequest(
  args: SelectorRequestArgs & { contextWindowChars: number },
): BuiltSelectorRequest {
  let messages = args.messages;
  let built = buildSelectorRequest({ ...args, messages });
  let previousInputChars = Number.POSITIVE_INFINITY;

  while (true) {
    const inputChars = selectorRequestInputChars(built.selectorRequest);
    if (inputChars <= args.contextWindowChars) return built;
    if (inputChars >= previousInputChars) {
      throw new RouterCoreError(
        "configuration",
        `Selector prompt cannot fit configured context window for ${args.selectorModel}.`,
        "configuration_error"
      );
    }
    previousInputChars = inputChars;
    messages = trimMessagesFromFront(messages, inputChars - args.contextWindowChars);
    built = buildSelectorRequest({ ...args, messages });
  }
}

function selectorRequestInputChars(request: ChatCompletionRequest): number {
  return JSON.stringify(request.messages ?? []).length;
}

export function buildSelectorRequest(args: SelectorRequestArgs): BuiltSelectorRequest {
  const isCustom = args.customRules !== undefined;
  const selectorInput = buildSelectorInput({
    candidates: args.candidates,
    evals: args.evals ?? [],
    previousDecision: args.previousDecision ?? null,
    costEstimates: args.costEstimates ?? null,
    messages: args.messages,
    customRules: args.customRules,
    defaultTarget: args.defaultTarget,
    modelFallbackEnabled: args.modelFallbackEnabled,
    fallbackRequiresDifferentProvider:
      args.fallbackRequiresDifferentProvider,
    imputeEvalScores: args.imputeEvalScores,
    imputationReferenceEvals: args.imputationReferenceEvals,
  });
  return {
    selectorInput,
    selectorRequest: {
      model: args.selectorModel,
      messages: [
        { role: "system", content: isCustom ? CUSTOM_SELECTOR_SYSTEM_PROMPT : SELECTOR_SYSTEM_PROMPT },
        {
          role: "user",
          content: JSON.stringify(selectorInput),
        },
      ],
      // The selector is an internal fixed-purpose JSON pick; it must not inherit
      // the client-facing implicit-medium reasoning default. Selector models
      // must support a true off mode.
      reasoning_effort: "off",
      response_format: {
        type: "json_schema",
        json_schema: {
          name: "routing_decision",
          strict: true,
          schema: {
            type: "object",
            additionalProperties: false,
            properties: {
              selected_model: { type: "string" },
              reasoning_effort: {
                type: "string",
                enum: [...REASONING_EFFORTS],
              },
              reason: { type: "string" },
            },
            required: [
              "selected_model",
              "reasoning_effort",
              "reason",
            ],
          },
        },
      },
    },
  };
}
