import { RouterCoreError } from "./errors.js";
import { CUSTOM_SELECTOR_SYSTEM_PROMPT, SELECTOR_SYSTEM_PROMPT } from "./prompts.js";
import {
  buildSelectorInput,
  type CustomSelectorInput,
  type SelectorInput,
  type SelectorLeaseHistoryEntry,
} from "./selector_input.js";
import {
  olderMessageChars,
  trimMessagesFromFront,
  trimTaskMessage,
} from "./selector_truncation.js";
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
  task?: ChatMessage | null;
  leaseHistory?: SelectorLeaseHistoryEntry[];
  modelFallbackEnabled?: boolean;
  fallbackRequiresDifferentProvider?: boolean;
  imputeEvalScores?: boolean;
  imputationReferenceEvals?: RouterEval[];
};

// Builds the selector request, trimming old conversation history first and an
// oversized retained task only when necessary. The window size is an explicit
// input (in characters) so the core stays free of model-registry lookups.
export function buildSizedSelectorRequest(
  args: SelectorRequestArgs & { contextWindowChars: number },
): BuiltSelectorRequest {
  let messages = args.messages;
  let task = args.task;
  const taskExceedsWindow =
    task !== undefined
    && task !== null
    && JSON.stringify(task).length > args.contextWindowChars;
  let built = buildSelectorRequest({ ...args, messages, task });

  while (true) {
    const inputChars = selectorRequestInputChars(built.selectorRequest);
    if (inputChars <= args.contextWindowChars) return built;
    const dropChars = inputChars - args.contextWindowChars;

    const removableHistoryChars = olderMessageChars(messages);
    if (removableHistoryChars > 0) {
      const trimmedMessages = trimMessagesFromFront(
        messages,
        Math.min(dropChars, removableHistoryChars),
      );
      const messagesBuilt = buildSelectorRequest({ ...args, messages: trimmedMessages, task });
      if (selectorRequestInputChars(messagesBuilt.selectorRequest) < inputChars) {
        messages = trimmedMessages;
        built = messagesBuilt;
        continue;
      }
    }

    // If the retained task cannot fit the whole window by itself, compact it
    // before touching the latest conversation turn. Smaller tasks keep the
    // previous behavior: the conversation view shrinks first.
    if (taskExceedsWindow && task !== undefined && task !== null) {
      const taskChars = JSON.stringify(task).length;
      const maxTaskDropChars = Math.max(
        0,
        taskChars - Math.floor(args.contextWindowChars / 2),
      );
      const trimmedTask = trimTaskMessage(task, Math.min(dropChars, maxTaskDropChars));
      const taskBuilt = buildSelectorRequest({ ...args, messages, task: trimmedTask });
      if (selectorRequestInputChars(taskBuilt.selectorRequest) < inputChars) {
        task = trimmedTask;
        built = taskBuilt;
        continue;
      }
    }

    const trimmedMessages = trimMessagesFromFront(messages, dropChars);
    const messagesBuilt = buildSelectorRequest({ ...args, messages: trimmedMessages, task });
    if (selectorRequestInputChars(messagesBuilt.selectorRequest) < inputChars) {
      messages = trimmedMessages;
      built = messagesBuilt;
      continue;
    }

    if (task !== undefined && task !== null) {
      const trimmedTask = trimTaskMessage(task, dropChars);
      const taskBuilt = buildSelectorRequest({ ...args, messages, task: trimmedTask });
      if (selectorRequestInputChars(taskBuilt.selectorRequest) < inputChars) {
        task = trimmedTask;
        built = taskBuilt;
        continue;
      }
    }

    throw new RouterCoreError(
      "configuration",
      `Selector prompt cannot fit configured context window for ${args.selectorModel}.`,
      "configuration_error"
    );
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
    task: args.task,
    leaseHistory: args.leaseHistory,
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
