// TypeSafe Jev selector over the anonymous-action protocol. Jev
// (docs.typesafe.ai) is a System One model: it answers typed questions about
// a state instead of generating text, so the routing decision is two Choice
// questions over the same state — which anonymous candidate action serves
// the next turns, and how long a lease to commit to — and the answer comes
// back as calibrated probabilities rather than parsed JSON.
//
// This is an alternative to Dari's trained routing policy. Candidates are
// de-identified through the same anonymous-action protocol Dari uses
// (anonymous_actions.ts), so Jev decides on benchmark and cost evidence and
// never sees a model identity; comparisons Jev is weak at (cost ordering,
// benchmark standing) are computed here and passed as text.

import {
  anonymizeSelectorInput,
  assignAnonymousActions,
  sessionRng,
  type AnonymousActionCost,
  type AnonymousActionSlot,
  type AnonymousSelectorInput,
} from "./anonymous_actions.js";
import { RouterCoreError } from "./errors.js";
import { RouterFrameworkError } from "./framework_error.js";
import { FIXED_TURN_COST_COMPARISON_TURNS, SELECTOR_LEASE_TURNS } from "./fixed_turn_cost_config.js";
import type { JsonObject, JsonValue } from "./json.js";
import type { Selector } from "./route.js";
import type { CustomSelectorInput, SelectorInput } from "./selector_input.js";
import {
  isRecord,
  type ChatCompletionRequest,
  type RoutingCandidate,
  type RoutingDecision,
} from "./types.js";

export const DEFAULT_JEV_ENDPOINT = "https://api.typesafe.ai/v1";
export const DEFAULT_JEV_MODEL = "jev-latest";
const DEFAULT_JEV_TIMEOUT_MS = 30_000;
// Jev evaluates up to 32k tokens of state plus the longest question, and
// JSON-shaped conversation state tokenizes at roughly 2.7 characters per
// token. The selector request is sized on its JSON-escaped messages, which
// overstate the state Jev receives, so this budget leaves room for the
// questions and the action descriptions.
export const JEV_SELECTOR_CONTEXT_WINDOW_CHARS = 64_000;

// Rate-limit and overload responses get one backoff retry; anything else is
// the caller's failure to surface.
const JEV_RETRY_STATUSES = new Set([429, 529]);
const JEV_ATTEMPTS = 2;
const JEV_RETRY_DELAY_MS = 500;
const JEV_RETRY_MAX_DELAY_MS = 5_000;

export type JevSelectorConfig = {
  apiKey: string;
  endpoint?: string;
  model?: string;
  // Turn commitments offered as the lease question. Empty routes one turn at
  // a time without asking.
  leaseTurns?: readonly number[];
  timeoutMs?: number;
};

export type ResolvedJevSelectorConfig = {
  apiKey: string;
  endpoint: string;
  model: string;
  leaseTurns: readonly number[];
  timeoutMs: number;
};

export type JevChoiceQuestion = {
  type: "choice";
  instructions: JsonValue;
  criteria: Record<string, JsonValue>;
};

export type JevSystemOneRequest = {
  model: string;
  state: JsonValue;
  questions: {
    action: JevChoiceQuestion;
    lease?: JevChoiceQuestion;
  };
};

export type JevChoiceAnswer = {
  type: "choice";
  choice: string;
  confidence: number;
  probabilities: Record<string, number>;
};

export type JevTokenUsage = { input_tokens: number; output_tokens: number };

export type JevSystemOneResponse = {
  model: string;
  answers: {
    action: JevChoiceAnswer;
    lease?: JevChoiceAnswer;
  };
  usage: JevTokenUsage;
};

// selector_output after TypeSafe returned 200. Token counts are on the body
// even when the action or lease answer is rejected, so hosts can still bill
// the selector call.
export class JevSelectorOutputError extends RouterCoreError {
  readonly usage: JevTokenUsage | null;

  constructor(message: string, code: string, usage: JevTokenUsage | null = null) {
    super("selector_output", message, code);
    this.name = "JevSelectorOutputError";
    this.usage = usage;
  }
}

export type JevSelection = {
  decision: RoutingDecision;
  request: JevSystemOneRequest;
  response: JevSystemOneResponse;
};

export function resolveJevSelectorConfig(input: JevSelectorConfig): ResolvedJevSelectorConfig {
  const apiKey = input.apiKey?.trim() ?? "";
  if (!apiKey) throw configurationError("Jev selector requires an apiKey.", "apiKey");
  const endpoint = (input.endpoint ?? DEFAULT_JEV_ENDPOINT).trim().replace(/\/+$/, "");
  let url: URL;
  try {
    url = new URL(endpoint);
  } catch {
    throw configurationError("Jev selector endpoint must be a valid URL.", "endpoint");
  }
  if ((url.protocol !== "https:" && url.protocol !== "http:") || url.username || url.password) {
    throw configurationError(
      "Jev selector endpoint must be an HTTP(S) URL without embedded credentials.",
      "endpoint",
    );
  }
  const model = (input.model ?? DEFAULT_JEV_MODEL).trim();
  if (!model) throw configurationError("Jev selector model must not be empty.", "model");
  const leaseTurns = input.leaseTurns ?? SELECTOR_LEASE_TURNS;
  if (
    leaseTurns.some((turns, index) =>
      !Number.isSafeInteger(turns) || turns <= 0 || (index > 0 && turns <= leaseTurns[index - 1]!)
    )
  ) {
    throw configurationError(
      "Jev selector leaseTurns must be ascending positive integers.",
      "leaseTurns",
    );
  }
  const timeoutMs = input.timeoutMs ?? DEFAULT_JEV_TIMEOUT_MS;
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs <= 0) {
    throw configurationError("Jev selector timeoutMs must be a positive integer.", "timeoutMs");
  }
  return { apiKey, endpoint, model, leaseTurns, timeoutMs };
}

// Renders the de-identified selector input as a Jev request. Every candidate
// identity has already been replaced by its action letter; the only evidence
// an action carries is its projected cost and its benchmark standing.
export function buildJevSelectorRequest(
  anonymousInput: AnonymousSelectorInput,
  options: { model: string; leaseTurns: readonly number[] },
): JevSystemOneRequest {
  const costs = actionCosts(anonymousInput.cost_estimates);
  const criteria: Record<string, JsonValue> = {};
  for (const { action } of anonymousInput.candidate_actions) {
    const cost = costs.get(action);
    const benchmarks: Record<string, JsonValue> = {};
    for (const card of anonymousInput.imported_evals) {
      const score = card.scores.find((entry) => entry.action === action);
      if (score === undefined) continue;
      benchmarks[card.name] = {
        score: score.score,
        rank: `${score.rank} of ${score.rank_total}`,
      };
    }
    criteria[action] = cost === undefined && Object.keys(benchmarks).length === 0
      ? null
      : {
          ...(cost === undefined ? {} : { cost }),
          ...(Object.keys(benchmarks).length === 0 ? {} : { benchmarks }),
        };
  }

  const state: Record<string, JsonValue> = {
    task: anonymousInput.task,
    previous_action: anonymousInput.previous_action === null
      ? null
      : anonymousInput.previous_action.action,
    lease_history: anonymousInput.lease_history as unknown as JsonValue,
    conversation: anonymousInput.messages,
  };

  const glossary: Record<string, JsonValue> = {};
  for (const card of anonymousInput.imported_evals) {
    glossary[card.name] = {
      ...(card.description === null ? {} : { description: card.description }),
      scale: `${card.min_score} to ${card.max_score}, higher is better`,
    };
  }

  const questions: JevSystemOneRequest["questions"] = {
    action: {
      type: "choice",
      instructions: {
        question: "Which action should serve the coding agent's next turns?",
        rules: [
          "Capability first: the action must be able to handle the task; capability is never traded away for cost.",
          "Judge the task's difficulty and how much work remains from `task`, `lease_history`, and the latest turns of `conversation`.",
          "Use each action's benchmark scores and ranks as evidence of capability. A missing benchmark means the action was not measured on it, not that it scored poorly.",
          "When several actions are capable enough, choose the one with the best (lowest-numbered) cost rank.",
          "Prefer `previous_action` while the task and its current phase have not changed.",
        ],
        ...(Object.keys(glossary).length === 0 ? {} : { benchmarks: glossary }),
      },
      criteria,
    },
    ...(options.leaseTurns.length === 0 ? {} : { lease: leaseQuestion(options.leaseTurns) }),
  };
  return { model: options.model, state, questions };
}

function leaseQuestion(leaseTurns: readonly number[]): JevChoiceQuestion {
  const last = leaseTurns.length - 1;
  const criteria: Record<string, JsonValue> = {};
  leaseTurns.forEach((turns, index) => {
    criteria[String(turns)] = index === 0
      ? `${turns} turns: the task is nearly done, or the next step may change what the agent needs.`
      : index === last
        ? `${turns} turns: a long stretch of similar work remains, such as a large implementation or many edit-and-test cycles.`
        : `${turns} turns: a bounded stretch of similar work remains.`;
  });
  return {
    type: "choice",
    instructions: {
      question: "How many consecutive turns should the chosen action serve before the router reconsiders?",
      guidance: "A longer lease amortizes the cold start paid on its first turn across warm cache hits on the turns that follow. The lease ends early only if the task finishes or the provider fails. Judge from how much similar work remains after the latest turn of `conversation`.",
    },
    criteria,
  };
}

export function parseJevSelectorResponse(
  body: unknown,
  args: { slots: readonly AnonymousActionSlot[]; leaseTurns: readonly number[] },
): { decision: RoutingDecision; response: JevSystemOneResponse } {
  if (!isRecord(body) || typeof body.model !== "string" || !isRecord(body.answers)) {
    throw invalidResponse("Jev returned an unrecognized response body.");
  }
  const usage = body.usage;
  if (
    !isRecord(usage)
    || typeof usage.input_tokens !== "number"
    || typeof usage.output_tokens !== "number"
  ) {
    throw invalidResponse("Jev response is missing token usage.");
  }
  const tokenUsage: JevTokenUsage = {
    input_tokens: usage.input_tokens,
    output_tokens: usage.output_tokens,
  };
  const action = choiceAnswer(body.answers.action, "action", tokenUsage);
  const slot = args.slots.find((entry) => entry.action === action.choice);
  if (slot === undefined) {
    throw new JevSelectorOutputError(
      "Jev selected an action outside this router's candidates.",
      "selector_invalid_candidate",
      tokenUsage,
    );
  }
  let lease: JevChoiceAnswer | undefined;
  let turns: number | undefined;
  if (args.leaseTurns.length > 0) {
    lease = choiceAnswer(body.answers.lease, "lease", tokenUsage);
    turns = Number(lease.choice);
    if (!args.leaseTurns.includes(turns)) {
      throw invalidResponse(
        `Jev lease answer ${lease.choice} is not on the offered menu.`,
        tokenUsage,
      );
    }
  }
  const response: JevSystemOneResponse = {
    model: body.model,
    answers: { action, ...(lease === undefined ? {} : { lease }) },
    usage: tokenUsage,
  };
  return {
    decision: {
      selectedModel: slot.candidate.model,
      reasoningEffort: slot.candidate.reasoningEffort as RoutingDecision["reasoningEffort"],
      // The reason is persisted into lease history and replayed to later
      // selectors, so it stays anonymous like the rest of the protocol.
      reason: decisionReason(action, turns, lease),
      ...(turns === undefined ? {} : { leaseTurnsRemaining: turns - 1 }),
    },
    response,
  };
}

function decisionReason(
  action: JevChoiceAnswer,
  turns: number | undefined,
  lease: JevChoiceAnswer | undefined,
): string {
  const pick = `Jev selected Action ${action.choice} `
    + `(probability ${percent(action.probabilities[action.choice] ?? 0)}, confidence ${percent(action.confidence)})`;
  if (turns === undefined || lease === undefined) return `${pick}.`;
  return `${pick} with a ${turns}-turn lease (probability ${percent(lease.probabilities[lease.choice] ?? 0)}).`;
}

// Anonymizes the named selector input against the slots, asks Jev, and maps
// the answers back onto the slots' candidates.
export async function selectWithJev(
  selectorInput: SelectorInput | CustomSelectorInput,
  slots: readonly AnonymousActionSlot[],
  config: ResolvedJevSelectorConfig,
  signal?: AbortSignal,
): Promise<JevSelection> {
  if ("custom_rules" in selectorInput || "default_target" in selectorInput) {
    throw new RouterCoreError(
      "configuration",
      "The Jev selector serves only the default routing policy and cannot apply custom routing rules.",
      "custom_rules_not_supported",
    );
  }
  const anonymousInput = anonymizeSelectorInput(selectorInput as unknown as JsonObject, slots);
  const request = buildJevSelectorRequest(anonymousInput, config);
  const body = await postSystemOne(request, config, signal);
  const { decision, response } = parseJevSelectorResponse(body, {
    slots,
    leaseTurns: config.leaseTurns,
  });
  return { decision, request, response };
}

// A Selector for createDariRoutingPolicy: decodes the selector input from the
// chat-shaped request and returns a routing_decision JSON the parser already
// understands, including lease_turns_remaining and the raw Jev answers.
// Letters are assigned per call: the framework path carries no conversation
// identity, and every candidate-bearing field is re-lettered together, so
// correctness never depends on cross-turn stability.
export function createJevSelector(input: JevSelectorConfig): Selector {
  const config = resolveJevSelectorConfig(input);
  return {
    async select(request, signal) {
      const selectorInput = selectorInputFromRequest(request);
      const single = singleCandidate(selectorInput);
      if (single !== undefined) {
        return JSON.stringify({
          selected_model: single.model,
          reasoning_effort: single.reasoningEffort,
          reason: "The router has only one compatible model/thinking-level pair.",
        });
      }
      const candidates = selectorInput.candidate_pairs.map((pair) => ({
        model: pair.model,
        reasoningEffort: pair.thinking_level,
      }));
      const rng = typeof request.prompt_cache_key === "string" && request.prompt_cache_key !== ""
        ? sessionRng(request.prompt_cache_key)
        : Math.random;
      let selection: JevSelection;
      try {
        selection = await selectWithJev(
          selectorInput,
          assignAnonymousActions(candidates, rng),
          config,
          signal,
        );
      } catch (error) {
        throw frameworkError(error);
      }
      const { decision, response } = selection;
      return JSON.stringify({
        selected_model: decision.selectedModel,
        reasoning_effort: decision.reasoningEffort,
        reason: decision.reason,
        ...(decision.leaseTurnsRemaining === undefined
          ? {}
          : { lease_turns_remaining: decision.leaseTurnsRemaining }),
        jev: response,
      });
    },
  };
}

function singleCandidate(
  selectorInput: SelectorInput,
): RoutingCandidate | undefined {
  if (selectorInput.candidate_pairs.length !== 1) return undefined;
  const pair = selectorInput.candidate_pairs[0]!;
  return { model: pair.model, reasoningEffort: pair.thinking_level };
}

async function postSystemOne(
  request: JevSystemOneRequest,
  config: ResolvedJevSelectorConfig,
  signal: AbortSignal | undefined,
): Promise<unknown> {
  for (let attempt = 1; ; attempt += 1) {
    const response = await fetchSystemOne(request, config, signal);
    if (response.ok) {
      try {
        return await response.json();
      } catch {
        throw invalidResponse("Jev returned a non-JSON response body.");
      }
    }
    if (attempt < JEV_ATTEMPTS && JEV_RETRY_STATUSES.has(response.status)) {
      await delay(retryDelayMs(response.headers.get("retry-after")), signal);
      continue;
    }
    const body = await response.text().catch(() => "");
    throw new RouterFrameworkError(
      "policy",
      `Jev returned HTTP ${response.status}: ${body}`,
      "jev_http_error",
    );
  }
}

async function fetchSystemOne(
  request: JevSystemOneRequest,
  config: ResolvedJevSelectorConfig,
  signal: AbortSignal | undefined,
): Promise<Response> {
  const timeout = AbortSignal.timeout(config.timeoutMs);
  try {
    return await fetch(`${config.endpoint}/systemone`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${config.apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
      signal: signal === undefined ? timeout : AbortSignal.any([signal, timeout]),
    });
  } catch (error) {
    if (signal?.aborted) throw error;
    throw new RouterFrameworkError(
      "policy",
      `Jev request failed: ${error instanceof Error ? error.message : String(error)}`,
      "jev_request_failed",
      undefined,
      { cause: error },
    );
  }
}

function retryDelayMs(retryAfter: string | null): number {
  const seconds = Number(retryAfter);
  if (retryAfter === null || !Number.isFinite(seconds) || seconds <= 0) return JEV_RETRY_DELAY_MS;
  return Math.min(seconds * 1000, JEV_RETRY_MAX_DELAY_MS);
}

function delay(ms: number, signal: AbortSignal | undefined): Promise<void> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(timer);
      reject(signal?.reason);
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

function selectorInputFromRequest(request: ChatCompletionRequest): SelectorInput {
  const user = request.messages?.find((message) => message.role === "user");
  if (typeof user?.content !== "string") {
    throw new RouterCoreError(
      "invalid_request",
      "Selector request is missing the JSON user message.",
      "invalid_request_error",
    );
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(user.content);
  } catch {
    throw new RouterCoreError(
      "invalid_request",
      "Selector request user message is not valid JSON.",
      "invalid_request_error",
    );
  }
  if (!isRecord(parsed) || !Array.isArray(parsed.candidate_pairs) || !Array.isArray(parsed.messages)) {
    throw new RouterCoreError(
      "invalid_request",
      "Selector input is missing candidate_pairs or messages.",
      "invalid_request_error",
    );
  }
  return parsed as unknown as SelectorInput;
}

// Cost per action as text Jev can compare literally: the loop projections at
// each lease length, plus a rank across actions on the comparison horizon.
function actionCosts(costs: readonly AnonymousActionCost[]): Map<string, JsonValue> {
  const scalars = new Map<string, number>();
  const rendered = new Map<string, Record<string, JsonValue>>();
  for (const cost of costs) {
    if (cost.projections.length === 0) continue;
    const comparison = cost.projections.find(
      (projection) => projection.projected_turns === FIXED_TURN_COST_COMPARISON_TURNS,
    ) ?? cost.projections[cost.projections.length - 1]!;
    scalars.set(cost.action, comparison.total_cost_usd);
    rendered.set(cost.action, {
      projected_loop_cost: Object.fromEntries(
        cost.projections.map((projection) => [
          `${projection.projected_turns} turns`,
          usd(projection.total_cost_usd),
        ]),
      ),
    });
  }
  const ranked = [...scalars.entries()].sort((left, right) => left[1] - right[1]);
  ranked.forEach(([, value], index) => {
    // Competition ranking: tied costs share the better rank.
    const rank = ranked.findIndex(([, other]) => other === value) + 1;
    const suffix = ranked.length === 1
      ? ""
      : rank === 1
        ? " (cheapest)"
        : index === ranked.length - 1 && rank === ranked.length
          ? " (most expensive)"
          : "";
    const action = ranked[index]![0];
    rendered.get(action)!.rank = `${rank} of ${ranked.length}${suffix}`;
  });
  return rendered;
}

function choiceAnswer(
  value: unknown,
  name: string,
  usage: JevTokenUsage,
): JevChoiceAnswer {
  if (
    !isRecord(value)
    || value.type !== "choice"
    || typeof value.choice !== "string"
    || typeof value.confidence !== "number"
    || !isRecord(value.probabilities)
  ) {
    throw invalidResponse(`Jev response is missing a choice answer for ${name}.`, usage);
  }
  const probabilities: Record<string, number> = {};
  for (const [label, probability] of Object.entries(value.probabilities)) {
    if (typeof probability === "number") probabilities[label] = probability;
  }
  return {
    type: "choice",
    choice: value.choice,
    confidence: value.confidence,
    probabilities,
  };
}

function usd(value: number): string {
  if (value >= 1) return `$${value.toFixed(2)}`;
  const trimmed = value.toPrecision(3).replace(/0+$/, "").replace(/\.$/, "");
  return `$${trimmed}`;
}

function percent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function invalidResponse(message: string, usage: JevTokenUsage | null = null): JevSelectorOutputError {
  return new JevSelectorOutputError(message, "selector_invalid_response", usage);
}

function configurationError(message: string, param: string): RouterCoreError {
  return new RouterCoreError("configuration", message, "jev_configuration_invalid", param);
}

function frameworkError(error: unknown): unknown {
  if (!(error instanceof RouterCoreError)) return error;
  const kind = error.kind === "selector_output" ? "policy" : error.kind;
  return new RouterFrameworkError(kind, error.message, error.code, error.param, { cause: error });
}
