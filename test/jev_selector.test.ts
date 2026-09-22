import { afterEach, expect, test } from "bun:test";

import {
  createJevRouter,
  createRouter,
  RouterFrameworkError,
  type RouterModel,
} from "../src/index.js";
import {
  anonymizeSelectorInput,
  assignAnonymousActions,
  type AnonymousActionSlot,
  type Rng,
} from "../src/anonymous_actions.js";
import {
  buildJevSelectorRequest,
  createJevSelector,
  JevSelectorOutputError,
  parseJevSelectorResponse,
  resolveJevSelectorConfig,
  selectWithJev,
  type JevSystemOneRequest,
} from "../src/jev_selector.js";
import { buildSelectorRequest } from "../src/selector_request.js";
import type { SelectorInput } from "../src/selector_input.js";
import { RouterCoreError } from "../src/errors.js";
import type { RouterEval, RoutingCandidate } from "../src/types.js";

const SOL = "openai/gpt-5.6-sol";
const SONNET = "anthropic/claude-sonnet-5";
const FLASH = "fireworks/deepseek-ai/DeepSeek-V4-Flash";

const CANDIDATES: RoutingCandidate[] = [
  { model: SOL, reasoningEffort: "high" },
  { model: SONNET, reasoningEffort: "medium" },
  { model: FLASH, reasoningEffort: "high" },
];

const EVALS: RouterEval[] = [
  {
    id: "swe",
    name: "SWE-bench Verified",
    description: "Resolved rate on real GitHub issues",
    min_score: 0,
    max_score: 100,
    scores: [
      { model_id: SOL, thinking_level: "high", score: 78 },
      { model_id: SONNET, thinking_level: "medium", score: 72 },
    ],
  },
];

const originalFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = originalFetch;
});

// Deterministic rng so letter assignments are reproducible in tests.
function rng(seed: number): Rng {
  let state = seed % 2147483647;
  if (state <= 0) state += 2147483646;
  return () => {
    state = (state * 16807) % 2147483647;
    return (state - 1) / 2147483646;
  };
}

function slots(): AnonymousActionSlot[] {
  return assignAnonymousActions(CANDIDATES, rng(42));
}

function actionFor(candidate: RoutingCandidate): string {
  const slot = slots().find(
    (entry) => entry.candidate.model === candidate.model
      && entry.candidate.reasoningEffort === candidate.reasoningEffort,
  );
  if (slot === undefined) throw new Error("no slot");
  return slot.action;
}

function selectorInput(overrides: Partial<SelectorInput> = {}): SelectorInput {
  const { selectorInput: built } = buildSelectorRequest({
    candidates: CANDIDATES,
    evals: EVALS,
    previousDecision: { model: SONNET, reasoningEffort: "medium", reason: "held" },
    costEstimates: CANDIDATES.map((candidate, index) => ({
      model: candidate.model,
      reasoning_effort: candidate.reasoningEffort,
      warm_tokens: 0,
      est_prompt_tokens: 1_000,
      est_input_cost_usd: 0.01 * (3 - index),
      output_cost_per_mtok: 10,
      pricing_known: true,
      fixed_turn_cost_estimate: {
        output_tokens_per_turn: 500,
        assumed_reasoning_effort: candidate.reasoningEffort,
        projections: [5, 10, 30].map((turns) => ({
          projected_turns: turns,
          total_cost_usd: 0.03 * turns * (3 - index),
        })),
      },
    })),
    selectorModel: "typesafe/jev-latest",
    messages: [
      { role: "user", content: "Fix the failing auth tests." },
      { role: "assistant", content: "Running the suite." },
    ],
    task: { role: "user", content: "Fix the failing auth tests." },
    leaseHistory: [
      {
        candidate: { model: SONNET, thinking_level: "medium" },
        requested_turns: 10,
        completed_turns: 10,
        rationale: "held",
        last_agent_thought: null,
        last_tool_call: null,
        last_tool_result: null,
        tool_errors: 1,
        files_changed: 2,
        tests: { command: "bun test", status: "failed", result: "3 failing" },
      },
    ],
  });
  return { ...(built as SelectorInput), ...overrides };
}

function jevAnswer(
  action: string,
  lease: string | undefined,
  overrides: Record<string, unknown> = {},
): Record<string, unknown> {
  return {
    model: "jev-1.13.0",
    answers: {
      action: {
        type: "choice",
        choice: action,
        confidence: 0.6,
        probabilities: Object.fromEntries(
          slots().map((slot) => [slot.action, slot.action === action ? 0.7 : 0.15]),
        ),
      },
      ...(lease === undefined
        ? {}
        : {
            lease: {
              type: "choice",
              choice: lease,
              confidence: 0.5,
              probabilities: { "5": 0.2, "10": 0.6, "30": 0.2 },
            },
          }),
    },
    usage: { input_tokens: 1_000, output_tokens: 40 },
    ...overrides,
  };
}

test("resolveJevSelectorConfig validates the API key, endpoint, and lease menu", () => {
  expect(() => resolveJevSelectorConfig({ apiKey: " " })).toThrow("requires an apiKey");
  expect(() => resolveJevSelectorConfig({ apiKey: "k", endpoint: "not a url" })).toThrow(
    "must be a valid URL",
  );
  expect(() =>
    resolveJevSelectorConfig({ apiKey: "k", endpoint: "https://user:pw@api.typesafe.ai/v1" }),
  ).toThrow("without embedded credentials");
  expect(() => resolveJevSelectorConfig({ apiKey: "k", leaseTurns: [10, 5] })).toThrow(
    "ascending positive integers",
  );
  const resolved = resolveJevSelectorConfig({ apiKey: "k", endpoint: "https://api.typesafe.ai/v1/" });
  expect(resolved.endpoint).toBe("https://api.typesafe.ai/v1");
  expect(resolved.model).toBe("jev-latest");
  expect(resolved.leaseTurns).toEqual([5, 10, 30]);
});

test("renders the anonymized input as a Jev state with action and lease choices", () => {
  const anonymous = anonymizeSelectorInput(selectorInput() as never, slots());
  const request = buildJevSelectorRequest(anonymous, {
    model: "jev-latest",
    leaseTurns: [5, 10, 30],
  });

  expect(request.model).toBe("jev-latest");
  const state = request.state as Record<string, unknown>;
  expect(state.task).toEqual({ role: "user", content: "Fix the failing auth tests." });
  expect(state.previous_action).toBe(actionFor({ model: SONNET, reasoningEffort: "medium" }));
  expect(state.lease_history).toEqual([
    expect.objectContaining({
      action: actionFor({ model: SONNET, reasoningEffort: "medium" }),
      requested_turns: 10,
      completed_turns: 10,
    }),
  ]);
  expect(state.conversation).toEqual([
    { role: "user", content: "Fix the failing auth tests." },
    { role: "assistant", content: "Running the suite." },
  ]);

  const actions = slots().map((slot) => slot.action);
  const action = request.questions.action;
  expect(Object.keys(action.criteria).sort()).toEqual([...actions].sort());
  // The most expensive candidate carries its projections, its rank, and its
  // benchmark standing; the cheapest has no benchmark row at all.
  expect(action.criteria[actionFor(CANDIDATES[0]!)]).toEqual({
    cost: {
      projected_loop_cost: { "5 turns": "$0.45", "10 turns": "$0.9", "30 turns": "$2.70" },
      rank: "3 of 3 (most expensive)",
    },
    benchmarks: { "SWE-bench Verified": { score: 78, rank: "1 of 2" } },
  });
  expect(action.criteria[actionFor(CANDIDATES[2]!)]).toEqual({
    cost: {
      projected_loop_cost: { "5 turns": "$0.15", "10 turns": "$0.3", "30 turns": "$0.9" },
      rank: "1 of 3 (cheapest)",
    },
  });
  const instructions = action.instructions as Record<string, unknown>;
  expect(instructions.benchmarks).toEqual({
    "SWE-bench Verified": {
      description: "Resolved rate on real GitHub issues",
      scale: "0 to 100, higher is better",
    },
  });

  const lease = request.questions.lease!;
  expect(Object.keys(lease.criteria)).toEqual(["5", "10", "30"]);
  // The whole request is de-identified: no model identity reaches Jev.
  for (const model of [SOL, SONNET, FLASH]) {
    expect(JSON.stringify(request)).not.toContain(model);
  }
  expect(JSON.stringify(request)).not.toContain("thinking_level");
});

test("omits the lease question when no menu is offered", () => {
  const anonymous = anonymizeSelectorInput(selectorInput() as never, slots());
  const request = buildJevSelectorRequest(anonymous, { model: "jev-latest", leaseTurns: [] });
  expect(request.questions.lease).toBeUndefined();
});

test("selectWithJev rejects custom-rule selector inputs", async () => {
  const custom = { ...selectorInput(), custom_rules: [], default_target: null };
  const error = await selectWithJev(
    custom,
    slots(),
    resolveJevSelectorConfig({ apiKey: "k" }),
  ).catch((caught: unknown) => caught);
  expect(error).toBeInstanceOf(RouterCoreError);
});

test("maps the Jev answers onto the slots as a leased, anonymous decision", () => {
  const chosen = actionFor(CANDIDATES[2]!);
  const { decision, response } = parseJevSelectorResponse(jevAnswer(chosen, "10"), {
    slots: slots(),
    leaseTurns: [5, 10, 30],
  });
  expect(decision).toEqual({
    selectedModel: FLASH,
    reasoningEffort: "high",
    reason: `Jev selected Action ${chosen} (probability 70%, confidence 60%) with a 10-turn lease (probability 60%).`,
    leaseTurnsRemaining: 9,
  });
  expect(decision.reason).not.toContain(FLASH);
  expect(response.model).toBe("jev-1.13.0");
  expect(response.usage).toEqual({ input_tokens: 1_000, output_tokens: 40 });
});

test("rejects answers outside the slots or the lease menu", () => {
  expect(() =>
    parseJevSelectorResponse(jevAnswer("D", "10"), { slots: slots(), leaseTurns: [5, 10, 30] }),
  ).toThrow("outside this router's candidates");
  try {
    parseJevSelectorResponse(jevAnswer("D", "10"), { slots: slots(), leaseTurns: [5, 10, 30] });
    throw new Error("expected JevSelectorOutputError");
  } catch (error) {
    expect(error).toBeInstanceOf(JevSelectorOutputError);
    expect((error as JevSelectorOutputError).usage).toEqual({
      input_tokens: 1_000,
      output_tokens: 40,
    });
  }
  expect(() =>
    parseJevSelectorResponse(jevAnswer(actionFor(CANDIDATES[0]!), "7"), {
      slots: slots(),
      leaseTurns: [5, 10, 30],
    }),
  ).toThrow("not on the offered menu");
  expect(() =>
    parseJevSelectorResponse(jevAnswer(actionFor(CANDIDATES[0]!), undefined), {
      slots: slots(),
      leaseTurns: [5, 10, 30],
    }),
  ).toThrow("missing a choice answer for lease");
  expect(() =>
    parseJevSelectorResponse(jevAnswer(actionFor(CANDIDATES[0]!), "10", { usage: null }), {
      slots: slots(),
      leaseTurns: [5, 10, 30],
    }),
  ).toThrow("missing token usage");
});

test("createJevSelector posts an anonymized request and returns a routing_decision document", async () => {
  let captured: { url: string; auth: string | null; body: JevSystemOneRequest } | undefined;
  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    captured = {
      url: String(input),
      auth: new Headers(init?.headers).get("authorization"),
      body: JSON.parse(String(init?.body)),
    };
    // Whatever letters this call's rng drew, answer with an offered action.
    const offered = Object.keys(captured.body.questions.action.criteria);
    return Response.json(jevAnswerFor(offered, offered[0]!, "30"));
  }) as typeof fetch;

  const selector = createJevSelector({ apiKey: "ts-key" });
  const { selectorRequest } = buildSelectorRequest({
    candidates: CANDIDATES,
    selectorModel: "typesafe/jev-latest",
    messages: [{ role: "user", content: "hello" }],
  });
  const output = JSON.parse(await selector.select(selectorRequest));

  expect(captured?.url).toBe("https://api.typesafe.ai/v1/systemone");
  expect(captured?.auth).toBe("Bearer ts-key");
  expect(captured?.body.model).toBe("jev-latest");
  for (const model of [SOL, SONNET, FLASH]) {
    expect(JSON.stringify(captured?.body)).not.toContain(model);
  }
  expect(output.jev).toMatchObject({ model: "jev-1.13.0" });
  expect(CANDIDATES.map((candidate) => candidate.model)).toContain(output.selected_model);
  expect(output.reason).toMatch(/^Jev selected Action [A-Z]+ /);
  expect(output.lease_turns_remaining).toBe(29);
});

test("createJevSelector answers a single candidate without calling Jev", async () => {
  globalThis.fetch = (async (_input: RequestInfo | URL): Promise<Response> => {
    throw new Error("must not call Jev");
  }) as typeof fetch;
  const selector = createJevSelector({ apiKey: "ts-key" });
  const { selectorRequest } = buildSelectorRequest({
    candidates: [CANDIDATES[0]!],
    selectorModel: "typesafe/jev-latest",
    messages: [{ role: "user", content: "hello" }],
  });
  const output = JSON.parse(await selector.select(selectorRequest));
  expect(output).toEqual({
    selected_model: SOL,
    reasoning_effort: "high",
    reason: "The router has only one compatible model/thinking-level pair.",
  });
});

test("createJevSelector retries once on rate limits and surfaces other HTTP errors", async () => {
  const statuses = [429, 200];
  let calls = 0;
  globalThis.fetch = (async (_input: RequestInfo | URL, init?: RequestInit) => {
    calls += 1;
    const status = statuses.shift()!;
    if (status !== 200) {
      return new Response("slow down", { status, headers: { "retry-after": "0" } });
    }
    const body = JSON.parse(String(init?.body)) as JevSystemOneRequest;
    const offered = Object.keys(body.questions.action.criteria);
    return Response.json(jevAnswerFor(offered, offered[0]!, "5"));
  }) as typeof fetch;
  const selector = createJevSelector({ apiKey: "ts-key" });
  const { selectorRequest } = buildSelectorRequest({
    candidates: CANDIDATES,
    selectorModel: "typesafe/jev-latest",
    messages: [{ role: "user", content: "hello" }],
  });
  const output = JSON.parse(await selector.select(selectorRequest));
  expect(calls).toBe(2);
  expect(CANDIDATES.map((candidate) => candidate.model)).toContain(output.selected_model);

  globalThis.fetch = (async (_input: RequestInfo | URL) =>
    new Response("bad key", { status: 401 })) as typeof fetch;
  const error = await selector.select(selectorRequest).catch((caught: unknown) => caught);
  expect(error).toBeInstanceOf(RouterFrameworkError);
  expect((error as RouterFrameworkError).code).toBe("jev_http_error");
  expect((error as RouterFrameworkError).message).toContain("401");
});

const MODELS: RouterModel[] = [
  {
    id: SOL,
    executor: "mock",
    api: "openai-responses",
    reasoningEfforts: ["high"],
    capabilities: { imageInput: true, toolUse: true },
  },
  {
    id: SONNET,
    executor: "mock",
    api: "anthropic-messages",
    reasoningEfforts: ["medium"],
    capabilities: { imageInput: true, toolUse: true },
  },
];

const PRICING = {
  [SOL]: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
  [SONNET]: { input: 1.5, output: 7.5, cacheRead: 0.15, cacheWrite: 1.9 },
};

const COST_OPTIONS = {
  pricing: (model: string) => PRICING[model as keyof typeof PRICING] ?? null,
  averageOutputTokensByModel: {
    [SOL]: { high: 800 },
    [SONNET]: { medium: 800 },
  },
};

test("createJevRouter routes through Jev and holds the lease it granted", async () => {
  let policyCalls = 0;
  globalThis.fetch = (async (_input: RequestInfo | URL, init?: RequestInit) => {
    policyCalls += 1;
    const body = JSON.parse(String(init?.body)) as JevSystemOneRequest;
    const offered = Object.keys(body.questions.action.criteria);
    return Response.json(jevAnswerFor(offered, offered[0]!, "5"));
  }) as typeof fetch;

  const router = createRouter({
    models: MODELS,
    policy: createJevRouter({ apiKey: "ts-key", ...COST_OPTIONS }),
    executors: {
      mock: {
        execute: ({ model }) => ({
          type: "complete",
          output: { content: [{ type: "text", text: model.id }], finishReason: "stop" },
        }),
      },
    },
  });
  const body = {
    model: "dari/routing",
    messages: [{ role: "user", content: "hello" }],
    prompt_cache_key: "conversation-1",
  };
  const request = () =>
    new Request("https://test.test/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

  const first = await router.fetch(request());
  expect(first.status).toBe(200);
  const served = first.headers.get("X-Router-Selected-Model")!;
  expect([SOL, SONNET]).toContain(served);
  expect(policyCalls).toBe(1);
  const second = await router.fetch(request());
  expect(second.headers.get("X-Router-Selected-Model")).toBe(served);
  expect(policyCalls).toBe(1);
});

test("createJevRouter requires an API key and host-owned accounting", () => {
  expect(() => createJevRouter({ apiKey: "", ...COST_OPTIONS })).toThrow(RouterCoreError);
  expect(() => createJevRouter({ apiKey: "ts-key" } as never)).toThrow(RouterFrameworkError);
});

// Like jevAnswer, but for a request whose letter permutation was drawn by a
// different rng than the test fixture's.
function jevAnswerFor(
  offered: readonly string[],
  action: string,
  lease: string,
): Record<string, unknown> {
  return {
    model: "jev-1.13.0",
    answers: {
      action: {
        type: "choice",
        choice: action,
        confidence: 0.8,
        probabilities: Object.fromEntries(
          offered.map((entry) => [entry, entry === action ? 0.9 : 0.1 / (offered.length - 1)]),
        ),
      },
      lease: {
        type: "choice",
        choice: lease,
        confidence: 0.7,
        probabilities: { "5": 0.8, "10": 0.1, "30": 0.1 },
      },
    },
    usage: { input_tokens: 300, output_tokens: 30 },
  };
}
