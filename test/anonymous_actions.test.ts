import { expect, test } from "bun:test";

import type { JsonObject } from "../src/json.js";
import { ANONYMOUS_ACTION_SYSTEM_PROMPT } from "../src/prompts.js";
import {
  assignAnonymousActions,
  anonymizeSelectorInput,
  buildAnonymousPolicyPrompt,
  parseAnonymousActionSelection,
  type Rng,
} from "../src/anonymous_actions.js";

const CHEAP = "fireworks/cheap";
const STRONG = "fireworks/strong";
const MEDIUM = "openai/medium";

const CANDIDATES = [
  { model: CHEAP, reasoningEffort: "low" },
  { model: STRONG, reasoningEffort: "high" },
  { model: MEDIUM, reasoningEffort: "medium" },
];

// Deterministic multiplicative-congruential rng; the protocol only requires a
// stable [0, 1) stream.
function rng(seed: number): Rng {
  let state = seed % 2147483647;
  if (state <= 0) state += 2147483646;
  return () => {
    state = (state * 16807) % 2147483647;
    return (state - 1) / 2147483646;
  };
}

function selectorInput(): JsonObject {
  return {
    candidate_pairs: CANDIDATES.map((candidate) => ({
      model: candidate.model,
      thinking_level: candidate.reasoningEffort,
    })),
    imported_evals: [
      {
        id: "evl_swe",
        name: "SWE-bench",
        description: "Coding reliability",
        min_score: 0,
        max_score: 100,
        scores: [
          {
            model_id: CHEAP,
            thinking_level: "low",
            score: 40,
            rank: 22,
            rank_total: 28,
            z_score: -0.94,
            notes: "cheap model",
          },
          {
            model_id: STRONG,
            thinking_level: "high",
            score: 90,
            rank: 1,
            rank_total: 28,
            z_score: 1.6,
            notes: "strong model",
          },
          {
            model_id: MEDIUM,
            thinking_level: "medium",
            score: 70,
            rank: 5,
            rank_total: 28,
            z_score: 1.13,
            notes: "medium model",
          },
        ],
      },
    ],
    previous_decision: {
      model: MEDIUM,
      thinking_level: "medium",
      reason: `previously selected ${MEDIUM}`,
    },
    cost_estimates: [
      {
        model: CHEAP,
        reasoning_effort: "low",
        pricing_known: true,
        fixed_turn_cost_estimate: {
          output_tokens_per_turn: 700,
          assumed_reasoning_effort: "low",
          projections: [
            { projected_turns: 1, total_cost_usd: 0.006 },
            { projected_turns: 10, total_cost_usd: 0.0432 },
            { projected_turns: 100, total_cost_usd: 0.55 },
          ],
        },
      },
      {
        model: STRONG,
        reasoning_effort: "high",
        pricing_known: true,
        fixed_turn_cost_estimate: {
          output_tokens_per_turn: 700,
          assumed_reasoning_effort: "high",
          projections: [
            { projected_turns: 1, total_cost_usd: 0.02 },
            { projected_turns: 10, total_cost_usd: 0.191 },
            { projected_turns: 100, total_cost_usd: 2.4 },
          ],
        },
      },
      // No configured average output tokens means no loop projection at all.
      {
        model: MEDIUM,
        reasoning_effort: "medium",
        pricing_known: true,
        est_input_cost_usd: 0.09,
        output_cost_per_mtok: 75,
        fixed_turn_cost_estimate: null,
      },
    ],
    messages: [{ role: "user", content: "Fix the failing test." }],
  };
}

test("deterministically anonymizes every candidate-bearing selector field", () => {
  const slots = assignAnonymousActions(CANDIDATES, rng(42));
  const replay = assignAnonymousActions(CANDIDATES, rng(42));
  expect(replay).toEqual(slots);
  expect(slots.map((slot) => slot.action)).toEqual(["A", "B", "C"]);

  const anonymous = anonymizeSelectorInput(selectorInput(), slots);
  const prompt = buildAnonymousPolicyPrompt(anonymous, slots);
  const [systemMessage, userMessage] = prompt.messages;
  expect(systemMessage).toEqual({ role: "system", content: ANONYMOUS_ACTION_SYSTEM_PROMPT });
  expect(prompt.actions).toEqual(["A", "B", "C"]);

  const serialized = userMessage.content;
  expect(serialized).not.toContain(CHEAP);
  expect(serialized).not.toContain(STRONG);
  expect(serialized).not.toContain(MEDIUM);
  expect(serialized).not.toContain("thinking_level");
  expect(serialized).not.toContain("reasoning_effort");
  expect(serialized).not.toContain("cheap model");
  expect(serialized).toContain("## Action A");
  expect(serialized).toContain("## Action B");
  expect(serialized).toContain("## Action C");
  expect(serialized).toContain("<conversation>");
  expect(serialized).toContain("Fix the failing test.");
});

test("renders each action as a block of benchmark standing and one projected cost", () => {
  const slots = assignAnonymousActions(CANDIDATES, rng(42));
  const anonymous = anonymizeSelectorInput(selectorInput(), slots);
  const prompt = buildAnonymousPolicyPrompt(anonymous, slots);
  const serialized = prompt.messages[1].content;

  expect(serialized).toContain("<benchmarks>\n- SWE-bench: Coding reliability\n</benchmarks>");
  expect(block(serialized, actionFor(slots, CHEAP))).toEqual([
    "- Cost: 1 turn $0.006, 10 turns $0.0432, 100 turns $0.55",
    "- SWE-bench: Rank 22/28, Z -0.94",
  ]);
  expect(block(serialized, actionFor(slots, STRONG))).toEqual([
    "- Cost: 1 turn $0.02, 10 turns $0.191, 100 turns $2.40",
    "- SWE-bench: Rank 1/28, Z +1.60",
  ]);
  // An action with no loop projection simply has no cost line; per-request
  // rates would be a different unit from every other block.
  expect(block(serialized, actionFor(slots, MEDIUM))).toEqual([
    "- SWE-bench: Rank 5/28, Z +1.13",
  ]);
  expect(serialized).toContain(`<previous_action>\n${actionFor(slots, MEDIUM)}\n</previous_action>`);
});

test("omits benchmark sections and cost details it has no data for", () => {
  const slots = assignAnonymousActions(CANDIDATES, rng(42));
  const bare = {
    ...selectorInput(),
    imported_evals: [],
    previous_decision: null,
    cost_estimates: null,
  };
  const prompt = buildAnonymousPolicyPrompt(anonymizeSelectorInput(bare, slots), slots);
  const serialized = prompt.messages[1].content;

  expect(serialized).not.toContain("<benchmarks>");
  expect(serialized).not.toContain("<previous_action>");
  expect(block(serialized, "A")).toEqual([]);
});

function actionFor(slots: ReturnType<typeof assignAnonymousActions>, model: string): string {
  const slot = slots.find((entry) => entry.candidate.model === model);
  if (slot === undefined) throw new Error(`no slot for ${model}`);
  return slot.action;
}

// The lines of one action's block, so assertions do not depend on the order
// the rng shuffled the candidates into.
function block(prompt: string, action: string): string[] {
  const lines = prompt.split("\n");
  const start = lines.indexOf(`## Action ${action}`);
  expect(start).toBeGreaterThanOrEqual(0);
  const body = lines.slice(start + 1);
  const end = body.findIndex((line) => line === "" || line === "</candidates>");
  return body.slice(0, end === -1 ? undefined : end);
}

test("different rng streams change the mapping but keep the candidates", () => {
  const first = assignAnonymousActions(CANDIDATES, rng(42));
  const second = assignAnonymousActions(CANDIDATES, rng(43));
  expect(
    new Set(second.map((slot) => JSON.stringify(slot.candidate))),
  ).toEqual(new Set(first.map((slot) => JSON.stringify(slot.candidate))));
});

test("resolves the strict JSON action back to the real pair", () => {
  const slots = assignAnonymousActions(CANDIDATES, rng(42));
  const selection = parseAnonymousActionSelection('{"action":"B"}', slots);

  expect(selection.action).toBe("B");
  expect(selection.candidate).toEqual(slots[1]!.candidate);
  expect(() =>
    parseAnonymousActionSelection('{"action":"B","reason":"extra"}', slots),
  ).toThrow("must contain exactly");
  expect(() => parseAnonymousActionSelection('{"action":"Z"}', slots)).toThrow(
    "unavailable action Z",
  );
});

test("parses the lease turns beside the action when a menu is offered", () => {
  const slots = assignAnonymousActions(CANDIDATES, rng(42));
  const menu = [5, 10, 30];

  const selection = parseAnonymousActionSelection('{"action":"B","turns":10}', slots, menu);
  expect(selection.action).toBe("B");
  expect(selection.turns).toBe(10);

  // With a menu the turns field is mandatory and must be on the menu.
  expect(() => parseAnonymousActionSelection('{"action":"B"}', slots, menu)).toThrow(
    "must contain exactly",
  );
  expect(() =>
    parseAnonymousActionSelection('{"action":"B","turns":7}', slots, menu),
  ).toThrow("must be one of 5, 10, 30");
  // Without a menu the turns field is rejected, preserving the strict shape.
  expect(() =>
    parseAnonymousActionSelection('{"action":"B","turns":10}', slots),
  ).toThrow("must contain exactly");
});

test("supports more than 26 anonymous actions", () => {
  const candidates = Array.from({ length: 53 }, (_, index) => ({
    model: `fireworks/model-${index}`,
    reasoningEffort: "low",
  }));
  const slots = assignAnonymousActions(candidates, () => 0.999999);

  expect(slots.map((slot) => slot.action).slice(24)).toEqual([
    "Y", "Z", "AA", "AB", "AC", "AD", "AE", "AF", "AG", "AH",
    "AI", "AJ", "AK", "AL", "AM", "AN", "AO", "AP", "AQ", "AR",
    "AS", "AT", "AU", "AV", "AW", "AX", "AY", "AZ", "BA",
  ]);
});

test("rejects candidate sets that cannot form anonymous actions", () => {
  expect(() => assignAnonymousActions(CANDIDATES.slice(0, 1), rng(42))).toThrow(
    "at least two candidates",
  );
  const duplicate = [CANDIDATES[0]!, CANDIDATES[0]!];
  expect(() => assignAnonymousActions(duplicate, rng(42))).toThrow(
    "Duplicate anonymous action candidate",
  );
});

test("rejects malformed or identity-leaking selector shapes", () => {
  const slots = assignAnonymousActions(CANDIDATES, rng(42));

  const badPairs = { ...selectorInput(), candidate_pairs: null };
  expect(() => anonymizeSelectorInput(badPairs, slots)).toThrow(
    "candidate_pairs must be an array",
  );

  const mismatch = { ...selectorInput(), candidate_pairs: [] };
  expect(() => anonymizeSelectorInput(mismatch, slots)).toThrow(
    "Routing and selector candidates do not match",
  );

  const badEvals = { ...selectorInput(), imported_evals: null };
  expect(() => anonymizeSelectorInput(badEvals, slots)).toThrow(
    "imported_evals must be an array",
  );

  const badScores = { ...selectorInput(), imported_evals: [{ id: "eval", scores: null }] };
  expect(() => anonymizeSelectorInput(badScores, slots)).toThrow(
    "scores must be an array",
  );

  const badCosts = { ...selectorInput(), cost_estimates: {} };
  expect(() => anonymizeSelectorInput(badCosts, slots)).toThrow(
    "cost_estimates must be an array or null",
  );

  // Pre-projection estimates (a bare scalar shape) must fail loudly instead of
  // silently rendering the candidate's cost as unavailable.
  const legacyCost = selectorInput();
  (legacyCost["cost_estimates"] as JsonObject[])[0]!["fixed_turn_cost_estimate"] = {
    projected_turns: 10,
    total_cost_usd: 0.05,
  };
  expect(() => anonymizeSelectorInput(legacyCost, slots)).toThrow(
    "fixed_turn_cost_estimate must carry a projections array",
  );

  const staleEval = selectorInput();
  (staleEval["imported_evals"] as JsonObject[])[0]!["scores"] = [
    { model_id: "openai/not-a-candidate", thinking_level: "low", score: 1 },
  ];
  expect(() => anonymizeSelectorInput(staleEval, slots)).toThrow(
    "references unavailable candidate openai/not-a-candidate/low",
  );
});

test("an empty lease menu parses per-turn completions", () => {
  const slots = assignAnonymousActions(CANDIDATES, rng(42));

  expect(parseAnonymousActionSelection('{"action":"A"}', slots, [])).toEqual({
    action: "A",
    candidate: slots[0]!.candidate,
  });
  expect(() => parseAnonymousActionSelection('{"action":"A","turns":5}', slots, [])).toThrow(
    'must contain exactly {"action":...}',
  );
});

test("rejects malformed action completions", () => {
  const slots = assignAnonymousActions(CANDIDATES, rng(42));
  expect(() => parseAnonymousActionSelection("not json", slots)).toThrow("not valid JSON");
  expect(() => parseAnonymousActionSelection('{"action":1}', slots)).toThrow(
    "action must be a string",
  );
  expect(() => parseAnonymousActionSelection("[]", slots)).toThrow("must be an object");
});
