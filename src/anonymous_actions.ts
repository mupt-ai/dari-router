// Anonymous candidate-action selector protocol. Candidates are shuffled into
// anonymous action slots ("A", "B", ..., "AA", ...) and every model-identifying
// field in the selector input is replaced by its slot, so a trained policy
// cannot memorize model names or list positions. This is the canonical
// open-weight selector protocol used by Dari's SLM routing strategy; the
// named-candidate protocol in selector_input.ts/prompts.ts remains available
// for custom routers that use a general-purpose LLM selector.

import { ANONYMOUS_ACTION_SYSTEM_PROMPT } from "./prompts.js";
import {
  jsonRecord,
  jsonValue,
  requiredString,
  type JsonObject,
  type JsonValue,
} from "./json.js";
import type { TurnCostProjection } from "./types.js";

// Deterministic random source in [0, 1). Callers own seeding so experiment
// harnesses can reproduce recorded permutations with their own PRNG.
export type Rng = () => number;

export type AnonymousActionCandidate = {
  model: string;
  reasoningEffort: string;
};

export type AnonymousActionSlot = {
  action: string;
  candidate: AnonymousActionCandidate;
};

export type AnonymousSelectorMessage = {
  role: "system" | "user";
  content: string;
};

export type AnonymousPolicyPrompt = {
  messages: [system: AnonymousSelectorMessage, user: AnonymousSelectorMessage];
  actions: string[];
};

export type AnonymousActionSelection = {
  action: string;
  candidate: AnonymousActionCandidate;
  // Turns the selection commits to when the caller offered a lease menu;
  // absent for single-turn selectors.
  turns?: number;
};

// The de-identified selector input. Every field is JSON-serializable so it can
// cross a process boundary, but it is a concrete type rather than a bag: it is
// produced by anonymizeSelectorInput and consumed by the prompt formatter, so
// nothing downstream needs to re-parse it.
export type AnonymousEvalScore = {
  action: string;
  rank: number;
  rank_total: number;
  z_score: number;
};

export type AnonymousEvalCard = {
  name: string;
  description: string | null;
  scores: AnonymousEvalScore[];
};

export type AnonymousActionCost = {
  action: string;
  projections: TurnCostProjection[];
};

export type AnonymousSelectorInput = {
  candidate_actions: Array<{ action: string }>;
  imported_evals: AnonymousEvalCard[];
  previous_action: { action: string } | null;
  cost_estimates: AnonymousActionCost[];
  messages: JsonValue;
};

// Shuffles candidates into spreadsheet-style action slots A..Z, AA..AZ, ...
// with the injected rng.
export function assignAnonymousActions(
  candidates: readonly AnonymousActionCandidate[],
  rng: Rng,
): AnonymousActionSlot[] {
  if (candidates.length < 2) {
    throw new Error(`Anonymous action routing requires at least two candidates, got ${candidates.length}`);
  }
  const keys = new Set<string>();
  for (const candidate of candidates) {
    const key = candidateKey(candidate.model, candidate.reasoningEffort);
    if (keys.has(key)) throw new Error(`Duplicate anonymous action candidate ${key}`);
    keys.add(key);
  }
  return shuffled(candidates, rng).map((candidate, index) => ({
    action: actionForIndex(index),
    candidate,
  }));
}

function actionForIndex(index: number): string {
  let remaining = index;
  let action = "";
  do {
    action = String.fromCharCode("A".charCodeAt(0) + (remaining % 26)) + action;
    remaining = Math.floor(remaining / 26) - 1;
  } while (remaining >= 0);
  return action;
}

// Rewrites a named-candidate selector input (see selector_input.ts) into its
// anonymous-action form. The input must reference exactly the candidates in
// `slots`; any mismatch is an error rather than a silent partial rewrite.
export function anonymizeSelectorInput(
  selectorInput: JsonObject,
  slots: readonly AnonymousActionSlot[],
): AnonymousSelectorInput {
  const input = jsonRecord(structuredClone(selectorInput), "selector input");
  const byCandidate = new Map(
    slots.map((slot) => [candidateKey(slot.candidate.model, slot.candidate.reasoningEffort), slot.action]),
  );
  assertSelectorCandidates(input["candidate_pairs"], byCandidate);

  return {
    candidate_actions: slots.map((slot) => ({ action: slot.action })),
    imported_evals: anonymizeEvals(input["imported_evals"], byCandidate),
    previous_action: anonymizePreviousDecision(input["previous_decision"], byCandidate),
    cost_estimates: anonymizeCosts(input["cost_estimates"], byCandidate),
    messages: jsonValue(input["messages"], "selector input messages"),
  };
}

export function buildAnonymousPolicyPrompt(
  anonymousSelectorInput: AnonymousSelectorInput,
  slots: readonly AnonymousActionSlot[],
): AnonymousPolicyPrompt {
  return {
    messages: [
      { role: "system", content: ANONYMOUS_ACTION_SYSTEM_PROMPT },
      { role: "user", content: formatAnonymousPolicyInput(anonymousSelectorInput) },
    ],
    actions: slots.map((slot) => slot.action),
  };
}

// Renders the anonymous selector input as sectioned text rather than raw JSON.
// One block per action carries that action's benchmark standing and its
// projected cost, so the policy compares like against like instead of
// re-deriving comparisons from scores on incomparable scales and six cost
// fields. How to read those blocks lives in ANONYMOUS_ACTION_SYSTEM_PROMPT,
// keeping this message pure data. The conversation goes last: it is the only
// section that grows turn over turn, so keeping it at the end preserves the
// shared prefix.
export function formatAnonymousPolicyInput(input: AnonymousSelectorInput): string {
  assertValidBenchmarkStandings(input.imported_evals);
  const sections = [
    ...(input.imported_evals.length > 0
      ? [section("benchmarks", input.imported_evals.map(glossaryLine).join("\n"))]
      : []),
    section(
      "candidates",
      input.candidate_actions
        .map((candidate) => candidateBlock(candidate.action, input))
        .join("\n\n"),
    ),
    ...(input.previous_action === null
      ? []
      : [section("previous_action", input.previous_action.action)]),
    section("conversation", JSON.stringify(input.messages)),
  ];
  return sections.join("\n\n");
}

function section(name: string, body: string): string {
  return `<${name}>\n${body}\n</${name}>`;
}

function candidateBlock(action: string, input: AnonymousSelectorInput): string {
  const cost = input.cost_estimates.find((entry) => entry.action === action);
  const lines = [`## Action ${action}`];
  if (cost !== undefined) lines.push(costLine(cost));
  for (const card of input.imported_evals) {
    const score = card.scores.find((entry) => entry.action === action);
    if (score !== undefined) {
      lines.push(
        `- ${card.name}: Rank ${score.rank}/${score.rank_total}, Z ${signed(score.z_score)}`,
      );
    }
  }
  return lines.join("\n");
}

// One line, one unit, the same horizons for every action.
function costLine(cost: AnonymousActionCost): string {
  const horizons = cost.projections.map(
    (projection) =>
      `${projection.projected_turns} turn${projection.projected_turns === 1 ? "" : "s"} ${usd(projection.total_cost_usd)}`,
  );
  return `- Cost: ${horizons.join(", ")}`;
}

function signed(value: number): string {
  return value >= 0 ? `+${value.toFixed(2)}` : value.toFixed(2);
}

function usd(value: number): string {
  if (value >= 1) return `$${value.toFixed(2)}`;
  const trimmed = value.toPrecision(3).replace(/0+$/, "").replace(/\.$/, "");
  return `$${trimmed}`;
}

function glossaryLine(card: AnonymousEvalCard): string {
  return card.description === null ? `- ${card.name}` : `- ${card.name}: ${card.description}`;
}

function assertValidBenchmarkStandings(cards: readonly AnonymousEvalCard[]): void {
  for (const [cardIndex, card] of cards.entries()) {
    const resolvedCount = card.scores.length;
    for (const [scoreIndex, score] of card.scores.entries()) {
      assertValidBenchmarkStanding(
        score.rank,
        score.rank_total,
        resolvedCount,
        `anonymous eval[${cardIndex}].scores[${scoreIndex}]`,
      );
    }
  }
}

function assertValidBenchmarkStanding(
  rank: number,
  rankTotal: number,
  resolvedCount: number,
  label: string,
): void {
  if (!Number.isInteger(rank) || rank < 1 || rank > rankTotal) {
    throw new Error(`${label}.rank must be an integer between 1 and rank_total`);
  }
  if (!Number.isInteger(rankTotal) || rankTotal !== resolvedCount) {
    throw new Error(
      `${label}.rank_total must equal the resolved score count ${resolvedCount}`,
    );
  }
}

function previousAction(value: unknown): string | null {
  if (value === null || value === undefined) return null;
  return requiredString(
    jsonRecord(value, "anonymous previous_action")["action"],
    "anonymous previous_action.action",
  );
}

function isRecordValue(value: unknown): value is Record<string, JsonValue> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

// Parses a policy completion against the permutation's slots. Without a lease
// menu the exact shape is {"action":"B"}; with one it is
// {"action":"B","turns":10} and turns must be on the menu.
export function parseAnonymousActionSelection(
  completion: string,
  slots: readonly AnonymousActionSlot[],
  leaseTurns?: readonly number[],
): AnonymousActionSelection {
  let parsed: unknown;
  try {
    parsed = JSON.parse(completion);
  } catch {
    throw new Error("Anonymous action completion is not valid JSON");
  }
  const record = jsonRecord(parsed, "anonymous action completion");
  const keys = Object.keys(record);
  // An empty menu means the caller routes one turn at a time, same as no menu.
  const offeredLeaseTurns =
    leaseTurns !== undefined && leaseTurns.length > 0 ? leaseTurns : undefined;
  const expected = offeredLeaseTurns === undefined ? ["action"] : ["action", "turns"];
  if (keys.length !== expected.length || expected.some((key, i) => keys[i] !== key)) {
    throw new Error(
      `Anonymous action completion must contain exactly {${expected
        .map((key) => `"${key}":...`)
        .join(",")}}`,
    );
  }
  const action = record["action"];
  if (typeof action !== "string") {
    throw new Error("Anonymous action completion action must be a string");
  }
  const slot = slots.find((candidate) => candidate.action === action);
  if (slot === undefined) {
    throw new Error(`Anonymous action completion selected unavailable action ${action}`);
  }
  if (offeredLeaseTurns === undefined) return { action, candidate: slot.candidate };
  const turns = record["turns"];
  if (typeof turns !== "number" || !offeredLeaseTurns.includes(turns)) {
    throw new Error(
      `Anonymous action completion turns must be one of ${offeredLeaseTurns.join(", ")}`,
    );
  }
  return { action, candidate: slot.candidate, turns };
}

// Fisher-Yates copy, drawing from the end, matching d3-array's shuffler so an
// injected rng reproduces permutations recorded by harnesses that used d3.
function shuffled<T>(items: readonly T[], rng: Rng): T[] {
  const array = [...items];
  let m = array.length;
  while (m) {
    const i = (rng() * m--) | 0;
    const t = array[m]!;
    array[m] = array[i]!;
    array[i] = t;
  }
  return array;
}

function assertSelectorCandidates(
  value: unknown,
  byCandidate: ReadonlyMap<string, string>,
): void {
  if (!Array.isArray(value)) {
    throw new Error("Selector input candidate_pairs must be an array");
  }
  const selectorKeys = new Set(
    value.map((candidate, index) => {
      const record = jsonRecord(candidate, `selector candidate[${index}]`);
      return candidateKey(
        requiredString(record["model"], `selector candidate[${index}].model`),
        requiredString(
          record["thinking_level"],
          `selector candidate[${index}].thinking_level`,
        ),
      );
    }),
  );
  if (selectorKeys.size !== byCandidate.size || [...byCandidate.keys()].some((key) => !selectorKeys.has(key))) {
    throw new Error("Routing and selector candidates do not match");
  }
}

function anonymizeEvals(
  value: unknown,
  byCandidate: ReadonlyMap<string, string>,
): AnonymousEvalCard[] {
  if (!Array.isArray(value)) {
    throw new Error("Selector input imported_evals must be an array");
  }
  return value.map((entry, evalIndex) => {
    const card = jsonRecord(entry, `selector eval[${evalIndex}]`);
    const scores = card["scores"];
    if (!Array.isArray(scores)) {
      throw new Error(`selector eval[${evalIndex}].scores must be an array`);
    }
    const resolvedCount = scores.length;
    const anonymousScores = scores.map((entryScore, scoreIndex) => {
      const label = `selector eval[${evalIndex}].scores[${scoreIndex}]`;
      const score = jsonRecord(entryScore, label);
      const rank = requiredNumber(score["rank"], `${label}.rank`);
      const rankTotal = requiredNumber(score["rank_total"], `${label}.rank_total`);
      assertValidBenchmarkStanding(rank, rankTotal, resolvedCount, label);
      return {
        action: actionForCandidate(
          byCandidate,
          requiredString(score["model_id"], `${label}.model_id`),
          requiredString(score["thinking_level"], `${label}.thinking_level`),
        ),
        rank,
        rank_total: rankTotal,
        z_score: requiredNumber(score["z_score"], `${label}.z_score`),
      };
    });
    return {
      name: requiredString(card["name"], `selector eval[${evalIndex}].name`),
      description: typeof card["description"] === "string" ? card["description"] : null,
      // Explicit allowlist: notes and any future field can name the model.
      scores: anonymousScores,
    };
  });
}

function anonymizePreviousDecision(
  value: unknown,
  byCandidate: ReadonlyMap<string, string>,
): { action: string } | null {
  if (value === null || value === undefined) return null;
  const previous = jsonRecord(value, "selector previous_decision");
  const model = requiredString(previous["model"], "selector previous_decision.model");
  const thinkingLevel = requiredString(
    previous["thinking_level"],
    "selector previous_decision.thinking_level",
  );
  return {
    action: actionForCandidate(byCandidate, model, thinkingLevel),
  };
}

function anonymizeCosts(
  value: unknown,
  byCandidate: ReadonlyMap<string, string>,
): AnonymousActionCost[] {
  if (value === null || value === undefined) return [];
  if (!Array.isArray(value)) {
    throw new Error("Selector input cost_estimates must be an array or null");
  }
  // Only the loop projections survive. The per-request fields the prompt does
  // not show would be dead weight, and assumed_reasoning_effort names the
  // candidate's thinking level, which the protocol exists to hide.
  return value.flatMap((entry, index) => {
    const label = `selector cost_estimates[${index}]`;
    const cost = jsonRecord(entry, label);
    const action = actionForCandidate(
      byCandidate,
      requiredString(cost["model"], `${label}.model`),
      requiredString(cost["reasoning_effort"], `${label}.reasoning_effort`),
    );
    const fixedTurn = cost["fixed_turn_cost_estimate"];
    if (fixedTurn === null || fixedTurn === undefined) return [];
    if (!isRecordValue(fixedTurn) || !Array.isArray(fixedTurn["projections"])) {
      // A present estimate without projections is the pre-projection schema.
      // Reject it loudly: silently dropping it would show "Cost: unavailable"
      // for a candidate whose cost the caller thinks it supplied.
      throw new Error(
        `${label}.fixed_turn_cost_estimate must carry a projections array`,
      );
    }
    return [
      {
        action,
        projections: fixedTurn["projections"].map((projection, projectionIndex) => {
          const projectionLabel = `${label}.fixed_turn_cost_estimate.projections[${projectionIndex}]`;
          const record = jsonRecord(projection, projectionLabel);
          return {
            projected_turns: requiredNumber(
              record["projected_turns"],
              `${projectionLabel}.projected_turns`,
            ),
            total_cost_usd: requiredNumber(
              record["total_cost_usd"],
              `${projectionLabel}.total_cost_usd`,
            ),
          };
        }),
      },
    ];
  });
}

function requiredNumber(value: JsonValue | undefined, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${label} must be a finite number`);
  }
  return value;
}

function actionForCandidate(
  byCandidate: ReadonlyMap<string, string>,
  model: string,
  thinkingLevel: string,
): string {
  const action = byCandidate.get(candidateKey(model, thinkingLevel));
  if (action === undefined) {
    throw new Error(`Selector field references unavailable candidate ${model}/${thinkingLevel}`);
  }
  return action;
}

function candidateKey(model: string, thinkingLevel: string): string {
  return `${model}\u0000${thinkingLevel}`;
}
