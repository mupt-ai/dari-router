import { expect, test } from "bun:test";

import { buildSelectorInput } from "../src/selector_input.js";
import type { RouterEval, RoutingCandidate } from "../src/types.js";

const CHEAP: RoutingCandidate = { model: "fireworks/cheap", reasoningEffort: "low" };
const STRONG: RoutingCandidate = { model: "fireworks/strong", reasoningEffort: "high" };

function evalCard(scores: RouterEval["scores"]): RouterEval {
  return {
    id: "evl_swe",
    name: "SWE-bench",
    description: "Coding reliability",
    min_score: 0,
    max_score: 100,
    scores,
  };
}

function scores(input: SelectorInputResult): Array<Record<string, unknown>> {
  const card = input.imported_evals[0];
  expect(card).toBeDefined();
  return card!["scores"] as Array<Record<string, unknown>>;
}

type SelectorInputResult = ReturnType<typeof buildSelectorInput>;

function selectorInput(card: RouterEval, candidates: RoutingCandidate[]): SelectorInputResult {
  return buildSelectorInput({
    candidates,
    evals: [card],
    previousDecision: null,
    costEstimates: null,
    messages: [{ role: "user", content: "Fix the failing test." }],
  });
}

test("ranks and normalizes candidate scores against the whole scorecard", () => {
  // Candidate-relative normalization would report +/-1 for these two whatever
  // the gap; the other 8 scored models are what make the numbers mean anything.
  const others = Array.from({ length: 8 }, (_, index) => ({
    model_id: `other/model-${index}`,
    score: 30 + index * 5,
  }));
  const input = selectorInput(
    evalCard([
      ...others,
      { model_id: CHEAP.model, thinking_level: "low", score: 40 },
      { model_id: STRONG.model, thinking_level: "high", score: 90 },
    ]),
    [CHEAP, STRONG],
  );

  expect(scores(input)).toEqual([
    {
      model_id: CHEAP.model,
      thinking_level: "low",
      score: 40,
      rank: 7,
      rank_total: 10,
      z_score: -0.66,
      notes: null,
    },
    {
      model_id: STRONG.model,
      thinking_level: "high",
      score: 90,
      rank: 1,
      rank_total: 10,
      z_score: 2.33,
      notes: null,
    },
  ]);
});

test("gives tied scores the same rank and a flat scorecard a zero z-score", () => {
  const input = selectorInput(
    evalCard([
      { model_id: "other/best", score: 70 },
      { model_id: CHEAP.model, thinking_level: "low", score: 50 },
      { model_id: STRONG.model, thinking_level: "high", score: 50 },
    ]),
    [CHEAP, STRONG],
  );
  expect(scores(input).map((score) => score["rank"])).toEqual([2, 2]);

  const flat = selectorInput(
    evalCard([
      { model_id: CHEAP.model, thinking_level: "low", score: 50 },
      { model_id: STRONG.model, thinking_level: "high", score: 50 },
    ]),
    [CHEAP, STRONG],
  );
  expect(scores(flat).map((score) => score["z_score"])).toEqual([0, 0]);
});

test("normalizes a thinking-level-agnostic score row against the same population", () => {
  const input = selectorInput(
    evalCard([
      { model_id: "other/weak", score: 10 },
      { model_id: "other/strong", score: 90 },
      { model_id: CHEAP.model, score: 50 },
    ]),
    [CHEAP],
  );

  expect(scores(input)[0]).toMatchObject({
    thinking_level: "low",
    score: 50,
    rank: 2,
    rank_total: 3,
    z_score: 0,
  });
});

const MID: RoutingCandidate = { model: "fireworks/mid", reasoningEffort: "medium" };

function imputedInput(
  card: RouterEval,
  candidates: RoutingCandidate[],
): SelectorInputResult {
  return buildSelectorInput({
    candidates,
    evals: [card],
    previousDecision: null,
    costEstimates: null,
    messages: [{ role: "user", content: "Fix the failing test." }],
    imputeEvalScores: true,
  });
}

test("leaves the scorecard empty when imputation is off and only other levels are scored", () => {
  const input = selectorInput(
    evalCard([
      { model_id: MID.model, thinking_level: "low", score: 40 },
      { model_id: MID.model, thinking_level: "high", score: 80 },
    ]),
    [MID],
  );
  // No exact medium and no generic Any -> no row for the candidate.
  expect(input.imported_evals).toHaveLength(0);
});

test("learns ratios from hidden reference evals without serializing them", () => {
  const selected = {
    ...evalCard([
      { model_id: MID.model, thinking_level: "high", score: 80 },
    ]),
    id: "evl_selected",
    name: "Selected Eval",
  };
  const reference = {
    ...evalCard([
      { model_id: "other/calibrator", thinking_level: "high", score: 80 },
      { model_id: "other/calibrator", thinking_level: "medium", score: 60 },
    ]),
    id: "evl_reference",
    name: "Unselected Reference Eval",
  };

  const input = buildSelectorInput({
    candidates: [MID],
    evals: [selected],
    imputationReferenceEvals: [reference],
    previousDecision: null,
    costEstimates: null,
    messages: [{ role: "user", content: "Fix the failing test." }],
    imputeEvalScores: true,
  });

  expect(input.imported_evals).toHaveLength(1);
  expect(input.imported_evals[0]?.["id"]).toBe("evl_selected");
  expect(scores(input)[0]).toMatchObject({
    model_id: MID.model,
    thinking_level: "medium",
    score: 60,
    imputed: true,
  });
});

test("imputes from average pairwise ratios on a negative score scale", () => {
  const card = {
    ...evalCard([
      { model_id: MID.model, thinking_level: "low", score: -20 },
      { model_id: "other/a", thinking_level: "low", score: -60 },
      { model_id: "other/a", thinking_level: "medium", score: -20 },
      { model_id: "other/b", thinking_level: "low", score: 0 },
      { model_id: "other/b", thinking_level: "medium", score: 50 },
    ]),
    min_score: -100,
    max_score: 100,
  };
  const input = imputedInput(card, [MID]);

  // Normalized low->medium ratios are 2 and 1.5. Applying their 1.75
  // average to MID's normalized low score (0.4) gives 0.7, or raw score 40.
  expect(scores(input)[0]).toMatchObject({
    model_id: MID.model,
    thinking_level: "medium",
    score: 40,
    notes: null,
    imputed: true,
  });
});

test("averages estimates from every measured anchor level", () => {
  const card = {
    ...evalCard([
      { model_id: MID.model, thinking_level: "low", score: -20 },
      { model_id: MID.model, thinking_level: "high", score: 60 },
      { model_id: "other/a", thinking_level: "low", score: -60 },
      { model_id: "other/a", thinking_level: "medium", score: -20 },
      { model_id: "other/a", thinking_level: "high", score: 60 },
      { model_id: "other/b", thinking_level: "low", score: 0 },
      { model_id: "other/b", thinking_level: "medium", score: 50 },
      { model_id: "other/b", thinking_level: "high", score: 100 },
    ]),
    min_score: -100,
    max_score: 100,
  };

  // The low anchor estimates 40 and the high anchor estimates 0.
  expect(scores(imputedInput(card, [MID]))[0]).toMatchObject({ score: 20 });
});

test("uses pairwise ratios learned from other evals", () => {
  const target = evalCard([
    { model_id: MID.model, thinking_level: "low", score: 40 },
  ]);
  const evidence: RouterEval = {
    ...evalCard([
      { model_id: "other/a", thinking_level: "low", score: 20 },
      { model_id: "other/a", thinking_level: "medium", score: 40 },
    ]),
    id: "evl_other",
  };
  const input = buildSelectorInput({
    candidates: [MID],
    evals: [target, evidence],
    previousDecision: null,
    costEstimates: null,
    messages: [],
    imputeEvalScores: true,
  });

  expect(scores(input)[0]).toMatchObject({ score: 80, imputed: true });
});

test("drops invalid anchor estimates before averaging valid anchors", () => {
  const card = {
    ...evalCard([
      { model_id: MID.model, thinking_level: "off", score: 29.28 },
      { model_id: MID.model, thinking_level: "max", score: 51.77 },
      { model_id: "other/a", thinking_level: "off", score: 5 },
      { model_id: "other/a", thinking_level: "high", score: 50 },
      { model_id: "other/a", thinking_level: "max", score: 50 },
    ]),
    min_score: 0,
    max_score: 100,
  };

  const high = { ...MID, reasoningEffort: "high" } satisfies RoutingCandidate;
  const row = scores(imputedInput(card, [high]))[0];
  // The off anchor predicts 292.8 and is discarded; max predicts 51.77.
  expect(row).toMatchObject({ score: 51.77, imputed: true });
});

test("leaves an out-of-range ratio estimate missing instead of clamping it", () => {
  const target = evalCard([
    { model_id: MID.model, thinking_level: "low", score: 40 },
  ]);
  const outlier: RouterEval = {
    ...evalCard([
      { model_id: "other/a", thinking_level: "low", score: 20 },
      { model_id: "other/a", thinking_level: "medium", score: 60 },
    ]),
    id: "evl_outlier",
  };
  const input = buildSelectorInput({
    candidates: [MID],
    evals: [target, outlier],
    previousDecision: null,
    costEstimates: null,
    messages: [],
    imputeEvalScores: true,
  });

  // 40 * (60 / 20) = 120, outside this scorecard's [0, 100] range.
  expect(input.imported_evals).toHaveLength(0);
});

test("does not impute from an anchor at the score range floor", () => {
  const target = evalCard([
    { model_id: MID.model, thinking_level: "low", score: 0 },
    { model_id: "other/a", thinking_level: "low", score: 20 },
    { model_id: "other/a", thinking_level: "medium", score: 40 },
  ]);

  expect(imputedInput(target, [MID]).imported_evals).toHaveLength(0);
});

test("does not impute when no observed pair includes the requested level", () => {
  const input = imputedInput(
    evalCard([{ model_id: MID.model, thinking_level: "high", score: 80 }]),
    [MID],
  );
  expect(input.imported_evals).toHaveLength(0);
});

test("prefers an exact score over imputation and a generic Any over imputation", () => {
  const exactCard = evalCard([
    { model_id: MID.model, thinking_level: "medium", score: 77 },
    { model_id: MID.model, thinking_level: "low", score: 40 },
    { model_id: MID.model, thinking_level: "high", score: 80 },
  ]);
  const exactRow = scores(imputedInput(exactCard, [MID]))[0];
  expect(exactRow).toMatchObject({ score: 77 });
  expect("imputed" in exactRow).toBe(false);

  const anyCard = evalCard([
    { model_id: MID.model, score: 55 },
    { model_id: MID.model, thinking_level: "low", score: 40 },
    { model_id: MID.model, thinking_level: "high", score: 80 },
  ]);
  const anyRow = scores(imputedInput(anyCard, [MID]))[0];
  expect(anyRow).toMatchObject({ score: 55 });
  expect("imputed" in anyRow).toBe(false);
});
