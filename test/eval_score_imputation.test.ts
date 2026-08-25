import { expect, test } from "bun:test";

import {
  createThinkingLevelRatios,
  resolveRouterEvalScore,
} from "../src/eval_score_imputation.js";
import type { RouterEval } from "../src/types.js";

function evalCard(
  id: string,
  scores: RouterEval["scores"],
  minScore = 0,
  maxScore = 100,
): RouterEval {
  return {
    id,
    name: id,
    min_score: minScore,
    max_score: maxScore,
    scores,
  };
}

test("resolves exact and generic rows before imputation", () => {
  const scores = [
    { model_id: "openai/exact", thinking_level: "high" as const, score: 84 },
    { model_id: "anthropic/generic", thinking_level: null, score: 91 },
  ];
  const common = { scores, minScore: 0, maxScore: 100 };

  expect(
    resolveRouterEvalScore({
      ...common,
      modelId: "openai/exact",
      thinkingLevel: "high",
      impute: true,
    }),
  ).toEqual({ score: 84, imputed: false });
  expect(
    resolveRouterEvalScore({
      ...common,
      modelId: "anthropic/generic",
      thinkingLevel: "low",
      impute: true,
    }),
  ).toEqual({ score: 91, imputed: false });
});

test("uses ratios learned from every supplied calibration eval", () => {
  const target = evalCard("target", [
    { model_id: "openai/target", thinking_level: "low", score: 40 },
  ]);
  const reference = evalCard("reference", [
    { model_id: "other/calibrator", thinking_level: "low", score: 20 },
    { model_id: "other/calibrator", thinking_level: "medium", score: 40 },
  ]);

  expect(
    resolveRouterEvalScore({
      scores: target.scores,
      modelId: "openai/target",
      thinkingLevel: "medium",
      minScore: target.min_score,
      maxScore: target.max_score,
      impute: true,
      ratios: createThinkingLevelRatios([target, reference]),
    }),
  ).toEqual({ score: 80, imputed: true });
});

test("chains pairwise ratios through intermediate thinking levels", () => {
  const designArena = evalCard(
    "designarena",
    [{
      model_id: "openai/gpt-5.6-sol",
      thinking_level: "medium",
      score: 1351,
    }],
    0,
    3000,
  );
  const mediumToMax = evalCard("medium-to-max", [
    { model_id: "other/a", thinking_level: "medium", score: 50 },
    { model_id: "other/a", thinking_level: "max", score: 60 },
  ]);
  const maxToXhigh = evalCard("max-to-xhigh", [
    { model_id: "other/b", thinking_level: "max", score: 40 },
    { model_id: "other/b", thinking_level: "xhigh", score: 44 },
  ]);

  expect(
    resolveRouterEvalScore({
      scores: designArena.scores,
      modelId: "openai/gpt-5.6-sol",
      thinkingLevel: "xhigh",
      minScore: designArena.min_score,
      maxScore: designArena.max_score,
      impute: true,
      ratios: createThinkingLevelRatios([
        designArena,
        mediumToMax,
        maxToXhigh,
      ]),
    }),
  ).toEqual({ score: 1783.32, imputed: true });
});

test("prefers a direct ratio over transitive calibration paths", () => {
  const target = evalCard("target", [
    { model_id: "openai/target", thinking_level: "medium", score: 50 },
  ]);
  const mediumToMax = evalCard("medium-to-max", [
    { model_id: "other/a", thinking_level: "medium", score: 25 },
    { model_id: "other/a", thinking_level: "max", score: 50 },
  ]);
  const maxToXhigh = evalCard("max-to-xhigh", [
    { model_id: "other/b", thinking_level: "max", score: 25 },
    { model_id: "other/b", thinking_level: "xhigh", score: 50 },
  ]);
  const direct = evalCard("direct", [
    { model_id: "other/c", thinking_level: "medium", score: 50 },
    { model_id: "other/c", thinking_level: "xhigh", score: 60 },
  ]);

  expect(
    resolveRouterEvalScore({
      scores: target.scores,
      modelId: "openai/target",
      thinkingLevel: "xhigh",
      minScore: target.min_score,
      maxScore: target.max_score,
      impute: true,
      ratios: createThinkingLevelRatios([
        target,
        mediumToMax,
        maxToXhigh,
        direct,
      ]),
    }),
  ).toEqual({ score: 60, imputed: true });
});

test("normalizes negative and differently-sized calibration scales", () => {
  const target = evalCard(
    "target",
    [{ model_id: "openai/target", thinking_level: "low", score: -20 }],
    -100,
    100,
  );
  const reference = evalCard(
    "reference",
    [
      { model_id: "other/calibrator", thinking_level: "low", score: 20 },
      { model_id: "other/calibrator", thinking_level: "medium", score: 40 },
    ],
    0,
    100,
  );

  expect(
    resolveRouterEvalScore({
      scores: target.scores,
      modelId: "openai/target",
      thinkingLevel: "medium",
      minScore: target.min_score,
      maxScore: target.max_score,
      impute: true,
      ratios: createThinkingLevelRatios([target, reference]),
    }),
  ).toEqual({ score: 60, imputed: true });
});

test("rejects estimates beyond a bounded scorecard", () => {
  const target = evalCard("target", [
    { model_id: "openai/target", thinking_level: "low", score: 40 },
  ]);
  const reference = evalCard("reference", [
    { model_id: "other/calibrator", thinking_level: "low", score: 20 },
    { model_id: "other/calibrator", thinking_level: "medium", score: 60 },
  ]);

  expect(
    resolveRouterEvalScore({
      scores: target.scores,
      modelId: "openai/target",
      thinkingLevel: "medium",
      minScore: target.min_score,
      maxScore: target.max_score,
      impute: true,
      ratios: createThinkingLevelRatios([target, reference]),
    }),
  ).toBeNull();
});
