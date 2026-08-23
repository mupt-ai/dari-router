import type { ReasoningEffort } from "./types.js";

export type EvalScoreForImputation = {
  model_id: string;
  score: number;
  thinking_level?: ReasoningEffort | null;
};

export type EvalForImputation = {
  min_score: number;
  max_score: number;
  scores: readonly EvalScoreForImputation[];
};

export type ThinkingLevelRatios = ReadonlyMap<string, number>;

export type ResolvedRouterEvalScore = {
  score: number;
  imputed: boolean;
};

export function createThinkingLevelRatios(
  evals: readonly EvalForImputation[],
): ThinkingLevelRatios {
  const samples = new Map<string, number[]>();
  for (const evalCard of evals) {
    const range = evalCard.max_score - evalCard.min_score;
    if (!Number.isFinite(range) || range <= 0) continue;
    const byModel = explicitScoresByModel(
      evalCard.scores,
      evalCard.min_score,
      range,
    );
    for (const levels of byModel.values()) {
      for (const [targetLevel, targetScore] of levels) {
        for (const [anchorLevel, anchorScore] of levels) {
          if (targetLevel === anchorLevel || anchorScore <= 0) continue;
          const ratio = targetScore / anchorScore;
          if (!Number.isFinite(ratio) || ratio < 0) continue;
          const key = thinkingLevelPairKey(targetLevel, anchorLevel);
          const values = samples.get(key) ?? [];
          values.push(ratio);
          samples.set(key, values);
        }
      }
    }
  }
  return new Map([...samples].map(([key, values]) => [key, mean(values)]));
}

export function resolveRouterEvalScore(args: {
  scores: readonly EvalScoreForImputation[];
  modelId: string;
  thinkingLevel: ReasoningEffort;
  minScore: number;
  maxScore: number;
  impute?: boolean;
  ratios?: ThinkingLevelRatios;
}): ResolvedRouterEvalScore | null {
  const modelScores = args.scores.filter(
    (score) => score.model_id === args.modelId,
  );
  const exact = modelScores.find(
    (score) => score.thinking_level === args.thinkingLevel,
  );
  const generic = modelScores.find((score) => score.thinking_level == null);
  const match = exact ?? generic;
  if (match) return { score: match.score, imputed: false };
  if (!(args.impute ?? false)) return null;

  const score = pairwiseRatioScore(
    modelScores,
    args.thinkingLevel,
    args.minScore,
    args.maxScore,
    args.ratios ?? new Map(),
  );
  return score === null ? null : { score, imputed: true };
}

function pairwiseRatioScore(
  modelScores: readonly EvalScoreForImputation[],
  targetLevel: ReasoningEffort,
  minScore: number,
  maxScore: number,
  ratios: ThinkingLevelRatios,
): number | null {
  const range = maxScore - minScore;
  if (!Number.isFinite(range) || range <= 0) return null;

  const estimates: number[] = [];
  for (const anchor of modelScores) {
    if (anchor.thinking_level == null || !Number.isFinite(anchor.score)) continue;
    const ratio = ratios.get(
      thinkingLevelPairKey(targetLevel, anchor.thinking_level),
    );
    if (ratio === undefined) continue;
    const normalizedAnchor = (anchor.score - minScore) / range;
    if (normalizedAnchor <= 0) continue;
    const estimate = normalizedAnchor * ratio;
    if (Number.isFinite(estimate) && estimate >= 0 && estimate <= 1) {
      estimates.push(estimate);
    }
  }
  if (estimates.length === 0) return null;
  return round2(minScore + mean(estimates) * range);
}

function explicitScoresByModel(
  rows: readonly EvalScoreForImputation[],
  minScore: number,
  range: number,
): Map<string, Map<ReasoningEffort, number>> {
  const result = new Map<string, Map<ReasoningEffort, number>>();
  for (const row of rows) {
    if (row.thinking_level == null || !Number.isFinite(row.score)) continue;
    const levels = result.get(row.model_id) ?? new Map();
    if (!levels.has(row.thinking_level)) {
      levels.set(row.thinking_level, (row.score - minScore) / range);
    }
    result.set(row.model_id, levels);
  }
  return result;
}

function thinkingLevelPairKey(
  target: ReasoningEffort,
  anchor: ReasoningEffort,
): string {
  return `${target}\u0000${anchor}`;
}

function mean(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function round2(value: number): number {
  return Math.round(value * 100) / 100;
}
