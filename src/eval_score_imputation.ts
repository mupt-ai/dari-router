import { REASONING_EFFORTS, type ReasoningEffort } from "./types.js";

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

// Only named, same-family predecessors may stand in for a new model while
// benchmark sources catch up. Never infer a predecessor from a slug.
const PREVIOUS_GENERATION: Readonly<Record<string, string>> = {
  "openai/gpt-6-sol": "openai/gpt-5.6-sol",
  "openai/gpt-6-luna": "openai/gpt-5.6-luna",
  "anthropic/claude-opus-5-5": "anthropic/claude-opus-5",
  "anthropic/claude-fable-5-1": "anthropic/claude-fable-5",
  "meta/muse-spark-1.3": "meta/muse-spark-1.2",
  "zai-org/GLM-5.3": "zai-org/GLM-5.2",
  "xai/grok-4.7": "xai/grok-4.6",
};

export type ResolvedRouterEvalScore = {
  score: number;
  imputed: boolean;
  sourceModelId?: string;
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

  const ratios = args.ratios ?? new Map();
  const score = pairwiseRatioScore(
    modelScores,
    args.thinkingLevel,
    args.minScore,
    args.maxScore,
    ratios,
  );
  if (score !== null) return { score, imputed: true };

  // A measured current-generation score or a valid same-model level
  // estimate always wins. Borrow only for a still-missing candidate level.
  const previous = PREVIOUS_GENERATION[args.modelId];
  if (!previous) return null;
  const previousScores = args.scores.filter((row) => row.model_id === previous);
  const predecessor = previousScores.find((row) => row.thinking_level === args.thinkingLevel)
    ?? previousScores.find((row) => row.thinking_level == null);
  const borrowed = predecessor?.score ?? pairwiseRatioScore(
    previousScores,
    args.thinkingLevel,
    args.minScore,
    args.maxScore,
    ratios,
  );
  return borrowed === null || borrowed === undefined
    ? null
    : { score: borrowed, imputed: true, sourceModelId: previous };
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
    const ratio = resolveThinkingLevelRatio(
      ratios,
      targetLevel,
      anchor.thinking_level,
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

// Prefer a directly observed relationship. When none exists, compose the
// shortest available chains (for example medium -> max -> xhigh) and average
// their products. This lets sparse calibration cards connect levels without a
// longer chain overriding stronger direct evidence.
function resolveThinkingLevelRatio(
  ratios: ThinkingLevelRatios,
  targetLevel: ReasoningEffort,
  anchorLevel: ReasoningEffort,
): number | undefined {
  const direct = ratios.get(thinkingLevelPairKey(targetLevel, anchorLevel));
  if (direct !== undefined) return direct;

  type RatioPaths = { sum: number; count: number };
  let frontier = new Map<ReasoningEffort, RatioPaths>([
    [anchorLevel, { sum: 1, count: 1 }],
  ]);
  const visited = new Set<ReasoningEffort>([anchorLevel]);

  while (frontier.size > 0) {
    const next = new Map<ReasoningEffort, RatioPaths>();
    for (const [currentLevel, paths] of frontier) {
      for (const nextLevel of REASONING_EFFORTS) {
        if (visited.has(nextLevel)) continue;
        const edge = ratios.get(
          thinkingLevelPairKey(nextLevel, currentLevel),
        );
        if (edge === undefined || !Number.isFinite(edge) || edge < 0) continue;
        const existing = next.get(nextLevel) ?? { sum: 0, count: 0 };
        existing.sum += paths.sum * edge;
        existing.count += paths.count;
        next.set(nextLevel, existing);
      }
    }

    const targetPaths = next.get(targetLevel);
    if (targetPaths !== undefined) {
      return targetPaths.sum / targetPaths.count;
    }
    for (const level of next.keys()) visited.add(level);
    frontier = next;
  }
  return undefined;
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
