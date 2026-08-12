// Builds the selector input: candidate pairs, matched eval scores, cost
// estimates, the previous decision, and the (image-scrubbed) conversation.
// The system prompts live in prompts.ts.

import {
  isRecord,
  REASONING_EFFORTS,
  routingCandidateKey,
  type CandidateCostEstimate,
  type ChatMessage,
  type CustomRouterRule,
  type PreviousDecision,
  type ReasoningEffort,
  type RouterEval,
  type RouterEvalScore,
  type RoutingCandidate,
} from "./types.js";

const IMAGE_OMITTED_PLACEHOLDER = "<image omitted>";

type SelectorCandidate = {
  model: string;
  thinking_level: RoutingCandidate["reasoningEffort"];
};

type SelectorPreviousDecision = SelectorCandidate & { reason: string };

export type SelectorInput = {
  candidate_pairs: SelectorCandidate[];
  imported_evals: Array<Record<string, unknown>>;
  previous_decision: SelectorPreviousDecision | null;
  cost_estimates: CandidateCostEstimate[] | null;
  messages: ChatMessage[];
};

type SelectorCustomRule = Omit<CustomRouterRule, "thinking_level"> & {
  thinking_level: ReasoningEffort | null;
};

export type CustomSelectorInput = SelectorInput & {
  custom_rules: SelectorCustomRule[];
  default_target: {
    model: string;
    thinking_level: ReasoningEffort | null;
  } | null;
};

// Selector serialization must not include image bytes. The provider-facing
// request and prefix fingerprint still use the original content, while this
// stable placeholder preserves the selector's own prompt cache behavior.
export function selectorSafeMessages(messages: ChatMessage[]): ChatMessage[] {
  return messages.map((message) => {
    if (!Array.isArray(message.content)) return message;
    return {
      ...message,
      content: message.content.map((part) =>
        isRecord(part) && part.type === "image_url"
          ? { type: "image_url", image_url: { url: IMAGE_OMITTED_PLACEHOLDER } }
          : part
      ),
    };
  });
}

export function buildSelectorInput(args: {
  candidates: RoutingCandidate[];
  evals: RouterEval[];
  previousDecision: PreviousDecision | null;
  costEstimates: CandidateCostEstimate[] | null;
  messages: ChatMessage[];
  customRules?: CustomRouterRule[];
  defaultTarget?: {
    model: string;
    thinkingLevel: ReasoningEffort | null;
  } | null;
  modelFallbackEnabled?: boolean;
  fallbackRequiresDifferentProvider?: boolean;
  // When true, candidates with no exact-level and no generic "Any" score use
  // cross-model thinking-level ratios from the same eval card. Off by default
  // so existing routers keep measured-only scores.
  imputeEvalScores?: boolean;
}): SelectorInput | CustomSelectorInput {
  const input: SelectorInput = {
    candidate_pairs: args.candidates.map(({ model, reasoningEffort }) => ({
      model,
      thinking_level: reasoningEffort,
    })),
    imported_evals: formatImportedEvals(
      args.evals,
      args.candidates,
      args.imputeEvalScores ?? false,
    ),
    previous_decision: args.previousDecision
      ? {
          model: args.previousDecision.model,
          thinking_level: args.previousDecision.reasoningEffort,
          reason: args.previousDecision.reason,
        }
      : null,
    cost_estimates: args.costEstimates,
    messages: args.messages,
  };
  if (args.customRules !== undefined) {
    (input as CustomSelectorInput).custom_rules = args.customRules.map(
      (rule) => ({
        ...rule,
        thinking_level: rule.thinking_level ?? null,
      }),
    );
    (input as CustomSelectorInput).default_target = args.defaultTarget
      ? {
          model: args.defaultTarget.model,
          thinking_level: args.defaultTarget.thinkingLevel,
        }
      : null;
  }
  return input;
}

function formatImportedEvals(
  evals: RouterEval[],
  candidates: RoutingCandidate[],
  imputeEvalScores: boolean,
): Array<Record<string, unknown>> {
  const thinkingLevelRatios = averageThinkingLevelRatios(evals);
  return evals.flatMap((evalCard) => {
    const scores = matchingEvalScores(
      evalCard.scores,
      candidates,
      imputeEvalScores,
      evalCard.min_score,
      evalCard.max_score,
      thinkingLevelRatios,
    );
    if (scores.length === 0) return [];
    return [
      {
        id: evalCard.id,
        name: evalCard.name,
        description: evalCard.description ?? null,
        min_score: evalCard.min_score,
        max_score: evalCard.max_score,
        scores,
      },
    ];
  });
}

function matchingEvalScores(
  scores: RouterEvalScore[],
  candidates: RoutingCandidate[],
  imputeEvalScores: boolean,
  minScore: number,
  maxScore: number,
  thinkingLevelRatios: ReadonlyMap<string, number>,
): Array<Record<string, unknown>> {
  const uniqueCandidates = new Map(
    candidates.map((candidate) => [routingCandidateKey(candidate), candidate]),
  );
  return [...uniqueCandidates.values()].flatMap((candidate) => {
    const modelScores = scores.filter(
      (score) => score.model_id === candidate.model,
    );
    const exact = modelScores.find(
      (score) => score.thinking_level === candidate.reasoningEffort,
    );
    const generic = modelScores.find((score) => score.thinking_level == null);
    const match = exact ?? generic;
    if (match) {
      return [
        {
          model_id: candidate.model,
          thinking_level: candidate.reasoningEffort,
          score: match.score,
          ...normalizedScore(match.score, scores),
          notes: match.notes ?? null,
        },
      ];
    }
    if (!imputeEvalScores) return [];
    const imputed = pairwiseRatioScore(
      candidate,
      modelScores,
      minScore,
      maxScore,
      thinkingLevelRatios,
    );
    if (imputed === null) return [];
    return [
      {
        model_id: candidate.model,
        thinking_level: candidate.reasoningEffort,
        score: imputed,
        ...normalizedScore(imputed, scores),
        notes: null,
        imputed: true,
      },
    ];
  });
}

// Learns every directed thinking-level ratio from all eval/model observations
// that contain both levels. Scores are first mapped to [0, 1] using each eval's
// declared range, so negative scales and differently-sized scales are safe.
function averageThinkingLevelRatios(evals: RouterEval[]): Map<string, number> {
  const samples = new Map<string, number[]>();
  for (const evalCard of evals) {
    const range = evalCard.max_score - evalCard.min_score;
    if (!Number.isFinite(range) || range <= 0) continue;
    const byModel = new Map<string, Map<ReasoningEffort, number>>();
    for (const row of evalCard.scores) {
      if (row.thinking_level == null || !Number.isFinite(row.score)) continue;
      const levels = byModel.get(row.model_id) ?? new Map();
      if (!levels.has(row.thinking_level)) {
        levels.set(row.thinking_level, (row.score - evalCard.min_score) / range);
      }
      byModel.set(row.model_id, levels);
    }
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

function pairwiseRatioScore(
  candidate: RoutingCandidate,
  modelScores: RouterEvalScore[],
  minScore: number,
  maxScore: number,
  thinkingLevelRatios: ReadonlyMap<string, number>,
): number | null {
  const range = maxScore - minScore;
  if (!Number.isFinite(range) || range <= 0) return null;

  const estimates: number[] = [];
  for (const anchor of modelScores) {
    if (anchor.thinking_level == null || !Number.isFinite(anchor.score)) continue;
    const ratio = thinkingLevelRatios.get(
      thinkingLevelPairKey(candidate.reasoningEffort, anchor.thinking_level),
    );
    if (ratio === undefined) continue;
    const normalizedAnchor = (anchor.score - minScore) / range;
    // Ratios cannot carry information from the scale floor: ratio learning
    // excludes non-positive anchors for the same reason.
    if (normalizedAnchor <= 0) continue;
    const normalizedEstimate = normalizedAnchor * ratio;
    // A single anchor can be invalid even when another anchor gives a useful
    // estimate. Discard only this anchor rather than poisoning the average.
    if (
      Number.isFinite(normalizedEstimate) &&
      normalizedEstimate >= 0 &&
      normalizedEstimate <= 1
    ) {
      estimates.push(normalizedEstimate);
    }
  }
  if (estimates.length === 0) return null;

  const normalizedScore = mean(estimates);
  // A ratio-derived score outside this eval's declared range is not credible;
  // leave it missing rather than disguising the bad estimate as an endpoint.
  if (normalizedScore < 0 || normalizedScore > 1) return null;
  return round2(minScore + normalizedScore * range);
}

function thinkingLevelPairKey(
  target: ReasoningEffort,
  anchor: ReasoningEffort,
): string {
  return `${target}\u0000${anchor}`;
}

function mean(values: number[]): number {
  return values.reduce((total, value) => total + value, 0) / values.length;
}

// Positions a candidate's score against every model on the scorecard, not just
// the candidates that survived pruning. Candidate-relative normalization would
// move with the pruned set (with two candidates a z-score is always +/-1), so
// the selector would read pure ordering as if it carried magnitude.
function normalizedScore(
  score: number,
  population: RouterEvalScore[],
): { rank: number; rank_total: number; z_score: number } {
  const values = population.map((entry) => entry.score);
  const mean = values.reduce((total, value) => total + value, 0) / values.length;
  const variance =
    values.reduce((total, value) => total + (value - mean) ** 2, 0) /
    values.length;
  const stdDev = Math.sqrt(variance);
  return {
    // Competition ranking, best score first; tied scores share the low rank.
    rank: 1 + values.filter((value) => value > score).length,
    rank_total: values.length,
    z_score: stdDev === 0 ? 0 : round2((score - mean) / stdDev),
  };
}

function round2(value: number): number {
  return Math.round(value * 100) / 100;
}
