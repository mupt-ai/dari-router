// Builds the selector input: candidate pairs, matched eval scores, cost
// estimates, the previous decision, and the (image-scrubbed) conversation.
// The system prompts live in prompts.ts.

import {
  createThinkingLevelRatios,
  resolveRouterEvalScore,
  type ThinkingLevelRatios,
} from "./eval_score_imputation.js";
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

type ResolvedEvalScore = {
  model_id: string;
  thinking_level: ReasoningEffort;
  score: number;
  notes: string | null;
  imputed?: true;
};

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
  // Calibrates cross-level ratios without adding these cards to imported_evals.
  imputationReferenceEvals?: RouterEval[];
}): SelectorInput | CustomSelectorInput {
  const imputeEvalScores = args.imputeEvalScores ?? false;
  const input: SelectorInput = {
    candidate_pairs: args.candidates.map(({ model, reasoningEffort }) => ({
      model,
      thinking_level: reasoningEffort,
    })),
    imported_evals: formatImportedEvals(
      args.evals,
      args.candidates,
      imputeEvalScores,
      args.imputationReferenceEvals ?? [],
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
  imputationReferenceEvals: RouterEval[],
): Array<Record<string, unknown>> {
  const selectedEvalIds = new Set(evals.map((evalCard) => evalCard.id));
  const ratioEvals = imputeEvalScores
    ? [
        ...evals,
        ...imputationReferenceEvals.filter(
          (evalCard) => !selectedEvalIds.has(evalCard.id),
        ),
      ]
    : [];
  const thinkingLevelRatios = createThinkingLevelRatios(ratioEvals);
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
  thinkingLevelRatios: ThinkingLevelRatios,
): Array<Record<string, unknown>> {
  const uniqueCandidates = new Map(
    candidates.map((candidate) => [routingCandidateKey(candidate), candidate]),
  );
  const resolvedScores = [...uniqueCandidates.values()]
    .map((candidate): ResolvedEvalScore | null => {
      const resolved = resolveRouterEvalScore({
        scores,
        modelId: candidate.model,
        thinkingLevel: candidate.reasoningEffort,
        minScore,
        maxScore,
        impute: imputeEvalScores,
        ratios: thinkingLevelRatios,
      });
      if (resolved === null) return null;
      const modelScores = scores.filter(
        (score) => score.model_id === candidate.model,
      );
      const measured = resolved.imputed
        ? null
        : (modelScores.find(
            (score) => score.thinking_level === candidate.reasoningEffort,
          ) ?? modelScores.find((score) => score.thinking_level == null));
      return {
        model_id: candidate.model,
        thinking_level: candidate.reasoningEffort,
        score: resolved.score,
        notes: measured?.notes ?? null,
        ...(resolved.imputed ? { imputed: true as const } : {}),
      };
    })
    .filter((score): score is ResolvedEvalScore => score !== null);
  return normalizeResolvedScores(resolvedScores);
}

// The selector compares current actions, so exact, generic, and imputed rows
// share one candidate-only population. Unresolved actions never enter the
// denominator or the distribution.
function normalizeResolvedScores(
  scores: ResolvedEvalScore[],
): Array<Record<string, unknown>> {
  if (scores.length === 0) return [];

  const values = scores.map((entry) => entry.score);
  const populationMean = mean(values);
  const variance =
    values.reduce((total, value) => total + (value - populationMean) ** 2, 0) /
    values.length;
  const stdDev = Math.sqrt(variance);
  const ranks = new Map<number, number>();
  [...values]
    .sort((left, right) => right - left)
    .forEach((score, index) => {
      // Competition ranking, best score first; tied scores share the low rank.
      if (!ranks.has(score)) ranks.set(score, index + 1);
    });

  return scores.map((score) => ({
    ...score,
    rank: ranks.get(score.score)!,
    rank_total: scores.length,
    z_score: stdDev === 0 ? 0 : round2((score.score - populationMean) / stdDev),
  }));
}

function mean(values: number[]): number {
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function round2(value: number): number {
  return Math.round(value * 100) / 100;
}
