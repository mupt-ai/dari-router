// Routing-only cost estimates for comparing candidate models before selection.
import {
  ANTHROPIC_LOOKBACK_BLOCKS,
  CROSS_TOKENIZER_RATIO,
  isAnthropicFamily,
  isOpenAiFamily,
  openAiCachedPrefixTokens,
  prefixHitMatchesProvider,
  promptCacheProviderForModel,
  providerCacheableTokens,
  providerMinCacheTokens,
  tokenizerFamily,
  type ModelProviderLookup,
  type PromptCacheProvider,
} from "./cache_behavior.js";
import { reasoningCacheScope, type ReasoningCacheScopeLookup } from "./cache_scope.js";
import {
  FIXED_TURN_CACHE_HIT_PROBABILITY,
  FIXED_TURN_COST_PROJECTED_TURNS,
} from "./fixed_turn_cost_config.js";
import { routingCandidateKey } from "./types.js";
import type {
  CandidateCostEstimate,
  ChainHits,
  FixedTurnCostEstimate,
  PrefixHit,
  ReasoningEffort,
  RouterPrefixHit,
  RoutingCandidate,
  TurnCostProjection,
} from "./types.js";

export const CHARS_PER_TOKEN = 4;
const MTOK = 1_000_000;

export type ModelPricing = {
  input: number;
  output: number;
  cacheRead: number;
  cacheWrite: number;
};

// A model missing from the resolved catalog must surface as null (pricing
// unknown), never as $0.
export type PricingLookup = (model: string) => ModelPricing | null;

export type CandidatePromptEstimate = {
  chars: number;
  reusesStoredPromptTokens: boolean;
};

// perModel/perModelBuckets key on raw model-id strings deliberately:
// entry.model is a verbatim copy of the router's enabled_models entry that
// served the turn, and candidate lookups use members of the same list, so
// both sides are byte-identical by construction. Canonicalizing here could
// merge two distinct custom model ids differing only in provider-prefix case
// and cross-contaminate their warm prefixes (an unsafe over-claim), whereas
// raw keying can at worst under-claim warmth if an id is ever re-spelled
// mid-conversation.
// Provider-visible chains can differ when one candidate receives compatible
// opaque continuation state and another sees readable history only. A hit is
// accepted solely against the chain for the model that originally wrote it;
// client-declared continuation provenance cannot make another candidate warm.
export function deriveModelChainHits(
  chainsByModel: ReadonlyMap<string, string[]>,
  hits: RouterPrefixHit[] | undefined
): ChainHits {
  const depthsByModel = new Map(
    [...chainsByModel].map(([model, chain]) => [
      model,
      new Map(chain.map((hash, index) => [hash, index + 1])),
    ])
  );
  return collectChainHits(hits, (entry) => depthsByModel.get(entry.model)?.get(entry.hash));
}

// Conversation identity accepts either hash tier: the strict per-model chain
// (byte-faithful echoes) or the host's loose chain (clients that drop
// reasoning payloads or rewrite item ids). Warmth estimation must NOT use
// this — a loose-only match means the provider-visible bytes diverged, so
// the cache is cold even though the conversation is the same.
// Loose hashes are not unique: two transcript twins (identical runs whose
// bytes differ only in reasoning payloads, item ids, or whitespace) can both
// match at the same depth. Ties are content-equivalent, but the pick must be
// deterministic: a strict match (byte-identical, provably this transcript)
// beats a loose-only one, then the most recently active entry wins.
export function deepestIdentityHit(
  chainsByModel: ReadonlyMap<string, string[]>,
  looseChain: readonly string[] | undefined,
  hits: RouterPrefixHit[] | undefined
): PrefixHit | undefined {
  const depthsByModel = new Map(
    [...chainsByModel].map(([model, chain]) => [
      model,
      new Map(chain.map((hash, index) => [hash, index + 1])),
    ])
  );
  const looseDepths = new Map((looseChain ?? []).map((hash, index) => [hash, index + 1]));
  let deepest: PrefixHit | undefined;
  let deepestStrict = false;
  let deepestUpdatedAtMs = Number.NEGATIVE_INFINITY;
  for (const entry of hits ?? []) {
    const strictDepth = depthsByModel.get(entry.model)?.get(entry.hash) ?? 0;
    const looseDepth = (entry.loose_hash && looseDepths.get(entry.loose_hash)) || 0;
    const depth = Math.max(strictDepth, looseDepth);
    if (depth === 0) continue;
    const strict = strictDepth > 0;
    const parsedUpdatedAtMs = Date.parse(entry.updated_at);
    const updatedAtMs = Number.isFinite(parsedUpdatedAtMs)
      ? parsedUpdatedAtMs
      : Number.NEGATIVE_INFINITY;
    const wins =
      !deepest ||
      depth > deepest.depth ||
      (depth === deepest.depth &&
        (strict !== deepestStrict
          ? strict
          : updatedAtMs > deepestUpdatedAtMs));
    if (wins) {
      deepest = { entry, depth };
      deepestStrict = strict;
      deepestUpdatedAtMs = updatedAtMs;
    }
  }
  return deepest;
}

function collectChainHits(
  hits: RouterPrefixHit[] | undefined,
  depthFor: (entry: RouterPrefixHit) => number | undefined
): ChainHits {
  let deepest: PrefixHit | undefined;
  const perModel = new Map<string, PrefixHit>();
  const perModelBuckets = new Map<string, Map<string | null, PrefixHit>>();
  for (const entry of hits ?? []) {
    const depth = depthFor(entry);
    if (depth === undefined) continue;
    const hit: PrefixHit = { entry, depth };
    if (!deepest || depth > deepest.depth) deepest = hit;
    const existing = perModel.get(entry.model);
    if (!existing || depth > existing.depth) perModel.set(entry.model, hit);
    const bucket = entry.reasoning_bucket ?? null;
    let byBucket = perModelBuckets.get(entry.model);
    if (!byBucket) {
      byBucket = new Map();
      perModelBuckets.set(entry.model, byBucket);
    }
    const existingBucketHit = byBucket.get(bucket);
    if (!existingBucketHit || depth > existingBucketHit.depth) byBucket.set(bucket, hit);
  }
  return { deepest, perModel, perModelBuckets };
}

// The warm-prefix hit a candidate can actually re-read, given how its
// provider scopes cache identity across reasoning payloads. Shared-scope
// providers reuse one partition regardless of effort; effort-keyed providers
// only re-read the partition warmed under the same effective effort, so the
// hit must come from the matching bucket. Entries written before buckets
// existed (null) are accepted for any effort so a rollout never estimates a
// genuinely warm prefix as cold.
export function selectPrefixHit(
  hits: ChainHits,
  model: string,
  reasoningBucket: string,
  cacheScope: ReasoningCacheScopeLookup = reasoningCacheScope
): PrefixHit | undefined {
  if (cacheScope(model) === "shared") return hits.perModel.get(model);
  const byBucket = hits.perModelBuckets.get(model);
  if (!byBucket) return undefined;
  const matched = byBucket.get(reasoningBucket);
  const legacy = byBucket.get(null);
  if (matched) return matched;
  return legacy;
}

export function selectPromptAnchor(
  hits: ChainHits,
  model: string,
  reasoningBucket: string,
  cacheScope: ReasoningCacheScopeLookup = reasoningCacheScope
): PrefixHit | undefined {
  // Prompt-token baselines are provider/model-specific. Cross-model hits can
  // recover conversation identity elsewhere, but cannot anchor this model's
  // provider-visible prompt estimate.
  return selectPrefixHit(hits, model, reasoningBucket, cacheScope);
}

export function estimateCandidateCosts(args: {
  candidates: RoutingCandidate[];
  hits: ChainHits;
  incomingProviderBlockCountFor: (model: string) => number;
  promptEstimatesByCandidate: ReadonlyMap<string, CandidatePromptEstimate>;
  toolChoiceFp: string;
  responseFormatFp: string;
  pricing: PricingLookup;
  averageOutputTokensByModel?: Readonly<
    Record<string, Partial<Record<ReasoningEffort, number>> | null>
  >;
  cacheScope?: ReasoningCacheScopeLookup;
  modelProvider?: ModelProviderLookup;
}): CandidateCostEstimate[] {
  const cacheScope = args.cacheScope ?? ((model) =>
    reasoningCacheScope(model, args.modelProvider?.(model)));
  return args.candidates.map((candidate) => {
    const { model, reasoningEffort } = candidate;
    const price = args.pricing(model);
    const provider = args.modelProvider?.(model);
    const anthropicStyle = isAnthropicFamily(model, provider);
    const openAiStyle = isOpenAiFamily(model, provider);
    const promptEstimate = args.promptEstimatesByCandidate.get(
      routingCandidateKey(candidate),
    );
    if (promptEstimate === undefined) {
      throw new Error(
        `missing prompt estimate for ${model}/${reasoningEffort}`,
      );
    }
    const selectedAnchor = selectPromptAnchor(args.hits, model, reasoningEffort, cacheScope);
    const anchor = selectedAnchor && prefixHitMatchesProvider(selectedAnchor.entry, provider)
      ? selectedAnchor
      : undefined;
    const reusesStoredPromptTokens = promptEstimate.reusesStoredPromptTokens;
    const serializedTokens = Math.ceil(
      promptEstimate.chars / CHARS_PER_TOKEN,
    );
    const tokenizerReference = anchor ?? args.hits.deepest;
    const crossesTokenizerFamilies =
      tokenizerReference !== undefined
      && tokenizerFamily(model, provider)
        !== tokenizerFamily(
          tokenizerReference.entry.model,
          args.modelProvider?.(tokenizerReference.entry.model),
        );
    // Candidate-specific serialization captures provider wire-shape
    // differences. Preserve the separate pessimistic margin for tokenizer
    // differences without borrowing the foreign model's measured token count.
    const newTokens = crossesTokenizerFamilies
      ? Math.round(serializedTokens * CROSS_TOKENIZER_RATIO)
      : serializedTokens;

    let estPrompt = newTokens;
    if (anchor && reusesStoredPromptTokens) {
      const knownTokens = anchor.entry.prompt_tokens;
      estPrompt = knownTokens + newTokens;
    }

    const selectedHit = selectPrefixHit(args.hits, model, reasoningEffort, cacheScope);
    const knownCacheProvider = promptCacheProviderForModel(model, provider) !== null;
    const hit = selectedHit
      && (provider === undefined || knownCacheProvider)
      && prefixHitMatchesProvider(selectedHit.entry, provider)
      ? selectedHit
      : undefined;
    let warmTokens = 0;
    if (hit) {
      // tool_choice flips bust Anthropic's message cache but not automatic
      // vendor caches; response_format schemas are prompt content everywhere.
      const fingerprintsMatch = anthropicStyle
        ? hit.entry.tool_choice_fp === args.toolChoiceFp &&
          hit.entry.response_format_fp === args.responseFormatFp
        : hit.entry.response_format_fp === args.responseFormatFp;
      const withinLookback =
        !anthropicStyle
        || withinAnthropicLookback({
          hit: hit.entry,
          incomingProviderDepth: args.incomingProviderBlockCountFor(model),
        });
      // Provider minimums apply to the cached prefix being re-read — this
      // candidate's own hit, counted by its own provider — not to the global
      // prompt estimate (which may anchor on another provider's deeper entry).
      const meetsMinimum =
        hit.entry.prompt_tokens >= providerMinCacheTokens(model, provider);
      if (fingerprintsMatch && withinLookback && meetsMinimum) {
        warmTokens = openAiStyle
          ? openAiCachedPrefixTokens(hit.entry.prompt_tokens)
          : hit.entry.prompt_tokens;
      }
    }
    // Anchored estimates contain only a fresh suffix; complete-prompt
    // estimates (used for legacy rows without an input depth) already include
    // the warm prefix and must not add it a second time.
    estPrompt = Math.max(
      estPrompt,
      reusesStoredPromptTokens ? warmTokens + newTokens : warmTokens,
    );
    // An unanchored hit (for example, a legacy row) leaves only a coarse
    // complete-prompt estimate. If that estimate undercuts the confirmed warm
    // prefix, its fresh/cached split is unknowable. Conservatively charge the
    // coarse estimate as fresh instead of making the fresh portion free.
    if (
      warmTokens > 0
      && !reusesStoredPromptTokens
      && newTokens <= warmTokens
    ) {
      estPrompt = warmTokens + newTokens;
    }

    let estCost: number | null = null;
    if (price) {
      const remainder = Math.max(0, estPrompt - warmTokens);
      estCost =
        (warmTokens * price.cacheRead + remainder * freshInputRate(model, price, estPrompt, provider)) /
        1_000_000;
    }

    return {
      model,
      reasoning_effort: reasoningEffort,
      warm_tokens: warmTokens,
      est_prompt_tokens: estPrompt,
      est_input_cost_usd: estCost,
      output_cost_per_mtok: price?.output ?? null,
      pricing_known: price !== null,
      fixed_turn_cost_estimate: price
        ? fixedTurnCostEstimate({
            model,
            price,
            estPromptTokens: estPrompt,
            warmTokens,
            reasoningEffort,
            provider,
            outputTokens: args.averageOutputTokensByModel?.[model]?.[reasoningEffort],
          })
        : null,
    } satisfies CandidateCostEstimate;
  });
}

function withinAnthropicLookback(args: {
  hit: RouterPrefixHit;
  incomingProviderDepth: number;
}): boolean {
  const storedProviderDepth = args.hit.provider_block_depth;
  if (storedProviderDepth === undefined || storedProviderDepth === null) {
    return false;
  }
  const delta = args.incomingProviderDepth - storedProviderDepth;
  return delta >= 0 && delta <= ANTHROPIC_LOOKBACK_BLOCKS;
}

function fixedTurnCostEstimate(args: {
  model: string;
  price: ModelPricing;
  estPromptTokens: number;
  warmTokens: number;
  reasoningEffort: ReasoningEffort;
  provider?: string;
  outputTokens?: number;
}): FixedTurnCostEstimate | null {
  const provider = promptCacheProviderForModel(args.model, args.provider);
  if (!provider || args.outputTokens === undefined) return null;

  // One pass to the longest horizon, snapshotting the running total at each
  // shorter one: turn N's cost depends on every turn before it, so the
  // horizons are prefixes of the same accumulation rather than separate runs.
  const horizons: number[] = [...FIXED_TURN_COST_PROJECTED_TURNS].sort(
    (left, right) => left - right,
  );
  const longestHorizon = horizons[horizons.length - 1] ?? 0;
  const projections: TurnCostProjection[] = [];
  const outputTokens = args.outputTokens;
  let totalCost = 0;

  for (let turn = 1; turn <= longestHorizon; turn += 1) {
    const inputTokens = args.estPromptTokens + (turn - 1) * outputTokens;
    totalCost +=
      turn === 1
        ? deterministicTurnCost({
            model: args.model,
            price: args.price,
            inputTokens,
            cachedTokens: args.warmTokens,
            outputTokens,
            provider,
          })
        : laterTurnCost({
            model: args.model,
            price: args.price,
            provider,
            inputTokens,
            // Later requests can only read prefixes that appeared in prior
            // request inputs. The immediately previous output is fresh on the
            // next request.
            reusablePrefix: args.estPromptTokens + (turn - 2) * outputTokens,
            outputTokens,
          });
    if (horizons.includes(turn)) {
      projections.push({ projected_turns: turn, total_cost_usd: totalCost });
    }
  }

  return {
    output_tokens_per_turn: outputTokens,
    assumed_reasoning_effort: args.reasoningEffort,
    projections,
  };
}

function laterTurnCost(args: {
  model: string;
  price: ModelPricing;
  provider: PromptCacheProvider;
  inputTokens: number;
  reusablePrefix: number;
  outputTokens: number;
}): number {
  const cacheableTokens = providerCacheableTokens(
    args.model,
    args.provider,
    args.reusablePrefix,
  );
  const missCost = deterministicTurnCost({
    model: args.model,
    price: args.price,
    inputTokens: args.inputTokens,
    cachedTokens: 0,
    outputTokens: args.outputTokens,
    provider: args.provider,
  });
  if (cacheableTokens <= 0) return missCost;

  const hitCost = deterministicTurnCost({
    model: args.model,
    price: args.price,
    inputTokens: args.inputTokens,
    cachedTokens: cacheableTokens,
    outputTokens: args.outputTokens,
    provider: args.provider,
  });
  const hitProbability = FIXED_TURN_CACHE_HIT_PROBABILITY[args.provider];
  return hitCost * hitProbability + missCost * (1 - hitProbability);
}

// The loop cost at one horizon, for callers that need a single scalar rather
// than the whole curve.
export function turnCostUsdAt(
  estimate: FixedTurnCostEstimate | null | undefined,
  turns: number,
): number | null {
  const projection = estimate?.projections.find(
    (entry) => entry.projected_turns === turns,
  );
  return projection?.total_cost_usd ?? null;
}

// A caching provider writes whatever the prompt did not read, and several bill
// that write above their input rate. An effort-keyed cache partitions on effort,
// so a switch of either model or effort leaves nothing warm and writes it all.
function freshInputRate(
  model: string,
  price: ModelPricing,
  promptTokens: number,
  provider?: string,
): number {
  return price.cacheWrite > 0 && promptTokens >= providerMinCacheTokens(model, provider)
    ? price.cacheWrite
    : price.input;
}

function deterministicTurnCost(args: {
  model: string;
  price: ModelPricing;
  inputTokens: number;
  cachedTokens: number;
  outputTokens: number;
  provider?: string;
}): number {
  const freshTokens = args.inputTokens - args.cachedTokens;
  return (
    (args.cachedTokens * args.price.cacheRead) / MTOK +
    (freshTokens * freshInputRate(args.model, args.price, args.inputTokens, args.provider)) / MTOK +
    (args.outputTokens * args.price.output) / MTOK
  );
}
