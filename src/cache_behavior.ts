import { nativeModelId, providerForModel } from "./model_ids.js";
import type { RouterPrefixHit } from "./types.js";

// Provider prompt caches stop serving a prefix roughly five minutes after it
// was last written. Stored prefix entries outlive that: a host's lookback
// keeps them long enough to still name the conversation and the decision it
// continued from. Only entries inside this window describe warmth.
export const PREFIX_WARMTH_WINDOW_MS = 5 * 60 * 1000;

// Anthropic checks 20 positions per breakpoint, counting the breakpoint
// itself as the first, so an old cached breakpoint hits only when at most
// 19 content blocks were appended after it. These are routing estimates, never
// billing.
export const ANTHROPIC_LOOKBACK_BLOCKS = 19;
export const CROSS_TOKENIZER_RATIO = 1.1;
const OPENAI_MIN_CACHE_TOKENS = 1024;
const OPENAI_CACHE_INCREMENT_TOKENS = 128;

export type PromptCacheProvider = "openai" | "anthropic" | "fireworks" | "meta";

// providerForModel() lowercases the provider prefix, so family checks must
// compare case-insensitively too. Unlike providerForModel this never throws;
// malformed ids classify as no family.
function providerPrefix(model: string): string {
  const slash = model.indexOf("/");
  return slash <= 0 ? "" : model.slice(0, slash).toLowerCase();
}

export function isAnthropicFamily(model: string): boolean {
  return providerPrefix(model) === "anthropic";
}

export function isOpenAiFamily(model: string): boolean {
  return providerPrefix(model) === "openai";
}

function isFireworksFamily(model: string): boolean {
  return providerPrefix(model) === "fireworks";
}

function isMetaFamily(model: string): boolean {
  return providerPrefix(model) === "meta";
}

export function promptCacheProviderForModel(model: string): PromptCacheProvider | null {
  if (isOpenAiFamily(model)) return "openai";
  if (isAnthropicFamily(model)) return "anthropic";
  if (isFireworksFamily(model)) return "fireworks";
  if (isMetaFamily(model)) return "meta";
  return null;
}

export function openAiCachedPrefixTokens(promptTokens: number): number {
  if (promptTokens < OPENAI_MIN_CACHE_TOKENS) return 0;
  const extra = promptTokens - OPENAI_MIN_CACHE_TOKENS;
  return (
    OPENAI_MIN_CACHE_TOKENS +
    Math.floor(extra / OPENAI_CACHE_INCREMENT_TOKENS) * OPENAI_CACHE_INCREMENT_TOKENS
  );
}

// Anthropic cache minimums vary by model family. Keep this routing-side
// estimate conservative for known higher-minimum models; unknown future
// direct Anthropic models fall back to the common 1024-token threshold.
export function anthropicMinCacheTokens(model: string): number {
  if (!isAnthropicFamily(model)) return OPENAI_MIN_CACHE_TOKENS;
  const native = nativeModelId(model)
    .toLowerCase()
    .replace(/(?<=\d)\.(?=\d)/g, "-");

  if (
    native.includes("opus-4-6") ||
    native.includes("opus-4-5") ||
    native.includes("haiku-4-5")
  ) {
    return 4096;
  }
  if (native.includes("fable-5")) {
    return 512;
  }
  return OPENAI_MIN_CACHE_TOKENS;
}

export function tokenizerFamily(model: string): string {
  const provider = providerForModel(model);
  if (provider === "fireworks") {
    const native = nativeModelId(model).toLowerCase();
    const accountPrefix = "accounts/fireworks/models/";
    const modelFamily = native.startsWith(accountPrefix)
      ? native.slice(accountPrefix.length)
      : native;
    const slash = modelFamily.indexOf("/");
    return `${provider}/${slash === -1 ? modelFamily : modelFamily.slice(0, slash)}`;
  }
  return provider;
}

export function providerMinCacheTokens(model: string): number {
  if (isAnthropicFamily(model)) return anthropicMinCacheTokens(model);
  if (isFireworksFamily(model) || isMetaFamily(model)) return 0;
  return OPENAI_MIN_CACHE_TOKENS;
}

// The stored entries a provider cache can still be re-reading at nowMs. This
// is the definition of "warm", and every cost estimate rests on it, so an
// unparseable timestamp reads as cold: warmth is proven, never assumed.
// nowMs is supplied by the caller because this package reads no clock.
export function warmPrefixHits(
  hits: readonly RouterPrefixHit[] | undefined,
  nowMs: number
): RouterPrefixHit[] {
  const cutoffMs = nowMs - PREFIX_WARMTH_WINDOW_MS;
  return (hits ?? []).filter((hit) => {
    const updatedAtMs = Date.parse(hit.updated_at);
    return Number.isFinite(updatedAtMs) && updatedAtMs >= cutoffMs;
  });
}

export function providerCacheableTokens(
  model: string,
  provider: PromptCacheProvider,
  prefixTokens: number
): number {
  if (prefixTokens <= 0) return 0;
  if (provider === "openai") return openAiCachedPrefixTokens(prefixTokens);
  if (provider === "anthropic") {
    return prefixTokens >= anthropicMinCacheTokens(model) ? Math.floor(prefixTokens) : 0;
  }
  return Math.floor(prefixTokens);
}
