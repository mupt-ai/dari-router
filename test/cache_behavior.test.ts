import { expect, test } from "bun:test";

import {
  anthropicMinCacheTokens,
  isAnthropicFamily,
  isOpenAiFamily,
  openAiCachedPrefixTokens,
  promptCacheProviderForModel,
  providerMinCacheTokens,
} from "../src/cache_behavior.js";

test("classifies anthropic family for direct Anthropic only", () => {
  expect(isAnthropicFamily("anthropic/claude-sonnet-4-6")).toBe(true);
  expect(isAnthropicFamily("fireworks/deepseek-ai/DeepSeek-V4-Pro")).toBe(false);
  expect(isAnthropicFamily("openai/gpt-5.2")).toBe(false);
  expect(isAnthropicFamily("zai-org/GLM-5.2")).toBe(false);

  expect(isAnthropicFamily("Anthropic/claude-sonnet-4-6")).toBe(true);
  expect(isAnthropicFamily("ANTHROPIC/claude-sonnet-4-6")).toBe(true);
  expect(isAnthropicFamily("anthropic-proxy/claude")).toBe(false);
  // A bare native id carries no provider prefix: classification needs the
  // caller's provider, and guessing is an error rather than "no family".
  expect(isAnthropicFamily("claude-sonnet-4-6", "anthropic")).toBe(true);
  expect(() => isAnthropicFamily("claude-sonnet-4-6")).toThrow(
    "Unsupported model provider",
  );
});

test("classifies the subscription adapter as OpenAI cache behavior", () => {
  expect(isOpenAiFamily("openai/gpt-5.6-sol", "openai-codex")).toBe(true);
  expect(promptCacheProviderForModel("openai/gpt-5.6-sol", "openai-codex")).toBe(
    "openai",
  );
});

test("tracks provider-specific cache token minimums", () => {
  expect(anthropicMinCacheTokens("anthropic/claude-fable-5")).toBe(512);
  expect(anthropicMinCacheTokens("anthropic/claude-sonnet-4-6")).toBe(1024);
  expect(anthropicMinCacheTokens("anthropic/claude-opus-4.7")).toBe(1024);
  expect(anthropicMinCacheTokens("anthropic/claude-opus-4-6")).toBe(4096);
  expect(openAiCachedPrefixTokens(1023)).toBe(0);
  expect(openAiCachedPrefixTokens(1024)).toBe(1024);
  expect(openAiCachedPrefixTokens(2006)).toBe(1920);
});

test("classifies Meta as an automatic prompt-cache provider", () => {
  expect(promptCacheProviderForModel("meta/muse-spark-1.1")).toBe("meta");
  expect(promptCacheProviderForModel("Meta/muse-spark-1.1")).toBe("meta");
  expect(providerMinCacheTokens("meta/muse-spark-1.1")).toBe(0);
});
