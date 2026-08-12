import { expect, test } from "bun:test";

import { cachePartitionKey, type ReasoningCacheScopeLookup } from "../src/cache_scope.js";

const SOL = "openai/gpt-5.6-sol";
const GLM = "fireworks/zai-org/GLM-5.2";

test("partition keys follow the provider's reasoning cache scope", () => {
  // Effort-keyed: each effective effort warms its own partition, and an entry
  // written before buckets existed is its own rather than any effort's.
  expect(cachePartitionKey(SOL, "low")).not.toBe(cachePartitionKey(SOL, "high"));
  expect(cachePartitionKey(SOL, null)).not.toBe(cachePartitionKey(SOL, "low"));
  expect(cachePartitionKey(SOL, "low")).toBe(cachePartitionKey(SOL, "low"));

  // Shared: one partition per model, so effort cannot split it.
  expect(cachePartitionKey(GLM, "off")).toBe(cachePartitionKey(GLM, "medium"));
  expect(cachePartitionKey(GLM, "off")).not.toBe(cachePartitionKey(SOL, "off"));

  // A host that reclassifies a provider moves its storage with the estimator.
  const shared: ReasoningCacheScopeLookup = () => "shared";
  expect(cachePartitionKey(SOL, "low", shared)).toBe(cachePartitionKey(SOL, "high", shared));
});
