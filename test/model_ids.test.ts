import { expect, test } from "bun:test";

import { RouterCoreError } from "../src/errors.js";
import { canonicalModelId, nativeModelId, providerForModel } from "../src/model_ids.js";

test("splits provider prefixes case-insensitively", () => {
  expect(providerForModel("openai/gpt-4.1-mini")).toBe("openai");
  expect(providerForModel("OpenAI/gpt-4.1-mini")).toBe("openai");
  expect(providerForModel("fireworks/deepseek-ai/DeepSeek-V4-Pro")).toBe("fireworks");
  expect(nativeModelId("fireworks/deepseek-ai/DeepSeek-V4-Pro")).toBe(
    "deepseek-ai/DeepSeek-V4-Pro",
  );
});

test("rejects malformed model ids with a typed error", () => {
  for (const model of ["gpt-4.1-mini", "/gpt", "9bad/model", "UPPER CASE/model"]) {
    try {
      providerForModel(model);
      throw new Error(`expected ${model} to be rejected`);
    } catch (error) {
      expect(error).toBeInstanceOf(RouterCoreError);
      expect((error as RouterCoreError).code).toBe("unsupported_model");
      expect((error as RouterCoreError).kind).toBe("invalid_request");
    }
  }
});

test("canonicalizes only the provider segment", () => {
  expect(canonicalModelId("OpenAI/gpt-4.1-mini")).toBe("openai/gpt-4.1-mini");
  expect(canonicalModelId("fireworks/deepseek-ai/DeepSeek-V4-Pro")).toBe(
    "fireworks/deepseek-ai/DeepSeek-V4-Pro",
  );
  expect(canonicalModelId("no-slash")).toBe("no-slash");
});
