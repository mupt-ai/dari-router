import { RouterCoreError } from "./errors.js";
import type { Provider } from "./types.js";

const PROVIDER_PATTERN = /^[a-z][a-z0-9_.-]{0,119}$/;

// Legacy inference for provider-prefixed model ids. Canonical model ids may
// instead name their owner; explicit provider metadata must win when present.
export function providerForModel(model: string): Provider {
  const slash = model.indexOf("/");
  if (slash <= 0) {
    throw new RouterCoreError("invalid_request", `Unsupported model provider for ${model}`, "unsupported_model", "model");
  }
  const provider = model.slice(0, slash).toLowerCase();
  if (!PROVIDER_PATTERN.test(provider)) {
    throw new RouterCoreError("invalid_request", `Unsupported model provider for ${model}`, "unsupported_model", "model");
  }
  return provider;
}

// Legacy companion to providerForModel for ids whose first segment is the
// execution provider. Provider-neutral declarations use providerModelId.
export function nativeModelId(model: string): string {
  return model.slice(`${providerForModel(model)}/`.length);
}

// providerForModel() lowercases the provider prefix, so lookups keyed on
// model ids must use the same canonical spelling; the model segment stays
// case-sensitive (native ids can be), matching providerForModel semantics.
export function canonicalModelId(modelId: string): string {
  const slash = modelId.indexOf("/");
  if (slash <= 0) return modelId;
  return modelId.slice(0, slash).toLowerCase() + modelId.slice(slash);
}
