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

// Serving-provider adapters that execute a canonical model owner's models
// retain the owner's tokenizer, prompt-cache, and cost family. Warm-prefix
// provenance is checked separately against the actual serving provider, so
// an alias here never lets one provider claim another's cached prefix.
// Returns null when the serving provider is not an alias.
function aliasFamilyProvider(model: string, provider: string): string | null {
  if (provider === "openai-codex") return "openai";
  if (provider !== "amazon-bedrock") return null;
  const slash = canonicalModelId(model).indexOf("/");
  if (slash <= 0) return null;
  const owner = canonicalModelId(model).slice(0, slash).toLowerCase();
  return owner === "openai" || owner === "anthropic" ? owner : null;
}

// The provider whose family behavior (tokenizer, prompt cache, cost class)
// a model follows: the serving provider itself, or the canonical model owner
// when serving through a provider alias. Callers that know the serving
// provider pass it; otherwise the model id's prefix decides, and an id with
// no parseable prefix is an error, not a silently generic model.
export function modelFamilyProvider(model: string, provider?: string): string {
  const resolved = (provider ?? providerForModel(model)).toLowerCase();
  const aliased = aliasFamilyProvider(model, resolved);
  return aliased ?? resolved;
}
