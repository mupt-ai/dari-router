// Jev Router — TypeSafe's Jev model as the routing policy for a self-hosted
// createRouter, in place of Dari's trained routing policy. Selection runs against
// TypeSafe's API with the caller's own TypeSafe key.

import {
  createDariRoutingPolicyInternal,
  type DariRoutingPolicyOptions,
} from "./dari_policy.js";
import type { RoutingPolicy } from "./framework_types.js";
import {
  createJevSelector,
  JEV_SELECTOR_CONTEXT_WINDOW_CHARS,
  resolveJevSelectorConfig,
  type JevSelectorConfig,
} from "./jev_selector.js";

export type CreateJevRouterOptions<Metadata = unknown> = JevSelectorConfig & Omit<
  DariRoutingPolicyOptions<Metadata>,
  | "selector"
  | "runtime"
  | "selectorModel"
  | "selectorContextWindowChars"
  | "strategy"
  | "customConfig"
>;

export function createJevRouter<Metadata = unknown>(
  options: CreateJevRouterOptions<Metadata>,
): RoutingPolicy<Metadata> {
  const {
    apiKey,
    endpoint,
    model,
    leaseTurns,
    timeoutMs,
    ...policyOptions
  } = options;
  const config = resolveJevSelectorConfig({
    apiKey,
    ...(endpoint === undefined ? {} : { endpoint }),
    ...(model === undefined ? {} : { model }),
    ...(leaseTurns === undefined ? {} : { leaseTurns }),
    ...(timeoutMs === undefined ? {} : { timeoutMs }),
  });
  return createDariRoutingPolicyInternal<Metadata>({
    ...policyOptions,
    selectorModel: `typesafe/${config.model}`,
    selectorContextWindowChars: JEV_SELECTOR_CONTEXT_WINDOW_CHARS,
    selector: createJevSelector(config),
  }, true);
}
