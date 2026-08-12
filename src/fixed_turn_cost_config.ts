// Loop horizons every candidate is projected at, ascending. These match the
// lease lengths the selector can commit to, so every option is priced at
// exactly the horizon its action would hold for — the longer horizons are
// where cache behavior and output pricing separate models that look alike on
// a single turn.
export const FIXED_TURN_COST_PROJECTED_TURNS = [5, 10, 30] as const;

// Turn commitments the selector may choose between: how many turns the routed
// model keeps serving before the router reconsiders. Committing amortizes the
// cold start it pays on the first turn across the warm turns that follow.
export const SELECTOR_LEASE_TURNS = [5, 10, 30] as const;

// The horizon used wherever a single scalar cost is needed rather than the
// whole curve, notably warm-incumbent pruning.
export const FIXED_TURN_COST_COMPARISON_TURNS = 10;

export const FIXED_TURN_CACHE_HIT_PROBABILITY = {
  openai: 0.95,
  anthropic: 0.99,
  fireworks: 0.88,
  // 48/50 follow-up requests hit in the July 2026 25-agent Pi benchmark.
  // First turns are excluded because fixed-turn turn one uses live warmth.
  meta: 0.96,
} as const;
