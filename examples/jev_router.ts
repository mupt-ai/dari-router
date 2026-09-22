// Jev Router — route with TypeSafe's Jev System One model instead of Dari's
// trained routing policy. Jev answers two typed questions per request (which candidate,
// how long a lease) and returns calibrated probabilities; the decision is
// built from those answers in this package.
//
// Requires TYPESAFE_API_KEY, OPENAI_API_KEY, and ANTHROPIC_API_KEY.

import { createJevRouter, createPiRuntime, createRouter } from "../src/index.js";

const pi = await createPiRuntime({
  apiKey: ({ provider }) => {
    const key = process.env[`${provider.toUpperCase()}_API_KEY`];
    if (!key) throw new Error(`Missing ${provider.toUpperCase()}_API_KEY`);
    return key;
  },
});

const SOL = "openai/gpt-5.6-sol";
const SONNET = "anthropic/claude-sonnet-5";

const router = createRouter({
  models: [pi.model(SOL), pi.model(SONNET)],
  policy: createJevRouter({
    apiKey: process.env.TYPESAFE_API_KEY!,
    // Jev routes on cost evidence you supply; these are per-million-token USD
    // prices and observed output-token averages for each enabled pair.
    pricing: (model) => model === SOL
      ? { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 }
      : { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
    averageOutputTokensByModel: {
      [SOL]: { medium: 800 },
      [SONNET]: { medium: 800 },
    },
  }),
  executor: pi,
});

const response = await router.fetch(
  new Request("https://example.test/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "dari/routing",
      messages: [{ role: "user", content: "Explain speculative decoding in one paragraph." }],
    }),
  }),
);

if (!response.ok) throw new Error(await response.text());
console.log(response.headers.get("X-Router-Selected-Model"), await response.json());
