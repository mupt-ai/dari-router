// Dari Auto Router — use Dari's hosted routing model as a service.
//
// This is the simplest way to use @mupt-ai/dari-router: declare your models,
// drop createAutoRouter in as the policy, and Dari's routing model picks
// the best model for each request. You're billed per selection call.
//
// Requires DARI_API_KEY, OPENAI_API_KEY, and ANTHROPIC_API_KEY.

import { createAutoRouter, createPiRuntime, createRouter } from "../src/index.js";

const pi = await createPiRuntime({
  apiKey: ({ provider }) => {
    const key = process.env[`${provider.toUpperCase()}_API_KEY`];
    if (!key) throw new Error(`Missing ${provider.toUpperCase()}_API_KEY`);
    return key;
  },
});

const router = createRouter({
  models: [
    pi.model("openai/gpt-5.6-sol"),
    pi.model("anthropic/claude-sonnet-5"),
  ],
  policy: createAutoRouter({ apiKey: process.env.DARI_API_KEY! }),
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
console.log(await response.json());
