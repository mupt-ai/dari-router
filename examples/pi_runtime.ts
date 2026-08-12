// A real end-to-end router using the built-in Pi AI runtime.
//
// Run with:
//   OPENAI_API_KEY=... bun run example:pi

import {
  createPiRuntime,
  createRouter,
  type RoutingPolicy,
} from "../src/index.js";

const apiKey = process.env.OPENAI_API_KEY;
if (!apiKey) throw new Error("OPENAI_API_KEY is required.");

const pi = await createPiRuntime({ apiKey });
const models = [
  pi.model("openai/gpt-5.4-mini"),
  pi.model("openai/gpt-5.4"),
];
const policy: RoutingPolicy = ({ request, candidates }) => {
  const promptLength = request.items.reduce((length, item) =>
    item.type === "message"
      ? length + item.content.reduce(
          (total, part) => total + (part.type === "text" ? part.text.length : 0),
          0,
        )
      : length,
  0);
  const preferred = promptLength > 500
    ? "openai/gpt-5.4"
    : "openai/gpt-5.4-mini";
  return {
    model: candidates.find((candidate) => candidate.id === preferred)?.id
      ?? candidates[0]!.id,
    reason: promptLength > 500
      ? "Use the stronger model for the longer prompt."
      : "Use the smaller model for the short prompt.",
  };
};
const router = createRouter({ models, policy, executor: pi });

const response = await router.fetch(new Request(
  "http://router.local/v1/chat/completions",
  {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "my-router",
      messages: [{ role: "user", content: "Explain LLM routing in one sentence." }],
    }),
  },
));

if (!response.ok) throw new Error(await response.text());
console.log(await response.json());
