# Dari Router Framework

`@mupt-ai/dari-router` is a pre-1.0 TypeScript framework for serving OpenAI Chat Completions and Anthropic Messages through one endpoint while choosing among multiple models.

It gives you three replaceable pieces:

- **Models** describe candidates and capabilities.
- **Policies** choose an eligible model.
- **Executors** call the chosen model.

The root package also includes a Pi AI executor and a hosted Auto Router policy.

## Install

```bash
npm install @mupt-ai/dari-router
```

The package is ESM and requires Node.js 22.19 or later, or Bun. Pi AI is an
optional peer dependency: install `@mupt-ai/pi-ai` when using
`createPiRuntime`; deterministic subpaths do not install its provider SDKs.

## Minimal router

```ts
import { createRouter, type RouterExecutor } from "@mupt-ai/dari-router";

const executor: RouterExecutor = {
  execute({ model }) {
    return {
      type: "complete",
      output: {
        content: [{ type: "text", text: `Served by ${model.id}` }],
        finishReason: "stop",
      },
    };
  },
};

const router = createRouter({
  models: [{ id: "demo/model", executor: "demo" }],
  executors: { demo: executor },
  policy: ({ candidates }) => ({
    model: candidates[0]!.id,
    reason: "Use the only eligible candidate.",
  }),
});

Bun.serve({ port: 3000, fetch: router.fetch });
```

Send `POST /v1/chat/completions` or `POST /v1/messages`. The response uses the request's protocol and reports the selected model in routing metadata.

## Pi execution

Use the built-in runtime when your models are in Pi's catalog:

```ts
import { createPiRuntime, createRouter } from "@mupt-ai/dari-router";

const pi = await createPiRuntime({ apiKey: process.env.OPENAI_API_KEY! });
const router = createRouter({
  executor: pi,
  models: [pi.model("openai/gpt-5.4-mini"), pi.model("openai/gpt-5.4")],
  policy: ({ candidates }) => ({ model: candidates[0]!.id }),
});
```

Credentials are supplied by your application; the runtime does not read environment variables itself. A canonical model ID identifies the model independently of where it runs. If that ID names the model owner rather than the execution provider, set `provider` to the service that will execute it and `providerModelId` to that service's Pi catalog ID:

```ts
const fireworksModel = pi.model("deepseek-ai/DeepSeek-V4-Pro-0813", {
  provider: "fireworks",
  providerModelId: "accounts/fireworks/models/deepseek-v4-pro-0813",
});
```

For multiple providers, supply a credential callback. It receives the selected provider, model, API, and whether the call is for execution or selection:

```ts
const pi = await createPiRuntime({
  apiKey: ({ provider }) => {
    const key = provider === "fireworks"
      ? process.env.FIREWORKS_API_KEY
      : process.env.OPENAI_API_KEY;
    if (!key) throw new Error(`Missing API key for ${provider}`);
    return key;
  },
});
```

Explicit provider metadata always wins; provider-prefixed IDs keep legacy prefix inference. See the [framework documentation](https://docs.dari.dev/framework/overview). Managed-router YAML manifests for use with the Dari CLI are in [`examples/managed/`](examples/managed/).

## Explicit advanced boundaries

The root API is the end-to-end framework. Deterministic preparation, cache-aware evidence, selector parsing, and anonymous actions are under `/policy-engine`:

```ts
import { prepareRoute, finalizeRoute } from "@mupt-ai/dari-router/policy-engine";
```

Pure OpenAI/Anthropic adapters and continuation-state helpers are under `/protocols`:

```ts
import { openAIChatRequest, anthropicRequest } from "@mupt-ai/dari-router/protocols";
```

Browser-safe benchmark score imputation is available without Pi AI or provider
SDKs:

```ts
import {
  createThinkingLevelRatios,
  resolveRouterEvalScore,
} from "@mupt-ai/dari-router/eval-score-imputation";
```

Most applications should use `createRouter`. Use `router.evaluatePolicy()` for stateless previews and `router.select()` when you need the same lease-aware selection used by `router.fetch()`.

## Development

```bash
bun install --frozen-lockfile
bun run typecheck
bun test
bun run build
```

Apache-2.0. See [LICENSE](LICENSE).
