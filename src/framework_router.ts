import { RouterFrameworkError } from "./framework_error.js";
import { eligibleRouterModels, normalizeRouterModels } from "./framework_models.js";
import {
  anthropicRequest,
  anthropicResponse,
} from "./protocol_anthropic.js";
import {
  openAIChatRequest,
  openAIChatResponse,
} from "./protocol_openai_chat.js";
import {
  primeStream,
  streamResponse,
  type StartedExecution,
} from "./framework_streaming.js";
import {
  cancelled,
  closeAsyncIterable,
  closeAsyncIterator,
  errorMessage,
  executorStartError,
  normalizeError,
  safeHook,
  throwIfAborted,
} from "./framework_lifecycle.js";
import {
  validateCompletion,
  validatePolicyDecision,
} from "./framework_validation.js";
import type { ReasoningEffort } from "./types.js";
import type {
  CreateRouterOptions,
  LeaseStore,
  Router,
  RouterCandidate,
  RouterExecutor,
  RouterExecutorResult,
  RouterLease,
  RouterProtocol,
  RouterRequest,
  RouterSelection,
  RoutingPolicyDecision,
} from "./framework_types.js";

export function createRouter<Metadata = unknown>(
  options: CreateRouterOptions<Metadata>,
): Router<Metadata> {
  if (
    typeof options.policy !== "function" &&
    (!options.policy || typeof options.policy.select !== "function")
  ) {
    throw new RouterFrameworkError(
      "configuration",
      "policy must be a function or define a select function.",
      "policy_invalid",
    );
  }
  if (options.generateId !== undefined && typeof options.generateId !== "function") {
    throw new RouterFrameworkError(
      "configuration",
      "generateId must be a function.",
      "id_generator_invalid",
    );
  }
  const defaultExecutorName = "default";
  if (options.executors && Object.hasOwn(options.executors, defaultExecutorName)) {
    throw new RouterFrameworkError(
      "configuration",
      `executors cannot use the reserved name '${defaultExecutorName}'.`,
      "executor_name_reserved",
    );
  }
  const executors: Record<string, RouterExecutor<Metadata>> = {
    ...(options.executor === undefined ? {} : { [defaultExecutorName]: options.executor }),
    ...(options.executors ?? {}),
  };
  const modelDeclarations = options.models.map((model) => {
    if (model.executor !== undefined) return model;
    if (options.executor === undefined) {
      throw new RouterFrameworkError(
        "configuration",
        `Model '${model.id}' does not name an executor and no default executor was supplied.`,
        "executor_missing",
      );
    }
    return { ...model, executor: defaultExecutorName };
  });
  const models = Object.freeze(normalizeRouterModels(modelDeclarations, executors));
  // In-memory lease commitments keyed by request cacheKey. When a policy
  // returns leaseTurnsRemaining > 0, the router serves the same model for
  // subsequent requests sharing the cacheKey without calling the policy
  // again, decrementing the remaining turns each time. Leases dissolve when
  // the leased model is no longer an eligible candidate, when their turns run
  // out, or when they sit unused past LEASE_TTL_MS (abandoned conversations
  // must not accumulate).
  const LEASE_TTL_MS = 30 * 60_000;
  // Default in-memory lease store; callers can inject a persistent store.
  const leaseStore: LeaseStore = options.leaseStore ?? inMemoryLeaseStore();
  // Store calls are advisory: an injected store that throws must never fail the request.
  const tryLeaseStore = async <T>(op: () => T | Promise<T>): Promise<T | undefined> => {
    try {
      return await op();
    } catch {
      return undefined;
    }
  };
  const hooks = options.hooks ?? {};
  const fallbackConfig = options.fallback ?? {};
  const idGenerator = options.generateId ?? (() => globalThis.crypto.randomUUID());
  const generateId = (): string => {
    const id = idGenerator();
    if (typeof id !== "string" || !id.trim()) {
      throw new RouterFrameworkError(
        "configuration",
        "generateId must return a non-empty string.",
        "id_generator_invalid",
      );
    }
    return id;
  };

  const selectWithoutLeases = async (
    request: RouterRequest,
    signal: AbortSignal = new AbortController().signal,
  ): Promise<RouterSelection<Metadata>> => {
    throwIfAborted(signal);
    const candidates = eligibleRouterModels(request, models);
    let chosen: RoutingPolicyDecision;
    try {
      chosen = await (typeof options.policy === "function"
        ? options.policy({ request, candidates, signal })
        : options.policy.select({ request, candidates, signal }));
    } catch (error) {
      if (signal.aborted) throw cancelled(signal.reason);
      if (error instanceof RouterFrameworkError) throw error;
      throw new RouterFrameworkError(
        "policy",
        errorMessage(error, "Routing policy failed."),
        "policy_failed",
        undefined,
        { cause: error },
      );
    }
    throwIfAborted(signal);
    return validatePolicyDecision(chosen, candidates);
  };

  // Authoritative serving selection: consumes and commits leases keyed by
  // cacheKey. Stateless inspection uses evaluatePolicy/selectWithoutLeases.
  const select = async (
    request: RouterRequest,
    signal: AbortSignal = new AbortController().signal,
  ): Promise<RouterSelection<Metadata>> => {
    throwIfAborted(signal);
    const candidates = eligibleRouterModels(request, models);

    // Lease short-circuit: if a prior decision committed turns to a model that
    // is still eligible for this request, serve it without calling the policy.
    const leaseKey = leaseKeyFor(request);
    if (leaseKey !== undefined) {
      const now = Date.now();
      await tryLeaseStore(() => leaseStore.pruneExpired(now));
      const lease = await tryLeaseStore(() => leaseStore.get(leaseKey));
      if (lease !== undefined) {
        // Defensive expiry check: pruneExpired is advisory and injected
        // stores may not implement it strictly.
        const candidate = lease.expiresAt > now
          ? candidates.find(
              (item) => item.id === lease.model && item.reasoningEfforts.includes(lease.reasoningEffort),
            )
          : undefined;
        if (candidate !== undefined) {
          const turnsRemaining = lease.turnsRemaining - 1;
          if (turnsRemaining <= 0) await tryLeaseStore(() => leaseStore.delete(leaseKey));
          else {
            await tryLeaseStore(() =>
              leaseStore.set(leaseKey, {
                ...lease,
                turnsRemaining,
                expiresAt: now + LEASE_TTL_MS,
              })
            );
          }
          const selection = {
            decision: {
              selectedModel: candidate.id,
              reasoningEffort: lease.reasoningEffort,
              reason: `Serving committed lease (${turnsRemaining <= 0 ? "final" : `${turnsRemaining} left`}).`,
              ...(turnsRemaining <= 0 ? {} : { leaseTurnsRemaining: turnsRemaining }),
            },
            candidates,
          };
          safeHook(() => hooks.onSelection?.(selection, request));
          return selection;
        }
        await tryLeaseStore(() => leaseStore.delete(leaseKey));
      }
    }

    const selection = await selectWithoutLeases(request, signal);

    // Record a new lease when the policy committed turns and the request
    // carries a cacheKey to correlate subsequent turns.
    if (
      leaseKey !== undefined &&
      selection.decision.leaseTurnsRemaining !== undefined &&
      selection.decision.leaseTurnsRemaining > 0
    ) {
      await tryLeaseStore(() =>
        leaseStore.set(leaseKey, {
          model: selection.decision.selectedModel,
          reasoningEffort: selection.decision.reasoningEffort,
          turnsRemaining: selection.decision.leaseTurnsRemaining!,
          expiresAt: Date.now() + LEASE_TTL_MS,
        })
      );
    }

    safeHook(() => hooks.onSelection?.(selection, request));
    return selection;
  };

  const fetch = async (request: Request): Promise<Response> => {
    const protocol = protocolFor(request);
    if (protocol === null) {
      return errorResponse(
        "openai_chat_completions",
        new RouterFrameworkError("not_found", "Not found.", "not_found"),
      );
    }
    if (request.method !== "POST") {
      return Response.json(
        protocol === "anthropic_messages"
          ? {
              type: "error",
              error: { type: "invalid_request_error", message: "Method not allowed. Use POST." },
            }
          : {
              error: {
                message: "Method not allowed. Use POST.",
                type: "invalid_request_error",
                code: "method_not_allowed",
              },
            },
        { status: 405, headers: { Allow: "POST" } },
      );
    }

    const controller = new AbortController();
    const abort = () => controller.abort(request.signal.reason);
    if (request.signal.aborted) abort();
    else request.signal.addEventListener("abort", abort, { once: true });

    let streamOwnsAbortListener = false;
    let effectiveSelection: RouterSelection<Metadata> | undefined;
    try {
      const payload = await parseJson(request);
      const normalized = protocol === "openai_chat_completions"
        ? openAIChatRequest(payload)
        : anthropicRequest(payload);
      const selection = await select(normalized, controller.signal);
      effectiveSelection = selection;
      const model = selection.candidates.find(
        (candidate) => candidate.id === selection.decision.selectedModel,
      )!;
      // eligibleRouterModels guarantees each candidate's default effort is in
      // its (filtered) reasoningEfforts.
      const effectiveEffort = (candidate: RouterCandidate<Metadata>): ReasoningEffort =>
        candidate.reasoningEfforts.includes(selection.decision.reasoningEffort)
          ? selection.decision.reasoningEffort
          : candidate.defaultReasoningEffort;
      // Primes streams here: first-pull provider failures must surface inside
      // the fallback-covered region, before anything reaches the client.
      const startWithModel = async (
        candidate: RouterCandidate<Metadata>,
      ): Promise<StartedExecution> => {
        const attempt: RouterExecutorResult = await executors[candidate.executor]!.execute({
          request: normalized,
          model: candidate,
          decision: {
            ...selection.decision,
            selectedModel: candidate.id,
            reasoningEffort: effectiveEffort(candidate),
          },
          signal: controller.signal,
        });
        if (normalized.stream) {
          if (attempt.type !== "stream") {
            throw new RouterFrameworkError(
              "executor",
              `Executor '${candidate.executor}' returned a non-streaming result for a streaming request.`,
              "executor_mode_mismatch",
            );
          }
          return { type: "stream", primed: await primeStream(attempt.events, controller) };
        }
        if (attempt.type !== "complete") {
          await closeAsyncIterable(attempt.events);
          throw new RouterFrameworkError(
            "executor",
            `Executor '${candidate.executor}' returned a stream for a non-streaming request.`,
            "executor_mode_mismatch",
          );
        }
        validateCompletion(attempt.output);
        return { type: "complete", output: attempt.output };
      };
      const leaseKey = leaseKeyFor(normalized);
      let started: StartedExecution;
      let executedModel = model;
      try {
        started = await startWithModel(model);
      } catch (error) {
        if (controller.signal.aborted) throw cancelled(controller.signal.reason);
        const startError = executorStartError(error, model.executor);
        // Pure cancellation keeps the lease: the user aborted, the model did
        // not fail.
        if (startError.kind === "cancelled") throw startError;
        // Drop before the fallback attempt: any lease pins the failing primary.
        if (leaseKey !== undefined) await tryLeaseStore(() => leaseStore.delete(leaseKey));
        // Fallback: try a different model if configured and available.
        const fallback = fallbackCandidate(selection, model, fallbackConfig);
        if (fallback === undefined) throw startError;
        try {
          started = await startWithModel(fallback);
        } catch (fallbackError) {
          if (controller.signal.aborted) throw cancelled(controller.signal.reason);
          throw executorStartError(fallbackError, fallback.executor);
        }
        executedModel = fallback;
        effectiveSelection = {
          ...selection,
          decision: {
            selectedModel: fallback.id,
            reasoningEffort: effectiveEffort(fallback),
            reason: `Fell back to '${fallback.id}' after executor for '${selection.decision.selectedModel}' failed. Original reason: ${selection.decision.reason}`,
          },
        };
      }

      const served = effectiveSelection;

      if (started.type === "stream") {
        const primedStream = started.primed;
        try {
          throwIfAborted(controller.signal);
          const response = streamResponse({
            protocol,
            id: generateId(),
            model: executedModel.id,
            requestedModel: normalized.requestedModel,
            selection: served,
            primed: primedStream,
            controller,
            headers: selectionHeaders(served),
            // Not awaited: stream teardown must not stall on the store; turns/TTL backstop.
            onStreamFailure: leaseKey === undefined ? undefined : () => {
              void tryLeaseStore(() => leaseStore.delete(leaseKey));
            },
            onFinalize: (completion, error) => {
              request.signal.removeEventListener("abort", abort);
              if (hooks.onStreamClose) {
                safeHook(() => hooks.onStreamClose!(completion, served, error));
              }
            },
          });
          streamOwnsAbortListener = true;
          return response;
        } catch (error) {
          await closeAsyncIterator(primedStream.iterator);
          throw error;
        }
      }
      throwIfAborted(controller.signal);
      const id = generateId();
      const output = started.output;
      safeHook(() => hooks.onCompletion?.(output, served));
      return Response.json(
        protocol === "openai_chat_completions"
          ? openAIChatResponse({
              id,
              model: executedModel.id,
              requestedModel: normalized.requestedModel,
              output,
              selection: served,
            })
          : anthropicResponse({
              id,
              model: executedModel.id,
              requestedModel: normalized.requestedModel,
              output,
              selection: served,
            }),
        { headers: selectionHeaders(served) },
      );
    } catch (error) {
      const normalizedError = request.signal.aborted
        ? cancelled(request.signal.reason)
        : normalizeError(error);
      const errorSelection = effectiveSelection;
      if (errorSelection !== undefined) {
        safeHook(() => hooks.onError?.(normalizedError, errorSelection));
      }
      return errorResponse(protocol, normalizedError);
    } finally {
      if (!streamOwnsAbortListener) request.signal.removeEventListener("abort", abort);
    }
  };

  return { models, select, evaluatePolicy: selectWithoutLeases, fetch };
}
function protocolFor(request: Request): RouterProtocol | null {
  const path = new URL(request.url).pathname.replace(/\/+$/, "");
  if (path === "/v1/chat/completions" || path === "/chat/completions") {
    return "openai_chat_completions";
  }
  if (path === "/v1/messages" || path === "/messages") return "anthropic_messages";
  return null;
}

async function parseJson(request: Request): Promise<unknown> {
  try {
    return await request.json();
  } catch (error) {
    throw new RouterFrameworkError(
      "invalid_request",
      "Request body is not valid JSON.",
      "invalid_json",
      undefined,
      { cause: error },
    );
  }
}

function selectionHeaders<Metadata>(selection: RouterSelection<Metadata>): Headers {
  return new Headers({
    "X-Router-Selected-Model": selection.decision.selectedModel,
    "X-Router-Reasoning-Effort": selection.decision.reasoningEffort,
  });
}

function errorResponse(protocol: RouterProtocol, error: RouterFrameworkError): Response {
  if (protocol === "anthropic_messages") {
    return Response.json(
      {
        type: "error",
        error: {
          type: error.status >= 500 ? "api_error" : "invalid_request_error",
          message: error.message,
        },
      },
      { status: error.status },
    );
  }
  return Response.json(
    {
      error: {
        message: error.message,
        type: error.status >= 500 ? "server_error" : "invalid_request_error",
        code: error.code,
        ...(error.param === undefined ? {} : { param: error.param }),
      },
    },
    { status: error.status },
  );
}

function fallbackCandidate<Metadata>(
  selection: RouterSelection<Metadata>,
  primary: RouterCandidate<Metadata>,
  config: { enabled?: boolean; requiresDifferentProvider?: boolean },
): RouterCandidate<Metadata> | undefined {
  if (!config.enabled) return undefined;
  return selection.candidates.find((candidate) => {
    if (candidate.id === primary.id) return false;
    if (config.requiresDifferentProvider && candidate.provider === primary.provider) return false;
    return true;
  });
}

// Protocol adapters normalize empty cache keys away, but programmatic
// callers can still pass "": treat it as no cache key so unrelated requests
// never share a lease bucket.
function leaseKeyFor(request: RouterRequest): string | undefined {
  return request.cacheKey !== undefined && request.cacheKey.trim()
    ? request.cacheKey
    : undefined;
}

function inMemoryLeaseStore(): LeaseStore {
  const leases = new Map<string, RouterLease>();
  return {
    get: (cacheKey) => leases.get(cacheKey),
    set: (cacheKey, lease) => {
      leases.set(cacheKey, lease);
    },
    delete: (cacheKey) => {
      leases.delete(cacheKey);
    },
    pruneExpired: (nowMs) => {
      for (const [cacheKey, lease] of leases) {
        if (lease.expiresAt <= nowMs) leases.delete(cacheKey);
      }
    },
  };
}
