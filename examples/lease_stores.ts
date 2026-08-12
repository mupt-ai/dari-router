// Shared lease stores for ephemeral or replicated deployments: redis- and
// postgres-backed LeaseStore implementations against minimal structural client
// interfaces. The demo runs on in-file fakes (no external services) to show
// leases surviving a router "restart".
//
// Run with: bun run example:lease-stores

import {
  createRouter,
  type LeaseStore,
  type RouterExecutor,
  type RouterLease,
  type RoutingPolicy,
} from "../src/index.js";

// --- Redis-backed store -----------------------------------------------------
// Expiry is native (PX), so pruneExpired is a no-op.

type RedisLike = {
  get(key: string): Promise<string | null>;
  set(key: string, value: string, options: { PX: number }): Promise<unknown>;
  del(key: string): Promise<unknown>;
};

export function redisLeaseStore(redis: RedisLike, keyPrefix = "router-lease:"): LeaseStore {
  return {
    async get(cacheKey) {
      const raw = await redis.get(`${keyPrefix}${cacheKey}`);
      if (raw === null) return undefined;
      return JSON.parse(raw) as RouterLease;
    },
    async set(cacheKey, lease) {
      const ttlMs = lease.expiresAt - Date.now();
      if (ttlMs <= 0) return;
      await redis.set(`${keyPrefix}${cacheKey}`, JSON.stringify(lease), { PX: ttlMs });
    },
    async delete(cacheKey) {
      await redis.del(`${keyPrefix}${cacheKey}`);
    },
    pruneExpired() {},
  };
}

// --- Postgres-backed store --------------------------------------------------
// Table:
//   create table router_leases (
//     cache_key text primary key,
//     model text not null,
//     reasoning_effort text not null,
//     turns_remaining int not null,
//     expires_at bigint not null
//   );
//
// The router re-checks expiresAt itself before honoring a lease, so a lazy
// pruneExpired (or a cron) is safe. If concurrent same-cacheKey requests
// matter at your scale, do the turn decrement in SQL behind this interface.

type SqlLike = {
  query(text: string, params: unknown[]): Promise<{ rows: Record<string, unknown>[] }>;
};

export function postgresLeaseStore(sql: SqlLike): LeaseStore {
  return {
    async get(cacheKey) {
      const { rows } = await sql.query(
        "select model, reasoning_effort, turns_remaining, expires_at from router_leases where cache_key = $1",
        [cacheKey],
      );
      const row = rows[0];
      if (row === undefined) return undefined;
      return {
        model: String(row.model),
        reasoningEffort: row.reasoning_effort as RouterLease["reasoningEffort"],
        turnsRemaining: Number(row.turns_remaining),
        expiresAt: Number(row.expires_at),
      };
    },
    async set(cacheKey, lease) {
      await sql.query(
        `insert into router_leases (cache_key, model, reasoning_effort, turns_remaining, expires_at)
         values ($1, $2, $3, $4, $5)
         on conflict (cache_key) do update set
           model = excluded.model,
           reasoning_effort = excluded.reasoning_effort,
           turns_remaining = excluded.turns_remaining,
           expires_at = excluded.expires_at`,
        [cacheKey, lease.model, lease.reasoningEffort, lease.turnsRemaining, lease.expiresAt],
      );
    },
    async delete(cacheKey) {
      await sql.query("delete from router_leases where cache_key = $1", [cacheKey]);
    },
    async pruneExpired(nowMs) {
      await sql.query("delete from router_leases where expires_at <= $1", [nowMs]);
    },
  };
}

// --- Demo: leases survive a router restart ----------------------------------

function fakeRedis(): RedisLike {
  const data = new Map<string, { value: string; expiresAt: number }>();
  return {
    async get(key) {
      const entry = data.get(key);
      if (entry === undefined || entry.expiresAt <= Date.now()) return null;
      return entry.value;
    },
    async set(key, value, options) {
      data.set(key, { value, expiresAt: Date.now() + options.PX });
    },
    async del(key) {
      data.delete(key);
    },
  };
}

const policy: RoutingPolicy = ({ candidates }) => ({
  model: candidates[0]!.id,
  reason: "Fresh selection by the policy.",
  leaseTurnsRemaining: 4,
});

const executor: RouterExecutor = {
  execute: ({ decision }) => ({
    type: "complete",
    output: {
      content: [{ type: "text", text: decision.reason }],
      finishReason: "stop",
      usage: { inputTokens: 4, outputTokens: 4 },
    },
  }),
};

const models = [{ id: "demo/model", executor: "demo", reasoningEfforts: ["off" as const] }];
const sharedStore = redisLeaseStore(fakeRedis());
const routerOptions = { models, policy, executors: { demo: executor }, leaseStore: sharedStore };

const ask = (router: { fetch: (request: Request) => Promise<Response> }) =>
  router.fetch(
    new Request("https://example.test/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "dari/routing",
        prompt_cache_key: "conversation-1",
        messages: [{ role: "user", content: "hello" }],
      }),
    }),
  );

const first = createRouter(routerOptions);
const turn1 = await (await ask(first)).json() as { dari_routing: { reason: string } };
console.log("turn 1 (fresh process): ", turn1.dari_routing.reason);

// "Restart": a brand-new router instance sharing only the external store.
const second = createRouter(routerOptions);
const turn2 = await (await ask(second)).json() as { dari_routing: { reason: string } };
console.log("turn 2 (new process):   ", turn2.dari_routing.reason);
