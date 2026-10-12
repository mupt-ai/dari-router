// Selector system prompts live in prompts/*.md so prompt review never requires
// reading request-assembly code. Per-request candidate data rides in the user
// message, keeping these byte-stable for provider prompt caching.

import { readFileSync } from "node:fs";

import {
  FIXED_TURN_COST_COMPARISON_TURNS,
  FIXED_TURN_COST_PROJECTED_TURNS,
} from "./fixed_turn_cost_config.js";

// Placeholders interpolated where a prompt names the projection horizons, so
// the wording cannot drift from the config that produces the data.
const PROMPT_VARIABLES: Record<string, string> = {
  projected_turns: FIXED_TURN_COST_PROJECTED_TURNS.join(", "),
  comparison_turns: String(FIXED_TURN_COST_COMPARISON_TURNS),
};

function loadPrompt(name: string): string {
  const template = readFileSync(new URL(`./prompts/${name}.md`, import.meta.url), "utf8");
  return template.trimEnd().replace(/\{\{(\w+)\}\}/g, (_, key: string) => {
    const value = PROMPT_VARIABLES[key];
    if (value === undefined) throw new Error(`Unknown placeholder {{${key}}} in prompts/${name}.md`);
    return value;
  });
}

export const SELECTOR_SYSTEM_PROMPT = loadPrompt("selector");

export const CUSTOM_SELECTOR_SYSTEM_PROMPT = loadPrompt("custom_selector");

// System prompt for the anonymous candidate-action protocol (see
// anonymous_actions.ts), shared by GRPO training and the deployed policy. It
// carries every invariant instruction, including how to read the per-action
// blocks, so the user message stays pure data.
export const ANONYMOUS_ACTION_SYSTEM_PROMPT = loadPrompt("anonymous_action");
