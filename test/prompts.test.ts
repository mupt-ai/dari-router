import { expect, test } from "bun:test";

import {
  ANONYMOUS_ACTION_SYSTEM_PROMPT,
  CUSTOM_SELECTOR_SYSTEM_PROMPT,
  SELECTOR_SYSTEM_PROMPT,
} from "../src/prompts.js";

test("selector prompts explain that missing benchmark scores are not negative evidence", () => {
  const guidance =
    "A missing benchmark score means that candidate pair was not evaluated on that benchmark; do not treat it as zero, failure, or negative evidence.";

  expect(SELECTOR_SYSTEM_PROMPT).toContain(guidance);
  expect(CUSTOM_SELECTOR_SYSTEM_PROMPT).toContain(guidance);
});

test("selector prompts define benchmark standing over scored candidate actions", () => {
  const namedGuidance =
    "Rank is each candidate action's standing among the scored candidate actions on that benchmark, and z_score uses that same candidate group.";

  expect(SELECTOR_SYSTEM_PROMPT).toContain(namedGuidance);
  expect(CUSTOM_SELECTOR_SYSTEM_PROMPT).toContain(namedGuidance);
  expect(ANONYMOUS_ACTION_SYSTEM_PROMPT).toContain(
    "Rank is the action's position among the scored candidate actions on that benchmark, 1 being best.",
  );
});

test("anonymous action prompt explains missing benchmark scores for actions", () => {
  expect(ANONYMOUS_ACTION_SYSTEM_PROMPT).toContain(
    "A missing benchmark score means that action was not evaluated on that benchmark; do not treat it as zero, failure, or negative evidence.",
  );
});
