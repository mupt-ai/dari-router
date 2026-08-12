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

test("anonymous action prompt explains missing benchmark scores for actions", () => {
  expect(ANONYMOUS_ACTION_SYSTEM_PROMPT).toContain(
    "A missing benchmark score means that action was not evaluated on that benchmark; do not treat it as zero, failure, or negative evidence.",
  );
});
