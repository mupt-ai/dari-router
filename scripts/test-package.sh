#!/usr/bin/env bash
set -euo pipefail

package_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
temporary_dir="$(mktemp -d)"
trap 'rm -rf "$temporary_dir"' EXIT

cd "$package_root"
bun run build
package_archive="$(npm pack --silent --pack-destination "$temporary_dir")"

mkdir "$temporary_dir/consumer"
cd "$temporary_dir/consumer"
npm init --yes >/dev/null
npm install --ignore-scripts "$temporary_dir/$package_archive" >/dev/null

node --input-type=module <<'EOF'
import {
  createThinkingLevelRatios,
} from "@mupt-ai/dari-router/eval-score-imputation";

if (createThinkingLevelRatios([]).size !== 0) {
  throw new Error("Unexpected imputation result from the packed package");
}
EOF

test ! -e node_modules/@mupt-ai/pi-ai
