#!/usr/bin/env bash
# Fails when a size in target/sizes.json exceeds reports/sizes.json by more than the margin, or
# when either file has a key the other lacks. Improvements pass with a reminder: a baseline
# update is its own commit, so the history shows when and why a number moved.
set -euo pipefail
cd "$(dirname "$0")/.."

margin_percent=3
baseline=reports/sizes.json
current=target/sizes.json
status=0

for key in $(jq -r 'keys[]' "$baseline"); do
  base=$(jq -r --arg k "$key" '.[$k]' "$baseline")
  cur=$(jq -r --arg k "$key" '.[$k] // empty' "$current")
  if [ -z "$cur" ]; then
    echo "missing: $key not in $current"
    status=1
    continue
  fi
  slack=$(((base * margin_percent + 99) / 100))
  if [ "$cur" -gt $((base + slack)) ]; then
    echo "regression: $key = $cur bytes, limit $((base + slack)) (baseline $base)"
    status=1
  elif [ "$cur" -lt $((base - slack)) ]; then
    echo "improved: $key = $cur bytes, baseline $base; run 'just sizes-baseline' to lock it in"
  else
    echo "ok: $key = $cur bytes (baseline $base)"
  fi
done

for key in $(jq -r 'keys[]' "$current"); do
  if ! jq -e --arg k "$key" 'has($k)' "$baseline" >/dev/null; then
    echo "untracked: $key in $current but not in $baseline"
    status=1
  fi
done

exit $status
