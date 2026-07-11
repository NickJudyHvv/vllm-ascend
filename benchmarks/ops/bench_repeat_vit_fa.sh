#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run the head_dim=72 production benchmark in independent processes and report
# aggregate performance. Per-process stability is informational: every
# successful, parseable run contributes to the final statistics. Compact output
# is the default; pass ``--verbose`` after N to retain every child benchmark
# line. No log or JSON file is created.
# Usage: ./bench_repeat_vit_fa.sh [N] [--verbose] [benchmark args...], where N
# defaults to 5.
set -uo pipefail

if [ "$#" -gt 0 ]; then
  N="$1"
  shift
else
  N=5
fi
case "$N" in
  ''|*[!0-9]*)
    printf 'N must be an integer >= 5, got %s\n' "$N" >&2
    exit 2
    ;;
esac
if [ "$N" -lt 5 ]; then
  printf 'N must be >= 5 for the performance gate, got %s\n' "$N" >&2
  exit 2
fi

verbose=${VIT_FA_BENCH_VERBOSE:-0}
if [ "${1:-}" = "--verbose" ]; then
  verbose=1
  shift
fi
case "$verbose" in
  0|1) ;;
  *)
    printf 'VIT_FA_BENCH_VERBOSE must be 0 or 1, got %s\n' "$verbose" >&2
    exit 2
    ;;
esac

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

speedups=()
p10s=()
fia_p50s=()
triton_p50s=()
failed=0
noisy=0
header_printed=0

output_mode=compact
if [ "$verbose" -eq 1 ]; then
  output_mode=verbose
fi
printf '# ViT FA repeat: runs=%d output=%s\n' "$N" "$output_mode"

for i in $(seq 1 "$N"); do
  if out=$(python3 "$HERE/benchmark_vit_flash_attention.py" --isolated "$@" 2>&1); then
    rc=0
  else
    rc=$?
  fi
  if [ "$verbose" -eq 1 ]; then
    printf '\n# run %d/%d (isolated, independent process)\n%s\n' "$i" "$N" "$out"
  fi

  if [ "$header_printed" -eq 0 ]; then
    case_line=$(printf '%s\n' "$out" | sed -n '/^seq_lens=/p' | head -1)
    config_line=$(printf '%s\n' "$out" | sed -n '/^Triton .*path=/p' | head -1)
    if [ -n "$case_line" ]; then
      printf 'CASE %s\n' "$case_line"
    fi
    if [ -n "$config_line" ]; then
      printf 'CONFIG %s\n' "$config_line"
    fi
    header_printed=1
  fi

  speedup=$(printf '%s\n' "$out" | sed -n 's/.*epoch speedup (FIA \/ Triton)=\([0-9.]*\)x.*/\1/p' | head -1)
  p10=$(printf '%s\n' "$out" | sed -n 's/.*epoch p10-p90=\([0-9.]*\)-[0-9.]*x.*/\1/p' | head -1)
  fia_p50=$(printf '%s\n' "$out" | sed -n 's/^FIA .*p50=\([0-9.]*\) ms.*/\1/p' | head -1)
  triton_p50=$(printf '%s\n' "$out" | sed -n 's/^Triton:.*p50=\([0-9.]*\) ms.*/\1/p' | head -1)
  unstable=0
  if [[ "$out" == *UNSTABLE* ]]; then
    unstable=1
  fi

  if [ "$rc" -ne 0 ] || [ -z "$speedup" ] || \
     [ -z "$p10" ] || [ -z "$fia_p50" ] || [ -z "$triton_p50" ]; then
    failed=$((failed + 1))
    printf 'RUN %d/%d INVALID rc=%d parsed=%s/%s/%s/%s\n' \
      "$i" "$N" "$rc" "$speedup" "$p10" "$fia_p50" "$triton_p50" >&2
    if [ "$verbose" -eq 0 ]; then
      printf '%s\n' "$out" | tail -20 >&2
    fi
    continue
  fi

  if [ "$unstable" -ne 0 ]; then
    noisy=$((noisy + 1))
  fi
  absolute_ratio=$(awk -v fia="$fia_p50" -v triton="$triton_p50" \
    'BEGIN { if (triton == 0) print "nan"; else printf "%.4f", fia / triton }')
  printf 'RUN %d/%d OK noisy=%d FIA=%sms Triton=%sms abs=%sx epoch=%sx p10=%sx\n' \
    "$i" "$N" "$unstable" "$fia_p50" "$triton_p50" "$absolute_ratio" "$speedup" "$p10"
  speedups+=("$speedup")
  p10s+=("$p10")
  fia_p50s+=("$fia_p50")
  triton_p50s+=("$triton_p50")
done

python3 - \
  "${speedups[@]}" "::" "${p10s[@]}" "::" \
  "${fia_p50s[@]}" "::" "${triton_p50s[@]}" "::" "$failed" "$noisy" "$N" <<'PY'
import statistics
import sys

parts = []
current = []
for arg in sys.argv[1:]:
    if arg == "::":
        parts.append(current)
        current = []
    else:
        current.append(arg)
parts.append(current)

speedups = [float(value) for value in parts[0]]
p10s = [float(value) for value in parts[1]]
fia_p50s = [float(value) for value in parts[2]]
triton_p50s = [float(value) for value in parts[3]]
failed = int(parts[4][0])
noisy = int(parts[4][1])
expected = int(parts[4][2])

valid = len(speedups)
if not (valid == len(p10s) == len(fia_p50s) == len(triton_p50s)):
    print("RESULT: FAIL inconsistent parsed result counts")
    raise SystemExit(1)
if failed or valid != expected:
    print(f"RESULT: FAIL valid={valid}/{expected} invalid={failed}")
    raise SystemExit(1)

absolute_ratios = [fia / triton for fia, triton in zip(fia_p50s, triton_p50s)]
epoch_speedup_median = statistics.median(speedups)
worst_p10 = min(p10s)
target = "MET" if epoch_speedup_median >= 1.20 and worst_p10 >= 1.15 else "MISS"

print(
    f"RESULT: PASS runs={valid}/{expected} noisy={noisy} "
    f"epoch_speedup_median={epoch_speedup_median:.4f} "
    f"epoch_speedup_range={min(speedups):.4f}-{max(speedups):.4f} "
    f"worst_p10={worst_p10:.4f} "
    f"absolute_p50_ratio_median={statistics.median(absolute_ratios):.4f} "
    f"absolute_p50_ratio_range={min(absolute_ratios):.4f}-{max(absolute_ratios):.4f} "
    f"target={target}(median>=1.20_and_worst_p10>=1.15)"
)
raise SystemExit(0)
PY
