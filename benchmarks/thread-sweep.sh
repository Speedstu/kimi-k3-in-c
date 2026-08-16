#!/usr/bin/env bash
# Sweep the real K3 workload across OpenMP thread counts instead of guessing from cores.
#
# Usage:
#   benchmarks/thread-sweep.sh MODEL_DIR [normal k3 options ...]
#
# Example:
#   K3_SWEEP_REPEATS=3 benchmarks/thread-sweep.sh ~/k3model \
#     --trunk ~/k3trunk --preset laptop --incremental --ids 1,2,3 --gen 4
#
# Optional environment:
#   K3_BIN=./bin/k3                  executable (default ./bin/k3)
#   K3_SWEEP_REPEATS=3              repeats per candidate (default 2)
#   K3_SWEEP_THREADS="1 2 4 8 12"   exact candidates; default powers of two + CPU count
#   K3_SWEEP_OUT=thread-sweep.tsv    result file; default temporary + final summary stdout
#
# Every run is a normal exact K3 run. --threads changes only OpenMP worker count; kernels
# parallelise independent output rows/heads, so emitted tokens are expected to be identical.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 MODEL_DIR [normal k3 options ...]" >&2
  exit 2
fi

K3_BIN=${K3_BIN:-./bin/k3}
REPEATS=${K3_SWEEP_REPEATS:-2}
if ! [[ $REPEATS =~ ^[1-9][0-9]*$ ]]; then
  echo "K3_SWEEP_REPEATS must be a positive integer" >&2
  exit 2
fi
if [[ ! -x $K3_BIN ]]; then
  echo "K3 binary not executable: $K3_BIN" >&2
  exit 2
fi

MODEL=$1
shift
ARGS=("$@")

if printf '%s\n' "${ARGS[@]}" | grep -qx -- '--threads'; then
  echo "do not pass --threads to the sweep; it controls that option itself" >&2
  exit 2
fi

cpu_count=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)
if ! [[ $cpu_count =~ ^[1-9][0-9]*$ ]]; then cpu_count=1; fi

if [[ -n ${K3_SWEEP_THREADS:-} ]]; then
  read -r -a candidates <<< "$K3_SWEEP_THREADS"
else
  candidates=(1)
  t=2
  while (( t < cpu_count )); do
    candidates+=("$t")
    t=$((t * 2))
  done
  if (( cpu_count > 1 )); then candidates+=("$cpu_count"); fi
fi

# Validate, deduplicate and sort numerically. This also makes captured runs reproducible.
mapfile -t candidates < <(
  printf '%s\n' "${candidates[@]}" |
    awk -v max="$cpu_count" '/^[0-9]+$/ && $1 >= 1 && $1 <= max { print $1 }' |
    sort -nu
)
if (( ${#candidates[@]} == 0 )); then
  echo "no valid thread candidates (machine reports $cpu_count logical CPUs)" >&2
  exit 2
fi

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
out=${K3_SWEEP_OUT:-$work/thread-sweep.tsv}
mkdir -p "$(dirname "$out")"
printf 'threads\trepeat\tseconds_per_token\tgenerated_ids\n' > "$out"

printf 'K3 thread sweep: CPUs=%s, candidates=%s, repeats=%s\n' \
  "$cpu_count" "${candidates[*]}" "$REPEATS"
printf 'Each candidate runs the SAME exact request. Lower median s/token wins.\n\n'

reference_ids=''
for threads in "${candidates[@]}"; do
  printf '%3d threads: ' "$threads"
  vals=()
  for ((r=1; r<=REPEATS; r++)); do
    jf="$work/t${threads}-r${r}.json"
    lf="$work/t${threads}-r${r}.log"
    "$K3_BIN" "$MODEL" "${ARGS[@]}" --threads "$threads" --out "$jf" >"$lf" 2>&1
    read -r sec ids < <(python3 - "$jf" <<'PY'
import json, sys
x = json.load(open(sys.argv[1], encoding="utf-8"))
print(x["seconds_per_token"], ",".join(map(str, x["generated_ids"])))
PY
)
    if [[ -z $reference_ids ]]; then
      reference_ids=$ids
    elif [[ $ids != "$reference_ids" ]]; then
      echo >&2
      echo "REFUSING RESULT: generated ids changed at ${threads} threads, repeat ${r}" >&2
      echo "reference: $reference_ids" >&2
      echo "got      : $ids" >&2
      exit 1
    fi
    vals+=("$sec")
    printf '%s%s' "$sec" "$([[ $r -eq $REPEATS ]] && echo '' || echo ', ')"
    printf '%s\t%s\t%s\t%s\n' "$threads" "$r" "$sec" "$ids" >> "$out"
  done
  median=$(printf '%s\n' "${vals[@]}" | sort -n | awk '{a[NR]=$1} END {if (NR%2) print a[(NR+1)/2]; else printf "%.6f", (a[NR/2]+a[NR/2+1])/2}')
  printf '  median=%s s/token\n' "$median"
done

python3 - "$out" <<'PY'
import csv, statistics, sys
p = sys.argv[1]
rows = list(csv.DictReader(open(p, encoding="utf-8"), delimiter="\t"))
by = {}
for r in rows:
    by.setdefault(int(r["threads"]), []).append(float(r["seconds_per_token"]))
med = {k: statistics.median(v) for k, v in by.items()}
best = min(med, key=med.get)
base = med[min(med)]
print("\nSummary (median):")
for k in sorted(med):
    print(f"  {k:3d} threads  {med[k]:.4f} s/token  {base/med[k]:.3f}x vs {min(med)} thread(s)")
print(f"\nRECOMMENDED: --threads {best}  ({med[best]:.4f} s/token median)")
print("Keep the same storage/preset/prompt when comparing; changing them invalidates the sweep.")
PY

if [[ ${K3_SWEEP_OUT:-} ]]; then
  echo "raw results: $out"
fi
