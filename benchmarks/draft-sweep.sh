#!/usr/bin/env bash
# Measure probability-correct K3 draft settings on the REAL requested workload.
#
#   benchmarks/draft-sweep.sh MODEL_DIR DRAFT_TRUNK [normal bin/k3 arguments...]
#
# Example:
#   K3_DRAFT_SWEEP_REPEATS=3 benchmarks/draft-sweep.sh ~/k3model ~/k3draft-q4 \
#     --trunk ~/k3trunk-lossless --preset laptop --incremental --threads 8 \
#     --ids 1008,10484,318,15383,387 --gen 8
#
# The sweep pins temperature/top-p/seed through environment variables so every candidate
# measures the same benchmark profile. Different speculative settings are NOT required to
# emit identical stochastic token sequences: exact speculative sampling preserves the
# target distribution, not a particular coupling of RNG draws across algorithms.
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: $0 MODEL_DIR DRAFT_TRUNK [normal bin/k3 arguments...]" >&2
    exit 2
fi

MODEL=$1
DRAFT=$2
shift 2

BIN=${K3_BIN:-./bin/k3}
TOPKS=${K3_DRAFT_SWEEP_TOPKS:-"2 4 8"}
SPECS=${K3_DRAFT_SWEEP_SPECS:-"2 4 6"}
REPEATS=${K3_DRAFT_SWEEP_REPEATS:-3}
TEMP=${K3_DRAFT_SWEEP_TEMPERATURE:-1}
TOPP=${K3_DRAFT_SWEEP_TOP_P:-1}
SEED=${K3_DRAFT_SWEEP_SEED:-424242}
DRAFT_GB=${K3_DRAFT_SWEEP_GB:-32}
OUT=${K3_DRAFT_SWEEP_OUT:-draft-sweep.tsv}

case $REPEATS in
    ''|*[!0-9]*) echo "K3_DRAFT_SWEEP_REPEATS must be an integer" >&2; exit 2 ;;
esac
if [ "$REPEATS" -lt 1 ]; then
    echo "K3_DRAFT_SWEEP_REPEATS must be >= 1" >&2
    exit 2
fi

TMP=$(mktemp -d "${TMPDIR:-/tmp}/k3-draft-sweep.XXXXXX")
trap 'rm -rf "$TMP"' EXIT

printf 'topk\tspec\trepeat\tseconds_per_token\tacceptance\taccepted\tproposed\tdraft_seconds\tverify_seconds\n' >"$OUT"

echo "K3 exact sampled-spec draft sweep"
echo "  model       : $MODEL"
echo "  draft       : $DRAFT"
echo "  top-k       : $TOPKS"
echo "  spec lengths: $SPECS"
echo "  repeats     : $REPEATS"
echo "  sampling    : temperature=$TEMP top_p=$TOPP seed=$SEED"
echo

for k in $TOPKS; do
    for spec in $SPECS; do
        baseline_ids=""
        for r in $(seq 1 "$REPEATS"); do
            json="$TMP/k${k}-s${spec}-r${r}.json"
            log="$TMP/k${k}-s${spec}-r${r}.log"
            "$BIN" "$MODEL" "$@" \
                --temperature "$TEMP" --top-p "$TOPP" --seed "$SEED" \
                --draft-trunk "$DRAFT" --draft-trunk-gb "$DRAFT_GB" \
                --draft-topk "$k" --spec "$spec" --out "$json" >"$log"

            row=$(python3 - "$json" <<'PY'
import json, sys
x=json.load(open(sys.argv[1]))
ids=','.join(map(str,x['generated_ids']))
print('\t'.join([
    str(x['seconds_per_token']),
    str(x.get('draft_acceptance',0.0)),
    str(x.get('draft_accepted',0)),
    str(x.get('draft_proposed',0)),
    str(x.get('draft_seconds',0.0)),
    str(x.get('verify_seconds',0.0)),
    ids,
]))
PY
)
            IFS=$'\t' read -r spt acc accepted proposed draft_s verify_s ids <<<"$row"
            if [ "$r" -eq 1 ]; then
                baseline_ids=$ids
            elif [ "$ids" != "$baseline_ids" ]; then
                echo "ERROR: k=$k spec=$spec is not deterministic across identical seed/config repeats" >&2
                exit 1
            fi
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$k" "$spec" "$r" "$spt" "$acc" "$accepted" "$proposed" "$draft_s" "$verify_s" >>"$OUT"
            printf '  topk=%-2s spec=%-2s run=%s  %8.4f s/tok  accept=%6.2f%% (%s/%s)\n' \
                "$k" "$spec" "$r" "$spt" "$(python3 -c "print(100*float('$acc'))")" "$accepted" "$proposed"
        done
    done
done

echo
python3 - "$OUT" <<'PY'
import csv, statistics, sys
from collections import defaultdict
path=sys.argv[1]
groups=defaultdict(list)
acc=defaultdict(list)
with open(path,newline='') as f:
    for row in csv.DictReader(f,delimiter='\t'):
        key=(int(row['topk']),int(row['spec']))
        groups[key].append(float(row['seconds_per_token']))
        acc[key].append(float(row['acceptance']))
ranked=[]
for key,vals in groups.items():
    ranked.append((statistics.median(vals), statistics.median(acc[key]), key))
ranked.sort()
print('ranking by median seconds/token:')
for i,(spt,a,(k,s)) in enumerate(ranked,1):
    print(f'  {i}. topk={k:<2} spec={s:<2}  median={spt:.4f} s/tok  median acceptance={a*100:.2f}%')
if ranked:
    spt,a,(k,s)=ranked[0]
    print(f'\nRECOMMENDED: --draft-topk {k} --spec {s}  ({spt:.4f} s/tok median, {a*100:.2f}% acceptance)')
PY

echo "raw measurements: $OUT"
