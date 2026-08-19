#!/usr/bin/env bash
set -euo pipefail

binary=${1:-./build/hnsw_threshold}
output=${2:-results.csv}
shift $(( $# >= 2 ? 2 : $# ))

first=1
latency_list=${LATENCIES:-"0 10000 50000 100000 500000 1000000"}
for latency in $latency_list; do
    for scale in 1.0 1.25 1.5 2.0 3.0; do
        if (( first )); then
            "$binary" --latency-ns "$latency" --threshold-scale "$scale" --mode all "$@" > "$output"
            first=0
        else
            "$binary" --latency-ns "$latency" --threshold-scale "$scale" --mode all "$@" | tail -n +2 >> "$output"
        fi
    done
done

echo "wrote $output"
