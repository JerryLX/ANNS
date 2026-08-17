#!/usr/bin/env bash
set -euo pipefail

binary=${1:-./build/hnsw_threshold}
raw_output=${2:-shard_results.csv}
shift $(( $# >= 2 ? 2 : $# ))

summary_output=${raw_output%.csv}_summary.csv
shard_list=${SHARDS:-"1 2 4 8 16 32"}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
first=1

for shards in $shard_list; do
    echo "running shards=$shards" >&2
    if (( first )); then
        "$binary" --shards "$shards" --mode all "$@" > "$raw_output"
        first=0
    else
        "$binary" --shards "$shards" --mode all "$@" | tail -n +2 >> "$raw_output"
    fi
done

python3 "$script_dir/summarize_results.py" "$raw_output" "$summary_output"
echo "raw: $raw_output" >&2
echo "summary: $summary_output" >&2
