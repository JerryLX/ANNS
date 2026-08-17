#!/usr/bin/env python3
import csv
import sys

if len(sys.argv) != 3:
    raise SystemExit("usage: summarize_results.py RAW.csv SUMMARY.csv")

raw_path, summary_path = sys.argv[1:]
with open(raw_path, newline="") as f:
    rows = list(csv.DictReader(f))

numeric_keys = ("n", "queries", "dim", "shards", "k", "ef_search", "latency_ns", "threshold_scale")
keys = (("engine",) if "engine" in rows[0] else ()) + numeric_keys
groups = {}
for row in rows:
    key = tuple(row[k] if k == "engine" else float(row[k]) for k in keys)
    groups.setdefault(key, {})[row["mode"]] = row

fields = list(keys) + [
    "independent_recall", "shared_recall", "recall_delta",
    "independent_distance_computations", "shared_distance_computations", "distance_saving_pct",
    "independent_expansions", "shared_expansions", "expansion_saving_pct",
    "independent_latency_ns", "shared_latency_ns", "latency_saving_pct",
    "shared_messages", "shared_threshold_stops",
    "independent_posting_reads", "shared_posting_reads", "posting_read_saving_pct",
    "independent_bytes_read", "shared_bytes_read", "bytes_read_saving_pct",
    "shared_posting_prunes",
]

def number(row, name):
    return float(row[name])

def saving(base, shared):
    return 0.0 if base == 0 else 100.0 * (base - shared) / base

with open(summary_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for key, modes in groups.items():
        if "independent" not in modes or "shared" not in modes:
            continue
        base, shared = modes["independent"], modes["shared"]
        bd, sd = number(base, "avg_distance_computations"), number(shared, "avg_distance_computations")
        be, se = number(base, "avg_expansions"), number(shared, "avg_expansions")
        latency_field = "avg_latency_ns" if "avg_latency_ns" in base else "avg_simulated_latency_ns"
        bl, sl = number(base, latency_field), number(shared, latency_field)
        br, sr = number(base, "recall"), number(shared, "recall")
        out = {
            name: (value if name in ("engine", "threshold_scale") else int(value))
            for name, value in zip(keys, key)
        }
        out.update({
            "independent_recall": f"{br:.6f}", "shared_recall": f"{sr:.6f}", "recall_delta": f"{sr-br:.6f}",
            "independent_distance_computations": f"{bd:.3f}", "shared_distance_computations": f"{sd:.3f}",
            "distance_saving_pct": f"{saving(bd, sd):.3f}",
            "independent_expansions": f"{be:.3f}", "shared_expansions": f"{se:.3f}",
            "expansion_saving_pct": f"{saving(be, se):.3f}",
            "independent_latency_ns": f"{bl:.3f}", "shared_latency_ns": f"{sl:.3f}",
            "latency_saving_pct": f"{saving(bl, sl):.3f}",
            "shared_messages": shared["avg_messages"], "shared_threshold_stops": shared["avg_threshold_stops"],
        })
        if "avg_posting_reads" in base:
            bp, sp = number(base, "avg_posting_reads"), number(shared, "avg_posting_reads")
            bb, sb = number(base, "avg_bytes_read"), number(shared, "avg_bytes_read")
            out.update({
                "independent_posting_reads": f"{bp:.3f}", "shared_posting_reads": f"{sp:.3f}",
                "posting_read_saving_pct": f"{saving(bp, sp):.3f}",
                "independent_bytes_read": f"{bb:.3f}", "shared_bytes_read": f"{sb:.3f}",
                "bytes_read_saving_pct": f"{saving(bb, sb):.3f}",
                "shared_posting_prunes": shared["avg_posting_prunes"],
            })
        writer.writerow(out)
