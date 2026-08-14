#!/usr/bin/env python3
import math
import struct
import sys

prefix = sys.argv[1]
base = [[float(i), float(i % 3), float(i % 2), 1.0] for i in range(40)]
queries = [[1.1, 1.0, 1.0, 1.0], [17.2, 2.0, 1.0, 1.0], [32.8, 0.0, 1.0, 1.0]]

def write(path, rows, code):
    with open(path, "wb") as f:
        f.write(struct.pack("<II", len(rows), len(rows[0])))
        for row in rows:
            f.write(struct.pack("<" + code * len(row), *(int(x) if code != "f" else x for x in row)))

write(prefix + "_base.fbin", base, "f")
write(prefix + "_query.fbin", queries, "f")
write(prefix + "_base.u8bin", base, "B")
write(prefix + "_query.u8bin", queries, "B")

k = 5
ranked = []
for q in queries:
    values = sorted((sum((x-y)**2 for x, y in zip(v, q)), i) for i, v in enumerate(base))[:k]
    ranked.append(values)
with open(prefix + "_gt.bin", "wb") as f:
    f.write(struct.pack("<II", len(queries), k))
    for row in ranked:
        f.write(struct.pack("<" + "I" * k, *(i for _, i in row)))
    for row in ranked:
        f.write(struct.pack("<" + "f" * k, *(d for d, _ in row)))
