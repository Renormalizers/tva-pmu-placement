"""Turn the IonQ Forte 1 chip bitstrings into sensor placements.

Writes
  results/n1_qubit_map.json        for each N-1 piece: which bus each qubit stands for, the sensors
                                   already fixed by the reduction, and every optimal bitstring
  results/forte_n1_bitstrings.csv  every bitstring the chip returned, with its count, the buses it
                                   puts PMUs on, and whether it is valid / optimal

Bit order: qBraid keys are written qubit 11 ... qubit 0 (rightmost character = qubit 0), so
int(key, 2) has bit j = qubit j, and qubit j is the j-th remaining spot (sorted bus number) of the
piece after the exact reduction.

Run:  python bitstrings.py
"""
import csv
import json

from pmu_lib import load_grid
from stretch2 import reduce_need
from pieces import split_pieces

GRID = {"ieee14_N-1": "ieee14", "ieee30_N-1": "ieee30", "ieee300_N-1": "ieee300",
        "activsg2000_N-1": "activsg2000"}
S = json.load(open("lab/scoring.json"))
C = json.load(open("results/forte_n1_counts.json"))
J = json.load(open("results/forte_n1_jobs.json"))

qmap, rows = {}, []
for key, gname in GRID.items():
    g = load_grid(f"data/{gname}.json")
    red = reduce_need(g, {b: 2 for b in g["buses"]})
    n = S[key]["qubits"]
    opt = set(S[key]["optimal_indices"])
    piece = next(p for p in split_pieces(red) if len(p["cands"]) == n
                 and all(isinstance(v, int) for v in p["cands"]))
    # pick the piece whose optimal answers match scoring.json (several pieces can share a size)
    for p in split_pieces(red):
        if len(p["cands"]) != n:
            continue
        ok = []
        for x in opt:
            on = {p["cands"][j] for j in range(n) if x >> j & 1}
            ok.append(all(len(s & on) >= k for s, k, _ in p["cons"]))
        if all(ok):
            piece = p
            break
    cands, cons = piece["cands"], piece["cons"]

    def buses(x):
        return [cands[j] for j in range(n) if x >> j & 1]

    qmap[key] = {
        "title": S[key]["title"], "qubits": n,
        "qubit_to_bus": {f"q{j}": cands[j] for j in range(n)},
        "fixed_by_reduction": red["forced"],
        "optimal_bitstrings": {format(x, f"0{n}b"): buses(x) for x in sorted(opt)},
        "note": "bitstring is written q{n-1}...q0 (rightmost = q0), same as the qBraid counts",
    }
    if key not in C:
        continue
    job = J.get(f"{key}|multi", "")
    for b, cnt in sorted(C[key].items(), key=lambda kv: -kv[1]):
        x = int(b, 2)
        on = set(buses(x))
        valid = all(len(s & on) >= k for s, k, _ in cons)
        rows.append([S[key]["title"], b, cnt, len(on), " ".join(map(str, sorted(on))),
                     "yes" if valid else "no", "yes" if x in opt else "no", job])

json.dump(qmap, open("results/n1_qubit_map.json", "w"), indent=1)
with open("results/forte_n1_bitstrings.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["piece", "bitstring (q_last...q0)", "count (of 100 shots)", "PMUs in piece",
                "PMU buses in piece", "meets N-1 rules", "optimal", "IonQ Forte 1 job"])
    w.writerows(rows)
print("wrote results/n1_qubit_map.json and results/forte_n1_bitstrings.csv")
