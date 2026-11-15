"""Recheck the headline stretch-goal numbers.

1. Exactness of the N-1 reduction on every grid:
   sensors fixed by the rules + exact optimum of every piece == exact ILP on the whole grid.
2. Each piece answer combined with the fixed sensors really meets the goal (checked bus by bus).
3. The scored hardware results in results/stretch_qbraid_results.json match the raw chip counts
   in results/forte_n1_counts.json and the optimal answers in lab/scoring.json.

Run:  python verify.py          (about a minute; the Texas 2000 ILP is the slow part)
"""
import json
import time

from pmu_lib import load_grid
from stretch2 import reduce_need, ilp_need
from pieces import split_pieces
from stretch_quantum import ilp_cover, meets

GRIDS = ["ieee14", "ieee30", "ieee57", "ieee118", "ieee300", "activsg2000"]
ok = True
print(f"{'grid':12s} {'goal':5s} {'ILP':>5s} {'fixed':>6s} {'pieces':>7s} {'largest':>8s}  exact?")
for name in GRIDS:
    g = load_grid(f"data/{name}.json")
    goals = [("N-1", {b: 2 for b in g["buses"]})]
    for goal, need in goals:
        best, _, _ = ilp_need(g, need)
        red = reduce_need(g, need)
        P = split_pieces(red)
        chosen = list(red["forced"])
        for p in P:
            sol, _ = ilp_cover(p["cands"], p["cons"])
            chosen += sol
        exact = len(chosen) == len(best) and meets(g, chosen, need)
        ok &= exact
        largest = len(P[0]["cands"]) if P else 0
        print(f"{name:12s} {goal:5s} {len(best):5d} {len(red['forced']):6d} {len(P):7d} {largest:8d}  {exact}",
              flush=True)

# hardware scoring
S = json.load(open("lab/scoring.json"))
R = json.load(open("results/stretch_qbraid_results.json"))["results"]
C = json.load(open("results/forte_n1_counts.json"))
for key, counts in C.items():
    opt = set(S[key]["optimal_indices"])
    share = sum(v for b, v in counts.items() if int(b, 2) in opt) / sum(counts.values())
    match = abs(share - R[key]["multi"]["forte"]) < 1e-9
    ok &= match
    print(f"Forte chip {key:16s} optimal share {share:.0%}  matches results file: {match}")

print("\nALL CHECKS PASS" if ok else "\nSOME CHECKS FAILED")
