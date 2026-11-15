"""Independent checks of every number in the deck, using TVA's check.py as the referee.
Run with `python verify.py`; it raises on any mismatch."""
import json
from check import load_grid, is_valid, brute_force_best
from pmu_lib import ilp_placement, ilp_placement_zib, ilp_n1_zib, greedy_placement, reduce_problem, components, solve_piece_exact

expect = {"ieee14_piece6": (2, 2, 2, 5), "ieee14": (5, 4, 3, 9), "ieee30": (10, 10, 7, 21),
          "ieee57": (18, 17, 11, 33), "ieee118": (36, 32, 30, 68), "ieee300": (96, 87, 70, 202)}
for name, (gr, il, iz, n1) in expect.items():
    g = load_grid(f"data/{name}.json")
    a, b, c, d = greedy_placement(g), ilp_placement(g), ilp_placement_zib(g), ilp_placement(g, twice=True)
    assert is_valid(g, a) and is_valid(g, b) and is_valid(g, c, zib=True) and is_valid(g, d, n1=True), name
    assert (len(a), len(b), len(c), len(d)) == (gr, il, iz, n1), (name, len(a), len(b), len(c), len(d))
    red = reduce_problem(g)
    comps = components(red)
    if all(len(x["cands"]) <= 20 for x in comps):
        total = red["forced"] + [v for x in comps for v in solve_piece_exact(x["cands"], x["constraints"])]
        assert is_valid(g, total) and len(total) == il, name
    print(f"{name}: greedy {gr}, ILP {il}, ZIB {iz}, N-1 {n1}  OK")
g = load_grid("data/ieee14.json")
assert brute_force_best(g)[0] == 4 and brute_force_best(g, zib=True)[0] == 3
assert brute_force_best(g, n1=True)[0] == 9 and brute_force_best(g, zib=True, n1=True)[0] == 7
assert len(ilp_n1_zib(g)) == 7
print("IEEE 14 brute force agrees: 4, 3 (ZIB), 9 (N-1), 7 (N-1 + ZIB)")

# N-1 with zero-injection: recheck the saved exact answers with the checker
import json as _j
for r in _j.load(open("results/n1_zib.json")):
    g = load_grid(f"data/{r['grid']}.json")
    assert is_valid(g, r["buses"], zib=True, n1=True) and len(r["buses"]) == r["count"]
    print(f"{r['grid']}: N-1 + zero-injection {r['count']} PMUs, checker OK")
for r in _j.load(open("results/ieee300_study.json"))["window_search"]:
    g = load_grid("data/ieee300.json")
    rule = {"plain": {}, "zib": {"zib": True}, "n1": {"n1": True}}[r["variant"]]
    assert is_valid(g, r["buses"], **rule) and len(r["buses"]) == r["final"]
print("IEEE 300 window-search placements all pass the checker")
