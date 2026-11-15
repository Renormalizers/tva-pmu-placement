"""Split a reduced multi-cover problem into independent pieces (same idea as the 55k pipeline)."""
from collections import Counter
from stretch2 import reduce_need
from pmu_lib import load_grid

def split_pieces(red):
    parent = {v: v for v in red["cands"]}
    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]; v = parent[v]
        return v
    for s, k, u in red["cons"]:
        s = sorted(s)
        for v in s[1:]:
            parent[find(v)] = find(s[0])
    groups = {}
    for v in red["cands"]:
        groups.setdefault(find(v), []).append(v)
    pieces = []
    for root, cands in groups.items():
        cs = set(cands)
        cons = [(s, k, u) for s, k, u in red["cons"] if s & cs]
        pieces.append({"cands": sorted(cands), "cons": cons})
    return sorted(pieces, key=lambda p: -len(p["cands"]))

if __name__ == "__main__":
    import sys
    for name in sys.argv[1:]:
        g = load_grid(f"data/{name}.json")
        for goal, need in [("N-1", {b: 2 for b in g["buses"]})]:
            red = reduce_need(g, need); P = split_pieces(red)
            sz = Counter(len(p["cands"]) for p in P)
            small = sum(1 for p in P if len(p["cands"]) <= 16)
            print(f"{name:12s} {goal:15s} forced {len(red['forced']):5d} | pieces {len(P):4d} (<=16 q: {small}) | largest {[len(p['cands']) for p in P[:6]]}", flush=True)
