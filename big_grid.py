"""Sparse, scalable versions of our methods for grids with tens of thousands of buses
(Texas A&M synthetic grids ACTIVSg25k and ACTIVSg70k, converted with matpower_import.py).

    python big_grid.py data_big/activsg70k.json

Everything here is exact or clearly labeled as a heuristic. Results go to results/big_<name>.json.
"""
import heapq
import json
import sys
import time
from collections import defaultdict

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix, csr_matrix

from check import load_grid, is_valid, seen_buses


def closed_nbhd(g):
    return {b: {b} | g["neighbors"][b] for b in g["buses"]}


# ------------------------------------------------------------------ greedy (lazy priority queue)
def greedy_fast(g, twice=False):
    N = closed_nbhd(g)
    need = {b: (2 if twice else 1) for b in g["buses"]}
    gain = {b: len(N[b]) for b in g["buses"]}
    heap = [(-gain[b], b) for b in g["buses"]]
    heapq.heapify(heap)
    chosen = set()
    remaining = sum(need.values())
    while remaining > 0:
        neg, b = heapq.heappop(heap)
        if b in chosen:
            continue
        real = sum(1 for u in N[b] if need[u] > 0)
        if real != -neg:
            heapq.heappush(heap, (-real, b))
            continue
        chosen.add(b)
        for u in N[b]:
            if need[u] > 0:
                need[u] -= 1
                remaining -= 1
    return sorted(chosen)


# ------------------------------------------------------------------ exact ILPs (sparse)
def cover_ilp(rows_sets, cands, need, time_limit=600):
    """min sum x over cands s.t. for each row r: sum_{c in rows_sets[r]} x_c >= need[r]."""
    idx = {c: i for i, c in enumerate(cands)}
    ri, ci = [], []
    for r, s in enumerate(rows_sets):
        for c in s:
            ri.append(r); ci.append(idx[c])
    A = csr_matrix((np.ones(len(ri)), (ri, ci)), shape=(len(rows_sets), len(cands)))
    res = milp(c=np.ones(len(cands)), constraints=LinearConstraint(A, lb=np.array(need, float)),
               integrality=np.ones(len(cands)), bounds=Bounds(0, 1),
               options={"time_limit": time_limit, "mip_rel_gap": 0})
    sol = [c for c, x in zip(cands, res.x) if x > 0.5]
    return sol, res


def ilp_sparse(g, twice=False, time_limit=600):
    N = closed_nbhd(g)
    rows = [N[b] for b in g["buses"]]
    return cover_ilp(rows, g["buses"], [2 if twice else 1] * len(rows), time_limit)


def ilp_zib_sparse(g, time_limit=1200):
    """Same formulation as pmu_lib.ilp_placement_zib, built sparse."""
    buses, Z = g["buses"], g["zero_injection"]
    N = closed_nbhd(g)
    n = len(buses)
    idx = {b: i for i, b in enumerate(buses)}
    ypairs = [(z, m) for z in Z for m in sorted(N[z])]
    ny = len(ypairs)
    yof = defaultdict(list)
    for k, (z, m) in enumerate(ypairs):
        yof[m].append(k)
    zof = defaultdict(list)
    for k, (z, m) in enumerate(ypairs):
        zof[z].append(k)
    M = n + 1
    ri, ci, val, lb, ub = [], [], [], [], []
    r = 0
    for b in buses:
        for j in N[b]:
            ri.append(r); ci.append(idx[j]); val.append(1)
        for k in yof[b]:
            ri.append(r); ci.append(n + k); val.append(1)
        lb.append(1); ub.append(np.inf); r += 1
    for z in Z:
        for k in zof[z]:
            ri.append(r); ci.append(n + k); val.append(1)
        lb.append(-np.inf); ub.append(1); r += 1
    T = n + ny
    for k, (z, m) in enumerate(ypairs):
        for o in N[z] - {m}:
            ri += [r, r, r]; ci += [T + idx[m], T + idx[o], n + k]; val += [1, -1, -M]
            lb.append(1 - M); ub.append(np.inf); r += 1
    nv = n + ny + n
    A = coo_matrix((val, (ri, ci)), shape=(r, nv)).tocsr()
    integ = np.concatenate([np.ones(n + ny), np.zeros(n)])
    hi = np.concatenate([np.ones(n + ny), np.full(n, float(n))])
    res = milp(c=np.concatenate([np.ones(n), np.zeros(ny + n)]), constraints=LinearConstraint(A, lb, ub),
               integrality=integ, bounds=Bounds(np.zeros(nv), hi), options={"time_limit": time_limit})
    sol = [b for b, x in zip(buses, res.x[:n]) if x > 0.5]
    return sol, res


# ------------------------------------------------------------------ exact reduction (worklist version)
def reduce_fast(g):
    """R1 forced / R2 bus dominance / R3 candidate dominance, as in pmu_lib.reduce_problem,
    but only comparing sets that share a member, so it runs on 70,000 buses."""
    N = closed_nbhd(g)
    live = set(g["buses"])          # buses that still need a sighting
    cands = set(g["buses"])
    forced = []
    seers = {u: set(N[u]) for u in g["buses"]}       # candidates that see u
    covers = {v: set(N[v]) for v in g["buses"]}      # live buses v would see
    changed = True
    while changed:
        changed = False
        # R1
        for u in list(live):
            if u in live and len(seers[u]) == 1:
                v = next(iter(seers[u]))
                forced.append(v)
                for w in list(covers[v]):
                    live.discard(w)
                    for c in seers[w]:
                        if c != v:
                            covers[c].discard(w)
                cands.discard(v)
                for w in N[v]:
                    seers[w].discard(v)
                changed = True
        # R2: if seers[u] subset of seers[w], drop w (only w that share a candidate with u)
        for u in list(live):
            if u not in live:
                continue
            su = seers[u]
            pool = set()
            for c in su:
                pool |= covers[c]
            for w in pool:
                if w != u and w in live and su <= seers[w] and (su != seers[w] or u < w):
                    live.discard(w)
                    for c in seers[w]:
                        covers[c].discard(w)
                    changed = True
        # R3: candidate u dominated by v (only v sharing a live bus)
        for u in list(cands):
            if u not in cands:
                continue
            cu = covers[u]
            if not cu:
                cands.discard(u)
                for w in N[u]:
                    seers[w].discard(u)
                changed = True
                continue
            pool = set()
            for w in cu:
                pool |= seers[w]
            for v in pool:
                if v != u and cu <= covers[v] and (cu != covers[v] or v < u):
                    cands.discard(u)
                    for w in N[u]:
                        seers[w].discard(u)
                    changed = True
                    break
    cons = {}
    for u in live:
        cons.setdefault(frozenset(seers[u]), []).append(u)
    return {"forced": sorted(forced), "cands": sorted(cands), "constraints": [(set(s), 1, us) for s, us in cons.items()]}


def components_of(red):
    parent = {v: v for v in red["cands"]}

    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v
    for s, _, _ in red["constraints"]:
        s = list(s)
        for v in s[1:]:
            parent[find(v)] = find(s[0])
    groups = defaultdict(list)
    for v in red["cands"]:
        groups[find(v)].append(v)
    by_root = defaultdict(list)
    for c in red["constraints"]:
        if c[0]:
            by_root[find(next(iter(c[0])))].append(c)
    return [{"cands": sorted(v), "constraints": by_root[k]} for k, v in groups.items()]


if __name__ == "__main__":
    path = sys.argv[1]
    g = load_grid(path)
    name = g["name"].replace(" ", "_")
    out = {"grid": name, "buses": len(g["buses"]), "lines": len(g["lines"]), "zib": len(g["zero_injection"])}
    print(out, flush=True)

    t = time.perf_counter(); gr = greedy_fast(g); out["greedy"] = (len(gr), round(time.perf_counter() - t, 2))
    assert is_valid(g, gr); print("greedy", out["greedy"], flush=True)

    t = time.perf_counter(); red = reduce_fast(g); tr = time.perf_counter() - t
    comps = components_of(red)
    sizes = sorted((len(c["cands"]) for c in comps), reverse=True)
    out["reduction"] = {"forced": len(red["forced"]), "qubits_left": len(red["cands"]), "seconds": round(tr, 1),
                        "pieces": len(comps), "largest_pieces": sizes[:10],
                        "pieces_le_12": sum(1 for s in sizes if s <= 12)}
    print("reduction", out["reduction"], flush=True)

    t = time.perf_counter(); il, res = ilp_sparse(g); out["ilp"] = (len(il), round(time.perf_counter() - t, 2), res.status)
    assert is_valid(g, il); print("ILP", out["ilp"], flush=True)
    t = time.perf_counter(); n1, res = ilp_sparse(g, twice=True); out["n1"] = (len(n1), round(time.perf_counter() - t, 2), res.status)
    print("N-1 ILP", out["n1"], flush=True)
    json.dump({"out": out, "greedy": gr, "ilp": il, "n1": n1, "red_forced": red["forced"], "red_cands": red["cands"]},
              open(f"results/big_{name}.json", "w"))
    if "nozib" in sys.argv:
        raise SystemExit
    t = time.perf_counter(); iz, res = ilp_zib_sparse(g); out["ilp_zib"] = (len(iz), round(time.perf_counter() - t, 1), res.status, getattr(res, "mip_gap", None))
    print("ZIB ILP", out["ilp_zib"], "valid", is_valid(g, iz, zib=True), flush=True)
    json.dump({"out": out, "greedy": gr, "ilp": il, "n1": n1, "ilp_zib": iz, "red_forced": red["forced"], "red_cands": red["cands"]},
              open(f"results/big_{name}.json", "w"))
