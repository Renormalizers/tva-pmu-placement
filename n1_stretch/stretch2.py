"""N-1 core: exact reduction and exact ILP when a bus has to be seen more than once.

  reduce_need(g, need)   exact reduction for a per-bus need (N-1: need = 2 for every bus)
  ilp_need(g, need)      exact minimum by integer programming (the classical baseline)

Used by pieces.py, stretch_quantum.py, boost.py and verify.py.
"""
import time

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds, linprog

from pmu_lib import group

def popcount(a):
    return np.bitwise_count(a).astype(np.int64)


# ------------------------------------------------------------------------------------------
# Exact reduction for any per-bus need
# ------------------------------------------------------------------------------------------
def reduce_need(g, need):
    """Shrink a multi-cover problem without losing an optimal answer.
    R1: if a bus can be seen from at most need[u] remaining spots, all of them are forced.
    R2: if every spot that sees u also sees w and need[u] >= need[w], w's rule is implied: drop it.
    R0: a spot that sees no bus that still needs a sighting is never useful: drop it.
    R3': spot dominance, used only when every bus the weaker spot serves needs exactly 1 more sighting
    (plain R3 is not safe when a bus needs 2 sightings)."""
    G = {b: group(g, b) for b in g["buses"]}
    need = dict(need)
    cands, forced = set(g["buses"]), []
    while True:
        live = [b for b in need if need[b] > 0]
        seers = {u: frozenset(v for v in G[u] if v in cands) for u in live}
        bad = [u for u in live if len(seers[u]) < need[u]]
        if bad:
            raise ValueError(f"infeasible: bus {bad[0]} cannot get {need[bad[0]]} sightings")
        hit = next((u for u in live if len(seers[u]) == need[u]), None)
        if hit is not None:
            for v in sorted(seers[hit]):
                forced.append(v)
                cands.discard(v)
                for w in G[v]:
                    need[w] = max(0, need[w] - 1)
            continue
        done = False
        for u in live:
            for w in live:
                if u != w and seers[u] <= seers[w] and need[u] >= need[w] and \
                        (seers[u] != seers[w] or need[u] > need[w] or u < w):
                    need[w] = 0
                    done = True
                    break
            if done:
                break
        if done:
            continue
        dead = [v for v in cands if not any(need[u] > 0 for u in G[v])]
        if dead:
            cands -= set(dead)
            continue
        # R3': spot u is dominated by spot v (v sees every still-needed bus u sees) AND every bus u
        # still serves needs only 1 more sighting. Then u is never needed: if v is chosen, u adds
        # nothing; if v is not, swapping u -> v keeps every rule satisfied. Safe for mixed needs.
        cov = {v: frozenset(u for u in G[v] if need[u] > 0) for v in cands}
        drop = None
        for u in sorted(cands):
            if all(need[w] == 1 for w in cov[u]):
                for v in sorted(cands):
                    if v != u and cov[u] <= cov[v] and (cov[u] != cov[v] or v < u):
                        drop = u
                        break
            if drop is not None:
                break
        if drop is not None:
            cands.discard(drop)
            continue
        break
    cands = sorted(cands)
    cons = [(frozenset(v for v in G[u] if v in cands), need[u], u) for u in sorted(need) if need[u] > 0]
    return {"forced": sorted(forced), "cands": cands, "cons": cons}


def ilp_need(g, need):
    """Exact minimum: every bus b seen at least need[b] times."""
    B = g["buses"]
    A = np.array([[1 if c in group(g, b) else 0 for c in B] for b in B])
    lb = np.array([need[b] for b in B])
    t0 = time.perf_counter()
    res = milp(np.ones(len(B)), constraints=LinearConstraint(A, lb=lb), integrality=np.ones(len(B)),
               bounds=Bounds(0, 1))
    ms = 1000 * (time.perf_counter() - t0)
    lp = linprog(np.ones(len(B)), A_ub=-A, b_ub=-lb, bounds=(0, 1), method="highs")
    return [b for b, x in zip(B, res.x) if x > 0.5], ms, lp.fun


# ------------------------------------------------------------------------------------------
# Zero-injection, one level of inference (circuit-ready)
