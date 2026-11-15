"""Stretch goals: N-1, zero-injection, and fixed budget with NERC-style priority weights.
Run with `python stretch_study.py`. Writes results/stretch_study.json."""
import json
from pathlib import Path
from pmu_lib import *

OUT = Path("results"); OUT.mkdir(exist_ok=True)
g = load_grid("data/ieee14.json")
buses = g["buses"]
res = {}

# ---------- Priority table, NERC Table 4 style (weights from NERC's example, 10-point scale) ----------
# Our assumptions for IEEE 14: transformers 4-7, 4-9, 5-6 join the high- and low-voltage areas
# (angle monitoring); generator buses 1, 2, 3, 6, 8 (oscillation, model verification); buses 1 and 2
# make real power (frequency response); loads of 20 MW or more (disturbance monitoring).
APPS = {
    "State estimation":      (8,  set(buses)),
    "Oscillation monitoring": (10, set(g["generator_buses"])),
    "Angle monitoring":      (8,  {4, 5, 6, 7, 9}),
    "Model verification":    (10, set(g["generator_buses"])),
    "Disturbance monitoring": (10, {b for b in buses if g["load_mw"][str(b)] >= 20}),
    "Frequency response":    (5,  {1, 2}),
}
weights = {b: sum(w for w, s in APPS.values() if b in s) for b in buses}
res["priority_table"] = {"apps": {k: [w, sorted(s)] for k, (w, s) in APPS.items()}, "weights": weights}
print("bus weights:", weights)


def weighted_seen(chosen):
    return sum(weights[b] for b in seen_buses(g, chosen))


# ---------- Fixed budget: ILP vs XY-mixer QAOA vs penalty QAOA ----------
res["budget"] = []
n = len(buses)
X = all_bits(n)
for k in [2, 3]:
    ilp = ilp_budget(g, k, weights)
    best = weighted_seen(ilp)
    unweighted = ilp_budget(g, k)
    # XY mixer QAOA: every shot has exactly k sensors
    score = lambda s: weighted_seen([buses[i] for i in range(n) if (s >> i) & 1])
    for L in [1, 2, 3]:
        params, probs, states, f = run_budget_qaoa(score, n, k, layers=L)
        p_opt = float(probs[f == best].sum())
        exp_score = float(probs @ f)
        res["budget"].append({"k": k, "method": "QAOA, XY mixer", "layers": L, "p_optimal": p_opt,
                              "p_exactly_k": 1.0, "mean_share_of_best": exp_score / best})
        print(f"k={k} XY L={L}: p_opt {p_opt:.1%}, mean score {exp_score/best:.1%} of best", flush=True)
    # penalty QAOA on all 2^14 bitstrings: cost = -score/W + lam*(sum x - k)^2
    W = sum(weights.values())
    sc = np.array([weighted_seen([b for b, x in zip(buses, row) if x]) for row in X])
    cnt = X.sum(1)
    costs = -sc / sc.max() + 1.0 * (cnt - k) ** 2
    for L in [1, 2, 3]:
        params, probs = run_qaoa(costs, n, layers=L, alpha=0.25)
        p_opt = float(probs[(sc == best) & (cnt == k)].sum())
        res["budget"].append({"k": k, "method": "QAOA, penalty", "layers": L, "p_optimal": p_opt,
                              "p_exactly_k": float(probs[cnt == k].sum())})
        print(f"k={k} penalty L={L}: p_opt {p_opt:.1%}, exactly k {probs[cnt==k].sum():.1%}", flush=True)
    rnd = float(((sc == best) & (cnt == k)).sum() / comb(n, k))
    res["budget"].append({"k": k, "method": "random k sensors", "p_optimal": rnd})
    res.setdefault("budget_ilp", []).append({"k": k, "weighted": ilp, "weighted_score": best,
                                             "max_score": W, "unweighted": unweighted,
                                             "unweighted_score_w": weighted_seen(unweighted),
                                             "buses_seen_weighted": len(seen_buses(g, ilp)),
                                             "buses_seen_unweighted": len(seen_buses(g, unweighted))})
    print(f"k={k}: ILP weighted {ilp} score {best}/{W}; unweighted {unweighted} score {weighted_seen(unweighted)}; random p_opt {rnd:.2%}")

# ---------- N-1 ----------
red2 = reduce_problem(g, twice=True)
costs = cover_cost_table(red2["cands"], red2["constraints"], P=2.0)
m = len(red2["cands"])
Xm = all_bits(m)
sensors = Xm.sum(1) + len(red2["forced"])
valid = np.ones(2 ** m, bool)
idx = {b: i for i, b in enumerate(red2["cands"])}
for s, need, _ in red2["constraints"]:
    valid &= Xm[:, [idx[b] for b in s]].sum(1) >= need
best_n1 = int(sensors[valid].min())
res["n1"] = {"forced": red2["forced"], "qubits": m, "best": best_n1, "runs": []}
rnd = float((valid & (sensors == best_n1)).mean())
for L in [1, 2, 3]:
    params, probs = run_qaoa(costs, m, layers=L, alpha=0.25)
    r = {"layers": L, "p_optimal": float(probs[valid & (sensors == best_n1)].sum()),
         "p_valid": float(probs[valid].sum())}
    res["n1"]["runs"].append(r)
    print(f"N-1 kernel ({m} qubits) L={L}: p_opt {r['p_optimal']:.1%} valid {r['p_valid']:.1%} (random opt {rnd:.2%})", flush=True)
res["n1"]["random_p_optimal"] = rnd

# ---------- Zero-injection rule (simulation only: the cost is not a short polynomial) ----------
zc = zib_cost_table(g, P=2.0)
nz = n
sens = X.sum(1)
okz = zc == sens  # no unseen penalty
res["zib"] = {"best": 3, "runs": [], "random_p_optimal": float((okz & (sens == 3)).mean())}
for L in [1, 2, 3]:
    params, probs = run_qaoa(zc, nz, layers=L, alpha=0.25)
    r = {"layers": L, "p_optimal": float(probs[okz & (sens == 3)].sum()), "p_valid": float(probs[okz].sum())}
    res["zib"]["runs"].append(r)
    print(f"ZIB full 14 L={L}: p_opt {r['p_optimal']:.2%} valid {r['p_valid']:.1%}", flush=True)

json.dump(res, open(OUT / "stretch_study.json", "w"), indent=1, default=lambda o: sorted(o) if isinstance(o, set) else str(o))
