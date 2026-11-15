"""QAOA study: run with `python qaoa_study.py`. Writes results/qaoa_study.json."""
import json, sys, time
from pathlib import Path
from pmu_lib import *

OUT = Path("results"); OUT.mkdir(exist_ok=True)
P = 2.0


def shares(cands, constraints, probs, best_total, n_forced):
    n = len(cands)
    X = all_bits(n)
    idx = {b: i for i, b in enumerate(cands)}
    valid = np.ones(2 ** n, bool)
    for s, need, _ in constraints:
        valid &= X[:, [idx[b] for b in s]].sum(1) >= need
    sensors = X.sum(1) + n_forced
    return {"optimal": float(probs[valid & (sensors == best_total)].sum()),
            "valid, extra sensors": float(probs[valid & (sensors > best_total)].sum()),
            "leaves a bus unseen": float(probs[~valid].sum())}


def study(label, cands, constraints, best_total, n_forced=0, layers=(1, 2, 3), alphas=(0.25, 1.0)):
    n = len(cands)
    costs = cover_cost_table(cands, constraints, P)
    rows = []
    rnd = shares(cands, constraints, np.full(2 ** n, 2.0 ** -n), best_total, n_forced)
    rows.append({"case": label, "method": "random guess", "layers": 0, "alpha": None, **rnd})
    for a in alphas:
        prev = None
        for L in layers:
            t0 = time.perf_counter()
            starts = [interp_init(prev)] if prev is not None else None
            params, probs = run_qaoa(costs, n, layers=L, alpha=a, starts=starts)
            prev = params
            sh = shares(cands, constraints, probs, best_total, n_forced)
            rows.append({"case": label, "method": "QAOA", "layers": L, "alpha": a, **sh,
                         "params": [round(float(x), 4) for x in params],
                         "tune_seconds": round(time.perf_counter() - t0, 2)})
            print(f"{label:28s} L={L} alpha={a}: opt {sh['optimal']:.1%} valid+ {sh['valid, extra sensors']:.1%} "
                  f"unseen {sh['leaves a bus unseen']:.1%}  ({rows[-1]['tune_seconds']}s)", flush=True)
    print(f"{label:28s} random: opt {rnd['optimal']:.2%} unseen {rnd['leaves a bus unseen']:.1%}")
    return rows


if __name__ == "__main__":
    piece = load_grid("data/ieee14_piece6.json")
    g14 = load_grid("data/ieee14.json")
    red = reduce_problem(g14)
    rows = []
    rows += study("6-bus piece (6 qubits)", piece["buses"], grid_constraints(piece), 2)
    rows += study("IEEE 14 reduced (4 qubits)", red["cands"], red["constraints"], 4, len(red["forced"]))
    rows += study("IEEE 14 full (14 qubits)", g14["buses"], grid_constraints(g14), 4, layers=(1, 2, 3, 4))
    json.dump(rows, open(OUT / "qaoa_study.json", "w"), indent=1)
