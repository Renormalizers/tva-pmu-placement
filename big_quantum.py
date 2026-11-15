"""QAOA on every independent piece left after the exact reduction of a big grid.
    python big_quantum.py data_big/activsg70k.json
For each piece: tune QAOA (exact statevector), draw 200 shots, keep the best valid shot.
The full placement = sensors fixed by the rules + QAOA's pick in each piece; checked with check.py."""
import json, sys, time
import numpy as np
from check import load_grid, is_valid
from big_grid import reduce_fast, components_of, cover_ilp
from pmu_lib import cover_cost_table, run_qaoa, interp_init, all_bits, qaoa_circuit_cover, cx_count

g = load_grid(sys.argv[1]); name = g["name"].replace(" ", "_")
MAXQ = int(sys.argv[2]) if len(sys.argv) > 2 else 99  # pieces above this many qubits go to the ILP
rng = np.random.default_rng(0)
red = reduce_fast(g)
pieces = sorted(components_of(red), key=lambda c: -len(c["cands"]))
rows, pick = [], list(red["forced"])
t_all = time.perf_counter()
for i, pc in enumerate(pieces):
    cands, cons = pc["cands"], pc["constraints"]
    n = len(cands)
    exact, _ = cover_ilp([c[0] for c in cons], cands, [1] * len(cons))
    if n > MAXQ:
        pick += exact
        rows.append({"piece": i, "qubits": n, "constraints": len(cons), "exact": len(exact), "qaoa_best_shot": None,
                     "solver": "ILP (too many qubits to simulate)"})
        continue
    cost = cover_cost_table(cands, cons)
    X = all_bits(n); idx = {b: j for j, b in enumerate(cands)}
    valid = np.ones(2 ** n, bool)
    for s, _, _ in cons:
        valid &= X[:, [idx[b] for b in s]].sum(1) >= 1
    sizes = X.sum(1)
    t = time.perf_counter()
    prev, best = None, None
    for L in (1, 2, 3):
        p, probs = run_qaoa(cost, n, layers=L, alpha=1.0, maxiter=120 if n <= 12 else 50,
                            starts=[interp_init(prev)] if prev is not None else None)
        prev = p
        popt = float(probs[valid & (sizes == len(exact))].sum())
        if best is None or popt > best[1]:
            best = (p, popt, L, probs)
    p, popt, L, probs = best
    shots = rng.choice(2 ** n, size=200, p=probs / probs.sum())
    ok = [z for z in set(shots.tolist()) if valid[z]]
    z = min(ok, key=lambda z: sizes[z]) if ok else None
    chosen = [cands[j] for j in range(n) if z is not None and X[z, j]]
    pick += chosen if z is not None else exact
    ncx = cx_count(qaoa_circuit_cover(cands, cons, p, measure=False))[0] if n <= 20 else None
    rows.append({"piece": i, "qubits": n, "constraints": len(cons), "exact": len(exact), "qaoa_best_shot": len(chosen) if z is not None else None,
                 "layers": L, "p_optimal_per_shot": round(popt, 4), "random_p_optimal": round(float((valid & (sizes == len(exact))).mean()), 4),
                 "cx": ncx, "seconds": round(time.perf_counter() - t, 1), "params": [round(float(x), 4) for x in p]})
    r = rows[-1]
    print(f"piece {i:2d}: {n:2d} qubits, exact {r['exact']}, QAOA best of 200 shots {r['qaoa_best_shot']}, "
          f"p(opt)/shot {popt:.1%} (random {r['random_p_optimal']:.2%}), L={L}, {ncx} CX/circuit, {r['seconds']} s", flush=True)
ok = is_valid(g, pick)
q_rows = [r for r in rows if r.get("qaoa_best_shot") is not None]
summary = {"grid": name, "forced": len(red["forced"]), "pieces": len(pieces), "qubits": sum(r["qubits"] for r in rows),
           "pieces_by_qaoa": len(q_rows), "pmus_chosen_by_qaoa": sum(r["qaoa_best_shot"] for r in q_rows),
           "qaoa_pieces_optimal": sum(1 for r in q_rows if r["qaoa_best_shot"] == r["exact"]),
           "pmus_from_ilp_pieces": sum(r["exact"] for r in rows if r.get("qaoa_best_shot") is None),
           "total_pmus": len(set(pick)), "valid": ok, "all_qaoa_pieces_optimal": all(r["qaoa_best_shot"] == r["exact"] for r in q_rows),
           "seconds": round(time.perf_counter() - t_all, 1)}
print(summary)
json.dump({"summary": summary, "pieces": rows, "placement": sorted(set(pick))}, open(f"results/big_quantum_{name}.json", "w"), indent=1)
