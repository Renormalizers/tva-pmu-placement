"""IEEE 300: every stretch goal, classical and quantum.

Classical (exact): ILP, zero-injection, N-1, N-1 + zero-injection, and the fixed-budget curve with
NERC-style weights. Quantum: QAOA window search. Start from the greedy placement; repeatedly free a
window of up to 10 buses (everything else stays fixed), let QAOA propose a cheaper set of sensors for
the window, keep it if the checker says the whole grid is still covered. Each window is a small
circuit (constraints are short), so this is the hardware-sized way to put QAOA on a 300-bus grid.
Baseline with the same budget of shots: random sampling of the window.
Run with `python ieee300_study.py`. Writes results/ieee300_study.json."""
import json, sys
from collections import deque
from pmu_lib import *

g = load_grid("data/ieee300.json")
buses = g["buses"]; allb = set(buses)
rng = np.random.default_rng(0)
from pathlib import Path
OUTF = Path("results/ieee300_study.json")
res = json.load(open(OUTF)) if OUTF.exists() else {}
PART = sys.argv[1] if len(sys.argv) > 1 else "all"
save = lambda: json.dump(res, open(OUTF, "w"), indent=1)

# ---------------- classical, exact ----------------
CLASSICAL = [("ILP", lambda: ilp_placement(g)), ("ILP + zero-injection", lambda: ilp_placement_zib(g)),
                  ("N-1 ILP", lambda: ilp_placement(g, twice=True)), ("Greedy", lambda: greedy_placement(g)),
                  ("Greedy N-1", lambda: greedy_placement(g, twice=True))]
for label, fn in (CLASSICAL if PART in ("all", "classical") else []):
    sol, ms = timed(fn, repeat=1)
    res[label] = {"buses": sol, "count": len(sol), "ms": round(ms, 1)}
    print(label, len(sol), f"{ms:.0f} ms", flush=True)
save()

# ---------------- NERC-style weights and the budget curve ----------------
# Our assumptions for IEEE 300 (no transformer data, so no angle-monitoring row): every bus serves state
# estimation (8); generator buses serve oscillation monitoring (10), model verification (10) and frequency
# response (5); buses with 100 MW of load or more serve disturbance monitoring (10).
W = {b: 8 + (25 if b in g["generator_buses"] else 0) + (10 if g["load_mw"][str(b)] >= 100 else 0) for b in buses}
Wtot = sum(W.values())
curve = []
for k in ([] if PART not in ("all", "classical") else [5, 10, 20, 30, 40, 50, 60, 70, 80, 87]):
    wsol, ms = timed(ilp_budget, g, k, W, repeat=1)
    usol = ilp_budget(g, k)
    curve.append({"k": k, "weighted_share": sum(W[b] for b in seen_buses(g, wsol)) / Wtot,
                  "buses_seen_weighted": len(seen_buses(g, wsol)), "buses_seen_unweighted": len(seen_buses(g, usol)),
                  "weighted_share_of_unweighted": sum(W[b] for b in seen_buses(g, usol)) / Wtot, "ms": round(ms)})
    print(f"k={k}: weighted plan sees {curve[-1]['weighted_share']:.0%} of priority weight "
          f"({curve[-1]['buses_seen_weighted']} buses); count-only plan {curve[-1]['weighted_share_of_unweighted']:.0%} "
          f"({curve[-1]['buses_seen_unweighted']} buses)", flush=True)
res["weights"] = {"rule": "8 all + 25 generator + 10 if load >= 100 MW", "total": Wtot}
res["budget_curve"] = curve if curve else res.get("budget_curve")
save()


# ---------------- QAOA window search ----------------
def window(seed, size):
    seen_, order, q = {seed}, [], deque([seed])
    while q and len(order) < size:
        b = q.popleft(); order.append(b)
        for nb in sorted(g["neighbors"][b]):
            if nb not in seen_:
                seen_.add(nb); q.append(nb)
    return order


def window_problem(current, Wb, variant):
    """Cost table over the window's 2^|W| choices, everything outside the window fixed."""
    fixed = [b for b in current if b not in Wb]
    n = len(Wb); X = all_bits(n)
    if variant == "zib":
        cost = np.empty(2 ** n); valid = np.empty(2 ** n, bool)
        for z in range(2 ** n):
            ch = fixed + [Wb[i] for i in range(n) if X[z, i]]
            un = len(allb - seen_buses(g, ch, zib=True))
            cost[z] = X[z].sum() + 2.0 * un; valid[z] = un == 0
        return cost, valid, None
    need = 2 if variant == "n1" else 1
    cnt = {b: sum(1 for j in group(g, b) if j in fixed) for b in buses}
    cons = []
    for b in buses:
        short = need - cnt[b]
        if short > 0:
            s = {j for j in group(g, b) if j in Wb}
            cons.append((s, short, [b]))
    cost = cover_cost_table(Wb, cons, 2.0)
    idx = {b: i for i, b in enumerate(Wb)}
    valid = np.ones(2 ** n, bool)
    for s, k, _ in cons:
        valid &= X[:, [idx[b] for b in s]].sum(1) >= k
    return cost, valid, cons


def search(start, variant, sampler, size=10, shots=200, passes=3, log_every=0):
    cur = sorted(start); windows = 0; qubits = 0; cx_max = 0
    for p in range(passes):
        improved = False
        for seed in rng.permutation(sorted(cur)):
            if seed not in cur:
                continue
            Wb = window(int(seed), size)
            cost, valid, cons = window_problem(cur, Wb, variant)
            n = len(Wb)
            probs = sampler(cost, n)
            samp = rng.choice(2 ** n, size=shots, p=probs / probs.sum())
            X = all_bits(n)
            ok = [z for z in set(samp.tolist()) if valid[z]]
            windows += 1; qubits = max(qubits, n)
            if not ok:
                continue
            z = min(ok, key=lambda z: X[z].sum())
            inside_now = sum(1 for b in cur if b in Wb)
            if X[z].sum() < inside_now:
                cur = sorted([b for b in cur if b not in Wb] + [Wb[i] for i in range(n) if X[z, i]])
                improved = True
        if not improved:
            break
    return cur, windows, qubits


def qaoa_sampler(cost, n):
    _, probs = run_qaoa(cost, n, layers=2, alpha=0.25, maxiter=120,
                        starts=[np.array([0.2, 0.4, 0.6, 0.3]), np.array([0.5, 1.0, 0.5, 0.25])])
    return probs


def random_sampler(cost, n):
    return np.full(2 ** n, 1.0)


def exact_sampler(cost, n):
    p = np.zeros(2 ** n); p[int(np.argmin(cost))] = 1.0  # classical brute force of the window
    return p


res.setdefault("window_search", [])
for variant, start, rule, opt in [
    ("plain", res["Greedy"]["buses"], {}, res["ILP"]["count"]),
    ("zib", res["Greedy"]["buses"], {"zib": True}, res["ILP + zero-injection"]["count"]),
    ("n1", res["Greedy N-1"]["buses"], {"n1": True}, res["N-1 ILP"]["count"]),
]:
    size = 14 if PART in ("plain14", "exact14") else (8 if variant == "zib" else 10)
    if PART not in ("all", variant) and not (PART in ("plain14", "exact14") and variant == "plain"):
        continue
    samplers = [("QAOA, 2 layers", qaoa_sampler), ("random sampling", random_sampler)]
    if PART == "exact14":
        samplers = [("exact window (classical)", exact_sampler)]
    res["window_search"] = [r for r in res["window_search"] if not (r["variant"] == variant and r.get("window", 10 if variant != "zib" else 8) == size and r["sampler"] in [x[0] for x in samplers])]
    for name, sampler in samplers:
        t0 = time.perf_counter()
        final, nw, nq = search(start, variant, sampler, size=size)
        assert is_valid(g, final, **rule)
        row = {"variant": variant, "sampler": name, "start": len(start), "final": len(final), "optimum": opt,
               "windows": nw, "max_qubits": nq, "window": size, "seconds": round(time.perf_counter() - t0, 1), "buses": final}
        res["window_search"].append(row)
        print(f"{variant:5s} {name:16s}: {len(start)} -> {len(final)} PMUs (optimum {opt}), {nw} windows of up to {nq} qubits, {row['seconds']} s", flush=True)
        save()

# gate count of a typical plain window circuit (one QAOA layer)
if PART in ("all", "plain"):
  Wb = window(res["Greedy"]["buses"][0], 10)
  _, _, cons = window_problem(res["Greedy"]["buses"], Wb, "plain")
  res["window_cx_per_layer"] = cx_count(qaoa_circuit_cover(Wb, [c for c in cons if c[1] == 1], [0.4, 0.3], measure=False))[0]
  print("CX per layer for a 10-bus window:", res["window_cx_per_layer"])
save()
