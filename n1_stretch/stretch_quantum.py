"""Stretch goals on the quantum side, inside the team's pipeline.

Pipeline (same as plain coverage, generalized to any per-bus need):
  1. exact reduction  reduce_need(g, need)   need[b] = 1 (seen once) or 2 (seen twice)
  2. split            independent pieces     (as on the 55k / 70k grids)
  3. quantum          QAOA per piece <= MAX_Q qubits; bigger pieces go to ILP
  4. reassemble       forced + piece answers, checked with the hackathon checker

Goal: N-1 (every bus seen by at least two PMUs, so losing any one PMU keeps the grid observable).

Cost per piece:  C(x) = sensors + P * sum_rules max(0, need - sightings)
The cost is written as a sum of Pauli-Z terms (exact, no approximation), so every piece has a
real circuit: each Z-term of weight w costs 2(w-1) CX, compiled by Qiskit for all-to-all hardware.

Run: python stretch_quantum.py  -> results/stretch_quantum.json, hardware/stretch/*.qasm
"""
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds, minimize

from pmu_lib import load_grid, group, apply_rx_all, cvar, ilp_placement
from check import is_valid, seen_buses
from stretch2 import reduce_need
from pieces import split_pieces

P = 2.0
ALPHA = 0.25
MAX_Q = 16
SHOTS = 200
TWO_Q_ERR = 0.005       # same rough Forte-like two-qubit error the team used in noise_study
OUT_QASM = Path("hardware/stretch")


# ------------------------------------------------------------------------------------------
# piece -> cost vector and Pauli-Z terms
# ------------------------------------------------------------------------------------------
def piece_arrays(piece):
    cands = piece["cands"]
    n = len(cands)
    idx = {v: i for i, v in enumerate(cands)}
    Z = np.arange(2 ** n, dtype=np.int64)
    sensors = np.bitwise_count(Z).astype(np.int64)
    short = np.zeros(2 ** n, dtype=np.int64)
    rules = []
    for s, k, _ in piece["cons"]:
        local = sorted(idx[v] for v in s if v in idx)
        m = sum(1 << i for i in local)
        short += np.maximum(k - np.bitwise_count(Z & m).astype(np.int64), 0)
        rules.append((local, k))
    return sensors, short, rules


def pauli_terms(n, rules):
    """Exact Z-expansion of sensors + P*sum max(0, k - weight(S)).  Returns {tuple(qubits): coeff}."""
    terms = {}
    for i in range(n):                         # x_i = (1 - z_i)/2  -> constant dropped
        terms[(i,)] = terms.get((i,), 0) - 0.5
    for local, k in rules:
        m = len(local)
        xs = np.array(list(itertools.product([0, 1], repeat=m)))
        f = P * np.maximum(k - xs.sum(1), 0)
        zs = 1 - 2 * xs
        for r in range(1, m + 1):
            for T in itertools.combinations(range(m), r):
                c = float(np.mean(f * np.prod(zs[:, T], axis=1)))
                if abs(c) > 1e-12:
                    key = tuple(sorted(local[t] for t in T))
                    terms[key] = terms.get(key, 0) + c
    return {k: v for k, v in terms.items() if abs(v) > 1e-12}


def build_circuit(n, terms, gammas, betas, measure=True, multi=None):
    """QAOA circuit. multi = list (per layer) of dicts term->angle and per-qubit mixer angles (ma-QAOA)."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n, n) if measure else QuantumCircuit(n)
    qc.h(range(n))
    for L in range(len(gammas)):
        for T, c in terms.items():
            ang = 2 * (multi[L]["g"][T] if multi else gammas[L] * c)
            if len(T) == 1:
                qc.rz(ang, T[0])
            else:
                for a, b in zip(T[:-1], T[1:]):
                    qc.cx(a, b)
                qc.rz(ang, T[-1])
                for a, b in reversed(list(zip(T[:-1], T[1:]))):
                    qc.cx(a, b)
        for q in range(n):
            qc.rx(2 * (multi[L]["b"][q] if multi else betas[L]), q)
    if measure:
        qc.measure(range(n), range(n))
    return qc


def cx_count(qc):
    from qiskit import transpile
    t = transpile(qc, basis_gates=["cx", "rz", "sx", "x"], optimization_level=3)
    return t.count_ops().get("cx", 0), t


# ------------------------------------------------------------------------------------------
# simulation and tuning (numpy statevector; identical to the circuit up to a global phase)
# ------------------------------------------------------------------------------------------
def probs_std(costs, n, params):
    L = len(params) // 2
    st = np.full(2 ** n, 2 ** (-n / 2), dtype=complex)
    for gm, bt in zip(params[:L], params[L:]):
        st = apply_rx_all(st * np.exp(-1j * gm * costs), n, bt)
    return np.abs(st) ** 2


def term_matrix(n, terms):
    Z = np.arange(2 ** n, dtype=np.int64)
    keys = list(terms)
    M = np.empty((len(keys), 2 ** n))
    for r, T in enumerate(keys):
        par = np.zeros(2 ** n, dtype=np.int64)
        for q in T:
            par ^= (Z >> q) & 1
        M[r] = 1 - 2 * par                     # eigenvalue of Z_T
    return keys, M


def apply_rx_each(state, n, betas):
    psi = state.reshape([2] * n)
    for q in range(n):
        ax = n - 1 - q
        c, s = np.cos(betas[q]), -1j * np.sin(betas[q])
        a0, a1 = np.take(psi, 0, axis=ax), np.take(psi, 1, axis=ax)
        psi = np.stack([c * a0 + s * a1, s * a0 + c * a1], axis=ax)
    return psi.reshape(-1)


def probs_multi(M, n, params, nt):
    st = np.full(2 ** n, 2 ** (-n / 2), dtype=complex)
    per = nt + n
    for L in range(len(params) // per):
        g = params[L * per: L * per + nt]
        b = params[L * per + nt: (L + 1) * per]
        st = apply_rx_each(st * np.exp(-1j * (g @ M)), n, b)
    return np.abs(st) ** 2


def tune_std(costs, n, layers, maxiter=200, alpha=ALPHA):
    obj = lambda p: cvar(probs_std(costs, n, p), costs, alpha)
    out, prev = [], None
    for L in range(1, layers + 1):
        if prev is None:
            starts = [np.array([a, b]) for a in (0.2, 0.5, 1.0) for b in (0.3, 0.6)]
        else:
            g_, b_ = prev[:L - 1], prev[L - 1:]
            starts = [np.concatenate([g_, [g_[-1]], b_, [b_[-1] * 0.5]]),
                      np.concatenate([g_, [g_[-1]], b_, [b_[-1]]])]
        best = min((minimize(obj, x0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.2})
                    for x0 in starts), key=lambda r: r.fun)
        prev = best.x
        out.append((L, best.x))
    return out


def tune_multi(costs, n, terms, std_params, maxiter=600, alpha=ALPHA):
    """ma-QAOA, 1 layer, started from the standard 1-layer angles."""
    keys, M = term_matrix(n, terms)
    nt = len(keys)
    g0, b0 = std_params
    x0 = np.concatenate([g0 * np.array([terms[k] for k in keys]), np.full(n, b0)])
    obj = lambda p: cvar(probs_multi(M, n, p, nt), costs, alpha)
    r = minimize(obj, x0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.1})
    return keys, r.x, probs_multi(M, n, r.x, nt)


def noisy_p_opt(qc_t, is_opt, shots=4000):
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(TWO_Q_ERR, 2), ["cx"])
    nm.add_all_qubit_quantum_error(depolarizing_error(TWO_Q_ERR / 10, 1), ["sx", "x"])
    counts = AerSimulator(noise_model=nm).run(qc_t, shots=shots, seed_simulator=7).result().get_counts()
    hit = sum(c for k, c in counts.items() if is_opt[int(k.replace(" ", ""), 2)])
    return hit / shots


# ------------------------------------------------------------------------------------------
# one goal on one grid
# ------------------------------------------------------------------------------------------
def ilp_cover(cands, cons):
    idx = {v: i for i, v in enumerate(cands)}
    A = np.array([[1 if v in s else 0 for v in cands] for s, k, _ in cons])
    lb = np.array([k for s, k, _ in cons])
    t0 = time.perf_counter()
    r = milp(np.ones(len(cands)), constraints=LinearConstraint(A, lb=lb), integrality=np.ones(len(cands)),
             bounds=Bounds(0, 1))
    return [v for v, x in zip(cands, r.x) if x > 0.5], 1000 * (time.perf_counter() - t0)


def meets(g, chosen, need):
    cnt = {b: 0 for b in g["buses"]}
    for c in chosen:
        for b in group(g, c):
            cnt[b] += 1
    return all(cnt[b] >= need[b] for b in g["buses"])


def run_goal(g, name, goal, need, layers=3, export=True, rng=None):
    rng = rng or np.random.default_rng(0)
    t0 = time.perf_counter()
    red = reduce_need(g, need)
    t_red = 1000 * (time.perf_counter() - t0)
    pieces = split_pieces(red)
    full_ilp, full_ms = ilp_cover(g["buses"], [(group(g, b), need[b], b) for b in g["buses"]])
    row = {"grid": name, "goal": goal, "ilp": len(full_ilp), "ilp_ms": round(full_ms, 1),
           "forced": len(red["forced"]), "reduce_ms": round(t_red, 1), "pieces": []}
    chosen = list(red["forced"])
    exact_total = len(red["forced"])
    for pi, pc in enumerate(pieces):
        n = len(pc["cands"])
        best_sol, _ = ilp_cover(pc["cands"], pc["cons"])
        opt = len(best_sol)
        exact_total += opt
        prow = {"piece": pi, "qubits": n, "rules": len(pc["cons"]), "optimum": opt}
        if n > MAX_Q:
            prow["solver"] = "ILP (too big for quantum)"
            chosen += best_sol
            row["pieces"].append(prow)
            continue
        sensors, short, rules = piece_arrays(pc)
        costs = (sensors + P * short).astype(float)
        valid = short == 0
        is_opt = valid & (sensors == opt)
        terms = pauli_terms(n, rules)
        assert np.allclose(costs - costs.mean(),
                           sum(c * term_matrix(n, {T: 1})[1][0] for T, c in terms.items()) -
                           sum(c * term_matrix(n, {T: 1})[1][0] for T, c in terms.items()).mean())
        runs = []
        # tune with the plain mean (alpha=1) and with CVaR(0.25); keep the better one per depth
        tuned = {a: tune_std(costs, n, layers, alpha=a) for a in (1.0, 0.25)}
        res, used = [], []
        for L in range(1, layers + 1):
            a_best = max(tuned, key=lambda a: probs_std(costs, n, tuned[a][L - 1][1])[is_opt].sum())
            res.append(tuned[a_best][L - 1]); used.append(a_best)
        for (L, prm), a_used in zip(res, used):
            pr = probs_std(costs, n, prm)
            qc = build_circuit(n, terms, prm[:L], prm[L:])
            cx, qc_t = cx_count(qc)
            r = {"method": f"QAOA {L} layer", "layers": L, "cx": cx, "alpha": a_used,
                 "p_opt": round(float(pr[is_opt].sum()), 4), "p_valid": round(float(pr[valid].sum()), 4)}
            if cx <= 300 and n <= 14:
                r["p_opt_noisy"] = round(noisy_p_opt(qc_t, is_opt), 4)
            runs.append(r)
            if export and L <= 2:
                OUT_QASM.mkdir(parents=True, exist_ok=True)
                from qiskit import qasm2
                (OUT_QASM / f"{name}_{goal}_piece{pi}_L{L}.qasm").write_text(qasm2.dumps(qc_t))
        # multi-angle QAOA, 1 layer (the change that helped most on the Rigetti chip)
        if n <= 12:
            g1 = res[0][1]
            cand = [tune_multi(costs, n, terms, (g1[0], g1[1]), alpha=a) for a in (1.0, 0.25)]
            keys, prm, pr = max(cand, key=lambda c: c[2][is_opt].sum())
            multi = [{"g": {k: prm[i] for i, k in enumerate(keys)}, "b": prm[len(keys):]}]
            qc = build_circuit(n, terms, [0], [0], multi=multi)
            cx, qc_t = cx_count(qc)
            r = {"method": "multi-angle QAOA 1 layer", "layers": 1, "cx": cx,
                 "p_opt": round(float(pr[is_opt].sum()), 4), "p_valid": round(float(pr[valid].sum()), 4)}
            if cx <= 300:
                r["p_opt_noisy"] = round(noisy_p_opt(qc_t, is_opt), 4)
            runs.append(r)
            if export:
                from qiskit import qasm2
                (OUT_QASM / f"{name}_{goal}_piece{pi}_maqaoa.qasm").write_text(qasm2.dumps(qc_t))
        prow.update({"solver": "QAOA", "random_p_opt": round(float(is_opt.mean()), 6), "runs": runs})
        # sample 200 shots from the best-depth circuit and keep the best valid answer
        best_run = max(range(len(res)), key=lambda i: runs[i]["p_opt"])
        pr = probs_std(costs, n, res[best_run][1])
        shots = rng.choice(2 ** n, size=SHOTS, p=pr / pr.sum())
        vshots = [z for z in shots if valid[z]]
        pick = min(vshots, key=lambda z: sensors[z]) if vshots else None
        if pick is None:
            prow["sampled"] = "no valid shot, fell back to ILP"
            chosen += best_sol
        else:
            prow["sampled_sensors"] = int(sensors[pick])
            chosen += [pc["cands"][i] for i in range(n) if (pick >> i) & 1]
        row["pieces"].append(prow)
    row["exact_check"] = exact_total == len(full_ilp)     # reduction + pieces lose nothing
    row["assembled"] = len(chosen)
    row["assembled_valid"] = meets(g, chosen, need)
    q = [p for p in row["pieces"] if p["solver"] == "QAOA"]
    row["quantum_pieces"] = len(q)
    row["quantum_qubits"] = sum(p["qubits"] for p in q)
    row["classical_pieces"] = len(row["pieces"]) - len(q)
    return row


if __name__ == "__main__":
    import sys
    grids = sys.argv[1:] or ["ieee14", "ieee30", "ieee57", "ieee118", "ieee300", "activsg2000"]
    out = {"settings": {"P": P, "alpha": ALPHA, "max_qubits_quantum": MAX_Q, "two_qubit_error": TWO_Q_ERR},
           "rows": []}
    Path("results").mkdir(exist_ok=True)
    for name in grids:
        g = load_grid(f"data/{name}.json")
        for goal, need in [("N-1", {b: 2 for b in g["buses"]})]:
            t0 = time.perf_counter()
            row = run_goal(g, name, goal, need)
            row["seconds"] = round(time.perf_counter() - t0, 1)
            out["rows"].append(row)
            summ = {k: v for k, v in row.items() if k != "pieces"}
            print(json.dumps(summ), flush=True)
            for p in row["pieces"]:
                if p["solver"] == "QAOA":
                    best = max(p["runs"], key=lambda r: r["p_opt"])
                    print(f"   piece {p['piece']}: {p['qubits']} q, opt {p['optimum']}, random {p['random_p_opt']:.4f} | "
                          + " | ".join(f"{r['method']}: {r['p_opt']:.3f} ideal" +
                                       (f", {r['p_opt_noisy']:.3f} noisy" if 'p_opt_noisy' in r else "") +
                                       f", {r['cx']} CX" for r in p["runs"]), flush=True)
                else:
                    print(f"   piece {p['piece']}: {p['qubits']} q -> {p['solver']}", flush=True)
            json.dump(out, open("results/stretch_quantum.json", "w"), indent=1, default=float)
    json.dump(out, open("results/stretch_quantum.json", "w"), indent=1, default=float)
    print("wrote results/stretch_quantum.json")
