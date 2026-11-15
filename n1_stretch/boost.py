"""Raising QAOA accuracy on hard pieces (e.g. IEEE 300 N-1, 12 qubits).

Approaches compared on the same piece:
  A  baseline        multi-angle QAOA, 1 layer, start in |+>
  B  warm start      start each qubit tilted toward the LP-relaxation value; matching mixer (no extra CX)
  C  2-layer warm    B with a second layer
  D  RQAOA + reduce  run a short circuit, fix the qubit QAOA is most sure about, re-run our exact
                     reduction rules on what is left, re-split into pieces, repeat. Circuits shrink
                     every round; small leftovers get one last short circuit.
Run: python boost.py  -> results/boost_study.json
"""
import json
import time
import numpy as np
from scipy.optimize import minimize, linprog

import stretch_quantum as sq
from pmu_lib import load_grid, cvar
from stretch2 import reduce_need
from pieces import split_pieces

P = sq.P


# ---------------------------------------------------------------- piece-level exact reduction
def reduce_piece(cands, cons, fixed):
    """cons: list of (set, need). fixed: {bus: 0/1}. Applies the fixings, then R1/R2/R0/R3' rules.
    Returns (forced_on list, remaining pieces [(cands, cons)]) or None if infeasible."""
    cands = set(cands)
    cons = [(set(s), k) for s, k in cons]
    on = []
    for v, val in fixed.items():
        if v not in cands:
            continue
        cands.discard(v)
        cons = [(s - {v}, max(0, k - 1) if (val and v in s) else k) for s, k in cons]
        if val:
            on.append(v)
    while True:
        cons = [(s & cands, k) for s, k in cons if k > 0]
        if any(len(s) < k for s, k in cons):
            return None
        hit = next(((s, k) for s, k in cons if len(s) == k), None)
        if hit:
            for v in list(hit[0]):
                on.append(v); cands.discard(v)
                cons = [(s - {v}, max(0, k - 1)) if v in s else (s, k) for s, k in cons]
            continue
        drop = None
        for i, (s1, k1) in enumerate(cons):
            for j, (s2, k2) in enumerate(cons):
                if i != j and s1 <= s2 and k1 >= k2 and (s1 != s2 or k1 > k2 or i < j):
                    drop = j; break
            if drop is not None:
                break
        if drop is not None:
            cons.pop(drop); continue
        used = set().union(*[s for s, _ in cons]) if cons else set()
        if cands - used:
            cands &= used; continue
        dom = None
        for u in sorted(cands):
            ru = [i for i, (s, k) in enumerate(cons) if u in s]
            if all(cons[i][1] == 1 for i in ru):
                for v in sorted(cands):
                    if v != u and all(v in cons[i][0] for i in ru):
                        rv = [i for i, (s, k) in enumerate(cons) if v in s]
                        if set(ru) != set(rv) or v < u:
                            dom = u; break
            if dom is not None:
                break
        if dom is not None:
            cands.discard(dom); continue
        break
    # split
    parent = {v: v for v in cands}
    def f(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]; v = parent[v]
        return v
    for s, k in cons:
        s = sorted(s)
        for v in s[1:]:
            parent[f(v)] = f(s[0])
    groups = {}
    for v in cands:
        groups.setdefault(f(v), set()).add(v)
    pieces = [(sorted(gp), [(s, k) for s, k in cons if s & gp]) for gp in groups.values()]
    return on, pieces


def as_piece(cands, cons):
    return {"cands": list(cands), "cons": [(frozenset(s), k, None) for s, k in cons]}


def piece_opt(cands, cons):
    sol, _ = sq.ilp_cover(list(cands), [(frozenset(s), k, None) for s, k in cons])
    return len(sol)


# ---------------------------------------------------------------- warm-start multi-angle QAOA
def lp_values(cands, cons):
    idx = {v: i for i, v in enumerate(cands)}
    A = np.array([[1 if v in s else 0 for v in cands] for s, k, *_ in cons])
    lb = np.array([k for s, k, *_ in cons])
    x = linprog(np.ones(len(cands)), A_ub=-A, b_ub=-lb, bounds=(0, 1), method="highs").x
    return np.clip(x, 0.1, 0.9)            # keep every answer reachable (Egger et al. regularisation)


def ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(t):
    return np.array([[np.exp(-1j * t / 2), 0], [0, np.exp(1j * t / 2)]])


def apply_1q(state, n, mats):
    psi = state.reshape([2] * n)
    for q, U in enumerate(mats):
        ax = n - 1 - q
        psi = np.moveaxis(np.tensordot(U, np.moveaxis(psi, ax, 0), axes=1), 0, ax)
    return psi.reshape(-1)


def ws_probs(M, n, thetas, params, nt, layers):
    st = np.zeros(2 ** n, dtype=complex); st[0] = 1
    st = apply_1q(st, n, [ry(t) for t in thetas])
    per = nt + n
    for L in range(layers):
        g = params[L * per: L * per + nt]; b = params[L * per + nt:(L + 1) * per]
        st = st * np.exp(-1j * (g @ M))
        st = apply_1q(st, n, [ry(t) @ rz(2 * bb) @ ry(-t) for t, bb in zip(thetas, b)])
    return np.abs(st) ** 2


def ws_circuit(n, keys, thetas, params, nt, layers):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n, n)
    for q, t in enumerate(thetas):
        qc.ry(t, q)
    per = nt + n
    for L in range(layers):
        g = params[L * per: L * per + nt]; b = params[L * per + nt:(L + 1) * per]
        for T, ang in zip(keys, g):
            if len(T) == 1:
                qc.rz(2 * ang, T[0])
            else:
                for a, c in zip(T[:-1], T[1:]):
                    qc.cx(a, c)
                qc.rz(2 * ang, T[-1])
                for a, c in reversed(list(zip(T[:-1], T[1:]))):
                    qc.cx(a, c)
        for q, (t, bb) in enumerate(zip(thetas, b)):
            qc.ry(-t, q); qc.rz(2 * bb, q); qc.ry(t, q)
    qc.measure(range(n), range(n))
    return qc


def run_ws(piece, layers=1, alphas=(1.0, 0.25), maxiter=1500, warm=True):
    n = len(piece["cands"])
    sensors, short, rules = sq.piece_arrays(piece)
    costs = (sensors + P * short).astype(float)
    valid = short == 0
    opt = sensors[valid].min()
    is_opt = valid & (sensors == opt)
    terms = sq.pauli_terms(n, rules)
    keys, M = sq.term_matrix(n, terms)
    nt = len(keys)
    thetas = 2 * np.arcsin(np.sqrt(lp_values(piece["cands"], piece["cons"]))) if warm else np.full(n, np.pi / 2)
    best = None
    for a in alphas:
        prev = None
        for L in range(1, layers + 1):
            base = np.concatenate([0.3 * np.array([terms[k] for k in keys]), np.full(n, 0.4)])
            x0 = base if prev is None else np.concatenate([prev, 0.5 * base])
            obj = lambda p, L=L: cvar(ws_probs(M, n, thetas, p, nt, L), costs, a)
            r = minimize(obj, x0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.15})
            prev = r.x
        pr = ws_probs(M, n, thetas, prev, nt, layers)
        if best is None or pr[is_opt].sum() > best[1][is_opt].sum():
            best = (prev, pr)
    prm, pr = best
    qc = ws_circuit(n, keys, thetas, prm, nt, layers)
    cx, qc_t = sq.cx_count(qc)
    return {"p_opt": float(pr[is_opt].sum()), "p_valid": float(pr[valid].sum()), "cx": cx,
            "p_opt_noisy": sq.noisy_p_opt(qc_t, is_opt) if cx <= 700 else None,
            "random": float(is_opt.mean()), "probs": pr, "qc_t": qc_t}


# ---------------------------------------------------------------- RQAOA with exact re-reduction
def rqaoa(cands, cons, small=5, noisy=True, log=None):
    """Returns (sensors chosen, list of circuits run [(qubits, cx)], success flag vs exact optimum)."""
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(sq.TWO_Q_ERR, 2), ["cx"])
    nm.add_all_qubit_quantum_error(depolarizing_error(sq.TWO_Q_ERR / 10, 1), ["sx", "x"])
    sim = AerSimulator(noise_model=nm) if noisy else AerSimulator()
    circuits, chosen = [], []
    stack = [(list(cands), [(set(s), k) for s, k in cons])]
    while stack:
        cs, cn = stack.pop()
        if not cs:
            continue
        piece = as_piece(cs, cn)
        res = run_ws(piece, layers=1, maxiter=800)
        circuits.append((len(cs), res["cx"]))
        counts = sim.run(res["qc_t"], shots=1000, seed_simulator=11).result().get_counts()
        n = len(cs)
        if n <= small:
            # last round: keep the best valid measured answer (shots are checked classically)
            sensors, short, _ = sq.piece_arrays(piece)
            zs = [int(k, 2) for k in counts]
            ok = [z for z in zs if short[z] == 0]
            z = min(ok, key=lambda z: sensors[z]) if ok else None
            if z is None:
                return None, circuits, False
            chosen += [cs[i] for i in range(n) if (z >> i) & 1]
            continue
        # marginals <x_i> from the measured shots; fix the most confident qubit
        tot = sum(counts.values())
        m = np.zeros(n)
        for k, c in counts.items():
            z = int(k, 2)
            for i in range(n):
                m[i] += c * ((z >> i) & 1)
        m /= tot
        i = int(np.argmax(np.abs(m - 0.5)))
        val = int(m[i] > 0.5)
        out = reduce_piece(cs, cn, {cs[i]: val})
        if out is None:                       # that fix is impossible: take the other value
            out = reduce_piece(cs, cn, {cs[i]: 1 - val}); val = 1 - val
        on, pieces = out
        if log is not None:
            log.append(f"{n} qubits: fix bus {cs[i]} = {val} (QAOA {m[i]:.2f}); rules then place {len(on) - val} more -> pieces {[len(p[0]) for p in pieces]}")
        chosen += on
        stack += pieces
    return chosen, circuits, True


if __name__ == "__main__":
    out = {}
    for name, goal in [("ieee300", "N-1"), ("ieee14", "N-1")]:
        g = load_grid(f"data/{name}.json")
        need = {b: 2 for b in g["buses"]}
        pcs = [p for p in split_pieces(reduce_need(g, need)) if 8 <= len(p["cands"]) <= 16]
        for pc in pcs:
            key = f"{name} {goal} {len(pc['cands'])}q"
            row = {}
            t0 = time.perf_counter()
            for label, kw in [("A multi-angle, |+> start", dict(warm=False)), ("B warm start", dict(warm=True)),
                              ("C warm start, 2 layers", dict(warm=True, layers=2, maxiter=2500))]:
                r = run_ws(pc, **kw)
                row[label] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                              if k in ("p_opt", "p_valid", "cx", "p_opt_noisy", "random")}
                print(key, label, row[label], flush=True)
            log = []
            cons = [(set(s), k) for s, k, _ in pc["cons"]]
            chosen, circ, ok = rqaoa(pc["cands"], cons, log=log)
            opt = piece_opt(pc["cands"], cons)
            row["D RQAOA + exact reduction (noisy)"] = {
                "found": len(chosen) if chosen is not None else None, "optimum": opt,
                "optimal": chosen is not None and len(chosen) == opt,
                "circuits": circ, "max_cx": max(c for _, c in circ), "steps": log}
            print(key, "D", row["D RQAOA + exact reduction (noisy)"], flush=True)
            row["seconds"] = round(time.perf_counter() - t0, 1)
            out[key] = row
            json.dump(out, open("results/boost_study.json", "w"), indent=1, default=float)
