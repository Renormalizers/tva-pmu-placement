"""Our team's toolkit for the TVA PMU placement track.

Everything the notebook and the dashboard use lives here:
  classical: greedy, exact ILPs (plain, N-1, zero-injection, weighted fixed budget)
  reduction: exact preprocessing rules that shrink the problem before it goes to qubits
  quantum:   QAOA cost tables, a fast numpy statevector simulator, Qiskit circuits for hardware

Bus numbers are used everywhere at the interface. Qubit i stands for cands[i].
"""
import itertools
import time
from math import comb

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds, minimize

from check import load_grid, seen_buses, is_valid, brute_force_best, check_placement  # noqa: F401


# ----------------------------------------------------------------------------------------------
# Classical baselines
# ----------------------------------------------------------------------------------------------
def group(g, b):
    return {b} | g["neighbors"][b]


def greedy_placement(g, twice=False):
    """Add the bus that covers the most still-needed sightings until every bus is seen (twice if asked)."""
    need = {b: (2 if twice else 1) for b in g["buses"]}
    chosen = []
    while any(v > 0 for v in need.values()):
        b = max((c for c in g["buses"] if c not in chosen),
                key=lambda c: (sum(1 for u in group(g, c) if need[u] > 0), -c))
        chosen.append(b)
        for u in group(g, b):
            need[u] = max(0, need[u] - 1)
    return sorted(chosen)


def ilp_placement(g, twice=False, fixed=()):
    """Exact minimum: every bus needs >= 1 (or >= 2) sensors among itself and its neighbors."""
    buses = g["buses"]
    A = np.array([[1 if c in group(g, b) else 0 for c in buses] for b in buses])
    lb = np.zeros(len(buses))
    lb[[buses.index(b) for b in fixed]] = 1
    res = milp(c=np.ones(len(buses)), constraints=LinearConstraint(A, lb=2 if twice else 1),
               integrality=np.ones(len(buses)), bounds=Bounds(lb, 1))
    return [b for b, x in zip(buses, res.x) if x > 0.5]


def ilp_placement_zib(g):
    """Exact minimum with the zero-injection rule, as an ILP with a propagation order.

    x_b   sensor at b
    y_zm  zero-injection bus z is used to infer bus m (m in N[z]); each z is used at most once
    t_b   order in which b becomes seen; inferring m from z needs every other bus in N[z] earlier
    Every bus is seen directly or inferred once. The order rules out circular inferences, so a
    feasible solution is exactly a placement whose propagation (as in check.seen_buses) sees all.
    """
    buses, Z = g["buses"], g["zero_injection"]
    n = len(buses)
    idx = {b: i for i, b in enumerate(buses)}
    ypairs = [(z, m) for z in Z for m in sorted(group(g, z))]
    nx_, ny = n, len(ypairs)
    nv = nx_ + ny + n  # x, y, t
    rows, lbs, ubs = [], [], []

    def row():
        return np.zeros(nv)

    for b in buses:  # seen directly or inferred
        r = row()
        for j in group(g, b):
            r[idx[j]] = 1
        for k, (z, m) in enumerate(ypairs):
            if m == b:
                r[nx_ + k] = 1
        rows.append(r); lbs.append(1); ubs.append(np.inf)
    for z in Z:  # each zero-injection equation used once
        r = row()
        for k, (zz, m) in enumerate(ypairs):
            if zz == z:
                r[nx_ + k] = 1
        rows.append(r); lbs.append(-np.inf); ubs.append(1)
    M = n + 1
    for k, (z, m) in enumerate(ypairs):  # t_m >= t_o + 1 - M(1 - y)
        for o in group(g, z) - {m}:
            r = row()
            r[nx_ + ny + idx[m]] = 1
            r[nx_ + ny + idx[o]] = -1
            r[nx_ + k] = -M
            rows.append(r); lbs.append(1 - M); ubs.append(np.inf)
    c = np.concatenate([np.ones(nx_), np.zeros(ny + n)])
    integ = np.concatenate([np.ones(nx_ + ny), np.zeros(n)])
    lo = np.zeros(nv)
    hi = np.concatenate([np.ones(nx_ + ny), np.full(n, float(n))])
    res = milp(c=c, constraints=LinearConstraint(np.array(rows), lbs, ubs), integrality=integ,
               bounds=Bounds(lo, hi))
    return [b for b, x in zip(buses, res.x[:n]) if x > 0.5]


def ilp_n1_zib(g, max_rounds=2000):
    """Exact minimum that survives any one sensor failure, with the zero-injection rule.

    Lazy cuts: solve, check, and if losing sensor l from placement C breaks observability, every
    valid placement must have at least 2 sensors outside S = C - {l} (observability is monotone,
    so if all but one sensor sat inside S, losing that one would leave a subset of S).
    Buses outside every zero-injection group can only be seen directly, so they need 2 sensors
    nearby; buses inside a group may be inferred, so they start with no direct requirement.
    """
    buses = g["buses"]
    n = len(buses)
    zgroups = set().union(*[group(g, z) for z in g["zero_injection"]]) if g["zero_injection"] else set()
    rows, lbs = [], []
    for b in buses:
        rows.append([1 if c in group(g, b) else 0 for c in buses])
        lbs.append(0 if b in zgroups else 2)
    for _ in range(max_rounds):
        res = milp(c=np.ones(n), constraints=LinearConstraint(np.array(rows), lb=lbs),
                   integrality=np.ones(n), bounds=Bounds(0, 1))
        chosen = [b for b, x in zip(buses, res.x) if x > 0.5]
        if is_valid(g, chosen, zib=True, n1=True):
            return chosen
        if not is_valid(g, chosen, zib=True):
            S = set(chosen)
        else:
            lost = next(b for b in chosen if not is_valid(g, [c for c in chosen if c != b], zib=True))
            S = set(chosen) - {lost}
        rows.append([0 if b in S else 1 for b in buses])
        lbs.append(2)
    raise RuntimeError("did not converge")


def ilp_budget(g, k, weights=None, zib=False):
    """Fixed budget: exactly k sensors, maximize the total weight of buses seen (no ZIB)."""
    buses = g["buses"]
    n = len(buses)
    w = np.array([1.0 if weights is None else weights[b] for b in buses])
    # vars: x (n), s (n). max w.s  s.t. s_b <= sum_{j in N[b]} x_j, sum x = k
    A = []
    for i, b in enumerate(buses):
        r = np.zeros(2 * n)
        r[n + i] = 1
        for j in group(g, b):
            r[buses.index(j)] -= 1
        A.append(r)
    r = np.zeros(2 * n); r[:n] = 1
    cons = [LinearConstraint(np.array(A), ub=0), LinearConstraint(r, lb=k, ub=k)]
    res = milp(c=np.concatenate([np.zeros(n), -w]), constraints=cons, integrality=np.ones(2 * n),
               bounds=Bounds(0, 1))
    return [b for b, x in zip(buses, res.x[:n]) if x > 0.5]


def timed(fn, *a, repeat=5, **k):
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(*a, **k)
        best = min(best, time.perf_counter() - t0)
    return out, best * 1000


# ----------------------------------------------------------------------------------------------
# Exact reduction ("kernelization") before going to qubits
# ----------------------------------------------------------------------------------------------
def reduce_problem(g, twice=False, log=False):
    """Shrink the covering problem with rules that never lose an optimal answer.

    Elements are buses that still need a sighting; candidates are buses where a sensor could go.
      R1 forced:     a bus that only one remaining candidate can see forces that candidate
      R2 element:    if every candidate that sees u also sees w, covering u covers w: drop w
      R3 candidate:  if candidate u sees a subset of what candidate v sees, drop u
    (twice=True: each bus needs 2 sightings; only R1 is used, forcing when exactly 2 can see it.)

    Returns forced buses, remaining candidate buses, and constraints [(set of candidates, need, buses)].
    """
    need = {b: (2 if twice else 1) for b in g["buses"]}
    cands = set(g["buses"])
    forced, steps = [], []
    G = {b: group(g, b) for b in g["buses"]}

    while True:
        live = sorted(b for b in need if need[b] > 0)
        seers = {u: frozenset(v for v in G[u] if v in cands) for u in live}
        # R1
        hit = next((u for u in live if len(seers[u]) <= need[u]), None)
        if hit is not None:
            for v in sorted(seers[hit]):
                forced.append(v)
                cands.discard(v)
                for w in G[v]:
                    need[w] = max(0, need[w] - 1)
            steps.append(f"R1: bus {hit} can only be seen from {sorted(seers[hit])}, so put sensors there")
            continue
        if twice:
            break
        # R2
        done = False
        for u in live:
            for w in live:
                if u != w and seers[u] <= seers[w] and (seers[u] != seers[w] or u < w):
                    need[w] = 0
                    steps.append(f"R2: whatever sees bus {u} also sees bus {w}, so drop bus {w}'s rule")
                    done = True
                    break
            if done:
                break
        if done:
            continue
        # R3
        cov = {v: frozenset(u for u in G[v] if need[u] > 0) for v in cands}
        for u in sorted(cands):
            if not cov[u]:
                cands.discard(u)
                steps.append(f"R3: a sensor at {u} would see nothing still needed, so drop it")
                done = True
                break
            dom = [v for v in cands if v != u and cov[u] <= cov[v] and (cov[u] != cov[v] or v < u)]
            if dom:
                cands.discard(u)
                steps.append(f"R3: a sensor at {min(dom)} sees everything {u} would, so drop candidate {u}")
                done = True
                break
        if not done:
            break
    cands = sorted(cands)
    merged = {}
    for u in sorted(b for b in need if need[b] > 0):
        s_ = frozenset(v for v in cands if v in G[u])
        merged.setdefault((s_, need[u]), []).append(u)
    cons = [(set(s_), k, us) for (s_, k), us in merged.items()]
    if log:
        for st in steps:
            print(" ", st)
    return {"forced": sorted(forced), "cands": cands, "constraints": cons, "steps": steps}


def components(red):
    """Split the reduced problem into independent pieces (candidates linked by shared constraints)."""
    parent = {v: v for v in red["cands"]}

    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v
    for s_, _, _ in red["constraints"]:
        s_ = sorted(s_)
        for v in s_[1:]:
            parent[find(v)] = find(s_[0])
    comps = {}
    for v in red["cands"]:
        comps.setdefault(find(v), []).append(v)
    out = []
    for vs in comps.values():
        vs = sorted(vs)
        cons = [c for c in red["constraints"] if c[0] and min(c[0]) in vs]
        out.append({"cands": vs, "constraints": cons})
    return sorted(out, key=lambda c: -len(c["cands"]))


def solve_piece_exact(cands, constraints):
    """Brute-force optimum of a reduced piece (small pieces only)."""
    for k in range(len(cands) + 1):
        for c in itertools.combinations(cands, k):
            cs = set(c)
            if all(len(cs & s_) >= need for s_, need, _ in constraints):
                return list(c)


# ----------------------------------------------------------------------------------------------
# QAOA: cost tables, numpy statevector simulator, Qiskit circuits
# ----------------------------------------------------------------------------------------------
def all_bits(n):
    z = np.arange(2 ** n)
    return ((z[:, None] >> np.arange(n)) & 1).astype(np.int8)  # row z, column i = qubit i


def cover_cost_table(cands, constraints, P=2.0):
    """C(x) = #sensors + P * sum over constraints of max(0, need - sightings).
    Qubit i is cands[i]. For need=1 this is exactly sensors + P * unseen (the starter's cost)."""
    n = len(cands)
    X = all_bits(n)
    idx = {b: i for i, b in enumerate(cands)}
    cost = X.sum(1).astype(float)
    for s, need, _ in constraints:
        cols = [idx[b] for b in s]
        sight = X[:, cols].sum(1) if cols else np.zeros(len(X))
        cost += P * np.maximum(0, need - sight)
    return cost


def grid_constraints(g, twice=False):
    return [({b} | g["neighbors"][b], 2 if twice else 1, [b]) for b in g["buses"]]


def zib_cost_table(g, P=2.0, n1=False):
    """Cost with the zero-injection rule (and optionally N-1): sensors + P * buses left unseen.
    Not a low-degree polynomial, so it is simulation-only (no compact circuit)."""
    n = len(g["buses"])
    X = all_bits(n)
    out = np.empty(2 ** n)
    allb = set(g["buses"])
    for z in range(2 ** n):
        ch = [b for b, x in zip(g["buses"], X[z]) if x]
        unseen = len(allb - seen_buses(g, ch, zib=True))
        if n1 and unseen == 0:
            worst = max((len(allb - seen_buses(g, [c for c in ch if c != l], zib=True)) for l in ch), default=0)
            unseen = 0.5 * worst  # partial penalty for a placement that breaks under one failure
        out[z] = len(ch) + P * unseen
    return out


def apply_rx_all(state, n, beta):
    """Apply rx(2*beta) on every qubit of a statevector of length 2**n (qubit i = bit i)."""
    c, s = np.cos(beta), -1j * np.sin(beta)
    psi = state.reshape([2] * n)  # axis n-1-i is qubit i
    for ax in range(n):
        a0 = np.take(psi, 0, axis=ax)
        a1 = np.take(psi, 1, axis=ax)
        psi = np.stack([c * a0 + s * a1, s * a0 + c * a1], axis=ax)
    return psi.reshape(-1)


def qaoa_state(costs, params, n):
    L = len(params) // 2
    st = np.full(2 ** n, 2 ** (-n / 2), dtype=complex)
    for gm, bt in zip(params[:L], params[L:]):
        st = st * np.exp(-1j * gm * costs)
        st = apply_rx_all(st, n, bt)
    return st


def cvar(probs, costs, alpha):
    order = np.argsort(costs, kind="stable")
    pr, c = probs[order], costs[order]
    take = np.clip(alpha - (np.cumsum(pr) - pr), 0, pr)
    return take @ c / alpha


def run_qaoa(costs, n, layers=1, alpha=0.25, maxiter=300, starts=None, seed=0):
    """Tune angles on the exact statevector with several starts; return (params, probs)."""
    rng = np.random.default_rng(seed)
    obj = lambda p: cvar(np.abs(qaoa_state(costs, p, n)) ** 2, costs, alpha)
    k = (np.arange(layers) + 0.5) / layers
    x0s = starts or [np.concatenate([k * a, (1 - k) * b]) for a, b in
                     [(0.2, 0.2), (0.3, 0.6), (0.6, 0.3), (0.4, 0.4), (0.8, 0.5), (1.2, 0.4)]]
    x0s = list(x0s) + [rng.uniform(0, [1.5] * layers + [0.8] * layers) for _ in range(4)]
    best = None
    for x0 in x0s:
        r = minimize(obj, x0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.2})
        if best is None or r.fun < best.fun:
            best = r
    return best.x, np.abs(qaoa_state(costs, best.x, n)) ** 2


def interp_init(params):
    """INTERP warm start: grow an L-layer schedule into an (L+1)-layer one (Zhou et al. 2020)."""
    L = len(params) // 2
    out = []
    for half in (params[:L], params[L:]):
        new = np.zeros(L + 1)
        for i in range(L + 1):
            a = half[i - 1] if i - 1 >= 0 else 0
            b = half[i] if i < L else 0
            new[i] = (i / L) * a + ((L - i) / L) * b
        out.append(new)
    return np.concatenate(out)


def outcome_shares(probs, costs_valid, sensors, best):
    """Split probability into optimal / valid-but-extra / invalid."""
    valid = costs_valid
    return {"optimal": float(probs[valid & (sensors == best)].sum()),
            "valid, extra sensors": float(probs[valid & (sensors > best)].sum()),
            "leaves a bus unseen": float(probs[~valid].sum())}


def qaoa_circuit_cover(cands, constraints, params, P=2.0, measure=True):
    """Qiskit circuit for the cover cost (need=1 constraints only): one qubit per candidate."""
    from qiskit import QuantumCircuit
    n = len(cands)
    idx = {b: i for i, b in enumerate(cands)}
    L = len(params) // 2
    qc = QuantumCircuit(n, n) if measure else QuantumCircuit(n)
    qc.h(range(n))
    for gm, bt in zip(params[:L], params[L:]):
        for i in range(n):
            qc.p(-gm, i)
        for s, need, _ in constraints:
            assert need == 1
            q = sorted(idx[b] for b in s)
            qc.x(q)
            if len(q) == 1:
                qc.p(-gm * P, q[0])
            else:
                qc.mcp(-gm * P, q[:-1], q[-1])
            qc.x(q)
        qc.rx(2 * bt, range(n))
    if measure:
        qc.measure(range(n), range(n))
    return qc


def cx_count(qc):
    from qiskit import transpile
    t = transpile(qc, basis_gates=["rz", "sx", "x", "cx"], optimization_level=3, seed_transpiler=7)
    return t.count_ops().get("cx", 0), t


# ----------------------------------------------------------------------------------------------
# Fixed budget with a Hamming-weight-preserving (XY) mixer, simulated in the weight-k subspace
# ----------------------------------------------------------------------------------------------
def budget_subspace(n, k):
    states = [sum(1 << i for i in c) for c in itertools.combinations(range(n), k)]
    pos = {s: j for j, s in enumerate(states)}
    return states, pos


def xy_ring_hamiltonian(n, k, complete=True):
    """Sum over pairs (a, b) of (XX + YY)/2, i.e. moving a sensor from one bus to another,
    restricted to weight-k states. complete=True uses every pair; False uses a ring."""
    states, pos = budget_subspace(n, k)
    H = np.zeros((len(states), len(states)))
    pairs = list(itertools.combinations(range(n), 2)) if complete else [(i, (i + 1) % n) for i in range(n)]
    for j, s in enumerate(states):
        for a, b in pairs:
            if ((s >> a) & 1) != ((s >> b) & 1):
                t = s ^ (1 << a) ^ (1 << b)
                H[pos[t], j] = 1.0
    return H, states


def run_budget_qaoa(score_of_state, n, k, layers=2, maxiter=300):
    """Maximize score with exactly k sensors. Every shot uses exactly k sensors by construction."""
    from scipy.linalg import expm  # noqa
    H, states = xy_ring_hamiltonian(n, k)
    w, V = np.linalg.eigh(H)
    f = np.array([score_of_state(s) for s in states], dtype=float)
    cost = -f / f.max()   # scale to [-1, 0] so the angles live on a sensible range
    m = len(states)

    def state(p):
        L = len(p) // 2
        psi = np.full(m, 1 / np.sqrt(m), dtype=complex)  # Dicke state
        for gm, bt in zip(p[:L], p[L:]):
            psi = psi * np.exp(-1j * gm * cost)
            psi = V @ (np.exp(-1j * bt * w) * (V.conj().T @ psi))
        return psi

    obj = lambda p: cvar(np.abs(state(p)) ** 2, cost, 0.25)
    best = None
    k_ = (np.arange(layers) + 0.5) / layers
    for a, b in [(0.5, 0.3), (1.0, 0.3), (2.0, 0.2), (3.0, 0.15), (5.0, 0.1), (8.0, 0.1)]:
        r = minimize(obj, np.concatenate([k_ * a, (1 - k_) * b]), method="COBYLA",
                     options={"maxiter": maxiter, "rhobeg": 0.2})
        if best is None or r.fun < best.fun:
            best = r
    probs = np.abs(state(best.x)) ** 2
    return best.x, probs, states, f


# ----------------------------------------------------------------------------------------------
# N-1 with zero-injection, exact, by adding failure scenarios one at a time
# ----------------------------------------------------------------------------------------------
def ilp_n1_zib_scenarios(g, time_limit=1800, log=False, start_scenarios=()):
    """Fewest PMUs such that every bus stays observable (zero-injection rule on) after losing any one PMU.

    One shared x. For each failure scenario f we add a copy of the zero-injection ILP block
    (inference variables y^f and order variables t^f) in which bus f's sensor does not count.
    Requiring that block when x_f = 0 is harmless: then losing f changes nothing and the block is
    just the base observability condition. So the full model (all n scenarios) is exact; we add
    scenarios lazily, only for sensors whose loss actually breaks the current answer.
    """
    from scipy.sparse import lil_matrix
    buses, Z = g["buses"], g["zero_injection"]
    n = len(buses)
    idx = {b: i for i, b in enumerate(buses)}
    ypairs = [(z, m) for z in Z for m in sorted(group(g, z))]
    ny = len(ypairs)
    M = n + 1
    scen = [None] + list(start_scenarios)
    rounds = 0
    while True:
        rounds += 1
        S = len(scen)
        nv = n + S * (ny + n)
        rows = []  # (dict col->val, lb, ub)
        for s, f in enumerate(scen):
            yo, to = n + s * (ny + n), n + s * (ny + n) + ny
            for b in buses:
                r = {idx[j]: 1 for j in group(g, b) if j != f}
                for k, (z, m) in enumerate(ypairs):
                    if m == b:
                        r[yo + k] = 1
                rows.append((r, 1, np.inf))
            for z in Z:
                rows.append(({yo + k: 1 for k, (zz, _) in enumerate(ypairs) if zz == z}, -np.inf, 1))
            for k, (z, m) in enumerate(ypairs):
                for o in group(g, z) - {m}:
                    rows.append(({to + idx[m]: 1, to + idx[o]: -1, yo + k: -M}, 1 - M, np.inf))
        A = lil_matrix((len(rows), nv))
        lb, ub = np.empty(len(rows)), np.empty(len(rows))
        for i, (r, l, u) in enumerate(rows):
            for c, v in r.items():
                A[i, c] = v
            lb[i], ub[i] = l, u
        integ = np.zeros(nv); integ[:n] = 1
        hi = np.full(nv, float(n))
        for s in range(S):
            yo = n + s * (ny + n)
            integ[yo:yo + ny] = 1
            hi[yo:yo + ny] = 1
        hi[:n] = 1
        res = milp(c=np.concatenate([np.ones(n), np.zeros(nv - n)]),
                   constraints=LinearConstraint(A.tocsr(), lb, ub), integrality=integ,
                   bounds=Bounds(np.zeros(nv), hi), options={"time_limit": time_limit})
        chosen = [b for b, x in zip(buses, res.x[:n]) if x > 0.5]
        bad = [l for l in chosen if not is_valid(g, [c for c in chosen if c != l], zib=True)]
        if log:
            print(f"  round {rounds}: {S} scenarios, {len(chosen)} PMUs, {len(bad)} failures break it, "
                  f"solver status {res.status} ({res.message[:40]}), gap {getattr(res, 'mip_gap', None)}", flush=True)
        if not bad:
            ilp_n1_zib_scenarios.last_status = (res.status, getattr(res, "mip_gap", None), rounds, S)
            return chosen
        scen += [l for l in bad if l not in scen]


# ----------------------------------------------------------------------------------------------
# Grover search on a reduced core: the coverage check is the oracle, no angles to tune
# ----------------------------------------------------------------------------------------------
def cover_expression(cands, constraints, k):
    """Boolean formula 'every bus covered AND at most k sensors', variables q0..q{n-1} = cands."""
    var = {b: f"q{i}" for i, b in enumerate(cands)}
    clauses = []
    for s, need, _ in constraints:
        assert need == 1
        clauses.append("(" + " | ".join(var[b] for b in sorted(s)) + ")")
    n = len(cands)
    for combo in itertools.combinations(range(n), k + 1):  # forbid any k+1 sensors together
        clauses.append("~(" + " & ".join(f"q{i}" for i in combo) + ")")
    return " & ".join(clauses)


def grover_circuit(cands, constraints, k, iterations, measure=True):
    from qiskit import QuantumCircuit
    n = len(cands)
    expr = cover_expression(cands, constraints, k)
    try:  # Qiskit 2.x
        from qiskit.circuit.library import PhaseOracleGate
        oracle = PhaseOracleGate(expr, var_order=[f"q{i}" for i in range(n)])
    except ImportError:  # older Qiskit (needs tweedledum); the notebook falls back to hardware/*.qasm
        from qiskit.circuit.library import PhaseOracle
        oracle = PhaseOracle(expr, var_order=[f"q{i}" for i in range(n)])
    qc = QuantumCircuit(n, n) if measure else QuantumCircuit(n)
    qc.h(range(n))
    for _ in range(iterations):
        qc.append(oracle, range(n))
        qc.h(range(n)); qc.x(range(n))
        qc.h(n - 1); qc.mcx(list(range(n - 1)), n - 1); qc.h(n - 1)
        qc.x(range(n)); qc.h(range(n))
    if measure:
        qc.measure(range(n), range(n))
    return qc


def grover_best_iterations(n, m):
    """Iterations that maximize success for m marked states out of 2**n."""
    theta = np.arcsin(np.sqrt(m / 2 ** n))
    its = np.arange(0, int(np.ceil(np.pi / (4 * theta))) + 1)  # first peak only: fewest gates
    p = np.sin((2 * its + 1) * theta) ** 2
    return int(its[np.argmax(p)]), float(p.max())
