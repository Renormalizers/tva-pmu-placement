#!/usr/bin/env python3
"""
verify_qaoa_iqae.py: check a QAOA sensor-placement result two independent ways.

  1. Classical check: is each of QAOA's top answers a valid placement, and is its
     sensor count equal to the exact integer-program (ILP) minimum?
  2. Amplitude check: use Iterative Quantum Amplitude Estimation (IQAE, Grinko et al.
     2019) to estimate the probability that ONE QAOA shot lands on a "good" placement
     (valid, optimal, or a specific set of buses), with a confidence interval. Compare
     it with the exact value from the statevector and with plain shot sampling that
     uses the same number of circuit runs.

What this is and is not (say this in your slides):
  * It runs on a classical simulation. The "oracle" that marks good placements is a
    lookup over all 2^n bitstrings, so it is not a scalable quantum oracle.
  * IQAE needs deep circuits (Grover power k grows as the target error shrinks), so it
    is a fault-tolerant-era tool, not something to run on today's hardware.
  * What it verifies is the amplitude of the QAOA state: "what share of QAOA's output
    is optimal?" It does not prove QAOA found the best answer; the ILP check does that.

Needs only numpy and scipy. Optional: qiskit + qiskit-aer for --circuit-check.

Examples (from the sensor-placement folder, with data/ next to it):
  python verify_qaoa_iqae.py --grid data/ieee14_piece6.json
  python verify_qaoa_iqae.py --grid data/ieee14.json --layers 3 --good valid
  python verify_qaoa_iqae.py --grid data/ieee14_piece6.json --params=-1.5708,0.5004 --claimed 0.21
  python verify_qaoa_iqae.py --grid data/ieee14_piece6.json --circuit-check

From a notebook:
  from verify_qaoa_iqae import verify
  verify(grid, params=params14, P=2.0, good="optimal", claimed=None)
"""
import argparse
import re
import json
import math
import sys

import numpy as np
from scipy.optimize import LinearConstraint, Bounds, milp, minimize


# ---------------------------------------------------------------- problem setup
def load_grid(path):
    g = json.load(open(path))
    nb = {b: set() for b in g["buses"]}
    for u, v in g["lines"]:
        nb[u].add(v)
        nb[v].add(u)
    g["neighbors"] = nb
    return g


def bus_groups(g):
    """groups[i] = qubits of bus i and its neighbours (qubit i is bus g['buses'][i])."""
    idx = {b: i for i, b in enumerate(g["buses"])}
    return [sorted(idx[c] for c in {b} | set(g["neighbors"][b])) for b in g["buses"]]


def tables(g, P):
    """Per-bitstring sensors, unseen-bus count, and the cost sensors + P*unseen.
    Index z: bit i of z is qubit i (the starter notebook's convention)."""
    n = len(g["buses"])
    z = np.arange(2 ** n, dtype=np.int64)
    sensors = np.zeros(2 ** n, dtype=np.int64)
    for i in range(n):
        sensors += (z >> i) & 1
    unseen = np.zeros(2 ** n, dtype=np.int64)
    for grp in bus_groups(g):
        mask = sum(1 << j for j in grp)
        unseen += (z & mask) == 0
    return sensors, unseen, sensors + P * unseen


def ilp_minimum(g):
    buses = g["buses"]
    A = np.array([[1 if c == b or c in g["neighbors"][b] else 0 for c in buses] for b in buses])
    r = milp(np.ones(len(buses)), constraints=LinearConstraint(A, lb=1),
             integrality=np.ones(len(buses)), bounds=Bounds(0, 1))
    return int(round(r.fun)), [b for b, x in zip(buses, r.x) if x > 0.5]


# ---------------------------------------------------------------- QAOA state (numpy)
def qaoa_state(costs, n, params):
    """Exact QAOA statevector: |+>^n, then L x (cost phase exp(-i*gamma*C), rx(2*beta) mixer).
    Matches the starter notebook's circuit up to a global phase."""
    L = len(params) // 2
    gammas, betas = params[:L], params[L:]
    psi = np.full(2 ** n, 1 / np.sqrt(2 ** n), dtype=complex)
    for gam, bet in zip(gammas, betas):
        psi = psi * np.exp(-1j * gam * costs)
        c, s = np.cos(bet), -1j * np.sin(bet)
        U = np.array([[c, s], [s, c]])
        t = psi.reshape([2] * n)
        for ax in range(n):
            t = np.moveaxis(np.tensordot(U, t, axes=([1], [ax])), 0, ax)
        psi = t.reshape(-1)
    return psi


def cvar(probs, costs, alpha):
    order = np.argsort(costs)
    p, c = probs[order], costs[order]
    take = np.clip(alpha - (np.cumsum(p) - p), 0, p)
    return take @ c / alpha


def tune(costs, n, layers, alpha_cvar=0.25, maxiter=100):
    """Same recipe as the notebook's run_qaoa: ramp starts, COBYLA, CVaR objective."""
    obj = lambda p: cvar(np.abs(qaoa_state(costs, n, p)) ** 2, costs, alpha_cvar)
    k = (np.arange(layers) + 0.5) / layers
    starts = [np.concatenate([k * a, (1 - k) * b]) for a, b in [(0.2, 0.2), (0.3, 0.6), (0.6, 0.3), (0.4, 0.4)]]
    best = None
    for s in starts:
        r = minimize(obj, s, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.2})
        if best is None or r.fun < best.fun:
            best = r
    return best.x


# ---------------------------------------------------------------- Grover powers
class GroverPowers:
    """P(good) after k Grover rounds, Q = (2|psi><psi| - I) S_good, applied to |psi>.
    Equals sin^2((2k+1) theta) with sin^2(theta) = P(good). Cached; k only grows in IQAE."""

    def __init__(self, psi, good_mask):
        self.psi = psi
        self.sign = np.where(good_mask, -1.0, 1.0)
        self.good = good_mask
        self.v = psi.copy()
        self.k = 0

    def prob_good(self, k):
        if k < self.k:
            self.v, self.k = self.psi.copy(), 0
        while self.k < k:
            sv = self.sign * self.v
            self.v = 2 * self.psi * np.vdot(self.psi, sv) - sv
            self.k += 1
        return float(np.sum(np.abs(self.v[self.good]) ** 2))


# ---------------------------------------------------------------- IQAE (Grinko et al. 2019)
def _chernoff(p_hat, shots, max_rounds, alpha):
    eps = math.sqrt(math.log(2 * max_rounds / alpha) / (2 * shots))
    return max(0.0, p_hat - eps), min(1.0, p_hat + eps)


def _clopper_pearson(hits, shots, max_rounds, alpha):
    """Exact binomial interval, what Qiskit calls confint_method='beta' (tighter than Chernoff)."""
    from scipy.stats import beta
    a = alpha / max_rounds
    lo = 0.0 if hits == 0 else float(beta.ppf(a / 2, hits, shots - hits + 1))
    hi = 1.0 if hits == shots else float(beta.ppf(1 - a / 2, hits + 1, shots - hits))
    return lo, hi


def _next_k(k, upper, theta, min_ratio=2.0):
    tl, tu = theta
    old = 4 * k + 2
    max_scaling = int(1 / (2 * (tu - tl)))
    scaling = max_scaling - (max_scaling - 2) % 4
    while scaling >= min_ratio * old:
        tmin = scaling * tl - int(scaling * tl)
        tmax = scaling * tu - int(scaling * tu)
        if tmin <= tmax <= 0.5 and tmin <= 0.5:
            return int((scaling - 2) / 4), True
        if tmax >= 0.5 and tmax >= tmin >= 0.5:
            return int((scaling - 2) / 4), False
        scaling -= 4
    return k, upper


def iqae(prob_after_k, eps, alpha, shots, rng, min_ratio=2.0, max_iter=2000, confint="beta"):
    """Estimate a = P(good) to +/- eps with confidence 1-alpha.
    prob_after_k(k) -> exact P(good) after k Grover rounds; each round we draw `shots` samples.
    While the Grover power k stays the same, shots are pooled (as in Grinko et al.), so the
    interval keeps shrinking until it is narrow enough to jump to a larger k.
    Returns dict with estimate, interval, circuit runs, and a per-round log."""
    max_rounds = max(1, int(math.ceil(math.log(math.pi / (8 * eps)) / math.log(min_ratio))) + 1)
    theta = [0.0, 0.25]           # theta/(2*pi), so a = sin^2(2*pi*theta)
    base = list(theta)            # interval in force when the current k was chosen
    k, upper, runs, log = 0, True, 0, []
    k_prev, up_prev, hits_acc, shots_acc = None, None, 0, 0
    for _ in range(max_iter):
        if theta[1] - theta[0] <= eps / math.pi:
            break
        k, upper = _next_k(k, upper, theta, min_ratio)
        if (k, upper) != (k_prev, up_prev):          # new Grover power: start a fresh group
            base, hits_acc, shots_acc = list(theta), 0, 0
        k_prev, up_prev = k, upper
        p_true = prob_after_k(k)
        hits_acc += int(rng.binomial(shots, p_true))
        shots_acc += shots
        p_hat = hits_acc / shots_acc
        runs += shots * (2 * k + 1)             # each shot applies A once plus Q k times
        if confint == "chernoff":
            pl, pu = _chernoff(p_hat, shots_acc, max_rounds, alpha)
        else:
            pl, pu = _clopper_pearson(hits_acc, shots_acc, max_rounds, alpha)
        bl = math.acos(1 - 2 * pl) / (2 * math.pi)
        bu = math.acos(1 - 2 * pu) / (2 * math.pi)
        s = 4 * k + 2
        if upper:
            tu = (int(s * base[1]) + bu) / s
            tl = (int(s * base[0]) + bl) / s
        else:
            tu = (int(s * base[1]) + 1 - bl) / s
            tl = (int(s * base[0]) + 1 - bu) / s
        theta = [tl, tu]
        log.append((k, p_hat, theta[1] - theta[0]))
    a_lo = math.sin(2 * math.pi * theta[0]) ** 2
    a_hi = math.sin(2 * math.pi * theta[1]) ** 2
    return {"estimate": (a_lo + a_hi) / 2, "lo": a_lo, "hi": a_hi, "runs": runs, "rounds": log,
            "converged": theta[1] - theta[0] <= eps / math.pi}


def plain_sampling(p_true, runs, alpha, rng):
    """Baseline: just measure QAOA's output `runs` times (Hoeffding interval)."""
    runs = max(1, runs)
    p_hat = rng.binomial(runs, p_true) / runs
    half = math.sqrt(math.log(2 / alpha) / (2 * runs))
    return p_hat, max(0.0, p_hat - half), min(1.0, p_hat + half)


# ---------------------------------------------------------------- optional real-circuit check
def circuit_check(g, params, P, good_mask, ks=(0, 1, 2, 3), shots=20000):
    """Build the actual Grover-power circuits in Qiskit and run them on Aer, then compare
    P(good) with the numpy value. Only for small n (the diagonal oracle is exponential)."""
    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit.circuit.library import GroverOperator
        try:
            from qiskit.circuit.library import DiagonalGate as Diag
        except ImportError:
            from qiskit.circuit.library import Diagonal as Diag
        from qiskit_aer import AerSimulator
    except ImportError as e:
        print("  circuit check skipped (needs qiskit and qiskit-aer):", e)
        return
    n = len(g["buses"])
    if n > 10:
        print("  circuit check skipped: n > 10 (diagonal oracle too large).")
        return
    groups = bus_groups(g)
    A = QuantumCircuit(n)
    A.h(range(n))
    for gam, bet in zip(params[:len(params) // 2], params[len(params) // 2:]):
        for i in range(n):
            A.p(-gam, i)
        for grp in groups:
            A.x(grp)
            A.mcp(-gam * P, grp[:-1], grp[-1])
            A.x(grp)
        A.rx(2 * bet, range(n))
    oracle = QuantumCircuit(n)
    oracle.append(Diag([-1 if m else 1 for m in good_mask]), range(n))
    Q = GroverOperator(oracle, state_preparation=A)
    sim = AerSimulator()
    psi = qaoa_state(tables(g, P)[2], n, np.asarray(params))
    gp = GroverPowers(psi, good_mask)
    print("  k | numpy P(good) | Qiskit/Aer circuit P(good), %d shots" % shots)
    for k in ks:
        qc = QuantumCircuit(n, n)
        qc.compose(A, inplace=True)
        for _ in range(k):
            qc.compose(Q, inplace=True)
        qc.measure(range(n), range(n))
        counts = sim.run(transpile(qc, sim), shots=shots).result().get_counts()
        hit = sum(c for key, c in counts.items() if good_mask[int(key.replace(" ", ""), 2)])
        print("  %d | %.4f        | %.4f" % (k, gp.prob_good(k), hit / shots))


# ---------------------------------------------------------------- main routine
def good_mask_for(spec, g, sensors, unseen, opt):
    if spec == "optimal":
        return (unseen == 0) & (sensors == opt)
    if spec == "valid":
        return unseen == 0
    if spec.startswith("buses:"):
        want = {int(x) for x in spec[6:].split(",") if x}
        z = sum(1 << i for i, b in enumerate(g["buses"]) if b in want)
        m = np.zeros(2 ** len(g["buses"]), dtype=bool)
        m[z] = True
        return m
    raise SystemExit("--good must be optimal, valid, or buses:2,5,...")


def verify(g, params=None, layers=1, P=2.0, good="optimal", eps=0.01, alpha=0.05, shots=100,
           claimed=None, seed=1, circuit=False, alpha_cvar=0.25, confint="beta", min_ratio=2.0):
    rng = np.random.default_rng(seed)
    n = len(g["buses"])
    if n > 24:
        raise SystemExit("n = %d qubits is too large for a 2^n statevector on most Lab instances." % n)
    sensors, unseen, costs = tables(g, P)
    opt, opt_buses = ilp_minimum(g)
    if params is None:
        print("Tuning QAOA (layers=%d, P=%g, CVaR alpha=%g) ..." % (layers, P, alpha_cvar))
        params = tune(costs, n, layers, alpha_cvar)
    params = np.asarray(params, dtype=float)
    psi = qaoa_state(costs, n, params)
    probs = np.abs(psi) ** 2
    print("\n== Grid: %s | %d buses | QAOA angles %s" % (g.get("name", "?"), n, np.round(params, 4)))

    print("\n[1] Classical check of QAOA's top answers (exact ILP minimum = %d sensors, e.g. at %s)" % (opt, opt_buses))
    for z in np.argsort(probs)[::-1][:3]:
        buses = [b for i, b in enumerate(g["buses"]) if (z >> i) & 1]
        ok = unseen[z] == 0
        tag = ("valid, %d sensors%s" % (sensors[z], " = ILP optimum" if sensors[z] == opt else ", above the optimum")) \
            if ok else "NOT valid: %d bus(es) unseen" % unseen[z]
        print("    %5.1f%%  sensors at %-24s -> %s" % (100 * probs[z], buses, tag))

    mask = good_mask_for(good, g, sensors, unseen, opt)
    exact = float(probs[mask].sum())
    print("\n[2] Amplitude check: P(one QAOA shot is '%s')" % good)
    print("    Exact value from the statevector:      %.4f" % exact)
    if exact in (0.0, 1.0):
        print("    The target has probability %.0f: nothing to estimate." % exact)
        return
    gp = GroverPowers(psi, mask)
    res = iqae(gp.prob_good, eps, alpha, shots, rng, min_ratio=min_ratio, confint=confint)
    inside = res["lo"] <= exact <= res["hi"]
    print("    IQAE (epsilon_target=%g, confidence %d%%, %d shots/round, confint=%s, min_ratio=%g):" % (
        eps, round(100 * (1 - alpha)), shots, confint, min_ratio))
    print("        estimate %.4f, interval [%.4f, %.4f], converged: %s" % (res["estimate"], res["lo"], res["hi"], res["converged"]))
    ks = []
    for r in res["rounds"]:
        if not ks or ks[-1][0] != r[0]:
            ks.append([r[0], 1])
        else:
            ks[-1][1] += 1
    print("        circuit runs used: %d over %d rounds; Grover power k (x rounds at it): %s" % (
        res["runs"], len(res["rounds"]), ", ".join("%d(x%d)" % (a, b) for a, b in ks)))
    print("        exact value inside the interval: %s" % inside)
    pm, plo, phi = plain_sampling(exact, res["runs"], alpha, rng)
    print("    Plain sampling with the same %d runs: estimate %.4f, interval [%.4f, %.4f] (width %.4f vs IQAE %.4f)" % (
        res["runs"], pm, plo, phi, phi - plo, res["hi"] - res["lo"]))
    if claimed is not None:
        ok = res["lo"] <= claimed <= res["hi"]
        print("\n[3] Claimed value %.4f: %s the IQAE interval [%.4f, %.4f]" % (
            claimed, "INSIDE" if ok else "OUTSIDE", res["lo"], res["hi"]))
    if circuit:
        print("\n[4] Real-circuit cross-check (Qiskit Aer vs numpy)")
        circuit_check(g, params, P, mask)
    print("\nNotes: simulated classically; the oracle is a 2^n lookup; the interval is conservative"
          " (confint='chernoff' is the most conservative), so IQAE beats plain sampling only at tight epsilon.")
    return res


def _clean_argv(argv):
    """Let --params take spaces, commas, brackets or a pasted numpy/tuple repr, e.g.
    --params [-1.57, 0.50]   or   --params (-1.57 0.50)   or   --params=-1.57,0.50"""
    out, i = [], 0
    while i < len(argv):
        t = argv[i]
        if t == "--params" or t.startswith("--params="):
            vals = [t.split("=", 1)[1]] if "=" in t else []
            i += 1
            while i < len(argv) and not (argv[i].startswith("--") or argv[i].startswith("-") and argv[i][1:2].isalpha()):
                vals.append(argv[i]); i += 1
            nums = [x for x in re.split(r"[\s,()\[\]]+", " ".join(vals)) if x]
            out.append("--params=" + ",".join(nums))
        else:
            if t.startswith("--"):                      # --epsilon_target == --epsilon-target
                name, eq, val = t.partition("=")
                t = name.replace("_", "-") + eq + val
            out.append(t); i += 1
    return out


def main(argv=None):
    argv = _clean_argv(sys.argv[1:] if argv is None else list(argv))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", required=True, help="path to a grid json, e.g. data/ieee14_piece6.json")
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--P", type=float, default=2.0, help="unseen-bus penalty (> 1)")
    ap.add_argument("--params", help="comma list gamma_1..gamma_L,beta_1..beta_L; omit to tune")
    ap.add_argument("--good", default="optimal", help="optimal | valid | buses:2,5")
    ap.add_argument("--epsilon-target", "--eps", dest="eps", type=float, default=0.01,
                    help="epsilon_target: target half-width on the probability (0 < eps <= 0.5)")
    ap.add_argument("--confint", choices=["beta", "chernoff"], default="beta", help="confint_method")
    ap.add_argument("--min-ratio", type=float, default=2.0, help="min_ratio for the next Grover power (>= 2 is typical)")
    ap.add_argument("--alpha", type=float, default=0.05, help="alpha: 1 - confidence")
    ap.add_argument("--shots", type=int, default=100, help="shots per IQAE round")
    ap.add_argument("--claimed", type=float, help="the success probability you claim; checked against the interval")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--circuit-check", action="store_true", help="also run real Qiskit circuits on Aer (n <= 10)")
    a = ap.parse_args(argv)
    g = load_grid(a.grid)
    g.setdefault("name", a.grid)
    params = [float(x) for x in a.params.split(",")] if a.params else None
    if params and len(params) % 2:
        sys.exit("--params needs an even number of values: gammas then betas")
    if params:
        a.layers = len(params) // 2        # layers always follows the number of angles
    verify(g, params=params, layers=a.layers, P=a.P, good=a.good, eps=a.eps, alpha=a.alpha,
           shots=a.shots, claimed=a.claimed, seed=a.seed, circuit=a.circuit_check,
           confint=a.confint, min_ratio=a.min_ratio)


if __name__ == "__main__":
    main()
