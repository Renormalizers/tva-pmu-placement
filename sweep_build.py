"""Build the Saturday parameter sweep: many small circuits, each tuned on the exact statevector, exported
as QASM with a manifest saying how to score its shots. Run: python sweep_build.py
Writes hardware/sweep/*.qasm, hardware/sweep/manifest.json and results/sweep_predictions.json."""
import json
from pathlib import Path
import numpy as np
from qiskit import QuantumCircuit, qasm2, transpile
from qiskit_aer import AerSimulator
from pmu_lib import *
from noise_study import noise_model

OUT = Path("hardware/sweep"); OUT.mkdir(parents=True, exist_ok=True)
g14 = load_grid("data/ieee14.json"); piece = load_grid("data/ieee14_piece6.json"); g300 = load_grid("data/ieee300.json")
r14, r300 = reduce_problem(g14), reduce_problem(g300)
r14n1 = reduce_problem(g14, twice=True)
st = json.load(open("results/stretch_study.json"))
W14 = {int(k): v for k, v in st["priority_table"]["weights"].items()}


def walsh(cost):
    """Coefficients a_S with cost(x) = sum_S a_S prod_{i in S} (-1)^{x_i} (qubit i = bit i)."""
    a = cost.astype(float).copy()
    h = 1
    while h < len(a):
        for i in range(0, len(a), 2 * h):
            x, y = a[i:i + h].copy(), a[i + h:i + 2 * h].copy()
            a[i:i + h], a[i + h:i + 2 * h] = x + y, x - y
        h *= 2
    return a / len(a)


def diag_qaoa_circuit(cost, n, params, n_total=None):
    """QAOA with an arbitrary diagonal cost, compiled from its Walsh terms (CX ladders + rz)."""
    n_total = n_total or n
    a = walsh(cost)
    terms = [(s, a[s]) for s in range(1, len(a)) if abs(a[s]) > 1e-9]
    L = len(params) // 2
    qc = QuantumCircuit(n_total, n_total)
    qc.h(range(n_total))
    for gm, bt in zip(params[:L], params[L:]):
        for s, coef in terms:
            q = [i for i in range(n_total) if (s >> i) & 1]
            for i in range(len(q) - 1):
                qc.cx(q[i], q[i + 1])
            qc.rz(2 * gm * coef, q[-1])
            for i in reversed(range(len(q) - 1)):
                qc.cx(q[i], q[i + 1])
        qc.rx(2 * bt, range(n_total))
    qc.measure(range(n_total), range(n_total))
    return qc


def tune(cost, n, layers, good_mask, alphas=(1.0, 0.25)):
    best = None
    for a in alphas:
        p, pr = run_qaoa(cost, n, layers=layers, alpha=a, maxiter=200)
        po = float(pr[good_mask].sum())
        if best is None or po > best[1]:
            best = (p, po, a)
    return best


def masks_cover(cands, cons, best_total, nf):
    n = len(cands); X = all_bits(n); idx = {b: i for i, b in enumerate(cands)}
    valid = np.ones(2 ** n, bool)
    for s, need, _ in cons:
        valid &= X[:, [idx[b] for b in s]].sum(1) >= need
    return valid, valid & (X.sum(1) + nf == best_total)


def masks_grid(G, rule):
    n = len(G["buses"]); X = all_bits(n)
    valid = np.array([is_valid(G, [G["buses"][i] for i in range(n) if X[z, i]], **rule) for z in range(2 ** n)])
    sizes = X.sum(1)
    return valid, valid & (sizes == sizes[valid].min())


manifest, preds = [], []
noisy = AerSimulator(noise_model=noise_model(0.005), seed_simulator=3)


def add(name, desc, qc, n_place, ev, ideal_popt, max_cx=1500):
    t = transpile(qc, basis_gates=["rz", "sx", "x", "cx"], optimization_level=3, seed_transpiler=7)
    ncx = t.count_ops().get("cx", 0)
    if ncx > max_cx:
        print(f"{name:28s} {t.num_qubits:2d}q {ncx:4d} CX  too deep to send; ideal p(opt) {ideal_popt:.1%}  {desc}", flush=True)
        preds.append({"name": name, "desc": desc, "qubits": t.num_qubits, "cx": ncx, "ideal_p_opt": round(ideal_popt, 4), "skipped": True})
        return
    (OUT / f"{name}.qasm").write_text(qasm2.dumps(t))
    manifest.append({"name": name, "desc": desc, "qubits": t.num_qubits, "n_place": n_place, "cx": ncx, "eval": ev})
    preds.append({"name": name, "desc": desc, "qubits": t.num_qubits, "cx": ncx, "ideal_p_opt": round(ideal_popt, 4)})
    print(f"{name:28s} {t.num_qubits:2d}q {ncx:4d} CX  ideal p(opt) {ideal_popt:.1%}  {desc}", flush=True)


# ---- 1. layers sweep on the reduced cores ----
for tag, red, best in [("ieee14_core", r14, 4), ("ieee300_core", r300, 87)]:
    cands, cons, nf = red["cands"], red["constraints"], len(red["forced"])
    cost = cover_cost_table(cands, cons)
    valid, opt = masks_cover(cands, cons, best, nf)
    ev = {"kind": "cover", "cands": cands, "cons": [[sorted(s), k] for s, k, _ in cons], "best": best, "forced": nf}
    for L in ([1, 2, 5] if tag == "ieee14_core" else [4, 5]):
        p, po, a = tune(cost, len(cands), L, opt)
        add(f"{tag}_L{L}", f"QAOA {L} layers, P=2", qaoa_circuit_cover(cands, cons, p), len(cands), ev, po)
    if tag == "ieee14_core":
        # ---- 2. penalty sweep P ----
        for P in (1.5, 3.0, 5.0):
            c = cover_cost_table(cands, cons, P)
            p, po, a = tune(c, len(cands), 2, opt)
            add(f"{tag}_L2_P{str(P).replace('.', 'p')}", f"QAOA 2 layers, penalty P={P}",
                qaoa_circuit_cover(cands, cons, p, P=P), len(cands), ev, po)
        # ---- 3. Grover steps ----
        for its in (1, 3):
            qc = grover_circuit(cands, cons, 2, its)
            # ideal success from formula: 2 marked of 16
            th = np.arcsin(np.sqrt(2 / 16)); po = float(np.sin((2 * its + 1) * th) ** 2)
            add(f"{tag}_grover_{its}step", f"Grover k=2, {its} step(s)", qc, len(cands), ev, po)

# ---- 4. 6-bus piece: layers, penalty, QUBO ----
pc, pcons = piece["buses"], grid_constraints(piece)
pv, popt = masks_cover(pc, pcons, 2, 0)
evp = {"kind": "cover", "cands": pc, "cons": [[sorted(s), k] for s, k, _ in pcons], "best": 2, "forced": 0}
for L in (2, 3):
    p, po, a = tune(cover_cost_table(pc, pcons), 6, L, popt)
    add(f"piece6_L{L}", f"QAOA {L} layers, P=2", qaoa_circuit_cover(pc, pcons, p), 6, evp, po)


def ub_table(G, l1=1.0, l2=1.0):
    cands = G["buses"]; n = len(cands); X = all_bits(n); idx = {b: i for i, b in enumerate(cands)}
    c = X.sum(1).astype(float)
    for b in G["buses"]:
        h = X[:, [idx[j] for j in ({b} | G["neighbors"][b]) if j in idx]].sum(1) - 1
        c += -l1 * h + l2 * h * h
    return c


for G, tag, best in [(piece, "piece6", 2), (g14, "ieee14_full", 4)]:
    c = ub_table(G)
    v, o = masks_grid(G, {})
    ev = {"kind": "grid", "grid": "ieee14_piece6" if tag == "piece6" else "ieee14", "rule": {}, "best": best}
    for L in (1, 2):
        p, po, a = tune(c, len(G["buses"]), L, o)
        add(f"{tag}_qubo_L{L}", f"QUBO (unbalanced penalty, only 2-qubit terms), {L} layers",
            diag_qaoa_circuit(c, len(G["buses"]), p), len(G["buses"]), ev, po)

# slack-variable QUBO for the piece: sum_{j in N[b]} x_j - 1 - s_b = 0, s_b in binary
slack_bits = []
for s, _, _ in pcons:
    nb = int(np.ceil(np.log2(len(s)))) if len(s) > 1 else 0
    slack_bits.append(nb)
ns = sum(slack_bits); nt = 6 + ns
print("slack QUBO qubits:", nt)
if nt <= 16:
    X = all_bits(nt); idx = {b: i for i, b in enumerate(pc)}
    c = X[:, :6].sum(1).astype(float); off = 6
    for (s, _, _), nb in zip(pcons, slack_bits):
        val = X[:, [idx[b] for b in s]].sum(1) - 1 - sum((2 ** k) * X[:, off + k] for k in range(nb))
        c += 2.0 * val * val; off += nb
    good = np.zeros(2 ** nt, bool)
    for z in range(2 ** nt):
        good[z] = popt[z & 63]
    p, po, a = tune(c, nt, 1, good)
    add("piece6_qubo_slack_L1", f"QUBO with {ns} slack qubits, 1 layer", diag_qaoa_circuit(c, nt, p), 6, evp, po)

# ---- 5. stretch goals as circuits ----
# N-1 on the reduced core (need 2 sightings)
cands, cons, nf = r14n1["cands"], r14n1["constraints"], len(r14n1["forced"])
c = cover_cost_table(cands, cons)
v, o = masks_cover(cands, cons, 9, nf)
evn = {"kind": "cover", "cands": cands, "cons": [[sorted(s), k] for s, k, _ in cons], "best": 9, "forced": nf}
for L in (1, 2):
    p, po, a = tune(c, len(cands), L, o)
    add(f"n1_core_L{L}", f"N-1 core ({len(cands)} qubits), {L} layers", diag_qaoa_circuit(c, len(cands), p), len(cands), evn, po)

# zero-injection on all 14 buses (one ZIB: at most one unseen bus, and it must be in N[7])
c = zib_cost_table(g14)
v, o = masks_grid(g14, {"zib": True})
evz = {"kind": "grid", "grid": "ieee14", "rule": {"zib": True}, "best": 3}
a = walsh(c); nterms = int((np.abs(a[1:]) > 1e-9).sum())
print("ZIB cost Walsh terms:", nterms)
for L in (1,):
    p, po, al = tune(c, 14, L, o)
    add(f"zib_full_L{L}", f"zero-injection, all 14 buses, {L} layer", diag_qaoa_circuit(c, 14, p), 14, evz, po)

# fixed budget k=2, NERC weights, penalty QAOA
X = all_bits(14)
sc = np.array([sum(W14[b] for b in seen_buses(g14, [g14["buses"][i] for i in range(14) if X[z, i]])) for z in range(2 ** 14)], float)
cnt = X.sum(1); bestsc = sc[cnt == 2].max()
c = -sc / sc.max() + 1.0 * (cnt - 2) ** 2
o = (cnt == 2) & (sc == bestsc)
evb = {"kind": "budget", "grid": "ieee14", "k": 2, "weights": W14, "best_score": float(bestsc)}
for L in (1, 2):
    p, po, al = tune(c, 14, L, o, alphas=(0.25, 1.0))
    add(f"budget_k2_L{L}", f"fixed budget k=2 with NERC weights, penalty, {L} layers", diag_qaoa_circuit(c, 14, p), 14, evb, po)

# predictions under our own 0.5% noise model
for m, pr in zip(manifest, preds):
    qc = qasm2.load(str(OUT / f"{m['name']}.qasm"), custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    if m["cx"] <= 1500:
        counts = noisy.run(qc, shots=4000).result().get_counts()
        pr["noisy_counts_file"] = None
        pr["noisy_0p5_counts"] = counts
json.dump(manifest, open(OUT / "manifest.json", "w"), indent=1)
json.dump(preds, open("results/sweep_predictions.json", "w"), indent=1)
print(len(manifest), "circuits")
