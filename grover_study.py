"""Grover search on the reduced cores (IEEE 14: 4 qubits, IEEE 300: 3 qubits).
The oracle is the coverage check itself plus 'at most k sensors', synthesized by Qiskit from the
boolean formula. k is lowered until no placement is marked (Grover adaptive search, done here
classically over k because the cores are tiny). Run with `python grover_study.py`."""
import json
from qiskit import transpile
from qiskit_aer import AerSimulator
from pmu_lib import *
from qaoa_study import shares
from noise_study import noise_model, counts_to_probs

ideal = AerSimulator(method="statevector")
out = []
cases = []
g14 = load_grid("data/ieee14.json"); r14 = reduce_problem(g14)
g300 = load_grid("data/ieee300.json"); r300 = reduce_problem(g300)
for label, red, best_total in [("IEEE 14 core", r14, 4), ("IEEE 300 core", r300, 87)]:
    cands, cons, nf = red["cands"], red["constraints"], len(red["forced"])
    n = len(cands)
    for k in range(n, -1, -1):  # adaptive search over the sensor budget
        # count marked states only to pick the iteration count; the circuit itself never sees them
        X = all_bits(n); idx = {b: i for i, b in enumerate(cands)}
        ok = (X.sum(1) <= k)
        for s, _, _ in cons:
            ok &= X[:, [idx[b] for b in s]].sum(1) >= 1
        m = int(ok.sum())
        if m == 0:
            break
        its, p_ideal = grover_best_iterations(n, m)
        qc = grover_circuit(cands, cons, k, its)
        t = transpile(qc, basis_gates=["rz", "sx", "x", "cx"], optimization_level=3, seed_transpiler=7)
        ncx = t.count_ops().get("cx", 0)
        row = {"case": label, "k_core": k, "k_total": k + nf, "marked": m, "iterations": its, "cx": ncx}
        for p2 in [0.0, 0.005, 0.01]:
            sim = AerSimulator(noise_model=noise_model(p2), seed_simulator=2) if p2 else ideal
            counts = sim.run(t, shots=4000).result().get_counts()
            sh = shares(cands, cons, counts_to_probs(counts, n), best_total, nf)
            row[f"optimal@{p2}"] = sh["optimal"]
            row[f"valid@{p2}"] = sh["optimal"] + sh["valid, extra sensors"]
        out.append(row)
        print(f"{label}: k={k} (total {k+nf}) marked {m}, {its} iterations, {ncx} CX, "
              f"optimal ideal {row['optimal@0.0']:.0%}, at 0.5% error {row['optimal@0.005']:.0%}", flush=True)
        cases.append((label, k, its, t))
json.dump(out, open("results/grover_study.json", "w"), indent=1)
