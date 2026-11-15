"""Noise sweep: how often each circuit still returns an optimal placement as two-qubit error grows.
Noise model: depolarizing error on every gate plus readout flips. This is our own rough stand-in,
not IonQ's forte-1 model (that one runs on IonQ's simulator through qBraid)."""
import json
from pathlib import Path
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
from pmu_lib import *
from qaoa_study import shares

rows = json.load(open("results/qaoa_study.json"))


def best_params(case, alpha, L):
    return next(r["params"] for r in rows if r["case"] == case and r.get("alpha") == alpha and r["layers"] == L)


def noise_model(p2, p1=None, ro=None):
    p1 = p2 / 20 if p1 is None else p1
    ro = p2 if ro is None else ro
    nm = NoiseModel()
    if p2 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["rz", "sx", "x"])
        nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
        nm.add_all_qubit_readout_error(ReadoutError([[1 - ro, ro], [ro, 1 - ro]]))
    return nm


def counts_to_probs(counts, n):
    pr = np.zeros(2 ** n)
    for k, c in counts.items():
        pr[int(k.replace(" ", ""), 2)] += c
    return pr / pr.sum()


if __name__ == "__main__":
    piece = load_grid("data/ieee14_piece6.json"); g14 = load_grid("data/ieee14.json")
    red = reduce_problem(g14)
    cases = [
        ("6-bus piece, L=1", piece["buses"], grid_constraints(piece), 2, 0, best_params("6-bus piece (6 qubits)", 0.25, 1)),
        ("IEEE 14 reduced, L=1", red["cands"], red["constraints"], 4, 2, best_params("IEEE 14 reduced (4 qubits)", 1.0, 1)),
        ("IEEE 14 reduced, L=3", red["cands"], red["constraints"], 4, 2, best_params("IEEE 14 reduced (4 qubits)", 1.0, 3)),
        ("IEEE 14 full, L=1", g14["buses"], grid_constraints(g14), 4, 0, best_params("IEEE 14 full (14 qubits)", 0.25, 1)),
    ]
    out = []
    for p2 in [0.0, 0.002, 0.005, 0.01, 0.02, 0.05]:
        sim = AerSimulator(noise_model=noise_model(p2), seed_simulator=11)
        for label, cands, cons, best, nf, params in cases:
            qc = qaoa_circuit_cover(cands, cons, params)
            ncx, t = cx_count(qc)
            res = sim.run(t, shots=4000).result().get_counts()
            sh = shares(cands, cons, counts_to_probs(res, len(cands)), best, nf)
            out.append({"p2": p2, "case": label, "cx": ncx, **sh})
            print(f"p2={p2:<6} {label:24s} CX={ncx:4d} opt {sh['optimal']:.1%} unseen {sh['leaves a bus unseen']:.1%}", flush=True)
    json.dump(out, open("results/noise_study.json", "w"), indent=1)
