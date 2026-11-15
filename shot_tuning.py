"""Tune QAOA from measured shots only: the cost of a placement is computed only for bitstrings
that were actually sampled, never for all 2^n. This is how tuning would have to work at scale.
Run with `python shot_tuning.py`. Writes results/shot_tuning.json."""
import json
from qiskit import transpile
from qiskit_aer import AerSimulator
from pmu_lib import *
from qaoa_study import shares

sim = AerSimulator(seed_simulator=5)


def sampled_cvar(cands, constraints, params, shots, alpha, P=2.0):
    qc = transpile(qaoa_circuit_cover(cands, constraints, params), sim)
    counts = sim.run(qc, shots=shots).result().get_counts()
    idx = {b: i for i, b in enumerate(cands)}
    vals = []
    for key, c in counts.items():
        x = [int(ch) for ch in key[::-1]]
        cost = sum(x) + P * sum(max(0, need - sum(x[idx[b]] for b in s)) for s, need, _ in constraints)
        vals += [cost] * c
    vals = np.sort(vals)
    return vals[: max(1, int(alpha * len(vals)))].mean()


def tune(cands, constraints, L, shots=256, alpha=0.25, maxiter=60):
    used = [0]

    def obj(p):
        used[0] += shots
        return sampled_cvar(cands, constraints, p, shots, alpha)
    k = (np.arange(L) + 0.5) / L
    best = None
    for a, b in [(0.3, 0.6), (0.6, 0.3), (0.4, 0.4)]:
        r = minimize(obj, np.concatenate([k * a, (1 - k) * b]), method="COBYLA",
                     options={"maxiter": maxiter, "rhobeg": 0.2})
        if best is None or r.fun < best.fun:
            best = r
    return best.x, used[0]


if __name__ == "__main__":
    g14 = load_grid("data/ieee14.json")
    red = reduce_problem(g14)
    piece = load_grid("data/ieee14_piece6.json")
    out = []
    for label, cands, cons, best, nf, L, alpha in [
        ("IEEE 14 reduced (4 qubits)", red["cands"], red["constraints"], 4, 2, 3, 1.0),
        ("6-bus piece (6 qubits)", piece["buses"], grid_constraints(piece), 2, 0, 1, 0.25),
        ("IEEE 14 full (14 qubits)", g14["buses"], grid_constraints(g14), 4, 0, 2, 0.25),
    ]:
        params, shots_used = tune(cands, cons, L, alpha=alpha)
        probs = np.abs(qaoa_state(cover_cost_table(cands, cons), params, len(cands))) ** 2
        sh = shares(cands, cons, probs, best, nf)
        out.append({"case": label, "layers": L, "shots_used_for_tuning": shots_used,
                    "params": [round(float(x), 4) for x in params], **sh})
        print(f"{label}: L={L}, tuned with {shots_used} shots -> optimal {sh['optimal']:.1%}, unseen {sh['leaves a bus unseen']:.1%}", flush=True)
    json.dump(out, open("results/shot_tuning.json", "w"), indent=1)
