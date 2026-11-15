"""Verify a QAOA result with Qiskit's IterativeAmplitudeEstimation.

Usage (run from this folder):
  python3 iae_qiskit_demo.py --grid path/to/ieee14_piece6.json --params=-1.5708,0.5004
  python3 iae_qiskit_demo.py --grid ... --params=... --bits 010011   # one exact placement (bus0 first)

--params : tuned QAOA angles, ALL gammas first then ALL betas (floats, not a bit string).
"""
import argparse
import numpy as np
import verify_qaoa_iqae as v
from qiskit import QuantumCircuit
from qiskit.circuit.library import GroverOperator, DiagonalGate
from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

ap = argparse.ArgumentParser()
ap.add_argument("--grid", required=True)
ap.add_argument("--params", required=True, help="use --params=-1.5,0.5 (note the =)")
ap.add_argument("--P", type=float, default=2.0)
ap.add_argument("--bits", default=None, help="verify one exact placement, written bus0 first, e.g. 010011")
ap.add_argument("--epsilon-target", type=float, default=0.02)
ap.add_argument("--alpha", type=float, default=0.05)
ap.add_argument("--shots", type=int, default=100)
a = ap.parse_args()

g = v.load_grid(a.grid)
n = len(g["buses"]); P = a.P
params = [float(x) for x in a.params.split(",")]
assert len(params) % 2 == 0, "need gammas then betas (an even count)"
sensors, unseen, costs = v.tables(g, P)

if a.bits:                                     # one exact bit string
    assert len(a.bits) == n and set(a.bits) <= {"0", "1"}
    good = np.zeros(2**n, dtype=bool)
    good[int(a.bits[::-1], 2)] = True          # reverse: qubit 0 is the rightmost char in Qiskit
    print("good set: exactly", a.bits)
else:                                          # any minimum-size valid placement
    valid = unseen == 0
    good = valid & (sensors == sensors[valid].min())
    print("good set: all optimal placements (%d sensors), %d bit string(s)" % (sensors[valid].min(), good.sum()))

# A = the QAOA circuit with the tuned angles bound (no measurement)
A = QuantumCircuit(n); A.h(range(n))
L = len(params) // 2
for gam, bet in zip(params[:L], params[L:]):
    for i in range(n): A.p(-gam, i)
    for grp in v.bus_groups(g):
        A.x(grp); A.mcp(-gam * P, grp[:-1], grp[-1]); A.x(grp)
    A.rx(2 * bet, range(n))

oracle = QuantumCircuit(n)
oracle.append(DiagonalGate([-1 if m else 1 for m in good]), range(n))
Q = GroverOperator(oracle, state_preparation=A)
problem = EstimationProblem(state_preparation=A, grover_operator=Q, objective_qubits=list(range(n)),
                            is_good_state=lambda s: bool(good[int(s, 2)]))
iae = IterativeAmplitudeEstimation(
    epsilon_target=a.epsilon_target, alpha=a.alpha, confint_method="beta", min_ratio=2,
    sampler=SamplerV2(default_shots=a.shots),
    transpiler=generate_preset_pass_manager(optimization_level=1, backend=AerSimulator()))
res = iae.estimate(problem)
lo, hi = res.confidence_interval
psi = v.qaoa_state(costs, n, np.array(params))
exact = float(np.sum(np.abs(psi[good]) ** 2))
print("IQAE estimate   %.4f   %d%% interval [%.4f, %.4f]   oracle queries %d" % (
    res.estimation, round(100 * (1 - a.alpha)), lo, hi, res.num_oracle_queries))
print("exact (statevector) %.4f   inside interval: %s" % (exact, lo <= exact <= hi))
