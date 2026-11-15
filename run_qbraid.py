"""Send our circuits to qBraid devices, wait, save counts, log the run record, and print results.

Use in qBraid Lab with the qBraid-SDK kernel, from a notebook cell:
    %run run_qbraid.py ionq          # IonQ simulator with the forte-1 noise model (free)
    %run run_qbraid.py quantinuum    # Quantinuum H2 emulator (free, accept its terms once)
    %run run_qbraid.py forte         # IonQ Forte QPU (credits; 200 shots)
Add "nowait" to submit without waiting, then collect with %run collect.py
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from qiskit import qasm2, transpile

from pmu_lib import load_grid, reduce_problem, grid_constraints, qaoa_circuit_cover
from qaoa_study import shares

warnings.filterwarnings("ignore", message="Circuit contains measurement gates")
which = sys.argv[1] if len(sys.argv) > 1 else "ionq"
DEVICES = {
    "ionq": ("ionq:ionq:sim:simulator", {"noise_model": "forte-1"}, "simulator (forte-1 noise)"),
    "quantinuum": ("quantinuum:quantinuum:sim:h2-emulator", None, "emulator"),
    "forte": ("ionq:ionq:qpu:forte-1", None, "QPU"),
    "forte-aws": ("aws:ionq:qpu:forte-1", None, "QPU via AWS"),
}
dev_id, opts, label = DEVICES[which]
SHOTS = 200

piece = load_grid("data/ieee14_piece6.json")
g14 = load_grid("data/ieee14.json"); r14 = reduce_problem(g14)
g300 = load_grid("data/ieee300.json"); r300 = reduce_problem(g300)
meta = json.load(open("hardware/circuits.json"))
study = json.load(open("results/qaoa_study.json"))
full_params = next(r["params"] for r in study if r["case"] == "IEEE 14 full (14 qubits)" and r.get("alpha") == 0.25 and r["layers"] == 1)

CASES = {  # name: (circuit source, candidate buses, constraints, best total, number of fixed sensors)
    "piece6_L1": ("qasm", piece["buses"], grid_constraints(piece), 2, 0),
    "ieee14_core_L3": ("qasm", r14["cands"], r14["constraints"], 4, len(r14["forced"])),
    "ieee14_core_grover": ("qasm", r14["cands"], r14["constraints"], 4, len(r14["forced"])),
    "ieee300_core_L1": ("qasm", r300["cands"], r300["constraints"], 87, len(r300["forced"])),
    "ieee300_core_grover": ("qasm", r300["cands"], r300["constraints"], 87, len(r300["forced"])),
    "ieee14_full_L1": ("build", g14["buses"], grid_constraints(g14), 4, 0),  # unreduced, for comparison
}
if which.startswith("forte") or which == "quantinuum":
    CASES.pop("ieee14_full_L1")  # 336 CX: only worth showing on the free IonQ simulator

from qbraid.runtime import QbraidProvider
provider = QbraidProvider()
device = provider.get_device(dev_id)
print("Device:", dev_id, "| status:", device.status())

record = Path("results/run_record.csv")
if not record.exists():
    record.write_text("time,circuit,device,label,job_id,shots,status\n")

jobs = {}
for name, (src, cands, cons, best, nf) in CASES.items():
    if src == "qasm":
        qc = qasm2.load(f"hardware/{name}.qasm", custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    else:
        qc = transpile(qaoa_circuit_cover(cands, cons, full_params), basis_gates=["rz", "sx", "x", "cx"], optimization_level=3)
    ncx = qc.count_ops().get("cx", 0)
    try:
        job = device.run(qc, shots=SHOTS, **({"runtime_options": opts} if opts else {}))
        jobs[name] = job
        jid, status = job.id, str(job.status())
    except Exception as e:
        jid, status = "", ("FAILED: " + str(e)).replace(",", ";").replace("\n", " ")[:200]
    with record.open("a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M')},{name},{dev_id},{label},{jid},{SHOTS},{status}\n")
    print(f"{name:22s} {qc.num_qubits:2d} qubits {ncx:4d} CX  job {jid}  {status}")


def to_probs(counts, n):
    pr = np.zeros(2 ** n)
    for k, c in counts.items():
        pr[int(k.replace(" ", ""), 2)] += c
    return pr / pr.sum()


if "nowait" in sys.argv:
    print("\nSubmitted. Collect later with: %run collect.py")
    raise SystemExit
print("\nWaiting for results...")
rows = []
for name, job in jobs.items():
    src, cands, cons, best, nf = CASES[name]
    try:
        counts = job.result().data.get_counts()
    except Exception as e:
        print(f"{name}: no result yet ({e})")
        continue
    Path(f"results/counts_{name}_{dev_id.replace(':', '_')}.json").write_text(json.dumps(counts))
    sh = shares(cands, cons, to_probs(counts, len(cands)), best, nf)
    rows.append((name, job.id, sh))
    with record.open("a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M')},{name},{dev_id},{label},{job.id},{SHOTS},COMPLETED\n")

print(f"\n{label}: share of {SHOTS} shots")
print(f"{'circuit':22s} {'optimal':>8s} {'valid, extra':>13s} {'unseen':>8s}  job")
for name, jid, sh in rows:
    print(f"{name:22s} {sh['optimal']:8.1%} {sh['valid, extra sensors']:13.1%} {sh['leaves a bus unseen']:8.1%}  {jid}")
json.dump({n: {"job_id": j, **s} for n, j, s in rows}, open(f"results/summary_{which}.json", "w"), indent=1)
