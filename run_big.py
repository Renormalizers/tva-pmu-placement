"""Submit every piece circuit of a big grid to a qBraid device without waiting.
Usage: %run run_big.py quantinuum hardware/big_case_ACTIVSg70k"""
import json, sys, time, warnings
from pathlib import Path
from qiskit import qasm2
from qbraid.runtime import QbraidProvider
warnings.filterwarnings("ignore", message="Circuit contains measurement gates")
DEVICES = {"ionq": ("ionq:ionq:sim:simulator", {"noise_model": "forte-1"}),
           "quantinuum": ("quantinuum:quantinuum:sim:h2-emulator", None)}
dev_id, opts = DEVICES[sys.argv[1]]
folder = Path(sys.argv[2])
man = json.load(open(folder / "manifest.json"))
device = QbraidProvider().get_device(dev_id)
jobs_file = folder / f"jobs_{sys.argv[1]}.json"
jobs = json.load(open(jobs_file)) if jobs_file.exists() else {}
for p in man["pieces"]:
    key = str(p["piece"])
    if not p["qasm"] or key in jobs:
        continue
    qc = qasm2.load(str(folder / p["qasm"]), custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    try:
        job = device.run(qc, shots=200, **({"runtime_options": opts} if opts else {}))
        jobs[key] = job.id
        print(f"piece {key:>2s}: {qc.num_qubits:2d} qubits {p['cx']:4d} CX  {job.id}", flush=True)
    except Exception as e:
        print(f"piece {key:>2s}: FAILED {e}", flush=True)
    json.dump(jobs, open(jobs_file, "w"), indent=1)
print(len(jobs), "jobs recorded in", jobs_file)
