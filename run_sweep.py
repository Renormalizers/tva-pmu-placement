"""Submit every sweep circuit to a qBraid device without waiting.
Usage: %run run_sweep.py ionq      or      %run run_sweep.py quantinuum"""
import json, sys, warnings
from pathlib import Path
from qiskit import qasm2
from qbraid.runtime import QbraidProvider
warnings.filterwarnings("ignore", message="Circuit contains measurement gates")
DEVICES = {"ionq": ("ionq:ionq:sim:simulator", {"noise_model": "forte-1"}),
           "quantinuum": ("quantinuum:quantinuum:sim:h2-emulator", None)}
which = sys.argv[1]; dev_id, opts = DEVICES[which]
folder = Path("hardware/sweep"); man = json.load(open(folder / "manifest.json"))
device = QbraidProvider().get_device(dev_id)
jf = folder / f"jobs_{which}.json"
jobs = json.load(open(jf)) if jf.exists() else {}
for m in man:
    if m["name"] in jobs:
        continue
    if which == "quantinuum" and m["qubits"] > 26:
        continue
    qc = qasm2.load(str(folder / f"{m['name']}.qasm"), custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    try:
        job = device.run(qc, shots=200, **({"runtime_options": opts} if opts else {}))
        jobs[m["name"]] = job.id
        print(f"{m['name']:28s} {m['qubits']:2d}q {m['cx']:4d} CX  {job.id}", flush=True)
    except Exception as e:
        print(f"{m['name']:28s} FAILED {e}", flush=True)
    json.dump(jobs, open(jf, "w"), indent=1)
print(len(jobs), "jobs recorded in", jf)
