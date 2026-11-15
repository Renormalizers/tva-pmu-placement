"""Submit chosen circuits from hardware/*.qasm without waiting; collect later with %run collect.py.
Usage: %run run_more.py ionq ieee300_core_L2 ieee300_core_L3 ieee14_core_L4"""
import sys, time, warnings
from pathlib import Path
from qiskit import qasm2
from qbraid.runtime import QbraidProvider
warnings.filterwarnings("ignore", message="Circuit contains measurement gates")
DEVICES = {"ionq": ("ionq:ionq:sim:simulator", {"noise_model": "forte-1"}, "simulator (forte-1 noise)"),
           "quantinuum": ("quantinuum:quantinuum:sim:h2-emulator", None, "emulator")}
dev_id, opts, label = DEVICES[sys.argv[1]]
device = QbraidProvider().get_device(dev_id)
record = Path("results/run_record.csv")
for name in sys.argv[2:]:
    qc = qasm2.load(f"hardware/{name}.qasm", custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    job = device.run(qc, shots=200, **({"runtime_options": opts} if opts else {}))
    with record.open("a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M')},{name},{dev_id},{label},{job.id},200,SUBMITTED\n")
    print(f"{name:20s} {qc.num_qubits} qubits {qc.count_ops().get('cx', 0):3d} CX  {job.id}")
