"""Write each reduced piece of a big grid as a hardware circuit plus a manifest.
    python big_export.py data_big/activsg25k.json [max_cx]"""
import json, sys
from pathlib import Path
from qiskit import qasm2
from check import load_grid
from big_grid import reduce_fast, components_of
from pmu_lib import qaoa_circuit_cover, cx_count

g = load_grid(sys.argv[1]); name = g["name"].replace(" ", "_")
max_cx = int(sys.argv[2]) if len(sys.argv) > 2 else 400
res = json.load(open(f"results/big_quantum_{name}.json"))
red = reduce_fast(g)
pieces = sorted(components_of(red), key=lambda c: -len(c["cands"]))
out = Path(f"hardware/big_{name}"); out.mkdir(parents=True, exist_ok=True)
manifest = {"grid": name, "forced": red["forced"], "pieces": []}
for pc, r in zip(pieces, res["pieces"]):
    assert len(pc["cands"]) == r["qubits"]
    if "params" not in r:
        manifest["pieces"].append({"piece": r["piece"], "cands": pc["cands"], "constraints": [sorted(s) for s, _, _ in pc["constraints"]],
                                   "exact": r["exact"], "cx": None, "qasm": None, "solver": "ILP"})
        continue
    ncx, t = cx_count(qaoa_circuit_cover(pc["cands"], pc["constraints"], r["params"]))
    entry = {"piece": r["piece"], "cands": pc["cands"], "constraints": [sorted(s) for s, _, _ in pc["constraints"]],
             "exact": r["exact"], "cx": ncx, "qasm": None}
    if ncx <= max_cx:
        fn = out / f"piece{r['piece']:02d}.qasm"
        fn.write_text(qasm2.dumps(t)); entry["qasm"] = fn.name
    manifest["pieces"].append(entry)
    print(f"piece {r['piece']:2d}: {len(pc['cands']):2d} qubits, {ncx:4d} CX, {'exported' if entry['qasm'] else 'too deep, skipped'}")
json.dump(manifest, open(out / "manifest.json", "w"))
