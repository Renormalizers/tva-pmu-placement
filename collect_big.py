"""Collect piece results and assemble the full-grid placement from the best valid shot of each piece.
Usage: %run collect_big.py quantinuum hardware/big_case_ACTIVSg70k"""
import json, sys
from pathlib import Path
from qbraid.runtime import QbraidJob
folder = Path(sys.argv[2]); which = sys.argv[1]
man = json.load(open(folder / "manifest.json"))
jobs = json.load(open(folder / f"jobs_{which}.json"))
store = folder / f"counts_{which}.json"
counts_all = json.load(open(store)) if store.exists() else {}
done = pending = 0
rows = []
for p in man["pieces"]:
    key = str(p["piece"]); n = len(p["cands"])
    if key not in jobs:
        continue
    if key not in counts_all:
        job = QbraidJob(jobs[key])
        st = str(job.status()).split(".")[-1]
        if st != "COMPLETED":
            pending += 1
            print(f"piece {key:>2s}: {st}")
            continue
        counts_all[key] = job.result().data.get_counts()
        json.dump(counts_all, open(store, "w"))
    done += 1
    cons = [set(c) for c in p["constraints"]]
    best, opt_shots = None, 0
    for k, c in counts_all[key].items():
        bits = k.replace(" ", "")[::-1]
        chosen = {p["cands"][i] for i in range(n) if bits[i] == "1"}
        if all(chosen & s for s in cons):
            if len(chosen) == p["exact"]:
                opt_shots += c
            if best is None or len(chosen) < len(best):
                best = chosen
    rows.append({"piece": p["piece"], "qubits": n, "cx": p["cx"], "exact": p["exact"],
                 "best_valid": len(best) if best else None, "share_optimal": opt_shots / 200,
                 "chosen": sorted(best) if best else None})
    print(f"piece {key:>2s}: {n:2d} qubits {p['cx']:4d} CX  optimal shots {opt_shots:3d}/200  best valid {rows[-1]['best_valid']} (exact {p['exact']})")
json.dump(rows, open(folder / f"assembled_{which}.json", "w"), indent=1)
print(f"\n{done} done, {pending} pending. Pieces solved optimally: {sum(1 for r in rows if r['best_valid'] == r['exact'])}/{len(man['pieces'])}")
print("@@" + json.dumps({r['piece']: [r['share_optimal'], r['best_valid'], r['chosen']] for r in rows}) + "@@")
