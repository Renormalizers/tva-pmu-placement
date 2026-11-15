"""Collect sweep results and score every shot. Usage: %run collect_sweep.py ionq   (or quantinuum)"""
import json, sys
from pathlib import Path
from qbraid.runtime import QbraidJob
from check import load_grid, is_valid, seen_buses
which = sys.argv[1]
folder = Path("hardware/sweep"); man = {m["name"]: m for m in json.load(open(folder / "manifest.json"))}
jobs = json.load(open(folder / f"jobs_{which}.json"))
store = folder / f"counts_{which}.json"
counts_all = json.load(open(store)) if store.exists() else {}
grids = {}


def score(m, bits):
    ev = m["eval"]; x = [int(b) for b in bits[: m["n_place"]]]
    if ev["kind"] == "cover":
        ch = {c for c, v in zip(ev["cands"], x) if v}
        valid = all(len(ch & set(s)) >= k for s, k in ev["cons"])
        return valid, valid and len(ch) + ev["forced"] == ev["best"]
    g = grids.setdefault(ev["grid"], load_grid(f"data/{ev['grid']}.json"))
    ch = [b for b, v in zip(g["buses"], x) if v]
    if ev["kind"] == "grid":
        valid = is_valid(g, ch, **ev["rule"])
        return valid, valid and len(ch) == ev["best"]
    w = {int(k): v for k, v in ev["weights"].items()}
    ok = len(ch) == ev["k"]
    return ok, ok and sum(w[b] for b in seen_buses(g, ch)) == ev["best_score"]


lines, pending = [], 0
for name, jid in jobs.items():
    m = man[name]
    if name not in counts_all:
        job = QbraidJob(jid); st = str(job.status()).split(".")[-1]
        if st != "COMPLETED":
            pending += 1; lines.append(f"{name:28s} {st}"); continue
        counts_all[name] = job.result().data.get_counts()
        json.dump(counts_all, open(store, "w"))
    tot = v = o = 0
    for k, c in counts_all[name].items():
        bits = k.replace(" ", "")[::-1]
        a, b = score(m, bits); tot += c; v += c * a; o += c * b
    lines.append(f"{name:28s} {m['qubits']:2d}q {m['cx']:4d}CX opt {o / tot:6.1%} valid {v / tot:6.1%}")
print("\n".join(lines)); print(f"pending {pending}")
