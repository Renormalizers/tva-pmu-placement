"""Pull every qBraid job we submitted, keep its job ID, and save histogram-ready data.

Run inside qBraid Lab from ~/tva_run:   %run collect_all.py
Sources: results/run_record.csv (first two rounds), hardware/sweep/jobs_*.json (Saturday sweep),
hardware/big_*/jobs_*.json (pieces of the 25k, 70k and 55k grids).
Writes results/all_jobs.csv (one row per job, with job ID) and results/all_counts.json (raw counts by job ID).
Columns pmus_hist and valid_pmus_hist are {number of PMUs in the shot: shots}; for core circuits the
forced PMUs are included so the numbers are full-grid totals.
"""
import csv, json
from pathlib import Path
from qbraid.runtime import QbraidJob

R = Path("results"); R.mkdir(exist_ok=True)
cache_f = R / "all_counts.json"
cache = json.load(open(cache_f)) if cache_f.exists() else {}

sweep = {m["name"]: m for m in json.load(open("hardware/sweep/manifest.json"))}
by_cands = {}
for m in sweep.values():
    e = m["eval"]
    if e["kind"] == "cover":
        by_cands[tuple(e["cands"])] = (e["cons"], e.get("forced", 0), e["best"])
circ = json.load(open("hardware/circuits.json")) if Path("hardware/circuits.json").exists() else {}

def case_of(name):
    if name.startswith("ieee300"): return "IEEE 300"
    if name.startswith(("ieee14", "piece6", "n1_core", "budget")): return "IEEE 14"
    return name

jobs = []  # (job_id, source, case, circuit, device, cands, cons, forced, best)
if (R / "run_record.csv").exists():
    for r in csv.DictReader(open(R / "run_record.csv")):
        c = circ.get(r["circuit"], {})
        cands = c.get("cands")
        info = by_cands.get(tuple(cands)) if cands else None
        cons, forced, best = info if info else (None, len(c.get("forced", [])), None)
        jobs.append((r["job_id"], "rounds 1-2", case_of(r["circuit"]), r["circuit"], r["device"].split(":")[0],
                     cands, cons, len(c.get("forced", [])) if c else forced, best))
for dev in ("ionq", "quantinuum"):
    f = Path(f"hardware/sweep/jobs_{dev}.json")
    if f.exists():
        for name, jid in json.load(open(f)).items():
            e = sweep[name]["eval"]
            cons = e.get("cons"); cands = e.get("cands")
            jobs.append((jid, "sweep", case_of(name), name, dev, cands, cons, e.get("forced", 0),
                         e.get("best")))
for folder in sorted(Path("hardware").glob("big_*")):
    man = {str(p["piece"]): p for p in json.load(open(folder / "manifest.json"))["pieces"]}
    grid = folder.name.replace("big_", "")
    for dev in ("ionq", "quantinuum"):
        f = folder / f"jobs_{dev}.json"
        if f.exists():
            for piece, jid in json.load(open(f)).items():
                p = man[piece]
                jobs.append((jid, "grid pieces", grid, f"piece {piece}", dev, p["cands"],
                             [[c, 1] for c in p["constraints"]], 0, p["exact"]))

rows, done, pending = [], 0, 0
for jid, source, case, name, dev, cands, cons, forced, best in jobs:
    status = "COMPLETED" if jid in cache else None
    if jid not in cache:
        try:
            job = QbraidJob(jid)
            status = str(job.status()).split(".")[-1]
            if status == "COMPLETED":
                cache[jid] = job.result().data.get_counts()
        except Exception as ex:
            status = f"ERROR {ex}"[:80]
    counts = cache.get(jid)
    row = dict(job_id=jid, source=source, case=case, circuit=name, device=dev, status=status,
               shots="", qubits=len(cands) if cands else "", best=best if best is not None else "",
               optimal_shots="", valid_shots="", pmus_hist="", valid_pmus_hist="")
    if counts:
        done += 1
        ph, vh, opt, val, shots = {}, {}, 0, 0, 0
        for k, c in counts.items():
            bits = k.replace(" ", "")[::-1]
            shots += c
            w = forced + bits.count("1")
            ph[w] = ph.get(w, 0) + c
            if cons is not None and cands:
                chosen = {cands[i] for i in range(min(len(cands), len(bits))) if bits[i] == "1"}
                if all(len(chosen & set(s)) >= need for s, need in cons):
                    val += c
                    vh[w] = vh.get(w, 0) + c
                    if best is not None and w == best:
                        opt += c
        row.update(shots=shots, optimal_shots=opt if cons is not None else "",
                   valid_shots=val if cons is not None else "",
                   pmus_hist=json.dumps(dict(sorted(ph.items()))),
                   valid_pmus_hist=json.dumps(dict(sorted(vh.items()))) if cons is not None else "")
    else:
        pending += 1
    rows.append(row)
    if len(rows) % 50 == 0:
        print(len(rows), "jobs checked", flush=True)
        json.dump(cache, open(cache_f, "w"))

json.dump(cache, open(cache_f, "w"))
with open(R / "all_jobs.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
from collections import Counter
print(len(rows), "jobs total;", done, "with results;", pending, "not finished")
print(Counter((r["case"], r["device"], r["status"]) for r in rows).most_common())
