"""Collect results for every job in results/run_record.csv without blocking.
Usage in qBraid Lab: %run collect.py"""
import csv
import json
from pathlib import Path

import numpy as np
from qbraid.runtime import QbraidJob

from pmu_lib import load_grid, reduce_problem, grid_constraints
from qaoa_study import shares

piece = load_grid("data/ieee14_piece6.json")
g14 = load_grid("data/ieee14.json"); r14 = reduce_problem(g14)
g300 = load_grid("data/ieee300.json"); r300 = reduce_problem(g300)
def case(name):
    if name.startswith("ieee14_core"):
        return r14["cands"], r14["constraints"], 4, len(r14["forced"])
    if name.startswith("ieee300_core"):
        return r300["cands"], r300["constraints"], 87, len(r300["forced"])
    if name.startswith("piece6"):
        return piece["buses"], grid_constraints(piece), 2, 0
    return g14["buses"], grid_constraints(g14), 4, 0


def to_probs(counts, n):
    pr = np.zeros(2 ** n)
    for k, c in counts.items():
        pr[int(k.replace(" ", ""), 2)] += c
    return pr / pr.sum()


rows = list(csv.DictReader(open("results/run_record.csv")))
latest = {}
for r in rows:
    if r["job_id"]:
        latest[(r["circuit"], r["device"])] = r
summary = {}
print(f"{'circuit':22s} {'device':38s} {'status':12s} {'optimal':>8s} {'extra':>7s} {'unseen':>7s}")
for (name, dev), r in sorted(latest.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    fn = Path(f"results/counts_{name}_{dev.replace(':', '_')}.json")
    status = "COMPLETED"
    if fn.exists():
        counts = json.loads(fn.read_text())
    else:
        job = QbraidJob(r["job_id"])
        status = str(job.status()).split(".")[-1]
        if status != "COMPLETED":
            print(f"{name:22s} {dev:38s} {status:12s}")
            continue
        counts = job.result().data.get_counts()
        fn.write_text(json.dumps(counts))
    cands, cons, best, nf = case(name)
    sh = shares(cands, cons, to_probs(counts, len(cands)), best, nf)
    summary[f"{name}|{dev}"] = {"job_id": r["job_id"], "label": r["label"], "shots": sum(counts.values()), **sh}
    print(f"{name:22s} {dev:38s} {status:12s} {sh['optimal']:8.1%} {sh['valid, extra sensors']:7.1%} {sh['leaves a bus unseen']:7.1%}")
json.dump(summary, open("results/hardware_summary.json", "w"), indent=1)
