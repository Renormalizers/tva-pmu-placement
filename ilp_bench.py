"""ILP benchmark for histograms: IEEE 14, IEEE 300 and the team's 55,000-bus grid.

Every model is written once to an MPS file (bench/<grid>_<kind>.mps) so the CPU solver (HiGHS)
and the GPU solver (NVIDIA cuOpt, run_gpu_bench.py) solve the exact same problem.

    python ilp_bench.py export            # build and write the MPS files + bench/models.json
    python ilp_bench.py cpu [runs_small] [runs_big]   # repeated HiGHS solves with different seeds

Kinds: plain (every bus seen), n1 (every bus seen twice, survives one PMU failure),
zib (zero-injection rule, IEEE grids only; the 55k grid has no zero-injection buses).
Each run records solve time, objective, nodes and gap; the MILP and its LP relaxation are both timed.
Output: results/ilp_bench_cpu.csv and results/ilp_bench_solutions.json (per-bus coverage counts).
"""
import csv
import json
import os
import sys
import time
from collections import defaultdict

import highspy
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from check import load_grid, is_valid

GRIDS = {"ieee14": "data/ieee14.json", "ieee300": "data/ieee300.json",
         "synth55k": "data_big/synthetic55k.json"}
BENCH = "bench"


def closed_nbhd(g):
    return {b: {b} | g["neighbors"][b] for b in g["buses"]}


def model_cover(g, need):
    buses = g["buses"]
    N = closed_nbhd(g)
    idx = {b: i for i, b in enumerate(buses)}
    ri, ci = [], []
    for r, b in enumerate(buses):
        for j in N[b]:
            ri.append(r); ci.append(idx[j])
    n = len(buses)
    A = csr_matrix((np.ones(len(ri)), (ri, ci)), shape=(n, n))
    return dict(c=np.ones(n), A=A, lb=np.full(n, float(need)), ub=np.full(n, np.inf),
                integ=np.ones(n), lo=np.zeros(n), hi=np.ones(n), nx=n)


def model_zib(g):
    """Same formulation as big_grid.ilp_zib_sparse / pmu_lib.ilp_placement_zib."""
    buses, Z = g["buses"], g["zero_injection"]
    N = closed_nbhd(g)
    n = len(buses)
    idx = {b: i for i, b in enumerate(buses)}
    ypairs = [(z, m) for z in Z for m in sorted(N[z])]
    ny = len(ypairs)
    yof, zof = defaultdict(list), defaultdict(list)
    for k, (z, m) in enumerate(ypairs):
        yof[m].append(k); zof[z].append(k)
    M = n + 1
    ri, ci, val, lb, ub = [], [], [], [], []
    r = 0
    for b in buses:
        for j in N[b]:
            ri.append(r); ci.append(idx[j]); val.append(1)
        for k in yof[b]:
            ri.append(r); ci.append(n + k); val.append(1)
        lb.append(1); ub.append(np.inf); r += 1
    for z in Z:
        for k in zof[z]:
            ri.append(r); ci.append(n + k); val.append(1)
        lb.append(-np.inf); ub.append(1); r += 1
    T = n + ny
    for k, (z, m) in enumerate(ypairs):
        for o in N[z] - {m}:
            ri += [r, r, r]; ci += [T + idx[m], T + idx[o], n + k]; val += [1, -1, -M]
            lb.append(1 - M); ub.append(np.inf); r += 1
    nv = n + ny + n
    A = coo_matrix((val, (ri, ci)), shape=(r, nv)).tocsr()
    return dict(c=np.concatenate([np.ones(n), np.zeros(ny + n)]), A=A, lb=np.array(lb, float),
                ub=np.array(ub, float), integ=np.concatenate([np.ones(n + ny), np.zeros(n)]),
                lo=np.zeros(nv), hi=np.concatenate([np.ones(n + ny), np.full(n, float(n))]), nx=n)


def to_highs(m):
    lp = highspy.HighsLp()
    nv = len(m["c"])
    A = m["A"].tocsc()
    lp.num_col_, lp.num_row_ = nv, A.shape[0]
    lp.col_cost_ = m["c"]
    lp.col_lower_, lp.col_upper_ = m["lo"], m["hi"]
    inf = highspy.kHighsInf
    lp.row_lower_ = np.where(np.isinf(m["lb"]), -inf, m["lb"])
    lp.row_upper_ = np.where(np.isinf(m["ub"]), inf, m["ub"])
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr
    lp.a_matrix_.index_ = A.indices
    lp.a_matrix_.value_ = A.data.astype(float)
    lp.integrality_ = [highspy.HighsVarType.kInteger if v else highspy.HighsVarType.kContinuous
                       for v in m["integ"]]
    return lp


def export():
    os.makedirs(BENCH, exist_ok=True)
    meta = {}
    for gname, path in GRIDS.items():
        g = load_grid(path)
        kinds = {"plain": lambda: model_cover(g, 1), "n1": lambda: model_cover(g, 2)}
        if g.get("zero_injection"):
            kinds["zib"] = lambda: model_zib(g)
        for kind, build in kinds.items():
            m = build()
            h = highspy.Highs(); h.setOptionValue("output_flag", False)
            h.passModel(to_highs(m))
            f = f"{BENCH}/{gname}_{kind}.mps"
            h.writeModel(f)
            meta[f"{gname}_{kind}"] = dict(grid=gname, kind=kind, mps=f, buses=len(g["buses"]),
                                           vars=len(m["c"]), ints=int(m["integ"].sum()),
                                           rows=m["A"].shape[0], nnz=int(m["A"].nnz), nx=m["nx"])
            print(f, meta[f"{gname}_{kind}"])
    json.dump(meta, open(f"{BENCH}/models.json", "w"), indent=1)


def solve_cpu(mps, seed, relax=False, time_limit=600):
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("random_seed", int(seed))
    h.setOptionValue("time_limit", float(time_limit))
    h.setOptionValue("mip_rel_gap", 0.0)
    t0 = time.perf_counter()
    h.readModel(mps)
    if relax:
        h.setOptionValue("solve_relaxation", True)
    t1 = time.perf_counter()
    h.run()
    t2 = time.perf_counter()
    info = h.getInfo()
    x = np.array(h.getSolution().col_value)
    return dict(read_s=t1 - t0, solve_s=t2 - t1, obj=info.objective_function_value,
                status=h.modelStatusToString(h.getModelStatus()),
                nodes=int(getattr(info, "mip_node_count", 0) or 0),
                gap=float(getattr(info, "mip_gap", 0) or 0)), x


def coverage_hist(g, placement, need):
    N = closed_nbhd(g)
    S = set(placement)
    counts = [len(N[b] & S) for b in g["buses"]]
    deg = [len(g["neighbors"][b]) for b in placement]
    return dict(seen_count_per_bus=np.bincount(counts).tolist(),
                pmu_bus_degree=np.bincount(deg).tolist(), need=need)


def cpu(runs_small=50, runs_big=20):
    meta = json.load(open(f"{BENCH}/models.json"))
    os.makedirs("results", exist_ok=True)
    out = "results/ilp_bench_cpu.csv"
    fields = ["device", "solver", "model", "grid", "kind", "relax", "seed", "solve_s", "read_s", "obj",
              "status", "nodes", "gap", "valid"]
    sols = {}
    grids = {k: load_grid(p) for k, p in GRIDS.items()}
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for name, m in meta.items():
            g = grids[m["grid"]]
            N = closed_nbhd(g)
            runs = runs_big if m["grid"] == "synth55k" else runs_small
            for relax in (False, True):
                for seed in range(runs):
                    r, x = solve_cpu(m["mps"], seed, relax=relax)
                    valid = ""
                    if not relax:
                        place = [b for b, v in zip(g["buses"], x[:m["nx"]]) if v > 0.5]
                        if m["kind"] in ("plain", "n1"):
                            # no zero-injection rule here, so valid means every bus has at least
                            # need PMUs in its closed neighbourhood (for n1: still seen after any one loss)
                            need = 2 if m["kind"] == "n1" else 1
                            S = set(place)
                            valid = all(len(N[b] & S) >= need for b in g["buses"])
                        else:
                            valid = is_valid(g, place, zib=True)
                        if seed == 0:
                            sols[name] = dict(placement=place, pmus=len(place),
                                              **coverage_hist(g, place, 2 if m["kind"] == "n1" else 1))
                    row = dict(device=f"CPU ({os.cpu_count()} cores)", solver=f"HiGHS {highspy.Highs().version()}",
                               model=name, grid=m["grid"], kind=m["kind"], relax=relax, seed=seed,
                               valid=valid, **r)
                    w.writerow(row); fh.flush()
                    print(name, "LP" if relax else "MILP", seed, f"{r['solve_s']:.4f}s", r["obj"],
                          r["status"], valid, flush=True)
    json.dump(sols, open("results/ilp_bench_solutions.json", "w"))
    print("wrote", out)


if __name__ == "__main__":
    if sys.argv[1] == "export":
        export()
    else:
        cpu(*(int(a) for a in sys.argv[2:4]))
