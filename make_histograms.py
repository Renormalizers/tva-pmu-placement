"""Histograms for the deck: ILP solve times (CPU vs GPU) and what every qBraid job returned.

    python make_histograms.py
Inputs (whichever exist):
  results/ilp_bench_cpu.csv          local run, 2 CPU cores, HiGHS, raw per-run rows
  results/ilp_bench_cpu_lab_binned.txt  qBraid Lab Large (8 vCPU), HiGHS, binned on the same edges
  results/ilp_bench_gpu.csv          Google Colab T4 GPU, NVIDIA cuOpt, raw per-run rows
  results/all_jobs.csv               every qBraid job with its job ID (collect_all.py)
Outputs: figures/hist_*.png and results/hist_data.json (the binned numbers behind every figure).
"""
import csv, json, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

os.makedirs("figures", exist_ok=True)
E = np.logspace(-5, 3, 65)                 # shared log-spaced solve-time bins, seconds
INK, MUTED, GRID = "#1f2328", "#59636e", "#d8dee4"
DEV_COLORS = {"CPU 2 cores (HiGHS)": "#0072B2", "CPU 8 vCPU, qBraid Lab (HiGHS)": "#E69F00",
              "GPU T4, Colab (cuOpt)": "#009E73"}
JOB_COLORS = {"ionq": "#0072B2", "quantinuum": "#E69F00"}
JOB_LABEL = {"ionq": "IonQ simulator, Forte-1 noise", "quantinuum": "Quantinuum H2 emulator"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True})
TITLES = {"ieee14": "IEEE 14", "ieee300": "IEEE 300", "synth55k": "55,000-bus grid"}
KINDS = {"plain": "every bus seen", "n1": "N-1 (seen twice)", "zib": "zero-injection rule"}
out = {"bin_edges_s": E.tolist()}

# ---------------------------------------------------------------- solve times
counts = defaultdict(dict)   # (model, MILP/LP) -> device -> histogram counts on E
stats = defaultdict(dict)


def add_raw(path, device, ok=lambda r: True):
    if not os.path.exists(path):
        return
    vals = defaultdict(list)
    for r in csv.DictReader(open(path)):
        if not ok(r):
            continue
        vals[(r["model"], "LP" if r["relax"] == "True" else "MILP")].append(float(r["solve_s"]))
    for k, v in vals.items():
        v = np.clip(np.array(v), E[0], E[-1] * 0.999)
        counts[k][device] = np.histogram(v, bins=E)[0]
        stats[k][device] = dict(n=len(v), min_s=float(v.min()), median_s=float(np.median(v)),
                                max_s=float(v.max()))


add_raw("results/ilp_bench_cpu.csv", "CPU 2 cores (HiGHS)")
for binned, dev in (("results/ilp_bench_cpu_lab_binned.txt", "CPU 8 vCPU, qBraid Lab (HiGHS)"),
                    ("results/ilp_bench_gpu_binned.txt", "GPU T4, Colab (cuOpt)")):
    if not os.path.exists(binned):
        continue
    for line in open(binned):
        if line.startswith("#") or not line.strip():
            continue
        key, n, q, sp = line.split()[:4]
        model, what = key.split("|")
        h = np.zeros(len(E) - 1, int)
        for t in sp.split(";"):
            i, c = t.split(":"); h[int(i)] = int(c)
        lo, med, hi = map(float, q.split("/"))
        counts[(model, what)][dev] = h
        stats[(model, what)][dev] = dict(n=int(n[1:]), min_s=lo, median_s=med, max_s=hi,
                                         note=" ".join(line.split()[4:]))

order = [d for d in DEV_COLORS]
for (model, what), devs in sorted(counts.items()):
    grid, kind = model.split("_")
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    nz = [i for d in devs.values() for i in np.nonzero(d)[0]]
    lo, hi = max(min(nz) - 2, 0), min(max(nz) + 3, len(E) - 1)
    for dev in order:
        if dev not in devs:
            continue
        h = devs[dev]; s = stats[(model, what)][dev]
        ax.stairs(h, E, fill=True, alpha=0.30, color=DEV_COLORS[dev])
        ax.stairs(h, E, color=DEV_COLORS[dev], linewidth=1.6,
                  label=f"{dev}: median {s['median_s']:.3g} s (n={s['n']})"
                        + (", stopped at time limit, not optimal" if "not proven" in s.get("note", "") else ""))
    ax.set_xscale("log"); ax.set_xlim(E[lo], E[hi]); ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("solve time per run (seconds, log scale)"); ax.set_ylabel("runs")
    ax.set_title(f"{TITLES[grid]}, {KINDS[kind]}: {'exact ILP' if what == 'MILP' else 'LP relaxation'}",
                 color=INK, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    fig.tight_layout()
    fig.savefig(f"figures/hist_time_{model}_{what.lower()}.png", dpi=170)
    plt.close(fig)
    out[f"time|{model}|{what}"] = {d: dict(counts=devs[d].tolist(), **stats[(model, what)][d]) for d in devs}

# ---------------------------------------------------------------- qBraid jobs
jobs = list(csv.DictReader(open("results/all_jobs.csv"))) if os.path.exists("results/all_jobs.csv") else []
CASES = [("IEEE 14", "IEEE 14"), ("IEEE 300", "IEEE 300"), ("synthetic_55000-bus", "55,000-bus grid pieces")]
summary = []
for case, title in CASES:
    rows = [r for r in jobs if r["case"] == case]
    if not rows:
        continue
    # (a) excess PMUs over the optimum, per shot, pooled over all completed cover jobs
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    labels = ["invalid", "0 (optimal)", "+1", "+2", "+3", "+4 or more"]
    width = 0.38
    rec = {}
    for j, dev in enumerate(["ionq", "quantinuum"]):
        tot = np.zeros(len(labels))
        nj = 0
        for r in rows:
            if r["device"] != dev or not r["valid_pmus_hist"] or r["best"] == "":
                continue
            nj += 1
            best = int(float(r["best"]))
            vh = {int(k): v for k, v in json.loads(r["valid_pmus_hist"]).items()}
            tot[0] += int(r["shots"]) - int(r["valid_shots"])
            for w, c in vh.items():
                tot[min(max(w - best, 0), 4) + 1] += c
        if nj == 0:
            continue
        share = 100 * tot / tot.sum()
        ax.bar(np.arange(len(labels)) + (j - 0.5) * width, share, width=width * 0.92,
               color=JOB_COLORS[dev], label=f"{JOB_LABEL[dev]} ({nj} jobs, {int(tot.sum()):,} shots)")
        rec[dev] = dict(jobs=nj, shots=int(tot.sum()), labels=labels, shots_per_bar=tot.astype(int).tolist())
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_xlabel("PMUs in the measured placement minus the optimum"); ax.set_ylabel("% of shots")
    ax.set_title(f"{title}: every shot returned", color=INK, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    out[f"jobs|{case}|excess_pmus"] = rec
    # (b) per-job share of optimal shots
    ax = axes[1]
    bins = np.linspace(0, 100, 11)
    rec = {}
    for j, dev in enumerate(["ionq", "quantinuum"]):
        v = [100 * int(r["optimal_shots"]) / int(r["shots"]) for r in rows
             if r["device"] == dev and r["optimal_shots"] not in ("", None) and r["shots"]]
        if not v:
            continue
        h = np.histogram(v, bins=bins)[0]
        ax.bar(bins[:-1] + 2 + j * 4.2, h, width=3.8, color=JOB_COLORS[dev],
               label=f"{JOB_LABEL[dev]}: median {np.median(v):.0f}%")
        rec[dev] = dict(jobs=len(v), counts=h.tolist(), bin_edges_pct=bins.tolist(),
                        median_pct=float(np.median(v)), mean_pct=float(np.mean(v)))
        summary.append((title, JOB_LABEL[dev], len(v), float(np.mean(v)), float(np.median(v))))
    ax.set_xlim(0, 100); ax.set_xticks(bins); ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("% of a job's shots that are the optimal placement"); ax.set_ylabel("jobs")
    ax.set_title("per job", color=INK, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    out[f"jobs|{case}|share_optimal"] = rec
    fig.tight_layout()
    fig.savefig(f"figures/hist_jobs_{case.replace(' ', '_').replace(',', '')}.png", dpi=170)
    plt.close(fig)

out["jobs_summary"] = summary
json.dump(out, open("results/hist_data.json", "w"), indent=1)
print("figures:", sorted(f for f in os.listdir("figures") if f.startswith("hist_")))
for s in summary:
    print(s)
