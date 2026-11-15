"""One chart with every quantum result: % of shots that were the optimal PMU placement, per circuit and platform.

    python make_summary_chart.py   ->  figures/summary_all_results.png, results/summary_all_results.csv
Sources: results/all_jobs.csv (simulator jobs), results/sweep_results.json, results/forte_results.json,
noise-free values from statevector simulation of the same circuit files. 200 shots per job.
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NaN = np.nan
# label, two-qubit gates, noise-free, IonQ sim (Forte-1 noise), Quantinuum H2 emulator, random guess, [IonQ Forte 1 real chip]
ROWS = [
    ("IEEE 300 core, 3 qubits", None),
    ("QAOA 1 layer", 6, 72.0, 69.5, 71.0, 37.5),
    ("QAOA 2 layers", 12, 95.5, 93.0, 92.5, 37.5, 95.0),
    ("QAOA 4 layers", 24, 99.9, 93.5, 99.0, 37.5),
    ("IEEE 14 core, 4 qubits", None),
    ("QAOA 1 layer", 8, 33.0, 32.0, 35.0, 12.5),
    ("QAOA 2 layers, P=3", 16, 91.6, 81.0, 88.5, 12.5, 92.0),
    ("QAOA 3 layers", 24, 91.8, 75.0, 93.0, 12.5),
    ("IEEE 14, no reduction, 14 qubits", None),
    ("QUBO QAOA 1 layer", 98, 0.6, 1.0, NaN, 0.03),
    ("55,000-bus grid pieces (QAOA)", None),
    ("3 qubits (piece 372)", 18, 98.6, 91.0, 98.0, 37.5),
    ("6 qubits (piece 251)", 36, 44.2, 32.5, 42.0, 3.1),
    ("8 qubits (piece 214)", 54, 73.0, 56.5, 74.0, 4.7),
    ("12 qubits (piece 168)", 78, 46.8, 29.5, 38.5, 0.6),
]
SERIES = [("Noise-free simulation", "#9aa0a6"), ("IonQ simulator, Forte-1 noise", "#0072B2"),
          ("Quantinuum H2 emulator", "#E69F00"), ("IonQ Forte 1, real chip", "#D55E00")]
INK, MUTED, GRID = "#1f2328", "#59636e", "#d8dee4"

with open("results/summary_all_results.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["group", "circuit", "two_qubit_gates", "noise_free_pct", "ionq_sim_pct", "quantinuum_emu_pct",
                "random_guess_pct", "ionq_forte1_qpu_pct"])
    group = ""
    for r in ROWS:
        if r[1] is None:
            group = r[0]; continue
        r = tuple(r) + (("",) if len(r) == 6 else ())
        w.writerow([group, *["" if (isinstance(v, float) and np.isnan(v)) else v for v in r]])

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "xtick.color": MUTED, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False})
fig, ax = plt.subplots(figsize=(11, 9))
bar_h, gap = 0.19, 0.35
y = 0.0; yticks, ylabels, header_y = [], [], []
for r in ROWS:
    if r[1] is None:
        y -= 0.55; header_y.append((y, r[0])); y -= 0.55
        continue
    label, cx, *vals = r
    order = [0, 1, 2] + ([3] if len(vals) > 4 else [])
    plot_vals = vals[:3] + ([vals[4]] if len(vals) > 4 else [])
    nb = len(plot_vals)
    center = y - (nb - 1) / 2 * bar_h
    for k, ((name, col), v) in enumerate(zip([SERIES[i] for i in order], plot_vals)):
        yy = y - k * bar_h
        if np.isnan(v):
            ax.text(1.2, yy, "not scored",
                    va="center", fontsize=7, color=MUTED)
            continue
        ax.barh(yy, v, height=bar_h * 0.86, color=col, edgecolor="white", linewidth=0.5)
        if k == 3:
            ax.text(v + 1, yy, f"{v:g}%", va="center", fontsize=8, color=INK, fontweight="bold")
    rnd = vals[3]
    ax.plot([rnd, rnd], [y + bar_h * 0.55, y - (nb - 1) * bar_h - bar_h * 0.55], color=INK, lw=1.4, ls=(0, (2, 1.5)))
    yticks.append(center); ylabels.append(f"{label}\n{cx} two-qubit gates")
    y -= nb * bar_h + gap
for hy, text in header_y:
    ax.text(-1, hy, text, fontsize=11, fontweight="bold", color=INK, ha="right", va="center",
            transform=ax.get_yaxis_transform() if False else ax.transData)
ax.set_yticks(yticks, ylabels, fontsize=8.5)
ax.tick_params(axis="y", length=0)
ax.set_xlim(0, 105); ax.set_ylim(y + gap, 0.2)
ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.set_xlabel("% of 200 shots that are the optimal PMU placement", color=INK)
ax.xaxis.set_ticks_position("top"); ax.xaxis.set_label_position("top")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in SERIES] + \
          [plt.Line2D([0], [0], color=INK, lw=1.4, ls=(0, (2, 1.5)))]
ax.legend(handles, [n for n, _ in SERIES] + ["Random guessing"], frameon=False, fontsize=9,
          loc="upper center", bbox_to_anchor=(0.42, -0.015), ncol=3)
fig.suptitle("TVA PMU placement: every quantum run in one place", x=0.02, ha="left", fontsize=14,
             fontweight="bold", color=INK)
fig.text(0.02, 0.945, "Cores are what is left after exact classical reduction (IEEE 300: 85 PMUs fixed, 3 buses left; "
         "IEEE 14: 2 fixed, 4 left). Real-chip numbers in bold.\n200 shots per job, about plus or minus 3 to 5 points. "
         "The IonQ and Quantinuum runs are noisy simulators of those machines, not the machines themselves.",
         fontsize=8.5, color=MUTED, ha="left", va="top")
fig.tight_layout(rect=(0, 0.02, 1, 0.93))
fig.savefig("figures/summary_all_results.png", dpi=180)
print("saved")
