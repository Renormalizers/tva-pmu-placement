"""N-1 across grid sizes on qBraid: standard QAOA (2 layers) vs multi-angle QAOA (1 layer) vs random guess.
Reads results/stretch_qbraid_results.json -> deck_charts/n1_qbraid_benchmark.png  (run from the project folder)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = ["Inter", "DejaVu Sans"]
R = json.load(open("results/stretch_qbraid_results.json"))["results"]
GROUPS = [("ieee14_N-1", "IEEE 14 N-1"), ("ieee30_N-1", "IEEE 30 N-1"),
          ("ieee300_N-1", "IEEE 300 N-1"), ("activsg2000_N-1", "Texas 2000 N-1")]
BARS = [("standard", "noise_free", "Standard QAOA · noise-free sim", "#cfd5ee"),
        ("standard", "ionq", "Standard QAOA · IonQ sim (noisy)", "#a3acdb"),
        ("standard", "quantinuum", "Standard QAOA · Quantinuum emulator", "#6f78ad"),
        ("multi", "noise_free", "Multi-angle QAOA · noise-free sim", "#d9b9bf"),
        ("multi", "ionq", "Multi-angle QAOA · IonQ sim (noisy)", "#9b6470"),
        ("standard", "rigetti", "Standard QAOA · Rigetti real chip", "#7fb8a8"),
        ("multi", "quantinuum", "Multi-angle QAOA · Quantinuum emulator", "#4a2f35"),
        ("multi", "rigetti", "Multi-angle QAOA · Rigetti real chip", "#2f7a68"),
        ("multi", "forte", "Multi-angle QAOA · IonQ Forte 1 real chip", "#1d6f6a")]
# keep the standard bars together, then the multi-angle bars
BARS = [b for b in BARS if b[0] == "standard"] + [b for b in BARS if b[0] == "multi"]
BARS = [b for b in BARS if b[1] != "rigetti"]   # Rigetti results kept in the JSON but not shown
INK, SUB, LINE = "#3b2a2e", "#5a4a4e", "#d0873a"

bar_h, gap, sub_gap = 0.55, 1.0, 0.25
fig = plt.figure(figsize=(9.6, 12.2), facecolor="#dbe6f3")
bg = fig.add_axes([0, 0, 1, 1], zorder=-1); bg.axis("off")
bg.add_patch(FancyBboxPatch((0.012, 0.01), 0.976, 0.98, boxstyle="round,pad=0,rounding_size=0.02",
                            transform=bg.transAxes, facecolor="white", edgecolor="none"))
ax = fig.add_axes([0.27, 0.16, 0.65, 0.76]); ax.set_facecolor("none")

y = 0
for key, title in GROUPS:
    d = R[key]
    ys = []
    for i, (m, dev, lab, col) in enumerate(BARS):
        yy = y + i * bar_h + (sub_gap if i >= 3 else 0)
        ys.append(yy)
        v = d[m].get(dev)
        if v is None:
            if dev == "forte":
                ax.text(0.012, yy, "not run on chip", va="center", fontsize=9, color=SUB, style="italic")
            continue
        ax.barh(yy, v, height=bar_h * 0.86, color=col, zorder=2)
        ax.text(v + 0.012, yy, f"{v * 100:.1f}%".replace(".0%", "%"), va="center", fontsize=9.5, color=INK,
                fontweight="bold" if dev == "forte" else "normal")
    rnd = d["random"]
    ax.plot([rnd, rnd], [ys[0] - bar_h * 0.6, ys[-1] + bar_h * 0.6], ls=(0, (3, 2)), color=LINE, lw=1.8, zorder=3)
    mid = (ys[0] + ys[-1]) / 2
    ax.text(-0.04, mid - 0.45, title, transform=ax.get_yaxis_transform(), ha="right", va="center",
            fontsize=13, fontweight="bold", color=INK)
    ax.text(-0.04, mid + 0.15, f"{d['qubits']} qubits", transform=ax.get_yaxis_transform(), ha="right",
            va="center", fontsize=10, color=SUB)
    ax.text(-0.04, mid + 0.65, f"random {rnd * 100:.1f}%", transform=ax.get_yaxis_transform(), ha="right",
            va="center", fontsize=9, color=LINE)
    y = ys[-1] + bar_h + gap

ax.set_xlim(0, 1.0); ax.set_ylim(y - gap + 0.1, -0.6)
ax.set_xticks([0, .25, .5, .75, 1]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=10, color=INK)
ax.set_yticks([]); ax.tick_params(axis="x", length=0)
for s in ("top", "right", "bottom"):
    ax.spines[s].set_visible(False)
ax.spines["left"].set_color("#cfcfcf")
fig.text(0.5, 0.962, "N-1 sensor placement on qBraid: share of shots that are optimal", ha="center",
         fontsize=13.5, fontweight="bold", color=INK)
fig.text(0.5, 0.94, "Standard QAOA (2 layers) vs multi-angle QAOA (1 layer), 200 shots per job (100 on the Forte chip)",
         ha="center", fontsize=10, color=SUB)
h = [plt.Rectangle((0, 0), 1, 1, color=c) for *_, c in BARS] + [Line2D([0], [0], color=LINE, ls=(0, (3, 2)), lw=1.8)]
fig.legend(h, [b[2] for b in BARS] + ["Random guess"], loc="lower center", ncol=2, frameon=False, fontsize=9.5,
           bbox_to_anchor=(0.5, 0.025), handlelength=1.2, columnspacing=1.6)
fig.savefig("deck_charts/n1_qbraid_benchmark.png", dpi=220, facecolor=fig.get_facecolor())
print("wrote deck_charts/n1_qbraid_benchmark.png")
