"""Stretch-goal circuits on qBraid: the same test as the team's benchmark figure
(noise-free sim, IonQ simulator with the Forte-1 noise model, Quantinuum H2 emulator; 200 shots each).

Use in qBraid Lab, qBraid-SDK kernel, from this folder:
    %run run_stretch_qbraid.py submit ionq          # IonQ simulator + forte-1 noise (free)
    %run run_stretch_qbraid.py submit quantinuum    # Quantinuum H2 emulator (free; accept its terms once)
    %run run_stretch_qbraid.py collect              # rerun until every job is COMPLETED; makes the figures
Optional: add "standard" or "multi" after submit to send only one method.

Circuits (hardware/stretch/*.qasm, already compiled to cx/rz/sx/x):
    standard QAOA, 2 layers   (the method in the benchmark figure)
    multi-angle QAOA, 1 layer (our improvement)
Pieces: IEEE 14 N-1, IEEE 30 N-1 (8 qubits), IEEE 300 N-1, Texas 2000 N-1.
Dry run without qBraid (simulates the noise locally with Qiskit Aer):  python run_stretch_qbraid.py dryrun
"""
import csv
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

from pmu_lib import load_grid
from stretch2 import reduce_need
from pieces import split_pieces
import stretch_quantum as sq

warnings.filterwarnings("ignore", message="Circuit contains measurement gates")
SHOTS = 200
RECORD = Path("results/stretch_run_record.csv")
DEVICES = {
    "ionq": ("ionq:ionq:sim:simulator", {"noise_model": "forte-1"}, "IonQ sim (noisy)"),
    "quantinuum": ("quantinuum:quantinuum:sim:h2-emulator", None, "Quantinuum emulator"),
    "forte": ("ionq:ionq:qpu:forte-1", None, "IonQ Forte 1 chip"),          # costs credits
}
# (key, title, grid, goal, piece size in qubits, qasm stem)
PIECES = [
    ("ieee14_N-1", "IEEE 14 N-1", "ieee14", "N-1", 12, "ieee14_N-1_piece0"),
    ("ieee30_N-1", "IEEE 30 N-1", "ieee30", "N-1", 8, "ieee30_N-1_piece1"),
    ("ieee300_N-1", "IEEE 300 N-1", "ieee300", "N-1", 12, "ieee300_N-1_piece2"),
    ("activsg2000_N-1", "Texas 2000 N-1", "activsg2000", "N-1", 12, "activsg2000_N-1_piece3"),
]
METHODS = {"standard": ("L2", "standard QAOA, 2 layers"), "multi": ("maqaoa", "multi-angle QAOA, 1 layer")}


def load_qc(stem, suffix):
    from qiskit import qasm2
    return qasm2.load(f"hardware/stretch/{stem}_{suffix}.qasm", custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)


_scores = {}
def piece_scoring(grid, goal, nq):
    """is_opt / valid arrays over bitstring index (bit i = qubit i), plus random baseline."""
    key = (grid, goal, nq)
    if key not in _scores:
        g = load_grid(f"data/{grid}.json")
        need = {b: (2 if goal == "N-1" else 1) for b in g["buses"]}
        pc = next(p for p in split_pieces(reduce_need(g, need)) if len(p["cands"]) == nq)
        sensors, short, _ = sq.piece_arrays(pc)
        valid = short == 0
        is_opt = valid & (sensors == sensors[valid].min())
        _scores[key] = (is_opt, valid)
    return _scores[key]


def share(counts, is_opt):
    tot = sum(counts.values())
    hit = sum(c for k, c in counts.items() if is_opt[int(k.replace(" ", ""), 2)])
    return hit / tot, tot


def noise_free(qc, is_opt):
    from qiskit.quantum_info import Statevector
    pr = Statevector(qc.remove_final_measurements(inplace=False)).probabilities()
    return float(pr[is_opt].sum())


def submit(which, methods):
    from qbraid.runtime import QbraidProvider
    dev_id, opts, label = DEVICES[which]
    device = QbraidProvider().get_device(dev_id)
    print("Device:", dev_id, "| status:", device.status())
    RECORD.parent.mkdir(exist_ok=True)
    new = not RECORD.exists()
    with RECORD.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "piece", "method", "device", "label", "job_id", "shots"])
        for key, title, grid, goal, nq, stem in PIECES:
            for m in methods:
                qc = load_qc(stem, METHODS[m][0])
                job = device.run(qc, shots=SHOTS, **({"runtime_options": opts} if opts else {}))
                w.writerow([time.strftime("%Y-%m-%d %H:%M"), key, m, dev_id, label, job.id, SHOTS])
                print(f"{title:16s} {m:8s} {qc.num_qubits:2d} q {qc.count_ops().get('cx', 0):4d} CX  {job.id}")


def collect():
    from qbraid.runtime import QbraidJob
    rows = list(csv.DictReader(RECORD.open()))
    out = {}
    pending = 0
    for r in rows:
        meta = next(p for p in PIECES if p[0] == r["piece"])
        is_opt, _ = piece_scoring(meta[2], meta[3], meta[4])
        cache = Path(f"results/counts_stretch_{r['piece']}_{r['method']}_{r['device'].replace(':', '_')}.json")
        if cache.exists():
            counts = json.loads(cache.read_text())
        else:
            job = QbraidJob(r["job_id"])
            st = str(job.status()).split(".")[-1]
            if st != "COMPLETED":
                print(f"{r['piece']:16s} {r['method']:8s} {r['label']:20s} {st}")
                pending += 1
                continue
            counts = job.result().data.get_counts()
            cache.write_text(json.dumps(counts))
        s, tot = share(counts, is_opt)
        out.setdefault(f"{r['piece']}|{r['method']}", {})[r["label"]] = {"share": s, "shots": tot, "job_id": r["job_id"]}
    summarize(out)
    print(f"\n{pending} job(s) still running" if pending else "\nall jobs collected")


def summarize(out, path="results/stretch_hw_summary.json", tag=""):
    for key, title, grid, goal, nq, stem in PIECES:
        is_opt, _ = piece_scoring(grid, goal, nq)
        for m in METHODS:
            d = out.setdefault(f"{key}|{m}", {})
            d["Noise-free sim"] = {"share": noise_free(load_qc(stem, METHODS[m][0]), is_opt)}
            d["Random guess"] = {"share": float(is_opt.mean())}
            d["cx"] = load_qc(stem, METHODS[m][0]).count_ops().get("cx", 0)
    json.dump(out, open(path, "w"), indent=1)
    for m, (_, mlabel) in METHODS.items():
        print(f"\n{mlabel}")
        for key, title, *_ in PIECES:
            d = out[f"{key}|{m}"]
            cells = "  ".join(f"{lab}: {d[lab]['share']:.1%}" for lab in
                              ["Noise-free sim", "IonQ sim (noisy)", "Local noisy sim (dry run)", "Quantinuum emulator",
                               "IonQ Forte 1 chip", "Random guess"]
                              if lab in d)
            print(f"  {title:16s} {d['cx']:4d} CX  {cells}")
        figure(out, m, mlabel, tag)


def figure(out, method, mlabel, tag=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.lines import Line2D
    plt.rcParams["font.family"] = ["Inter", "DejaVu Sans"]
    BARS = [("Noise-free sim", "#c3c9e4"), ("IonQ sim (noisy)", "#9aa3d4"), ("Local noisy sim (dry run)", "#b9b9b9"),
            ("Quantinuum emulator", "#6f78ad"), ("IonQ Forte 1 chip", "#4a2f35")]
    INK, SUB, LINE = "#3b2a2e", "#5a4a4e", "#d0873a"
    present = [b for b in BARS if any(b[0] in out[f"{p[0]}|{method}"] for p in PIECES)]
    bar_h, gap = 0.62, 0.9
    fig = plt.figure(figsize=(9.2, 1.4 + len(PIECES) * (len(present) * bar_h + gap) * 0.42), facecolor="#dbe6f3")
    bg = fig.add_axes([0, 0, 1, 1], zorder=-1); bg.axis("off")
    bg.add_patch(FancyBboxPatch((0.012, 0.015), 0.976, 0.97, boxstyle="round,pad=0,rounding_size=0.025",
                                transform=bg.transAxes, facecolor="white", edgecolor="none"))
    ax = fig.add_axes([0.26, 0.16, 0.66, 0.74]); ax.set_facecolor("none")
    layers = "2 layers" if method == "standard" else "1 layer"
    y = 0
    for key, title, grid, goal, nq, stem in PIECES:
        d = out[f"{key}|{method}"]
        ys = [y + i * bar_h for i in range(len(present))]
        for (lab, col), yy in zip(present, ys):
            if lab not in d:
                continue
            v = d[lab]["share"]
            ax.barh(yy, v, height=bar_h * 0.86, color=col, zorder=2)
            ax.text(v + 0.012, yy, f"{v * 100:.1f}%".replace(".0%", "%"), va="center", fontsize=10, color=INK,
                    fontweight="bold" if lab == present[-1][0] else "normal")
        rnd = d["Random guess"]["share"]
        ax.plot([rnd, rnd], [ys[0] - bar_h * 0.6, ys[-1] + bar_h * 0.6], ls=(0, (3, 2)), color=LINE, lw=1.8, zorder=3)
        mid = (ys[0] + ys[-1]) / 2
        ax.text(-0.04, mid - 0.18, title, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=13, fontweight="bold", color=INK)
        ax.text(-0.04, mid + 0.42, f"{nq} qubits, {layers}", transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=10, color=SUB)
        y = ys[-1] + bar_h + gap
    ax.set_xlim(0, 1.0); ax.set_ylim(y - gap + 0.1, -0.6)
    ax.set_xticks([0, .25, .5, .75, 1]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=10, color=INK)
    ax.set_yticks([]); ax.tick_params(axis="x", length=0)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#cfcfcf")
    fig.text(0.5, 0.955, f"Stretch goals on qBraid: {mlabel}" + ("  (LOCAL DRY RUN, not qBraid)" if tag else ""), ha="center", fontsize=13, fontweight="bold", color=INK)
    h = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in present] + [Line2D([0], [0], color=LINE, ls=(0, (3, 2)), lw=1.8)]
    fig.legend(h, [l for l, _ in present] + ["Random guess"], loc="lower center", ncol=len(present) + 1,
               frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.03), handlelength=1.2, columnspacing=1.2)
    Path("deck_charts").mkdir(exist_ok=True)
    fn = f"deck_charts/stretch_qbraid_{method}{tag}.png"
    fig.savefig(fn, dpi=220, facecolor=fig.get_facecolor())
    print("wrote", fn)


def dryrun():
    """Local check of the whole pipeline: Aer with a Forte-like 0.5% two-qubit error stands in for the
    IonQ noisy simulator. Clearly labelled; not a qBraid result."""
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(0.005, 2), ["cx"])
    nm.add_all_qubit_quantum_error(depolarizing_error(0.0005, 1), ["sx", "x"])
    sim = AerSimulator(noise_model=nm)
    out = {}
    for key, title, grid, goal, nq, stem in PIECES:
        is_opt, _ = piece_scoring(grid, goal, nq)
        for m in METHODS:
            counts = sim.run(load_qc(stem, METHODS[m][0]), shots=SHOTS, seed_simulator=3).result().get_counts()
            out.setdefault(f"{key}|{m}", {})["Local noisy sim (dry run)"] = {"share": share(counts, is_opt)[0], "shots": SHOTS,
                                                                     "job_id": "local-dryrun"}
    summarize(out, path="results/stretch_hw_summary_DRYRUN.json", tag="_DRYRUN")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] == "dryrun":
        dryrun()
    elif args[0] == "submit":
        submit(args[1], [m for m in METHODS if len(args) < 3 or m == args[2]])
    elif args[0] == "collect":
        collect()
