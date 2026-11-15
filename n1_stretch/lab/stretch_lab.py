"""Stand-alone qBraid Lab runner for the TVA stretch-goal circuits (no other project files needed).
    python stretch_lab.py submit ionq        # IonQ simulator with the Forte-1 noise model (free)
    python stretch_lab.py submit quantinuum  # Quantinuum H2 emulator (free)
    python stretch_lab.py collect            # rerun until every job is COMPLETED; writes figures + JSON
"""
import csv, json, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore", message="Circuit contains measurement gates")
SHOTS = 200
HERE = Path(__file__).resolve().parent
RECORD = HERE / "stretch_run_record.csv"
SCORE = json.load(open(HERE / "scoring.json"))
DEVICES = {
    "ionq": ("ionq:ionq:sim:simulator", {"noise_model": "forte-1"}, "IonQ sim (noisy)"),
    "quantinuum": ("quantinuum:quantinuum:sim:h2-emulator", None, "Quantinuum emulator"),
    "forte": ("ionq:ionq:qpu:forte-1", None, "IonQ Forte 1 chip"),
    "forte-oq": ("openquantum:ionq:qpu:forte-1", None, "IonQ Forte 1 chip"),   # real QPU via OpenQuantum
    "forte-aws": ("aws:ionq:qpu:forte-1", None, "IonQ Forte 1 chip (AWS)"),     # the route our chip jobs used
}
METHODS = {"standard": "standard QAOA, 2 layers", "multi": "multi-angle QAOA, 1 layer"}
PIECES = [(k, v["title"], None, None, v["qubits"], k) for k, v in SCORE.items()]


def load_qc(key, m):
    from qiskit import qasm2
    return qasm2.load(str(HERE / "qasm" / f"{key}_{m}.qasm"), custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)


def share(counts, key):
    opt = set(SCORE[key]["optimal_indices"])
    tot = sum(counts.values())
    return sum(c for k, c in counts.items() if int(str(k).replace(" ", ""), 2) in opt) / tot, tot


def submit(which, methods, only=None):
    from qbraid.runtime import QbraidProvider
    dev_id, opts, label = DEVICES[which]
    device = QbraidProvider().get_device(dev_id)
    print("Device:", dev_id, "| status:", device.status(), flush=True)
    new = not RECORD.exists()
    with RECORD.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "piece", "method", "device", "label", "job_id", "shots"])
        for key in SCORE:
            if only and key not in only:
                continue
            for m in methods:
                qc = load_qc(key, m)
                try:
                    job = device.run(qc, shots=SHOTS, **({"runtime_options": opts} if opts else {}))
                    jid = job.id
                except Exception as e:
                    jid = "FAILED: " + str(e).replace(",", ";")[:150]
                w.writerow([time.strftime("%Y-%m-%d %H:%M"), key, m, dev_id, label, jid, SHOTS]); f.flush()
                print(f"{SCORE[key]['title']:16s} {m:8s} {qc.num_qubits:2d} q {qc.count_ops().get('cx', 0):4d} CX  {jid}", flush=True)


def collect():
    from qbraid.runtime import QbraidJob
    out, pending = {}, 0
    for r in csv.DictReader(RECORD.open()):
        if r["job_id"].startswith("FAILED"):
            continue
        cache = HERE / "counts" / f"{r['piece']}_{r['method']}_{r['device'].replace(':', '_')}.json"
        cache.parent.mkdir(exist_ok=True)
        if cache.exists():
            counts = json.loads(cache.read_text())
        else:
            job = QbraidJob(r["job_id"])
            st = str(job.status()).split(".")[-1]
            if st != "COMPLETED":
                print(f"{r['piece']:16s} {r['method']:8s} {r['label']:20s} {st}", flush=True)
                pending += 1
                continue
            counts = {str(k): int(v) for k, v in job.result().data.get_counts().items()}
            cache.write_text(json.dumps(counts))
        s, tot = share(counts, r["piece"])
        out.setdefault(f"{r['piece']}|{r['method']}", {})[r["label"]] = {"share": s, "shots": tot, "job_id": r["job_id"]}
    for key, v in SCORE.items():
        for m in METHODS:
            d = out.setdefault(f"{key}|{m}", {})
            d["Noise-free sim"] = {"share": v["noise_free"][m]}
            d["Random guess"] = {"share": v["random"]}
            d["cx"] = v["cx"][m]
    json.dump(out, open(HERE / "stretch_hw_summary.json", "w"), indent=1)
    for m, ml in METHODS.items():
        print("\n" + ml)
        for key, v in SCORE.items():
            d = out[f"{key}|{m}"]
            print(f"  {v['title']:16s} {d['cx']:4d} CX  " + "  ".join(
                f"{lab}: {d[lab]['share']:.1%}" for lab in ["Noise-free sim", "IonQ sim (noisy)", "Quantinuum emulator",
                                                            "IonQ Forte 1 chip", "Random guess"] if lab in d))
        figure(out, m, ml)
    print(f"\n{pending} job(s) still running" if pending else "\nall jobs collected", flush=True)


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
    fn = str(HERE / f"stretch_qbraid_{method}{tag}.png")
    fig.savefig(fn, dpi=220, facecolor=fig.get_facecolor())
    print("wrote", fn)



if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "submit":
        meth = [m for m in METHODS if len(a) < 3 or a[2] == "all" or m == a[2]]
        submit(a[1], meth, only=a[3].split(",") if len(a) > 3 else None)
    elif a[0] == "collect":
        collect()
