# Our solution: TVA Grid Sensor Placement

Main idea: shrink the problem with exact covering rules before it goes on qubits. The rules never lose an
optimal answer. On IEEE 14 they fix PMUs at buses 2 and 7 and leave a 4-qubit core, so the QAOA circuit for
the whole grid drops from 336 two-qubit gates per layer to 8. That let us send the full IEEE 14 problem (and
the full IEEE 300 problem, whose core is 3 qubits) to IonQ, not just the 6-bus piece.

## Files we added

| File | What it is |
|---|---|
| `solution.ipynb` | All milestones and stretch goals, executed with outputs. Section 10 sends the hardware jobs. |
| `pmu_lib.py` | Greedy, ILPs (plain, N-1, zero-injection, N-1 + zero-injection, weighted budget), the exact reduction, QAOA cost tables, a fast numpy QAOA simulator, Qiskit circuits, XY-mixer budget QAOA |
| `qaoa_study.py` | QAOA on the 6-bus piece, full IEEE 14 and the reduced core, layers 1 to 4, CVaR vs mean |
| `noise_study.py` | Two-qubit error sweep with our own depolarizing model |
| `shot_tuning.py` | Tuning QAOA from sampled shots only, no full cost table |
| `grover_study.py` | Grover search on the reduced cores: the coverage check is the oracle, no angles to tune |
| `ieee300_study.py` | IEEE 300, every stretch goal: exact answers, budget curve with NERC-style weights, QAOA window search vs random sampling vs exact window solve |
| `stretch_study.py` | NERC-style priority table, fixed budget (ILP vs XY mixer vs penalty QAOA), N-1 and zero-injection QAOA |
| `verify.py` | Rechecks every number in the deck against `check.py` and brute force |
| `results/` | JSON outputs of the studies, `run_record.csv` once cloud jobs run, and hardware counts |
| `hardware/` | The three hardware circuits as OpenQASM 2 plus their angles and bus mapping |
| `tools/grid_explorer.html` | Interactive explorer: place sensors, fail one, load greedy / ILP / QAOA answers, all six grids. Open in any browser. Rebuild with `python tools/build_tool_data.py`. |
| `matpower_import.py`, `big_grid.py`, `big_quantum.py`, `big_export.py` | Grid scale: convert MATPOWER cases, sparse greedy / ILP / reduction, QAOA on every reduced piece, export piece circuits |
| `synth55k.py` | Rebuilds the team's random 55,000-bus grid from `ilp_55000.ipynb` (same generator and seed) |
| `sweep_build.py`, `run_sweep.py`, `collect_sweep.py` | Saturday sweep: 21 circuits (layers, penalty P, Grover steps, QUBO forms, N-1 and budget circuits) for qBraid |
| `run_big.py`, `collect_big.py` | Submit every piece circuit of a big grid to qBraid and assemble the full placement from the shots |
| `data_big/` | ACTIVSg25k and ACTIVSg70k (Texas A&M synthetic grids from MATPOWER) in the challenge's JSON format |
| `deck_charts/` | Charts used in the pitch deck |

## Headline numbers

| Grid | Greedy | ILP | ZIB | N-1 | N-1 + ZIB | Qubits after rules |
|---|---|---|---|---|---|---|
| IEEE 14 | 5 | 4 | 3 | 9 | 7 | 4 |
| IEEE 30 | 10 | 10 | 7 | 21 | 14 | 0 |
| IEEE 57 | 18 | 17 | 11 | 33 | 23 | 28 |
| IEEE 118 | 36 | 32 | 30 | 68 | 63 | 41 |
| IEEE 300 | 96 | 87 | 70 | 202 | 159 | 3 |

N-1 + ZIB uses our scenario ILP (adds one failure scenario at a time): under 2 s up to IEEE 118, about
14 minutes on IEEE 300.

QAOA on IEEE 14 (ideal simulator): full 14 qubits, 4 layers, 16% of shots optimal (random 0.03%). Reduced
core, 3 layers, 24 CX: 92% optimal, 81% under our 0.5% two-qubit-error model. Grover on the same core,
2 steps, 84 CX, no tuning: 94% optimal, 67% under the same noise.

IEEE 300 QAOA window search (simulator): with 14-qubit windows QAOA goes from greedy 96 to 88 PMUs, the
same as solving each window exactly, while random sampling with the same shots stops at 92 (optimum 87).

## Grid scale (synthetic, 25,000 and 70,000 buses)

ACTIVSg70k: greedy 24,119 PMUs, exact ILP 22,777 in 0.7 s, N-1 53,612. The rules fix 22,676 sensors and leave
191 qubits in 38 independent pieces (3 to 19 qubits); QAOA solved every piece to the optimum in simulation,
so the assembled placement is 22,777. ACTIVSg25k: ILP 7,871 in 0.2 s, 6 pieces of at most 5 qubits, all
solved by QAOA. Zero-injection on 25k: 5,529, valid but not proven minimal (20-minute limit, 48% gap).

Team 55,000-bus random grid: greedy 17,767, ILP 16,119 in 6.5 s, N-1 33,972. Rules leave 7,020 qubits in
498 pieces (largest 172). QAOA solved the 372 pieces of up to 16 qubits (370 optimally); the ILP solved the rest.
Hybrid total 16,121, valid.

Sweep on IonQ's Forte-noise simulator: IEEE 14 core with 2 layers and penalty P = 3 reaches 81% optimal with 16
two-qubit gates (as good as 5 layers with 40); IEEE 300 core with 4 layers 93.5%; QUBO forms 0.5% to 16%; the
N-1 and budget circuits (hundreds of gates) near 0%.

## Running the hardware jobs

Five circuits: the 6-bus piece, the IEEE 14 core (QAOA and Grover), the IEEE 300 core (QAOA and Grover).

In qBraid Lab with the qBraid-SDK kernel, open `solution.ipynb`, run all, set `RUN_CLOUD = True` in section 10
and run that cell, then rerun the next cell until the jobs finish. Start with the IonQ simulator (forte-1
noise) and the Quantinuum H2 emulator, then uncomment the Forte line. 200 shots each.

## ILP benchmark and job histograms (Saturday afternoon)

`ilp_bench.py` writes the exact ILPs (every bus seen, N-1, zero-injection) for IEEE 14, IEEE 300 and the 55k grid
to MPS files and solves each one repeatedly with HiGHS on CPU (100 seeds for IEEE, 20 for 55k).
`run_gpu_bench.py` (in `hardware/gpu/` of the bundle) solves the same MPS files with NVIDIA cuOpt on a Colab T4.
`collect_all.py` pulls every qBraid job (678, with job IDs) into `results/all_jobs.csv`.
`make_histograms.py` draws `figures/hist_*.png` and writes the binned numbers to `results/hist_data.json`.

Every CPU and GPU run on IEEE 14 and 300 reached the proven optimum. On the 55k grid HiGHS proves 16,119 (median 4 to
5 s, a few seeds take up to 64 s) and N-1 33,972 (about 3 to 5 s). cuOpt on the T4 stops at its time limit there
(best 16,120 after 600 s, 16,157 and 16,223 after 120 s; N-1 33,978 and 33,982 after 120 s). GPU LP relaxations take
1.5 to 3 s versus about 1 s on CPU. For these covering ILPs the CPU solver is the right tool.
