# PMU placement for TVA with reduction plus QAOA

Our solution for the Tennessee Quantum Hackathon, qBraid challenge 2: placing PMUs (phasor measurement units) on TVA's grid so every substation is observable with as few sensors as possible.

## Team

<!-- Add names, roles and GitHub handles -->
| Name | Role | GitHub |
|---|---|---|
| | | |

## The idea

A PMU at a substation sees that substation and every neighbor. Finding the fewest PMUs that see the whole grid is the minimum dominating set problem, and it maps onto qubits as one qubit per substation (1 means put a PMU here).

The catch is that real grids are far too big for today's hardware. So before anything goes on qubits we shrink the problem with three exact rules that never throw away an optimal answer:

1. **Forced PMU.** If a bus can only be seen from one remaining candidate, that candidate gets a PMU.
2. **Covered for free.** If every location that sees bus u also sees bus w, then covering u covers w, so w's constraint is dropped.
3. **Dominated candidate.** If one candidate sees a subset of what another sees, the smaller one is dropped.

What survives splits into small independent pieces. Forced PMUs plus the best answer on each piece equals the true optimum. We then run QAOA (quantum approximate optimization algorithm) on the pieces, with cost = number of PMUs + P × number of unseen buses, where P is a penalty weight.

## Results

| Grid | Greedy | Exact ILP | After rules | Quantum part |
|---|---|---|---|---|
| IEEE 14 | 5 | 4 | 2 forced + 4-qubit core | IonQ Forte 1 chip: 92% of shots optimal, 98% valid |
| IEEE 300 | 96 | 87 | 85 forced + 3-qubit core | IonQ Forte 1 chip: 95% optimal, 96.5% valid |
| Texas 2,000-bus (ACTIVSg2000) | 558 | 512 (1.5 s) | 281 forced + pieces of 726 and 12 qubits | 12-qubit piece on simulators, see below |
| Texas 25,000-bus (ACTIVSg25k) | 8,399 | 7,871 (0.2 s) | 7,857 forced + 6 pieces, at most 5 qubits | all pieces optimal in simulation |
| Texas 70,000-bus (ACTIVSg70k) | 24,119 | 22,777 (0.7 s) | 22,676 forced + 38 pieces of 3 to 19 qubits | all pieces optimal in simulation |
| Team synthetic 55,000-bus | 17,767 | 16,119 (6.5 s) | 498 pieces, largest 172 | 372 pieces up to 16 qubits by QAOA, rest by ILP, total 16,121 |

ILP is the integer linear program, solved with HiGHS. Greedy picks the bus that covers the most unseen buses each step.

**2,000-bus, 12-qubit piece** (3 optimal placements out of 4,096, so random guessing hits 0.07%), P = 1.5, 200 shots per job:

| QAOA layers | Noise-free | IonQ simulator, Forte noise | Quantinuum H2 emulator |
|---|---|---|---|
| 1 | 1.2% | 0.0% | 2.0% |
| 2 | 5.8% | 1.5% | 4.5% |
| 3 | 11.2% | 1.0% | 11.0% |

On this grid the reduction is weaker: one 726-qubit piece is left and the ILP handles it, so the hybrid total is 281 + 227 + 4 = 512, same as the ILP.

### Hardware job IDs (IonQ Forte 1 via qBraid on AWS)

| Circuit | Job ID | Shots |
|---|---|---|
| IEEE 300 core, 2 layers | `aws:ionq:qpu:forte-1-a7b4-qjob-6af8d2e2c8f5265fb3f209ea` | 200 |
| IEEE 14 core, 2 layers, P = 3 | `aws:ionq:qpu:forte-1-a7b4-qjob-6af8d2e3c8f5265fb3f209ed` | 200 |
| 12 pieces of the 55k grid packed in one 36-qubit circuit | `aws:ionq:qpu:forte-1-a7b4-qjob-6af8dca7c8f5265fb3f20cfb` | 100 (queued at time of writing) |

Every simulator and emulator job (678 of them) is in `results/all_jobs.csv`.

### What to keep in mind

The cores that went on the chip are tiny (3 and 4 qubits), and the angles and P were tuned classically first, so the chip results show the circuits run cleanly, not a quantum speedup. Classical ILP solves every grid here in seconds. Where we think this matters is the reduction itself, which cuts a 70,000-bus problem down to pieces small enough for near-term hardware. The IonQ noise model also disagreed with the real chip on our cores (the chip did better), so we would not lean on simulator noise numbers alone.

## Repo map

| Path | What it is |
|---|---|
| `solution.ipynb` | All milestones and stretch goals, executed with outputs. Section 10 sends the hardware jobs |
| `pmu_lib.py` | Greedy, ILPs (plain, N-1, zero-injection, weighted budget), the exact reduction, QAOA cost tables, a numpy QAOA simulator, Qiskit circuits |
| `qaoa_study.py`, `noise_study.py`, `shot_tuning.py`, `grover_study.py` | QAOA layer and penalty studies, noise sweep, tuning from shots, Grover on the cores |
| `ieee300_study.py`, `stretch_study.py` | IEEE 300 and the stretch goals (N-1, zero-injection, fixed budget with NERC-style weights) |
| `matpower_import.py`, `big_grid.py`, `big_quantum.py`, `big_export.py` | Big grids: MATPOWER import, sparse greedy / ILP / reduction, QAOA on each piece, circuit export |
| `run_big.py`, `collect_big.py`, `run_sweep.py`, `collect_sweep.py`, `collect_all.py` | Submit and collect qBraid jobs |
| `hardware/` | Every circuit sent to qBraid as OpenQASM 2, with manifests (angles, bus mapping). `hardware/forte/` holds the real-chip jobs, `hardware/big_case_ACTIVSg2000/` the 2,000-bus runs |
| `ilp_bench.py`, `bench/`, `hardware/gpu/` | ILP benchmark on CPU (HiGHS) and GPU (NVIDIA cuOpt), MPS models |
| `verify.py` | Rechecks the headline numbers against `check.py` and brute force |
| `results/` | JSON and CSV outputs of every study, plus `TVA_PMU_results.xlsx` |
| `figures/`, `deck_charts/` | Charts |
| `tools/grid_explorer.html` | Interactive explorer: place sensors, fail one, compare greedy / ILP / QAOA. Open in a browser |
| `data/`, `data_big/` | IEEE test grids and the Texas synthetic grids in the challenge's JSON format |
| `starter.ipynb`, `qaoa_primer.ipynb`, `check.py`, `CHALLENGE.md` | Provided by the challenge organizers |

`SOLUTION.md` has more detail on each study.

## Running it

```bash
pip install -r requirements.txt
python verify.py            # recheck the headline numbers
python big_grid.py          # reduction and ILP on the big grids
```

For the cloud jobs, open `solution.ipynb` in qBraid Lab with the qBraid-SDK kernel, set `RUN_CLOUD = True` in section 10 and run that cell. Start with the free IonQ simulator (forte-1 noise) and the Quantinuum H2 emulator before spending credits on Forte.

## Data sources

IEEE 14, 30, 57, 118 and 300 bus test cases and the Texas A&M ACTIVSg synthetic grids come from MATPOWER. The 55,000-bus grid is a random grid our team generated (`synth55k.py`, fixed seed).
