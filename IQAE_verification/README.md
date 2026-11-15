# QAOA -> IQAE verification kit (TVA sensor placement)

Files
- verify_qaoa_iqae.py : stand-alone verifier (own IQAE implementation, no Qiskit needed except for --circuit-check)
- iae_qiskit_demo.py  : same check through qiskit_algorithms.IterativeAmplitudeEstimation
- requirements.txt

Setup:  pip install -r requirements.txt   (use --break-system-packages on qBraid if pip complains)

What you pass in: the tuned QAOA angles as floats, ALL gammas first then ALL betas. Not a bit string.
Also use the same grid file, penalty P, and number of layers as your QAOA run.

Run (from the repo root, with these files copied there or on PYTHONPATH):
  python3 verify_qaoa_iqae.py --grid 2-tva-sensor-placement/data/ieee14_piece6.json \
      --params=-1.5708,0.5004 --good optimal --epsilon-target 0.01 --claimed 0.21
  python3 iae_qiskit_demo.py --grid 2-tva-sensor-placement/data/ieee14_piece6.json --params=-1.5708,0.5004

Note the "=" after --params (negative numbers).

Verifying one exact placement: --good buses:2,5 (verifier) or --bits 010011 (demo, bus0 first; reversed internally).

Limits: simulated; the oracle is a classical lookup over 2^n entries, so practical up to ~20-24 buses.
This verifies/estimates QAOA's success probability, it does not solve the problem or show a speedup.
