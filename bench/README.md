# ILP benchmark models

MPS files written by `ilp_bench.py`. The two 55,000-bus models are gzipped to stay small; HiGHS reads `.mps.gz` directly (`h.readModel("bench/synth55k_plain.mps.gz")`). Running `python ilp_bench.py` rewrites all of them uncompressed.
