"""Reproduce the team's synthetic 55,000-bus grid (ilp_55000.ipynb, make_grid(55000), seed 0) and save it
in the challenge's JSON format so check.py and our pipeline can load it."""
import json
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import cKDTree


def make_grid(n, seed=0, extra=0.35):
    rng = np.random.default_rng(seed)
    pts = rng.random((n, 2))
    d, nb = cKDTree(pts).query(pts, k=7)
    rows = np.repeat(np.arange(n), 6)
    cols = nb[:, 1:].ravel()
    G = coo_matrix((d[:, 1:].ravel(), (rows, cols)), shape=(n, n)).tocsr()
    mst = minimum_spanning_tree(G).tocoo()
    edges = {(min(a, b), max(a, b)) for a, b in zip(mst.row.tolist(), mst.col.tolist())}
    cand = sorted({(min(a, b), max(a, b)) for a, b in zip(rows.tolist(), cols.tolist())} - edges)
    for i in rng.choice(len(cand), size=min(int(extra * n), len(cand)), replace=False):
        edges.add(cand[i])
    buses = list(range(1, n + 1))
    lines = [[a + 1, b + 1] for a, b in sorted(edges)]
    return {"name": f"synthetic {n}-bus", "buses": buses, "lines": lines, "zero_injection": [],
            "generator_buses": [], "load_mw": {}}


if __name__ == "__main__":
    g = make_grid(55000)
    json.dump(g, open("data_big/synthetic55k.json", "w"))
    print(g["name"], len(g["lines"]), "lines")
