"""Check: k independent sketches of B/k bins each (AND of flags for conflict/redundancy, OR for gaps,
which keeps every one-sided guarantee) against one sketch of B bins. Seed 0, rule decoder."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from dyfhl.policy import World, get_source
from dyfhl.model import Config, hash_bins
from dyfhl.fed import run, metrics
torch.set_num_threads(1)
TASKS = ["x-conflict", "x-redundancy", "x-irrelevancy", "x-gap"]
rows = []
for ds in ["agentic", "xu_stoller", "amazon"]:
    w = World(get_source(ds), seed=0); lab = w.labels(); R, Y, Q, gap = lab
    for Btot in (512, 4096):
        for k in (1, 2, 4):
            Ps, Pqs = [], []
            for j in range(k):
                *_, (P, Pq) = run(w, Config(h=0, B=Btot // k, seed=j), labels=lab, rule=True, return_all=True)
                Ps.append(P > .5); Pqs.append(Pq > .5)
            P, Pq = np.logical_and.reduce(Ps), np.logical_or.reduce(Pqs)
            m = metrics([Y[:, 0], Y[:, 1], Y[:, 2], gap], [P[:, 0], P[:, 1], P[:, 2], Pq])
            f1 = [m[t]["f1"] for t in TASKS if m[t]["pos"] > 0]
            rows.append(dict(dataset=ds, B=Btot, k=k, macro_f1=float(np.mean(f1)), **{t: m[t]["f1"] for t in TASKS}))
            print(ds, Btot, k, round(rows[-1]["macro_f1"], 3), flush=True)
json.dump(rows, open("results/multihash.json", "w"), indent=1)
