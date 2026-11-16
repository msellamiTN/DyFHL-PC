"""Recompute the x-irrelevancy metrics of rule-decoder rows with the exact per-attribute test.
Without noise this decision depends only on the world (rules, ownership, liveness), not on B,
the hash or the structural variants, so it is computed once per world and written into every
rule-decoder row of that world. Only XS-ABAC is affected (several entity attributes per rule)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from dyfhl.policy import World, get_source
from dyfhl.model import Config
from dyfhl.fed import run
torch.set_num_threads(1)
TASKS = ["x-conflict", "x-redundancy", "x-irrelevancy", "x-gap"]
cache = {}


def irr(ds, seed, natural):
    key = (ds, seed, natural)
    if key not in cache:
        kw = dict(p_conf=0.0, p_red=0.0, p_irr=0.0) if natural else {}
        w = World(get_source(ds), seed=seed, **kw)
        cache[key] = run(w, Config(h=0, B=64, seed=seed), labels=w.labels(), rule=True)["x-irrelevancy"]
    return cache[key]


def is_rule(r):
    return r.get("decoder") == "rule" or str(r.get("method", "")).startswith("DyFHL-PA-R") or r.get("group") == "structure"


changed = 0
for name, natural in [("main", False), ("sketch", False), ("ablation", False), ("natural", True)]:
    p = f"results/{name}.json"
    if not os.path.exists(p):
        continue
    rows = json.load(open(p))
    for r in rows:
        if r.get("dataset") != "xu_stoller" or not is_rule(r) or r.get("sigma", 0) not in (0, 0.0):
            continue
        new = irr("xu_stoller", r["seed"], natural)
        if abs(new["f1"] - r["x-irrelevancy"]["f1"]) > 1e-12:
            changed += 1
        r["x-irrelevancy"] = new
        r["macro_f1"] = float(np.mean([r[t]["f1"] for t in TASKS]))
        r["macro_ap"] = float(np.nanmean([r[t]["ap"] for t in TASKS]))
    json.dump(rows, open(p, "w"), indent=1)
print("rows changed:", changed)
