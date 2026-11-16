"""Rule-exposure experiment. A scope member that receives the context can read off a foreign rule
exactly when that rule is alone in its (scope, effect, bin) cell among the foreign rules the member
sees. We report the exposed fraction against the sketch size B, together with a case study of
anomalies found in the XS-ABAC benchmark."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from dyfhl.policy import World, get_source, pair_matrices
from dyfhl.model import hash_bins

rows = []
for ds in ["agentic", "xu_stoller", "amazon"]:
    for seed in [0, 1, 2]:
        w = World(get_source(ds), seed=seed)
        R = w.encode_rules()
        for B in [32, 64, 128, 256, 512, 1024, 2048, 4096, 16384]:
            bins = hash_bins(R["uid"], B, seed)
            exposed, total, occ, pred = 0, 0, [], []
            for s in range(w.S):
                for d in w.members[s]:
                    f = np.flatnonzero((R["scope"] == s) & (R["dom"] != d))      # foreign rules seen by d
                    if len(f) == 0:
                        continue
                    key = R["eff"][f] * B + bins[f]
                    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
                    exposed += int((cnt[inv] == 1).sum()); total += len(f)
                    occ.append(len(f) / (2 * B))
                    for e in (0, 1):
                        n = int((R["eff"][f] == e).sum())
                        pred.append((n, (1 - 1 / B) ** max(n - 1, 0)))
            rows.append(dict(dataset=ds, seed=seed, B=B, exposed=exposed / total, mean_rules_per_cell=float(np.mean(occ)),
                             predicted=float(sum(n * p for n, p in pred) / sum(n for n, _ in pred))))
    print(ds, [round(r["exposed"], 3) for r in rows if r["dataset"] == ds and r["seed"] == 0])
json.dump(rows, open("results/leakage.json", "w"), indent=1)

# ---- case study: anomalies between two organisations in the healthcare application of XS-ABAC
w = World(get_source("xu_stoller"), seed=0)
R, Y, Q, gap = w.labels()
inv = {v: k for k, v in w.vidx.items()}
def show(i):
    r = w.rules[i]
    conds = "; ".join(f"{a}: " + ", ".join(sorted(str(v).split('.')[-1] for v in vs)) for a, vs in sorted(r["conds"].items()))
    return dict(uid=r["uid"], dom=r["dom"], scope=r["scope"], effect="deny" if r["eff"] else "permit", conds=conds)
out = {}
for task, col, need_opp in [("x-conflict", 0, True), ("x-redundancy", 1, False)]:
    for i in np.flatnonzero(Y[:, col]):
        if "hc" not in w.rules[i]["conds"].get("app", set()):
            continue
        idx = np.flatnonzero(R["scope"] == R["scope"][i])
        ov, sub = pair_matrices(R["H"][idx], R["Wm"][idx], w.slices)
        k = int(np.flatnonzero(idx == i)[0])
        rel = ov[k] if need_opp else sub[k]
        ok = rel & (R["dom"][idx] != R["dom"][i]) & ((R["eff"][idx] != R["eff"][i]) if need_opp else (R["eff"][idx] == R["eff"][i]))
        js = idx[ok]
        js = [j for j in js if len(w.rules[j]["conds"]) >= 3 and len(w.rules[i]["conds"]) >= 3 and w.rules[j]["conds"] != w.rules[i]["conds"]]
        if js:
            out[task] = [show(i), show(js[0])]
            break
from dyfhl.model import Config
from dyfhl.fed import run
uid_pos = {u: i for i, u in enumerate(R["uid"])}
for B in (512, 4096):
    r, *_rest, (P, Pq) = run(w, Config(seed=0, h=0, B=B), labels=(R, Y, Q, gap), rule=True, return_all=True)
    for task in ("x-conflict", "x-redundancy"):
        for x in out.get(task, []):
            i = uid_pos[x["uid"]]
            x[f"flags_B{B}"] = {"x-conflict": bool(P[i, 0] > .5), "x-redundancy": bool(P[i, 1] > .5)}
            x["oracle"] = {"x-conflict": bool(Y[i, 0]), "x-redundancy": bool(Y[i, 1])}
    out[f"summary_B{B}"] = {t: dict(prec=r[t]["prec"], rec=r[t]["rec"], f1=r[t]["f1"]) for t in ("x-conflict", "x-redundancy")}
out["scope_members"] = w.members[out["x-conflict"][0]["scope"]]
out["n_rules_scope"] = int((R["scope"] == out["x-conflict"][0]["scope"]).sum())
json.dump(out, open("results/case_study.json", "w"), indent=1)
print(json.dumps(out, indent=1))
