"""Exact cross-domain checks under two-server secret sharing, against the pooled oracle and against the
sketch protocol. For every scope and member: flags must equal the exact (normal-form) reference.
Reports communication, rounds, triples and local running time per domain."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from dyfhl.policy import World, get_source
from dyfhl.mpc import MPC, expand

DS = {"agentic": "Agentic", "xu_stoller": "XS-ABAC", "amazon": "AMZ-Access"}
seeds = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0]
out = []
for ds in DS:
    for seed in seeds:
        w = World(get_source(ds), seed=seed)
        R, Y, Q, gapy = w.labels()
        X = expand(R["H"], R["Wm"], w.slices)
        n = len(R["eff"])
        conf, red, gap = np.zeros(n, bool), np.zeros(n, bool), np.zeros(len(gapy), bool)
        rconf, rred = np.zeros(n, bool), np.zeros(n, bool)
        per_dom = {d: MPC(seed=1000 * seed + d) for d in range(w.K)}
        secs = {d: 0.0 for d in range(w.K)}
        pairs = {d: 0 for d in range(w.K)}
        for s in range(w.S):
            idx = np.flatnonzero(R["scope"] == s)
            for d in w.members[s]:
                I = idx[R["dom"][idx] == d]
                J = idx[R["dom"][idx] != d]
                qi = np.flatnonzero((Q["scope"] == s) & (Q["dom"] == d))
                if len(I) == 0 or len(J) == 0:
                    gap[qi] = True
                    continue
                from dyfhl.mpc import exact_checks
                t = time.time()
                c, r, g = exact_checks(per_dom[d], X[I], R["eff"][I], X[J], R["eff"][J], w.slices,
                                       Q["H"][qi].astype(np.int64))
                secs[d] += time.time() - t
                pairs[d] += len(I) * len(J)
                conf[I], red[I] = c, r
                if g is not None:
                    gap[qi] = g
                # plaintext reference on the same expanded rows (normal form)
                xe = R["eff"][I][:, None] != R["eff"][J][None, :]
                ov = np.ones((len(I), len(J)), bool)
                for a, b in w.slices:
                    ov &= (X[I][:, a:b] @ X[J][:, a:b].T) > 0
                rconf[I] = (ov & xe).any(1)
                rred[I] = (((X[I].sum(1)[:, None] - X[I] @ X[J].T) == 0) & ~xe).any(1)
        costs = [per_dom[d].cost() for d in range(w.K)]
        row = dict(dataset=ds, seed=seed, V=int(w.V), A=int(w.A), rules=int(n),
                   conflict_equals_reference=bool((conf == rconf).all()),
                   redundancy_equals_reference=bool((red == rred).all()),
                   conflict_equals_oracle=bool((conf == Y[:, 0]).all()),
                   redundancy_vs_oracle_extra=int((red & ~Y[:, 1]).sum()),
                   redundancy_vs_oracle_missed=int((~red & Y[:, 1]).sum()),
                   gap_equals_oracle=bool((gap == gapy).all()),
                   pairs_per_domain=float(np.mean(list(pairs.values()))),
                   seconds_per_domain=float(np.mean(list(secs.values()))),
                   **{k: float(np.mean([c[k] for c in costs])) for k in costs[0]})
        out.append(row)
        print(json.dumps(row), flush=True)
json.dump(out, open("results/secure_exact.json", "w"), indent=1)
