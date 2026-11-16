"""E1, E4, E5, E6 on constructed federations.
E1 exactness: incremental admission equals full recomputation after every event; secure flags equal plaintext.
E4 scaling:   full recomputation against incremental admission, region size against the whole state.
E5 secure cost per event (region only) against a secure re-check of the whole state.
E6 output leakage: probes needed to learn a foreign decision row under per-class and existence-only output.
Usage: python3 scripts/federation_eval.py results/federation.json"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from dyfhl.federation import Federation
from dyfhl.secure_fed import secure_pairwise, secure_cross
from dyfhl.mpc import MPC

out = {"e1": [], "held": [], "e4": [], "e5": [], "e6": []}
KINDS = ("add", "del", "redescribe", "approve")

# ---------------- E1
for seed in range(10):
    f = Federation(K=4, rules_per_org=60, seed=seed)
    mism, sp, sc, n = 0, [], [], 0
    for i in range(300):
        ev = f.random_event()
        cells, new, ck = f.apply(ev)
        s1 = f.snapshot(); f.full(); s2 = f.snapshot()
        mism += s1 != s2; n += 1
        lay = ev[1].layer if ev[0] == "add" else None
        out["held"].append(dict(seed=seed, kind=ev[0] if ev[0] != "add" else "add-" + lay,
                                **{k: len(v) for k, v in new.items()}))
        if i % 3 == 0:
            m = MPC(1000 * seed + i)
            if ev[0] == "add":
                r = secure_pairwise(f, m, ev[1].id)
                if r is not None:
                    sp.append(r)
            for (t, r) in ck:
                if f.host[t] not in f.owners(r) or len(f.owners(r)) > 1:
                    sc.append(secure_cross(f, m, t, r))
    out["e1"].append(dict(seed=seed, events=n, mismatches=int(mism), secure_pairwise=len(sp),
                          secure_pairwise_agree=int(sum(sp)), secure_cross=len(sc), secure_cross_agree=int(sum(sc))))
    print("E1", out["e1"][-1], flush=True)

# ---------------- E4
grid = [(2, 200), (4, 200), (8, 200), (16, 200), (4, 50), (4, 800), (4, 3200)]
for K, rpo in grid:
    for seed in range(3):
        f = Federation(K=K, rules_per_org=rpo, seed=seed)
        t0 = time.perf_counter(); f.full(); tf = time.perf_counter() - t0
        tot = f.total_cells()
        ti, cells = [], []
        for i in range(200):
            ev = f.random_event()
            t0 = time.perf_counter(); c, _, _ = f.apply(ev, diff=False); ti.append(time.perf_counter() - t0)
            cells.append(c)
        out["e4"].append(dict(K=K, rules_per_org=rpo, seed=seed, rules=len(f.rules), t_full=tf,
                              t_inc_mean=float(np.mean(ti)), t_inc_median=float(np.median(ti)),
                              t_inc_p99=float(np.percentile(ti, 99)), total_cells=tot,
                              region_frac_median=float(np.median(cells) / tot), region_frac_mean=float(np.mean(cells) / tot)))
        print("E4", out["e4"][-1], flush=True)

# ---------------- E5
for K, rpo in ((4, 60), (4, 200), (8, 200)):
    f = Federation(K=K, rules_per_org=rpo, seed=0)
    # full secure re-check of the state
    m = MPC(7); t0 = time.time()
    for rid in list(f.rules):
        secure_pairwise(f, m, rid)
    pair_full = m.cost()
    m2 = MPC(8)
    for (t, r) in f.cross:
        if f.host[t] not in f.owners(r) or len(f.owners(r)) > 1:
            secure_cross(f, m2, t, r)
    cross_full = m2.cost(); t_full = time.time() - t0
    full_MB = sum(pair_full[k] + cross_full[k] for k in ("online_MB", "input_MB", "output_MB"))
    per = []
    for i in range(100):
        ev = f.random_event()
        _, _, ck = f.apply(ev, diff=False)
        m = MPC(100 + i); t0 = time.time()
        if ev[0] == "add":
            secure_pairwise(f, m, ev[1].id)
        for (t, r) in ck:
            if f.host[t] not in f.owners(r) or len(f.owners(r)) > 1:
                secure_cross(f, m, t, r)
        c = m.cost()
        per.append(dict(kind=ev[0], MB=c["online_MB"] + c["input_MB"] + c["output_MB"], rounds=c["rounds"],
                        seconds=time.time() - t0))
    mb = np.array([p["MB"] for p in per])
    out["e5"].append(dict(K=K, rules_per_org=rpo, rules=len(f.rules), full_MB=full_MB, full_seconds=t_full,
                          full_rounds=pair_full["rounds"] + cross_full["rounds"],
                          event_MB_median=float(np.median(mb)), event_MB_mean=float(mb.mean()),
                          event_MB_p95=float(np.percentile(mb, 95)),
                          event_rounds_median=float(np.median([p["rounds"] for p in per])),
                          event_seconds_median=float(np.median([p["seconds"] for p in per])),
                          by_kind={k: float(np.mean([p["MB"] for p in per if p["kind"] == k] or [0])) for k in KINDS}))
    print("E5", out["e5"][-1], flush=True)

# ---------------- E6: probes needed to learn a foreign decision row
rng = np.random.default_rng(0)
f = Federation(K=4, rules_per_org=60, seed=0)


def group_test(bits):
    """Adaptive binary splitting with an existence oracle: number of queries to recover every 1."""
    q = 0
    stack = [np.arange(len(bits))]
    while stack:
        s = stack.pop(); q += 1
        if bits[s].any() and len(s) > 1:
            h = len(s) // 2; stack += [s[:h], s[h:]]
    return q


for r in range(f.NR):
    row = ~f.D[r][:f.H]                      # classes denied on r (what a gap flag reveals)
    out["e6"].append(dict(res=r, classes=int(f.H), denied=int(row.sum()), per_class_probes=1,
                          existence_probes=group_test(row), entropy_bits=float(
                              0 if row.mean() in (0, 1) else f.H * -(row.mean() * np.log2(row.mean()) +
                                                                       (1 - row.mean()) * np.log2(1 - row.mean())))))
json.dump(out, open(sys.argv[1], "w"), indent=1)
print("done")
