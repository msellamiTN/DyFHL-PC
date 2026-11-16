"""Change admission under secret sharing on the three benchmarks.
After a batch of rule replacements, each member's conflict and redundancy flags are brought up to date by
checking only pairs that involve a changed rule (inserted or removed), plus a recheck of rows whose only
witnesses may have been removed. The result must equal the pooled oracle on the new state. Cost is compared
with a full secure re-analysis of the new state."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from dyfhl.policy import World, get_source
from dyfhl.mpc import MPC, expand, exact_checks

DS = ["agentic", "xu_stoller", "amazon"]
seeds = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0]
out = []


def chk(m, XI, eI, XJ, eJ, sl):
    if len(XI) == 0 or len(XJ) == 0:
        return np.zeros(len(XI), bool), np.zeros(len(XI), bool), 0
    c, r, _ = exact_checks(m, XI, eI, XJ, eJ, sl)
    return c, r, len(XI) * len(XJ)


for ds in DS:
    for seed in seeds:
        for churn in (0.01, 0.05, 0.10, 0.20):
            w = World(get_source(ds), seed=seed)
            R0, Y0, _, _ = w.labels()
            X0 = expand(R0["H"], R0["Wm"], w.slices)
            old = {u: (Y0[i, 0], Y0[i, 1]) for i, u in enumerate(R0["uid"])}
            w.step(churn, id_flip=0.0)
            R1, Y1, _, _ = w.labels()
            X1 = expand(R1["H"], R1["Wm"], w.slices)
            new1 = ~np.isin(R1["uid"], R0["uid"])
            gone0 = ~np.isin(R0["uid"], R1["uid"])
            conf, red = np.zeros(len(new1), bool), np.zeros(len(new1), bool)
            inc, full = MPC(seed), MPC(seed + 1)
            p_inc = p_full = 0
            t_inc = t_full = 0.0
            for s in range(w.S):
                i1 = np.flatnonzero(R1["scope"] == s)
                i0 = np.flatnonzero(R0["scope"] == s)
                for d in w.members[s]:
                    I = i1[R1["dom"][i1] == d]
                    J = i1[R1["dom"][i1] != d]
                    In, Ik = I[new1[I]], I[~new1[I]]
                    Jn, Jk = J[new1[J]], J[~new1[J]]
                    Jr = i0[(R0["dom"][i0] != d) & gone0[i0]]
                    t = time.time()
                    # inserted rules of d against everything foreign
                    c, r, p = chk(inc, X1[In], R1["eff"][In], X1[J], R1["eff"][J], w.slices)
                    conf[In], red[In] = c, r; p_inc += p
                    # kept rules of d against inserted and removed foreign rules
                    cn, rn, p = chk(inc, X1[Ik], R1["eff"][Ik], X1[Jn], R1["eff"][Jn], w.slices); p_inc += p
                    cr, rr, p = chk(inc, X1[Ik], R1["eff"][Ik], X0[Jr], R0["eff"][Jr], w.slices); p_inc += p
                    oc = np.array([old[u][0] for u in R1["uid"][Ik]], bool)
                    orr = np.array([old[u][1] for u in R1["uid"][Ik]], bool)
                    # rows that lost a witness: recheck against the kept foreign rules
                    rows = np.flatnonzero((oc & cr) | (orr & rr))
                    ck, rk, p = chk(inc, X1[Ik[rows]], R1["eff"][Ik[rows]], X1[Jk], R1["eff"][Jk], w.slices); p_inc += p
                    still_c, still_r = oc.copy(), orr.copy()
                    still_c[rows] = np.where(oc[rows] & cr[rows], ck, oc[rows])
                    still_r[rows] = np.where(orr[rows] & rr[rows], rk, orr[rows])
                    conf[Ik], red[Ik] = cn | still_c, rn | still_r
                    t_inc += time.time() - t
                    t = time.time()
                    _, _, p = chk(full, X1[I], R1["eff"][I], X1[J], R1["eff"][J], w.slices); p_full += p
                    t_full += time.time() - t
            ci, cf = inc.cost(), full.cost()
            row = dict(dataset=ds, seed=seed, churn=churn, changed_rules=int(new1.sum()), rules=int(len(new1)),
                       equals_oracle=bool((conf == Y1[:, 0]).all() and (red == Y1[:, 1]).all()),
                       pairs_inc=p_inc, pairs_full=p_full, MB_inc=ci["online_MB"] / w.K, MB_full=cf["online_MB"] / w.K,
                       s_inc=t_inc / w.K, s_full=t_full / w.K)
            out.append(row)
            print(json.dumps(row), flush=True)
json.dump(out, open("results/change_admission.json", "w"), indent=1)
