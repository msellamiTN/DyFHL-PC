"""Additions for the TOPS-style review:
(1) arithmetic-circuit form of the decoder (product over attributes, masked sum over bins, one zero test
    per rule), checked for equality with the rule decoder, with multiplication counts and an online
    communication estimate for Beaver-triple secret sharing. This is a plaintext simulation of the circuit,
    not a networked MPC implementation.
(2) measured false-positive rate of conflict detection against the bound N(N-1)/(2B).
(3) exposure of updated rules when two consecutive contexts are subtracted."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from dyfhl.policy import World, get_source
from dyfhl.model import Config, DyFHL, hash_bins
from dyfhl.fed import make_clients, exchange, evaluate
torch.set_num_threads(1)
P_ = (1 << 31) - 1
DS = {"agentic": "Agentic", "xu_stoller": "XS-ABAC", "amazon": "AMZ-Access"}
out = {"mpc": [], "fp": [], "upd": []}


def circuit(c, C, rng):
    """Flags from the arithmetic circuit over F_p for one client. Returns flags and multiplication count."""
    Cn = (C.numpy().astype(np.int64)) % P_                       # [G, V+A, 2, B]
    V, A, B = c.geo.V, c.geo.A, Cn.shape[-1]
    tot = np.stack([Cn[:, s:e].sum(1) % P_ for s, e in c.slices], 1)   # [G, A, 2, B]
    H, W, eff, sc = c.H.numpy().astype(bool), c.Wm.numpy().astype(bool), c.eff.numpy(), c.scope.numpy()
    QH, Qs = c.QH.numpy().astype(bool), c.Qscope.numpy()
    rho = rng.integers(1, P_, B)
    mults = 0
    conf, red = np.zeros(len(eff), bool), np.zeros(len(eff), bool)
    for i in range(len(eff)):
        g = sc[i]
        for kind, ch in (("ov", 1 - eff[i]), ("sub", eff[i])):
            acc = None
            for a, (s, e) in enumerate(c.slices):
                x = Cn[g, V + a, ch].copy()
                if W[i, a]:
                    if kind == "ov":
                        x = (x + tot[g, a, ch]) % P_
                else:
                    vs = np.flatnonzero(H[i, s:e]) + s
                    if kind == "ov":
                        x = (x + Cn[g, vs, ch].sum(0)) % P_
                    else:
                        pr = Cn[g, vs[0], ch]
                        for v in vs[1:]:
                            pr = pr * Cn[g, v, ch] % P_; mults += B
                        x = (x + pr) % P_
                if acc is None:
                    acc = x
                else:
                    acc = acc * x % P_; mults += B
            z = int((acc * rho % P_).sum() % P_); mults += B
            if kind == "ov":
                conf[i] = z != 0
            else:
                red[i] = z != 0
    gap = np.zeros(len(Qs), bool)
    for k in range(len(Qs)):
        g, z = Qs[k], 0
        for ch in (0, 1):
            acc = None
            for a, (s, e) in enumerate(c.slices):
                v = int(np.flatnonzero(QH[k, s:e])[0]) + s
                x = (Cn[g, V + a, ch] + Cn[g, v, ch]) % P_
                if acc is None:
                    acc = x
                else:
                    acc = acc * x % P_; mults += B
            z = (z + int((acc * rho % P_).sum())) % P_; mults += B
        gap[k] = z == 0
    return conf, red, gap, mults


rng = np.random.default_rng(0)
for ds in DS:
    w = World(get_source(ds), seed=0); lab = w.labels(); R, Y, Q, gapy = lab
    Nopp = np.zeros(len(R["eff"]))
    for s in range(w.S):
        idx = np.flatnonzero(R["scope"] == s)
        for i in idx:
            Nopp[i] = ((R["dom"][idx] != R["dom"][i]) & (R["eff"][idx] != R["eff"][i])).sum()
    for B in (512, 4096, 16384):
        cfg = Config(h=0, B=B, seed=0)
        clients, _ = make_clients(w, cfg, lab)
        model = DyFHL(clients[0].geo, cfg)
        ctx = exchange(clients, model, cfg, 0)
        res, (P, Pq) = evaluate(clients, model, lab, cfg, rule=True, ctx=ctx)
        neg = ~Y[:, 0]
        out["fp"].append(dict(dataset=ds, B=B, fpr=float((P[neg, 0] > .5).mean()),
                              bound=float(np.minimum(1.0, Nopp[neg] * (Nopp[neg] - 1) / (2 * B)).mean()),
                              medN=float(np.median(Nopp))))
        if B <= 4096:
            agree, mults = True, []
            for c, (C, al) in zip(clients, ctx):
                conf, red, gap, m = circuit(c, C, rng)
                agree &= bool((conf == (P[c.ridx, 0] > .5)).all() and (red == (P[c.ridx, 1] > .5)).all()
                              and (gap == (Pq[c.qidx] > .5)).all())
                mults.append(m)
            out["mpc"].append(dict(dataset=ds, B=B, agree=agree, mults_per_domain=float(np.mean(mults)),
                                   rules_per_domain=float(np.mean([c.n for c in clients])),
                                   online_MB=float(np.mean(mults)) * 32 / 1e6))
        print(ds, B, out["fp"][-1], out["mpc"][-1] if B <= 4096 else "", flush=True)
    for seed in (0, 1, 2):
        w2 = World(get_source(ds), seed=seed); old = set(w2.encode_rules()["uid"].tolist()); w2.step(0.05)
        R2 = w2.encode_rules(); new = ~np.isin(R2["uid"], list(old))
        for B in (64, 512, 4096):
            bins = hash_bins(R2["uid"], B, seed); ex = tot_ = 0
            for s in range(w2.S):
                for d in w2.members[s]:
                    f = np.flatnonzero((R2["scope"] == s) & (R2["dom"] != d) & new)
                    if len(f):
                        key = R2["eff"][f] * B + bins[f]; _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
                        ex += int((cnt[inv] == 1).sum()); tot_ += len(f)
            out["upd"].append(dict(dataset=ds, seed=seed, B=B, exposed=ex / tot_))
json.dump(out, open("results/revision_tops.json", "w"), indent=1)

G = "paper/generated"
with open(f"{G}/tab_fp.tex", "w") as f:
    for B in (512, 4096, 16384):
        row = [str(B)]
        for ds in DS:
            r = [x for x in out["fp"] if x["dataset"] == ds and x["B"] == B][0]
            row += [f"{r['fpr']:.3f}", f"{r['bound']:.3f}"]
        f.write(" & ".join(row) + "\\\\\n")
with open(f"{G}/tab_mpc.tex", "w") as f:
    for r in out["mpc"]:
        f.write(f"{DS[r['dataset']]} & {r['B']} & {r['rules_per_domain']:.0f} & {r['mults_per_domain'] / 1e6:.1f} & "
                f"{r['online_MB']:.0f} & {'yes' if r['agree'] else 'no'}\\\\\n")
with open(f"{G}/revision_numbers.tex", "w") as f:
    for B, nm in ((64, "lxiv"), (512, "dxii"), (4096, "mmmmxcvi")):
        v = [x["exposed"] for x in out["upd"] if x["B"] == B]
        f.write(f"\\newcommand{{\\rvupd{nm}}}{{{100 * np.mean(v):.0f}}}\n")
    f.write(f"\\newcommand{{\\rvmedN}}{{{np.mean([x['medN'] for x in out['fp'] if x['B'] == 512]):.0f}}}\n")
    f.write(f"\\newcommand{{\\rvagree}}{{{'all' if all(x['agree'] for x in out['mpc']) else 'NOT all'}}}\n")
print("agree:", [x["agree"] for x in out["mpc"]])
