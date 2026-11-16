"""Baselines: supervised random forests on audited labels, and configurations of the
federated model that reproduce weaker forms of cross-domain context."""
from __future__ import annotations
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from .model import Config, DyFHL, TASKS
from .fed import make_clients, exchange, metrics

# name -> Config overrides (all trained with the same label-free federated protocol)
VARIANTS = {
    "Local-only": dict(context=False, sketch=False, h=0, priors=True),
    "FedHGNN-count": dict(sketch=False, scope_part=False, conj=False, h=16, priors=True),   # FedHGL-style
    "FedGCN-clique": dict(sketch=False, scope_part=False, conj=False, h=16, priors=True, gcn=True),
    "DyFHL-PA-L": dict(),
}


def rf_baseline(world, labels, seed, context=False, audit=0.2):
    """Random forest trained on a globally audited fraction of labels (which our method does not
    need). Features: the hyperedge's own incidence row and, optionally, flat count context."""
    R, Y, Q, gap = labels
    cfg = Config(sketch=False, h=0, scope_part=False, conj=False, context=context, seed=seed)
    clients, _ = make_clients(world, cfg, labels)
    model = DyFHL(clients[0].geo, cfg)
    ctx = exchange(clients, model, cfg, seed)
    Xr = np.zeros((len(R["eff"]), 0)); Xq = None
    fr, fq = {}, {}
    for c, (C, al) in zip(clients, ctx):
        _, same, opp, irr = c.predict(model, C, al)
        n = c.n
        own_r = torch.cat([c.H, c.Wm, c.eff[:, None].float()], 1)
        own_q = c.QH
        ev = torch.cat([same.flatten(1), opp.flatten(1), irr], 1) if context else torch.zeros(len(irr), 0)
        fr[c.d] = (c.ridx, torch.cat([own_r, ev[:n]], 1).numpy())
        fq[c.d] = (c.qidx, torch.cat([own_q, ev[n:]], 1).numpy())
    Xr = np.zeros((len(R["eff"]), next(iter(fr.values()))[1].shape[1]))
    Xq = np.zeros((len(gap), next(iter(fq.values()))[1].shape[1]))
    for idx, f in fr.values():
        Xr[idx] = f
    for idx, f in fq.values():
        Xq[idx] = f
    rng = np.random.default_rng(seed)
    ys, ps = [], []
    for X, y in [(Xr, Y[:, 0]), (Xr, Y[:, 1]), (Xr, Y[:, 2]), (Xq, gap)]:
        tr = rng.random(len(y)) < audit
        if y[tr].min() == y[tr].max():
            p = np.full((~tr).sum(), float(y[tr].mean()))
        else:
            rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=2).fit(X[tr], y[tr])
            p = rf.predict_proba(X[~tr])[:, 1]
        ys.append(y[~tr]); ps.append(p)
    return metrics(ys, ps)
