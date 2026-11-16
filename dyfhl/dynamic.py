"""Incremental maintenance of the cross-domain context under policy churn.

Vertex aggregates are sums of per-rule messages, so a rule insertion or deletion changes the
aggregate by exactly that rule's contribution. A domain therefore uploads only a sparse delta."""
from __future__ import annotations
import time
import numpy as np
import torch
from .model import Config, DyFHL
from .fed import make_clients, exchange, evaluate


def delta_upload(old, new, model):
    """Sparse delta between two versions of a client's aggregate, computed from changed rules only."""
    keep_old = np.isin(old.uid, new.uid)
    keep_new = np.isin(new.uid, old.uid)
    rm, ad = torch.as_tensor(~keep_old), torch.as_tensor(~keep_new)
    with torch.no_grad():
        Mo = model.messages(old.H[rm], old.Wm[rm], old.eff[rm], old.sig[rm])
        Mn = model.messages(new.H[ad], new.Wm[ad], new.eff[ad], new.sig[ad])
        d = model.aggregate(new.H[ad], new.Wm[ad], new.eff[ad], new.scope[ad], Mn, new.S) - \
            model.aggregate(old.H[rm], old.Wm[rm], old.eff[rm], old.scope[rm], Mo, old.S)
    da = new.alive * new.own - old.alive * old.own
    return d, da


def run_stream(world, cfg: Config, T=10, churn=0.1, model=None, rule=True):
    """Policy stream: compare a stale context, full re-exchange and delta maintenance at each step."""
    clients, labels = make_clients(world, cfg)
    model = model or DyFHL(clients[0].geo, cfg)
    ups = [c.upload(model) for c in clients]
    U = [u.clone() for u, _ in ups]; AL = [a.clone() for _, a in ups]
    stale_ctx = exchange(clients, model, cfg, cfg.seed)
    rows = []
    for t in range(T + 1):
        if t > 0:
            world.step(churn)
            new_clients, labels = make_clients(world, cfg)
            t0 = time.perf_counter()
            deltas = [delta_upload(o, n, model) for o, n in zip(clients, new_clients)]
            t_delta = time.perf_counter() - t0
            for i, (d, da) in enumerate(deltas):
                U[i] += d; AL[i] += da
            t0 = time.perf_counter()
            full = [c.upload(model) for c in new_clients]
            t_full = time.perf_counter() - t0
            err = max(float((U[i] - full[i][0]).abs().max()) for i in range(len(full)))
            b_delta = sum(int((d != 0).sum()) + int((da != 0).sum()) for d, da in deltas) * 8
            b_full = sum(int((u != 0).sum()) + int((a != 0).sum()) for u, a in full) * 8
            clients = new_clients
        else:
            t_delta = t_full = err = 0.0
            b_delta = b_full = sum(int((u != 0).sum()) + int((a != 0).sum()) for u, a in ups) * 8
        g = torch.Generator().manual_seed(cfg.seed + t)
        totU, totA = sum(U), sum(AL)
        ctx = [(totU - U[i] + cfg.sigma * torch.randn(U[i].shape, generator=g),
                totA - AL[i] + cfg.sigma * torch.randn(AL[i].shape, generator=g) * c.id_mask)
               for i, c in enumerate(clients)]
        r_delta, _ = evaluate(clients, model, labels, cfg, rule=rule, ctx=ctx)
        r_stale, _ = evaluate(clients, model, labels, cfg, rule=rule, ctx=stale_ctx)
        rows.append(dict(t=t, f1_delta=r_delta["macro_f1"], f1_stale=r_stale["macro_f1"], bytes_delta=b_delta,
                         bytes_full=b_full, time_delta=t_delta, time_full=t_full, max_abs_err=err,
                         per_task_delta={k: r_delta[k]["f1"] for k in r_delta if isinstance(r_delta[k], dict)},
                         per_task_stale={k: r_stale[k]["f1"] for k in r_stale if isinstance(r_stale[k], dict)}))
    return rows
