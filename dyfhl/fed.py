"""Federated, label-free training of DyFHL-PA and cross-domain evaluation.

Training needs no global labels: each domain splits its own rules into two pseudo-domains,
adds synthetic variants, labels the pair exactly with its local oracle and learns to recover
those labels from the aggregated context of the other half. Model weights are averaged
across domains (FedAvg). At inference the real cross-domain context is used.
"""
from __future__ import annotations
import copy, time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
from .model import Config, DyFHL, Geometry, signature, t_cover, TASKS
from .policy import World

T = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), dtype=dt)


class Client:
    def __init__(self, world: World, d, R, Q, cfg: Config, n_train_q=300):
        self.d, self.cfg, self.S = d, cfg, world.S
        m = R["dom"] == d
        self.ridx = np.flatnonzero(m)
        self.H, self.Wm = T(R["H"][m]), T(R["Wm"][m])
        self.eff, self.scope = T(R["eff"][m], torch.long), T(R["scope"][m], torch.long)
        self.uid = R["uid"][m]
        self.sig = signature(self.uid, cfg)
        mq = Q["dom"] == d
        self.qidx = np.flatnonzero(mq)
        self.QH, self.Qscope = T(Q["H"][mq]), T(Q["scope"][mq], torch.long)
        tq = world.sample_requests(d, n_train_q)
        self.TQ, self.TQscope = T(tq["H"]), T(tq["scope"], torch.long)
        self.id_mask = T(world.id_mask)
        self.own = T((world.owner == d) & world.id_mask)
        self.alive = T(world.alive)
        self.id_attrs = [world.attrs.index(a) for a in world.source.id_attrs]
        self.slices = world.slices
        self.geo = Geometry(world.slices)
        self.block = torch.zeros(self.geo.A, self.geo.V)
        for a, (s, e) in enumerate(world.slices):
            self.block[a, s:e] = 1
        self.rng = torch.Generator().manual_seed(1000 * cfg.seed + d)

    @property
    def n(self):
        return len(self.eff)

    # -- irrelevancy features / labels
    def irr_feat(self, H, own, alive_tot):
        n = H.shape[0]
        amin, con_any, for_any = torch.ones(n), torch.zeros(n), torch.zeros(n)
        for a in self.id_attrs:
            s, e = self.slices[a]
            Ha, idm = H[:, s:e], self.id_mask[s:e]
            con = ((Ha * idm).sum(1) > 0) & ((Ha * (1 - idm)).sum(1) == 0)
            amax = (Ha * idm * alive_tot[s:e]).amax(1)
            amin = torch.where(con, torch.minimum(amin, amax), amin)
            con_any = torch.maximum(con_any, con.float())
            for_any = torch.maximum(for_any, (con & ((Ha * idm * (1 - own[s:e])).sum(1) > 0)).float())
        return torch.stack([amin, con_any, for_any], 1)

    def irr_label(self, H, own, alive):
        n = H.shape[0]
        y, valid = torch.zeros(n, dtype=torch.bool), torch.ones(n, dtype=torch.bool)
        for a in self.id_attrs:
            s, e = self.slices[a]
            Ha, idm = H[:, s:e], self.id_mask[s:e]
            con = ((Ha * idm).sum(1) > 0) & ((Ha * (1 - idm)).sum(1) == 0)
            all_dead = (Ha * idm * alive[s:e]).sum(1) == 0
            foreign = (Ha * idm * (1 - own[s:e])).sum(1) > 0
            y |= con & all_dead & foreign
            valid &= (Ha * idm * (1 - self.own[s:e])).sum(1) == 0   # only entities this domain really owns
        return y, valid

    def irr_exact(self, H, own, alive_tot):
        """Definition of x-irrelevancy, attribute by attribute: some entity attribute is constrained only by
        entity values, none of them alive, and at least one of them owned by another domain."""
        y = torch.zeros(H.shape[0], dtype=torch.bool)
        for a in self.id_attrs:
            s, e = self.slices[a]
            Ha, idm = H[:, s:e], self.id_mask[s:e]
            con = ((Ha * idm).sum(1) > 0) & ((Ha * (1 - idm)).sum(1) == 0)
            dead = (Ha * idm * alive_tot[s:e]).amax(1) < 0.5
            foreign = (Ha * idm * (1 - own[s:e])).sum(1) > 0
            y |= con & dead & foreign
        return y

    # -- synthetic pseudo-anomalies
    def augment(self, k):
        g, A = self.rng, self.geo.A
        i = torch.randint(0, self.n, (k,), generator=g)
        H, W, eff = self.H[i].clone(), self.Wm[i].clone(), self.eff[i].clone()
        flip = torch.rand(k, generator=g) < 0.5
        eff = torch.where(flip, 1 - eff, eff)
        u = torch.rand(k, generator=g)
        gen = (u < 0.45) & ((1 - W).sum(1) > 1)
        spec = (u > 0.75) & (W.sum(1) > 0)
        r = torch.arange(k)
        if gen.any():      # drop one constrained attribute
            a = torch.multinomial((1 - W[gen]) + 1e-9, 1, generator=g).squeeze(1)
            H[gen] = H[gen] * (1 - self.block[a])
            W[r[gen], a] = 1
        if spec.any():     # constrain one wildcard attribute to a single value
            a = torch.multinomial(W[spec] + 1e-9, 1, generator=g).squeeze(1)
            v = torch.multinomial(self.block[a], 1, generator=g).squeeze(1)
            H[r[spec], v] = 1
            W[r[spec], a] = 0
        # half of the copies refer to entities this domain owns, so that liveness can be supervised
        for a in self.id_attrs:
            s, e = self.slices[a]
            own = self.own[s:e]
            if own.sum() == 0:
                continue
            sel = torch.nonzero(torch.rand(k, generator=g) < 0.5 / len(self.id_attrs)).squeeze(1)
            if len(sel) == 0:
                continue
            v = torch.multinomial(own.expand(len(sel), -1), 1, generator=g).squeeze(1)
            H[sel, s:e] = 0
            H[sel, s + v] = 1
            W[sel, a] = 0
        return i, H, W, eff

    def recombine(self, m):
        """Synthetic unrelated rules: every attribute block is taken from a different random rule, so
        attribute marginals are preserved while the density of the pseudo-foreign side matches that
        of a real multi-domain context."""
        g = self.rng
        src = torch.randint(0, self.n, (m, self.geo.A), generator=g)
        H = torch.zeros(m, self.geo.V)
        W = torch.zeros(m, self.geo.A)
        for a, (s, e) in enumerate(self.slices):
            H[:, s:e] = self.H[src[:, a], s:e]
            W[:, a] = self.Wm[src[:, a], a]
        j = torch.randint(0, self.n, (m,), generator=g)
        return H, W, self.eff[j], self.scope[j]

    def pair_rect(self, H1, W1, H2, W2):
        ov = torch.ones(H1.shape[0], H2.shape[0], dtype=torch.bool)
        s12, s21 = ov.clone(), ov.clone()
        for a, (s, e) in enumerate(self.slices):
            inter = H1[:, s:e] @ H2[:, s:e].T
            w1, w2 = W1[:, a].bool()[:, None], W2[:, a].bool()[None, :]
            ov &= (inter > 0) | w1 | w2
            s12 &= w2 | (~w1 & (inter >= H1[:, s:e].sum(1)[:, None] - 1e-6))
            s21 &= w1 | (~w2 & (inter >= H2[:, s:e].sum(1)[None, :] - 1e-6))
        return ov, s12, s21

    def base_pairs(self):
        if not hasattr(self, "_bp"):
            ov, s12, _ = self.pair_rect(self.H, self.Wm, self.H, self.Wm)
            self._bp = (ov, s12)
        return self._bp

    # -- one local self-supervised step
    def pseudo_loss(self, model: DyFHL):
        cfg, g, S = self.cfg, self.rng, self.S
        n, k = self.n, int(cfg.aug * self.n)
        half = torch.randint(0, 2, (n,), generator=g)
        i, Hc, Wc, effc = self.augment(k)
        m = int(cfg.recomb * n)
        Hr, Wr, effr, scr = self.recombine(m)
        Hc, Wc, effc = torch.cat([Hc, Hr]), torch.cat([Wc, Wr]), torch.cat([effc, effr])
        H, W = torch.cat([self.H, Hc]), torch.cat([self.Wm, Wc])
        eff, scope = torch.cat([self.eff, effc]), torch.cat([self.scope, self.scope[i], scr])
        half = torch.cat([half, 1 - half[i], torch.randint(0, 2, (m,), generator=g)])
        k = k + m
        sig = torch.cat([self.sig, signature(torch.randint(1, 2 ** 31 - 1, (k,), generator=g), cfg)])
        # exact local labels between the two halves
        ovb, subb = self.base_pairs()
        ovc, s12, s21 = self.pair_rect(Hc, Wc, H, W)          # copies vs all
        N = n + k
        ov = torch.zeros(N, N, dtype=torch.bool); sub = torch.zeros(N, N, dtype=torch.bool)
        ov[:n, :n], sub[:n, :n] = ovb, subb
        ov[n:], ov[:, n:] = ovc, ovc.T
        sub[n:], sub[:, n:] = s12, s21.T
        cross = (scope[:, None] == scope[None, :]) & (half[:, None] != half[None, :])
        de = eff[:, None] != eff[None, :]
        y_conf = (ov & cross & de).any(1)
        y_red = (sub & cross & ~de).any(1)
        # pseudo ownership / liveness of this domain's own entities
        alive = (torch.rand(len(self.own), generator=g) < 0.5).float() * self.own
        pf = (torch.rand(len(self.own), generator=g) < 0.5).float() * self.own      # pseudo-foreign
        y_irr, v_irr = self.irr_label(H, self.own - pf, alive)
        # queries
        qh = torch.randint(0, 2, (len(self.TQscope),), generator=g)
        cov = t_cover(self.TQ, H, W) & (self.TQscope[:, None] == scope[None, :])
        own_cov = (cov & (qh[:, None] == half[None, :])).any(1)
        y_gap = ~(cov & (qh[:, None] != half[None, :])).any(1)
        # messages, context of the other half
        if cfg.context:
            M = model.messages(H, W, eff, sig)
            U = torch.stack([model.aggregate(H[half == h], W[half == h], eff[half == h], scope[half == h],
                                             M[half == h], S) for h in (0, 1)])
            C = torch.flip(U, [0])
            noise = cfg.sigma if cfg.noise_aware else 0.0
            if noise:
                C = C + noise * torch.randn(C.shape, generator=g)
            G = C.shape[1]
            C = C.reshape(2 * G, *C.shape[2:])
            ctx_alive = alive * pf + noise * torch.randn(len(pf), generator=g) * self.id_mask
        else:
            G = S if cfg.scope_part else 1
            C = torch.zeros(2 * G, self.geo.V + self.geo.A, 2, cfg.D)
            ctx_alive = torch.zeros(len(pf))
        gs = lambda sc, hf: hf * G + (sc if cfg.scope_part else torch.zeros_like(sc))
        # mini-batch of hyperedges for the loss (context always comes from all rules)
        br = torch.randperm(N, generator=g)[:cfg.batch]
        bq = torch.randperm(len(qh), generator=g)[:cfg.batch // 2]
        H, W, eff, scope, half = H[br], W[br], eff[br], scope[br], half[br]
        y_conf, y_red, y_irr, v_irr = y_conf[br], y_red[br], y_irr[br], v_irr[br]
        TQ, TQs, qh, own_cov, y_gap = self.TQ[bq], self.TQscope[bq], qh[bq], own_cov[bq], y_gap[bq]
        N = len(br)
        Ha, Wa = torch.cat([H, TQ]), torch.cat([W, torch.zeros(len(qh), self.geo.A)])
        effa = torch.cat([eff, torch.zeros(len(qh), dtype=torch.long)])
        isq = torch.cat([torch.zeros(N, dtype=torch.bool), torch.ones(len(qh), dtype=torch.bool)])
        grp = torch.cat([gs(scope, half), gs(TQs, qh)])
        irr = self.irr_feat(Ha, self.own - pf, alive * (self.own - pf) + ctx_alive)
        if not cfg.context:
            irr = irr * torch.tensor([0.0, 1.0, 1.0])
        adj = self.adjacency(Ha, half=torch.cat([half, qh])) if cfg.gcn else None
        self.last_pos = [float(y_conf.float().mean()), float(y_red.float().mean()),
                         float(y_irr[v_irr].float().mean()) if v_irr.any() else 0.0,
                         float(y_gap[~own_cov].float().mean()) if (~own_cov).any() else 0.0]
        logits, _ = model(Ha, Wa, effa, isq, grp, C, irr, adj)
        bce = F.binary_cross_entropy_with_logits
        loss = bce(logits[:N, 0], y_conf.float()) + bce(logits[:N, 1], y_red.float())
        if v_irr.any():      # entity-constrained rules carry the signal; the rest are easy negatives
            hard = v_irr & (irr[:N, 1] > 0)
            loss = loss + 0.2 * bce(logits[:N][v_irr, 2], y_irr[v_irr].float())
            if hard.any():
                loss = loss + bce(logits[:N][hard, 2], y_irr[hard].float())
        if (~own_cov).any():
            loss = loss + bce(logits[N:][~own_cov, 3], y_gap[~own_cov].float())
        return loss

    def adjacency(self, H, half=None):
        A = (H @ H.T > 0).float()
        if half is not None:
            A = A * (half[:, None] == half[None, :]).float()
        return A / A.sum(1, keepdim=True).clamp(min=1)

    # -- real exchange
    def upload(self, model: DyFHL):
        with torch.no_grad():
            M = model.messages(self.H, self.Wm, self.eff, self.sig)
            return model.aggregate(self.H, self.Wm, self.eff, self.scope, M, self.S), self.alive * self.own

    def predict(self, model: DyFHL, C, ctx_alive):
        cfg = self.cfg
        nq = len(self.Qscope)
        H = torch.cat([self.H, self.QH]); W = torch.cat([self.Wm, torch.zeros(nq, self.geo.A)])
        eff = torch.cat([self.eff, torch.zeros(nq, dtype=torch.long)])
        isq = torch.cat([torch.zeros(self.n, dtype=torch.bool), torch.ones(nq, dtype=torch.bool)])
        sc = torch.cat([self.scope, self.Qscope])
        grp = sc if cfg.scope_part else torch.zeros_like(sc)
        irr = self.irr_feat(H, self.own, self.alive * self.own + ctx_alive)
        if not cfg.context:
            irr = irr * torch.tensor([0.0, 1.0, 1.0])
        adj = self.adjacency(H) if cfg.gcn else None
        with torch.no_grad():
            if cfg.gcn:
                logits, (same, opp) = model(H, W, eff, isq, grp, C, irr, adj)
            else:   # decode in chunks to bound memory for large sketches
                ch = max(8, (256 * 512) // max(cfg.D, 512))     # keep the evidence tensors near 16 MB
                prep = model.prepare(C)                          # extended context, built once
                outs = [model(H[i:i + ch], W[i:i + ch], eff[i:i + ch], isq[i:i + ch], grp[i:i + ch], C, irr[i:i + ch],
                              prep=prep) for i in range(0, len(eff), ch)]
                logits = torch.cat([o[0] for o in outs])
                same, opp = torch.cat([o[1][0] for o in outs]), torch.cat([o[1][1] for o in outs])
        return torch.sigmoid(logits), same, opp, irr


# --------------------------------------------------------------------------- federation
def make_clients(world, cfg, labels=None):
    R, Y, Q, gap = labels or world.labels()
    return [Client(world, d, R, Q, cfg) for d in range(world.K)], (R, Y, Q, gap)


def fedavg(states, weights):
    w = np.asarray(weights, float) / sum(weights)
    return {k: sum(wi * s[k] for wi, s in zip(w, states)) for k in states[0]}


def train(clients, cfg: Config, model=None, rounds=None, log=None):
    torch.manual_seed(cfg.seed)
    geo = clients[0].geo
    glob = model or DyFHL(geo, cfg)
    locals_ = None if cfg.fedavg else [copy.deepcopy(glob) for _ in clients]
    opt_state = {}
    for r in range(rounds or cfg.rounds):
        states, losses = [], []
        for ci, c in enumerate(clients):
            m = copy.deepcopy(glob) if cfg.fedavg else locals_[ci]
            opt = torch.optim.Adam(m.parameters(), lr=cfg.lr)
            if ci in opt_state:      # keep each domain's optimiser moments across rounds
                opt.load_state_dict(opt_state[ci])
            for _ in range(cfg.local_steps):
                opt.zero_grad()
                loss = c.pseudo_loss(m)
                loss.backward()
                opt.step()
            losses.append(loss.item())
            states.append(m.state_dict())
            opt_state[ci] = opt.state_dict()
        if cfg.fedavg:
            glob.load_state_dict(fedavg(states, [c.n for c in clients]))
        if log is not None:
            log(r, float(np.mean(losses)), glob if cfg.fedavg else locals_)
    return glob if cfg.fedavg else locals_


def exchange(clients, models, cfg: Config, seed=0):
    """Each domain uploads its aggregates; each receives the sum of the others plus noise of std sigma."""
    g = torch.Generator().manual_seed(seed)
    ms = models if isinstance(models, list) else [models] * len(clients)
    ups = [c.upload(m) for c, m in zip(clients, ms)]
    totU, totA = sum(u for u, _ in ups), sum(a for _, a in ups)
    ctx = []
    for c, (u, a) in zip(clients, ups):
        if cfg.context:
            C = totU - u + cfg.sigma * torch.randn(u.shape, generator=g)
            al = totA - a + cfg.sigma * torch.randn(a.shape, generator=g) * c.id_mask
        else:
            C, al = torch.zeros_like(u), torch.zeros_like(a)
        ctx.append((C, al))
    return ctx


def metrics(y, p, thr=0.5):
    y = [np.asarray(v).astype(int) for v in y]
    out = {}
    for name, yy, pp in y_iter(y, p):
        hat = (pp >= thr).astype(int)
        out[name] = dict(f1=f1_score(yy, hat, zero_division=0), prec=precision_score(yy, hat, zero_division=0),
                         rec=recall_score(yy, hat, zero_division=0),
                         ap=average_precision_score(yy, pp) if yy.any() else float("nan"), n=int(len(yy)),
                         pos=float(yy.mean()))
    out["macro_f1"] = float(np.mean([out[t]["f1"] for t in TASKS]))
    out["macro_ap"] = float(np.nanmean([out[t]["ap"] for t in TASKS]))
    return out


def y_iter(y, p):
    for k, t in enumerate(TASKS):
        yield t, y[k], p[k]


def evaluate(clients, models, labels, cfg: Config, seed=0, rule=False, ctx=None):
    """Cross-domain evaluation against the exact global oracle. rule=True uses the
    non-learned threshold decoder on the signature sketch instead of the network."""
    R, Y, Q, gap = labels
    ms = models if isinstance(models, list) else [models] * len(clients)
    ctx = ctx if ctx is not None else exchange(clients, ms, cfg, seed)
    P = np.zeros((len(R["eff"]), 3)); Pq = np.zeros(len(gap))
    for c, m, (C, al) in zip(clients, ms, ctx):
        prob, same, opp, irr = c.predict(m, C, al)
        al = al
        n = c.n
        if rule:
            B = cfg.B if cfg.sketch else 1
            prob = torch.zeros_like(prob)
            prob[:, 0] = (opp[:, 0, :B].amax(1) > 0.5).float()
            prob[:, 1] = (same[:, 1, :B].amax(1) > 0.5).float()
            Hall = torch.cat([c.H, c.QH])
            prob[:, 2] = c.irr_exact(Hall, c.own, c.alive * c.own + al).float()
            prob[:, 3] = (same[:, 0, :B].amax(1) < 0.5).float()
        P[c.ridx] = prob[:n, :3].numpy()
        Pq[c.qidx] = prob[n:, 3].numpy()
    y = [Y[:, 0], Y[:, 1], Y[:, 2], gap]
    p = [P[:, 0], P[:, 1], P[:, 2], Pq]
    return metrics(y, p), (P, Pq)


def comm_cost(clients, model, cfg: Config):
    """Bytes per domain: one context upload and one model upload (float32)."""
    u, _ = clients[0].upload(model if not isinstance(model, list) else model[0])
    params = sum(p.numel() for p in (model if not isinstance(model, list) else model[0]).parameters())
    return dict(context_bytes=int(u.numel() * 4), context_nnz_bytes=int((u != 0).sum() * 4), model_bytes=int(params * 4))


def run(world: World, cfg: Config, labels=None, rule=False, return_all=False):
    t0 = time.time()
    clients, labels = make_clients(world, cfg, labels)
    model = DyFHL(clients[0].geo, cfg) if rule else train(clients, cfg)
    t1 = time.time()
    res, preds = evaluate(clients, model, labels, cfg, seed=cfg.seed, rule=rule)
    res["train_s"], res["infer_s"] = t1 - t0, time.time() - t1
    res.update(comm_cost(clients, model, cfg))
    return (res, clients, model, labels, preds) if return_all else res
