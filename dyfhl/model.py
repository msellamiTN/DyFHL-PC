"""DyFHL-PA model: hyperedge message encoder, cross-domain context and conjunctive decoder.

Every rule (hyperedge) emits a message made of a hashed signature sketch and a learned
embedding. Messages are summed at the vertices the rule is incident to (attribute values
and per-attribute wildcard vertices), separately per collaboration scope and effect. A
domain receives the sum of the other domains' aggregates and decodes, for each of its own
hyperedges, evidence that is disjunctive inside an attribute and conjunctive across attributes.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn

TASKS = ["x-conflict", "x-redundancy", "x-irrelevancy", "x-gap"]


@dataclass
class Config:
    B: int = 512             # sketch bins
    h: int = 16              # learned message width (0 disables the learned channel)
    sketch: bool = True      # False: a single count channel instead of the signature sketch
    scope_part: bool = True  # scope-partitioned context
    conj: bool = True        # attribute-wise conjunctive pooling (False: flat pooling over vertices)
    wild: bool = True        # wildcard vertices
    priors: bool = False     # let the decoder see the hyperedge's own features (effect, wildcard mask)
    gcn: bool = False        # local clique-expansion GCN on top of the decoder (baseline)
    context: bool = True     # False: no cross-domain exchange at all
    sigma: float = 0.0       # std of Gaussian noise added to every uploaded aggregate
    noise_aware: bool = True # train with the same noise that is used at inference
    fedavg: bool = True
    rounds: int = 20
    local_steps: int = 5
    lr: float = 1e-2
    batch: int = 256         # hyperedges per local step
    hidden: int = 32
    aug: float = 0.25        # fraction of synthetic pseudo-anomalies per local step
    recomb: float = 0.0      # synthetic unrelated rules per own rule (density matching)
    seed: int = 0

    @property
    def D(self):
        return (self.B if self.sketch else 1) + self.h


def hash_bins(uid, B, salt=0):
    """Salted pseudo-random hash of rule identifiers to bins (splitmix64 finaliser)."""
    with np.errstate(over="ignore"):
        x = (np.asarray(uid).astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15) * np.uint64(salt + 1))
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return (x % np.uint64(B)).astype(np.int64)


def signature(uid, cfg):
    n = len(uid)
    if not cfg.sketch:
        return torch.ones(n, 1)
    sig = torch.zeros(n, cfg.B)
    b = torch.as_tensor(hash_bins(np.asarray(uid), cfg.B, cfg.seed))
    sig[torch.arange(n), b] = 1.0
    return sig


class Geometry:
    """Padded attribute -> vertex index so that all attributes are processed in one tensor op."""

    def __init__(self, slices):
        self.slices = slices
        self.A = len(slices)
        self.V = slices[-1][1]
        vmax = max(e - s for s, e in slices)
        self.idx = torch.zeros(self.A, vmax, dtype=torch.long)
        self.mask = torch.zeros(self.A, vmax)
        for a, (s, e) in enumerate(slices):
            self.idx[a, :e - s] = torch.arange(s, e)
            self.mask[a, :e - s] = 1.0


class DyFHL(nn.Module):
    def __init__(self, geo: Geometry, cfg: Config, n_id_attrs=1):
        super().__init__()
        self.geo, self.cfg = geo, cfg
        fe = 2 + geo.A + 2
        self.enc = nn.Sequential(nn.Linear(fe + 16, 64), nn.ReLU(), nn.Linear(64, cfg.h)) if cfg.h else None
        self.vemb = nn.Embedding(geo.V + geo.A, 16) if cfg.h else None
        self.block = torch.zeros(geo.A, geo.V)
        for a, (s, e) in enumerate(geo.slices):
            self.block[a, s:e] = 1
        ns, hd = self.NSTAT, cfg.hidden
        # one narrow head per task; each sees only the evidence that is semantically relevant to it
        self.in_dims = [3 * ns, 3 * ns, 3, 3 * ns]
        extra = (fe if cfg.priors else 0) + (6 * cfg.h if cfg.h else 0)
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(d + extra, hd), nn.ReLU(), nn.Linear(hd, hd // 2), nn.ReLU(),
                                                  nn.Linear(hd // 2, 1)) for d in self.in_dims])
        if cfg.gcn:
            self.g1 = nn.Linear(extra + 6 * ns + 3, extra + 6 * ns + 3)

    # -- hyperedge features and messages
    def edge_feat(self, Wm, eff, isq):
        e = torch.zeros(len(eff), 2)
        r = ~isq
        e[r, eff[r]] = 1.0
        return torch.cat([e, Wm * r[:, None], (1 - Wm).sum(1, keepdim=True) / self.geo.A * r[:, None],
                          isq[:, None].float()], 1)

    def messages(self, H, Wm, eff, sig):
        """Message of a rule: its hashed signature and a learned embedding of its vertex set."""
        isq = torch.zeros(len(eff), dtype=torch.bool)
        if self.enc is None:
            return sig
        Hx = torch.cat([H, Wm], 1)
        z = (Hx @ self.vemb.weight) / Hx.sum(1, keepdim=True).clamp(min=1)
        return torch.cat([sig, torch.tanh(self.enc(torch.cat([self.edge_feat(Wm, eff, isq), z], 1)))], 1)

    def aggregate(self, H, Wm, eff, scope, M, S):
        """Vertex aggregates U[g, v, effect, :] of the messages of the given rules (g = scope)."""
        cfg = self.cfg
        G = S if cfg.scope_part else 1
        Hx = torch.cat([H, Wm if cfg.wild else torch.zeros_like(Wm)], 1)
        g = (scope if cfg.scope_part else torch.zeros_like(scope)) * 2 + eff
        B = cfg.B if cfg.sketch else 1
        e, v = torch.nonzero(Hx, as_tuple=True)                 # incidences
        Us = torch.zeros(G * 2, Hx.shape[1], B)
        if cfg.sketch:                                           # one-hot signatures: sparse scatter
            b = M[:, :B].argmax(1)
            Us.index_put_((g[e], v, b[e]), torch.ones(len(e)), accumulate=True)
        else:
            Us.index_put_((g[e], v, torch.zeros_like(e)), torch.ones(len(e)), accumulate=True)
        if cfg.h:
            onehot = torch.zeros(len(g), G * 2).scatter_(1, g[:, None], 1.0)
            Ul = torch.einsum("ng,nv,nd->gvd", onehot, Hx, M[:, B:])
            Us = torch.cat([Us, Ul], 2)
        return Us.view(G, 2, Hx.shape[1], -1).permute(0, 2, 1, 3)  # [G, V+A, 2, D]

    # -- decoding
    def extend(self, C):
        """Context rows extended with per-attribute totals and a zero row (computed once per context)."""
        V, A, G = self.geo.V, self.geo.A, C.shape[0]
        tot = torch.einsum("av,gvcd->gacd", self.block, C[:, :V])
        return torch.cat([C, tot, torch.zeros(G, 1, *C.shape[2:])], 1)

    def prepare(self, C):
        """Precompute the extended contexts used by forward(); pass the result as prep= when decoding in chunks."""
        B = self.cfg.B if self.cfg.sketch else 1
        prep = {"s": self.extend(C[..., :B])} if self.cfg.conj else {}
        if self.cfg.h and self.cfg.conj:
            prep["l"] = self.extend(C[..., B:] / (1 + C[..., :B].sum(-1, keepdim=True).clamp(min=0)))
        return prep

    def evidence(self, H, Wm, grp, C, Cx=None):
        """Per hyperedge pooled cross-domain evidence [n, 3, 2, D].
        Conjunctive mode: (min over attributes of the overlap evidence, min over attributes of the
        subsumption evidence, mean over attributes of the overlap evidence), where inside an attribute
        overlap sums and subsumption takes the minimum over the hyperedge's values, and wildcard
        vertices are added on both sides."""
        geo, cfg = self.geo, self.cfg
        V = geo.V
        if cfg.conj:
            n, A, G = H.shape[0], geo.A, C.shape[0]
            Ha = H[:, geo.idx] * geo.mask[None]                       # [n, A, Vmax]
            vals, pos = Ha.topk(2, dim=2)                             # value sets hold at most two values
            vid = geo.idx[torch.arange(A)[None, :, None], pos]        # [n, A, 2]
            has1, has2 = vals[..., 0] > 0, vals[..., 1] > 0
            if Cx is None:
                Cx = self.extend(C)                                   # rows: values | wildcards | totals | zero
            ar = torch.arange(A)[None, :].expand(n, A)
            i1 = torch.where(has1, vid[..., 0], V + A + ar)           # first value, or the attribute total
            i2 = torch.where(has2, vid[..., 1], torch.full_like(ar, V + 2 * A))
            gg = grp[:, None].expand(n, A)
            X1, X2, Wa = Cx[gg, i1], Cx[gg, i2], Cx[gg, V + ar]        # [n, A, 2, D]
            ov = X1 + X2 + Wa
            h1, h2 = has1[..., None, None], has2[..., None, None]
            sub = torch.where(h1, torch.where(h2, torch.minimum(X1, X2), X1) + Wa, Wa)
            return torch.stack([ov.amin(1), sub.amin(1), ov.mean(1)], 1)
        out = torch.zeros(H.shape[0], 3, 2, C.shape[-1])
        Hx = torch.cat([H, Wm if cfg.wild else torch.zeros_like(Wm)], 1)
        for g in range(C.shape[0]):
            idx = torch.nonzero(grp == g).squeeze(1)
            if len(idx) == 0:
                continue
            Hg, Cs = Hx[idx], C[g]
            deg = Hg.sum(1).clamp(min=1)[:, None, None]
            mean = torch.einsum("mv,vcd->mcd", Hg, Cs) / deg
            mx = (Cs[None] - (1 - Hg)[..., None, None] * 1e4).amax(1)
            mn = (Cs[None] + (1 - Hg)[..., None, None] * 1e4).amin(1)
            out[idx] = torch.stack([mn, mx, mean], 1)
        return out

    NSTAT = 8

    def bin_stats(self, x):
        """Permutation-invariant summary of the sketch bins: [.., B] -> [.., 8]:
        three largest bins, mean, std and the fraction of bins above 0.3 / 0.5 / 0.8."""
        if x.shape[-1] == 1:
            return torch.cat([torch.asinh(x)] * self.NSTAT, -1)
        top = x.topk(3, dim=-1).values
        soft = [torch.sigmoid((x - t) * 12).mean(-1, keepdim=True) * 20 for t in (0.3, 0.5, 0.8)]
        return torch.cat([torch.asinh(top), torch.asinh(x.mean(-1, keepdim=True)),
                          torch.asinh(x.std(-1, keepdim=True))] + soft, -1)

    def forward(self, H, Wm, eff, isq, grp, C, irr_feat, adj=None, prep=None):
        cfg = self.cfg
        B = cfg.B if cfg.sketch else 1
        prep = prep or {}
        with torch.no_grad():
            evs = self.evidence(H, Wm, grp, C[..., :B], prep.get("s"))
        if cfg.h:   # learned channel: per-vertex mean embedding of the foreign rules, bounded after pooling
            Cl = C[..., B:] / (1 + C[..., :B].sum(-1, keepdim=True).clamp(min=0))
            evs = torch.cat([evs, torch.tanh(self.evidence(H, Wm, grp, Cl, prep.get("l")) / 4)], -1)
        ev = evs
        n = len(eff)
        e = torch.where(isq, torch.zeros_like(eff), eff)
        same = ev[torch.arange(n), :, e]
        opp = ev[torch.arange(n), :, 1 - e]
        both = ev.sum(2)
        q = isq[:, None, None].float()
        same, opp = q * both + (1 - q) * same, q * both + (1 - q) * opp
        st_same, st_opp = self.bin_stats(same[..., :B]).flatten(1), self.bin_stats(opp[..., :B]).flatten(1)
        extra = []
        if cfg.priors:
            extra.append(self.edge_feat(Wm, eff, isq))
        if cfg.h:
            extra += [same[..., B:].flatten(1), opp[..., B:].flatten(1)]
        if cfg.gcn and adj is not None:   # baseline: propagate evidence over the local clique expansion
            x = torch.cat(extra + [st_same, st_opp, irr_feat], 1)
            x = x + torch.relu(adj @ self.g1(x))
            k = x.shape[1] - st_same.shape[1] - st_opp.shape[1] - 3
            extra = [x[:, :k]] if k else []
            st_same, st_opp, irr_feat = x[:, k:k + st_same.shape[1]], x[:, k + st_same.shape[1]:-3], x[:, -3:]
        ins = [st_opp, st_same, irr_feat, st_same]
        logits = torch.cat([hd(torch.cat(extra + [x], 1)) for hd, x in zip(self.heads, ins)], 1)
        return logits, (same, opp)


# --------------------------------------------------------------------------- torch oracle (local use)
def t_pairs(H, Wm, geo):
    Ha = H[:, geo.idx] * geo.mask[None]                 # [n, A, Vmax]
    inter = torch.einsum("iav,jav->aij", Ha, Ha)
    size = Ha.sum(2).T                                   # [A, n]
    w = Wm.T.bool()                                      # [A, n]
    ov = ((inter > 0) | w[:, :, None] | w[:, None, :]).all(0)
    sub = (w[:, None, :] | (~w[:, :, None] & (inter >= size[:, :, None] - 1e-6))).all(0)
    return ov, sub


def t_cover(Q, H, Wm):
    return (Q @ H.T + Wm.sum(1)[None]) >= Wm.shape[1] - 1e-6
