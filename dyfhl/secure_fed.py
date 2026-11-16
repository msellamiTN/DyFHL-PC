"""Flags-only cross-owner checks for the federation (two non-colluding servers, dealer, semi-honest).

secure_pairwise: same-layer conflict/redundancy between a changed rule and the foreign rules of its region
                 (exact inner-product tests of dyfhl.mpc.exact_checks, both directions).
secure_cross:    gap/amplification for one (tool, resource) cell when the tool host and the resource
                 owner(s) differ: each owner evaluates its OWN rules locally into bit vectors, and the servers
                 combine them with a few secure products ("decide locally, combine securely")."""
import numpy as np
from dyfhl.mpc import MPC, P, exact_checks


def secure_pairwise(fed, mpc, rid):
    r = fed.rules[rid]
    L, key = r.layer, {"data": "res", "tool": "tool", "usage": "res"}[r.layer]
    cand = sorted({k for v in r.c[key] for k in fed.idx[L].get(v, ()) if fed.rules[k].owner != r.owner})
    if not cand:
        return None
    X, eff, slices = fed.incidence([rid] + cand, L)
    c1, r1, _ = exact_checks(mpc, X[:1], eff[:1], X[1:], eff[1:], slices)
    c2, r2, _ = exact_checks(mpc, X[1:], eff[1:], X[:1], eff[:1], slices)
    plain_c1 = any(rid in p for p in fed.conf)
    plain_r1 = any(p[0] == rid for p in fed.red)
    plain_c2 = np.array([(min(k, rid), max(k, rid)) in fed.conf for k in cand])
    plain_r2 = np.array([(k, rid) in fed.red for k in cand])
    return bool(c1[0] == plain_c1 and r1[0] == plain_r1 and (c2 == plain_c2).all() and (r2 == plain_r2).all())


def _local_bits(fed, k, r, idxs):
    p = np.zeros(len(fed.so), bool); d = np.zeros(len(fed.so), bool)
    for j in fed.idx["data"].get(r, ()):
        x = fed.rules[j]
        if x.owner == k:
            (p if x.effect == "permit" else d)[:] |= fed.subj_mask(x)
    return p[idxs].astype(np.int64), d[idxs].astype(np.int64)


def secure_cross(fed, mpc, t, r):
    """Returns True if the revealed flags equal the plaintext findings of cell (t, r)."""
    H, h, desc = fed.H, int(fed.host[t]), fed.d1[t]
    idxs = np.r_[np.arange(H), H + h]                     # humans, then the host's service identity
    owners = fed.owners(r)
    U = mpc.share(fed.U[t][:H].astype(np.int64))           # input of the tool host
    if len(owners) == 1:                                   # the single owner decides locally
        D = mpc.share(fed.D[r][idxs].astype(np.int64))
    else:
        notP = notN = None
        for k in owners:
            p, d = _local_bits(fed, k, r, idxs)
            sp, sd = mpc.share(1 - p), mpc.share(1 - d)
            notP = sp if notP is None else mpc.mul(notP, sp)
            notN = sd if notN is None else mpc.mul(notN, sd)
        D = mpc.mul(mpc.cadd(mpc.neg(notP), 1), notN)    # (1 - prod(1-p_k)) * prod(1-d_k)
    Dh = (D[0][:H], D[1][:H])
    Ds = (np.repeat(D[0][H:], H), np.repeat(D[1][H:], H))
    one_minus = lambda x: mpc.cadd(mpc.neg(x), 1)
    if desc.binding == "obo":
        gap = mpc.reveal(mpc.mul(U, one_minus(Dh))) == 1
        amp = np.zeros(H, bool)
    else:
        gap = mpc.reveal(mpc.mul(U, one_minus(Ds))) == 1
        amp = mpc.reveal(mpc.mul(mpc.mul(U, Ds), one_minus(Dh))) == 1
    g, a = fed.cross[(t, r)]
    return set(np.flatnonzero(gap).tolist()) == g and set(np.flatnonzero(amp).tolist()) == a
