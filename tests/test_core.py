"""Unit tests: oracle semantics, exactness of delta maintenance, sketch decoder soundness."""
import numpy as np, torch, pytest
from dyfhl.policy import World, AgenticSource, pair_matrices, cover_matrix
from dyfhl.model import Config, DyFHL
from dyfhl.fed import make_clients, evaluate, exchange
from dyfhl.dynamic import delta_upload


def brute(world, R):
    """Brute-force oracle by enumerating matching value combinations pairwise."""
    n = len(R["eff"])
    ov = np.ones((n, n), bool); sub = np.ones((n, n), bool)
    for i in range(n):
        for j in range(n):
            for a, (s, e) in enumerate(world.slices):
                wi, wj = R["Wm"][i, a], R["Wm"][j, a]
                vi, vj = set(np.flatnonzero(R["H"][i, s:e])), set(np.flatnonzero(R["H"][j, s:e]))
                if not (wi or wj or vi & vj):
                    ov[i, j] = False
                if not (wj or (not wi and vi <= vj)):
                    sub[i, j] = False
    return ov, sub


def small(seed=0, **kw):
    return World(AgenticSource(n_tools=12), K=3, S=2, n_rules=40, n_queries=40, seed=seed, **kw)


def test_pair_matrices_match_bruteforce():
    w = small()
    R = w.encode_rules()
    ov, sub = pair_matrices(R["H"], R["Wm"], w.slices)
    bov, bsub = brute(w, R)
    assert (ov == bov).all() and (sub == bsub).all()


def test_cover_matrix():
    w = small()
    R, Q = w.encode_rules(), w.encode_queries()
    cov = cover_matrix(Q["H"], R["H"], R["Wm"])
    for q in range(10):
        for j in range(len(R["eff"])):
            ok = all(R["Wm"][j, a] or (R["H"][j, s:e] & Q["H"][q, s:e]).any() for a, (s, e) in enumerate(w.slices))
            assert ok == cov[q, j]


def test_sketch_decoder_has_no_false_negatives():
    """With noise-free context every true cross-domain conflict / redundancy is flagged and every
    flagged gap is a true gap: hash collisions can only add conflicts or hide gaps."""
    w = small(1)
    cfg = Config(h=0, B=64)
    clients, lab = make_clients(w, cfg)
    R, Y, Q, gap = lab
    _, (P, Pq) = evaluate(clients, DyFHL(clients[0].geo, cfg), lab, cfg, rule=True)
    assert (P[Y[:, 0], 0] == 1).all()
    assert (P[Y[:, 1], 1] == 1).all()
    assert (P[:, 2] == Y[:, 2]).all()
    assert gap[Pq == 1].all()


def test_large_sketch_is_exact():
    w = small(2)
    cfg = Config(h=0, B=1 << 16)
    clients, lab = make_clients(w, cfg)
    R, Y, Q, gap = lab
    _, (P, Pq) = evaluate(clients, DyFHL(clients[0].geo, cfg), lab, cfg, rule=True)
    assert (P[:, :3] == Y).mean() > 0.995 and ((Pq == 1) == gap).mean() > 0.995


def test_delta_equals_full_recompute():
    w = small(3)
    cfg = Config(h=0, B=128)
    old, _ = make_clients(w, cfg)
    m = DyFHL(old[0].geo, cfg)
    U = [c.upload(m)[0] for c in old]
    w.step(0.2)
    new, _ = make_clients(w, cfg)
    for o, n, u in zip(old, new, U):
        d, _ = delta_upload(o, n, m)
        assert torch.equal(u + d, n.upload(m)[0])


def test_irrelevancy_rule_matches_definition():
    """Proposition (irrelevancy): without noise the rule decoder reproduces the oracle exactly,
    also on XS-ABAC, where rules can constrain several entity attributes."""
    from dyfhl.policy import get_source
    for name in ("agentic", "xu_stoller"):
        w = World(get_source(name), seed=3, n_rules=150, n_queries=40)
        cfg = Config(h=0, B=64)
        clients, lab = make_clients(w, cfg)
        R, Y, Q, gap = lab
        _, (P, Pq) = evaluate(clients, DyFHL(clients[0].geo, cfg), lab, cfg, rule=True)
        assert ((P[:, 2] > .5) == Y[:, 2]).all(), name
