"""Randomized tests of dyfhl.composed against brute force. Run: python3 tests/test_composed.py"""
import os, random, sys
from itertools import product, combinations
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dyfhl.composed import (ALGOS, Tool, rule, matches, overlap, subsumed, decide, pair_anomalies,
                            cross_layer, usage_violations, drift_report, intended_authority,
                            chain_amplification)

DOM = {"role": ["analyst", "business", "auditor"], "region": ["eu", "us"],
       "tool": ["t0", "t1", "t2"], "res": ["sales", "hr", "fin"],
       "purpose": ["reporting", "training"], "sink.org": ["A", "B", "X"], "sink.region": ["eu", "us"]}
SVC = {"role": "svc", "region": "eu"}


def rand_rule(rng, layer, attrs, extra=()):
    c = {}
    for a in attrs:
        if rng.random() < 0.6:
            dom = DOM[a] + list(extra if a == "role" else ())
            c[a] = rng.sample(dom, rng.randint(1, len(dom) - 1))
    return rule(layer, "o", rng.choice(["permit", "permit", "deny"]), c, domains=DOM)


def reqs(attrs):
    return [dict(zip(attrs, v)) for v in product(*(DOM[a] for a in attrs))]


def run(seed):
    rng = random.Random(seed)
    U = reqs(("role", "region"))
    TR = [rand_rule(rng, "tool", ("role", "region", "tool")) for _ in range(8)]
    DR = [rand_rule(rng, "data", ("role", "region", "res"), ["svc"]) for _ in range(8)]
    UR = [rand_rule(rng, "usage", ("res", "purpose", "sink.org", "sink.region")) for _ in range(6)]
    sinks = lambda: frozenset(rng.sample(list(product(DOM["sink.org"], DOM["sink.region"])), rng.randint(1, 3)))
    tools = [Tool(t, frozenset(rng.sample(DOM["res"], rng.randint(1, 2))), "obo", rng.choice(DOM["purpose"]), rng.randint(1, 60), sinks())
             for t in DOM["tool"]]

    # Lemmas 1, 2 and Algorithm 1
    Q = reqs(("role", "region", "tool"))
    for a, b in combinations(TR, 2):
        assert overlap(a, b) == any(matches(a, q) and matches(b, q) for q in Q)
        assert subsumed(a, b) == all(matches(b, q) for q in Q if matches(a, q))
    c0, r0, k0 = pair_anomalies(TR)
    c1, r1, k1 = pair_anomalies(TR, pivot="role", pivot_domain=DOM["role"] + ["svc"])
    assert c0 == c1 and r0 == r1 and k1 <= k0

    # Theorem 1 (duality) under every combining algorithm
    for algo in ALGOS:
        okr = {(t.name, r) for t in tools for r in t.reads if decide(DR, {**SVC, "res": r}, algo) == "permit"}
        g_obo, a_obo = cross_layer(TR, DR, tools, U, SVC, algo)
        svc_tools = [Tool(t.name, t.reads, "svc", t.purpose, t.retention, t.sinks) for t in tools]
        g_svc, a_svc = cross_layer(TR, DR, svc_tools, U, SVC, algo)
        assert not a_obo                                                        # (1)
        assert not {k for k in g_svc if (k[1], k[2]) in okr}                    # (2)
        assert {k for k in a_svc if (k[1], k[2]) in okr} == {k for k in g_obo if (k[1], k[2]) in okr}   # (3)
        assert all((k[1], k[2]) in okr for k in a_svc)
        n_obo = len({k for k in g_obo | a_obo if (k[1], k[2]) in okr})
        n_svc = len({k for k in g_svc | a_svc if (k[1], k[2]) in okr})
        assert n_obo == n_svc                                                   # Corollary 1

    # Theorem 2 (drift locality) and Theorem 4 (potential covers confirmed)
    i = rng.randrange(len(svc_tools))
    drifted = list(svc_tools)
    t = drifted[i]
    drifted[i] = Tool(t.name, frozenset(rng.sample(DOM["res"], rng.randint(1, 2))), t.binding, t.purpose, t.retention, t.sinks)
    g2, a2 = cross_layer(TR, DR, drifted, U, SVC)
    g1, a1 = cross_layer(TR, DR, svc_tools, U, SVC)
    assert {k for k in g2 if k[1] != t.name} == {k for k in g1 if k[1] != t.name}
    assert {k for k in a2 if k[1] != t.name} == {k for k in a1 if k[1] != t.name}
    gd, ad = cross_layer(TR, DR, svc_tools, rng.sample(U, rng.randint(1, len(U))), SVC)
    assert gd <= g1 and ad <= a1

    # Theorem 3 (attenuation along chains)
    perms = {p: set(rng.sample(DOM["res"] + ["ops", "log"], rng.randint(1, 5))) for p in "uabc"}
    chain = ["u"] + rng.sample(["a", "b", "c"], rng.randint(1, 3))
    for k in range(2, len(chain) + 1):
        assert intended_authority(perms, chain[:k]) <= intended_authority(perms, chain[:k - 1])
    assert intended_authority(perms, chain) <= perms["u"]
    amp = chain_amplification(perms, chain)
    assert amp == perms[chain[-1]] - set.intersection(*(perms[p] for p in chain))
    assert chain_amplification(perms, ["u", "a"]) == perms["a"] - perms["u"]       # one-step case
    assert not chain_amplification(perms, chain, exercised=intended_authority(perms, chain))

    # Theorem 7 (observation is monotone) and drift report
    retain_max = {r: rng.randint(10, 50) for r in DOM["res"]}
    t = tools[0]
    extra = frozenset(rng.sample(list(product(DOM["sink.org"], DOM["sink.region"])), 2))
    observed = Tool(t.name, t.reads, t.binding, t.purpose, t.retention, t.sinks | extra)
    v1 = usage_violations(UR, [t], retain_max)
    vh = usage_violations(UR, [observed], retain_max)
    assert v1 <= vh
    rep = drift_report(UR, t, t, observed, retain_max)
    assert not rep["definition_drift"] and not rep["impact"]
    assert rep["behavioural_drift"] == (not observed.sinks <= t.sinks)
    assert rep["behavioural_findings"] == vh - v1
    brute = {(t.name, r, "sink", o, g) for r in observed.reads for o, g in observed.sinks
             if decide(UR, {"res": r, "purpose": t.purpose, "sink.org": o, "sink.region": g}) != "permit"}
    assert {x for x in vh if x[2] == "sink"} == brute


if __name__ == "__main__":
    for s in range(3000):
        run(s)
    print("3000 random instances passed")
