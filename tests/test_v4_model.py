"""Randomized property tests for the v4 formal model: every statement of
DyFHL-PA-v4-proofs-and-algorithms is checked against brute force.
Run: python3 tests/test_v4_model.py
"""
import random
from itertools import product, combinations

DOM = {"role": ["analyst", "business", "auditor"], "region": ["eu", "us"],
       "tool": ["t0", "t1", "t2"], "res": ["sales", "hr", "fin"]}
TOOL_ATTRS, DATA_ATTRS = ("role", "region", "tool"), ("role", "region", "res")
SVC = {"role": "svc", "region": "eu"}


def norm(cond):  # a full-domain set is the wildcard
    return {k: v for k, v in cond.items() if k not in DOM or v != set(DOM[k])}


def rand_cond(rng, attrs, extra_roles=()):
    c = {}
    for a in attrs:
        if rng.random() < 0.6:
            dom = DOM[a] + list(extra_roles if a == "role" else ())
            c[a] = set(rng.sample(dom, rng.randint(1, len(dom) - 1)))
    return norm(c)


def matches(c, q):
    return all(q[k] in v for k, v in c.items())


def overlap(c1, c2):
    return all(c1[k] & c2[k] for k in c1.keys() & c2.keys())


def subsumed(c1, c2):
    return all(k in c1 and c1[k] <= c2[k] for k in c2)


def requests(attrs):
    return [dict(zip(attrs, v)) for v in product(*(DOM[a] for a in attrs))]


def decide(rules, q):
    hit = [e for e, c in rules if matches(c, q)]
    return "deny" if "deny" in hit or not hit else "permit"


def cross_layer(tool_rules, data_rules, tools, population):
    """Types 4 (gap) and 5 (amplification) for every (user class, tool)."""
    gap, amp = set(), set()
    for u in population:
        for t, d in tools.items():
            if decide(tool_rules, {**u, "tool": t}) != "permit":
                continue
            user_ok = decide(data_rules, {**u, "res": d["reads"]}) == "permit"
            ident = u if d["binding"] == "obo" else SVC
            eff_ok = decide(data_rules, {**ident, "res": d["reads"]}) == "permit"
            key = (u["role"], u["region"], t)
            if not eff_ok:
                gap.add(key)
            elif not user_ok:
                amp.add(key)
    return gap, amp


def pair_anomalies(rules):
    conf, red = set(), set()
    for i, j in combinations(range(len(rules)), 2):
        (e1, c1), (e2, c2) = rules[i], rules[j]
        if e1 != e2 and overlap(c1, c2):
            conf.add((i, j))
        if e1 == e2:
            if subsumed(c1, c2):
                red.add((i, j))
            elif subsumed(c2, c1):
                red.add((j, i))
    return conf, red


def indexed_conflicts(rules, pivot="role"):
    """Algorithm 1: bucket rules by a pivot attribute; compare inside buckets only."""
    buckets = {}
    for i, (_, c) in enumerate(rules):
        for v in c.get(pivot, DOM[pivot] + ["svc"]):
            buckets.setdefault(v, []).append(i)
    conf, seen = set(), set()
    for ids in buckets.values():
        for i, j in combinations(ids, 2):
            if (i, j) in seen:
                continue
            seen.add((i, j))
            if rules[i][0] != rules[j][0] and overlap(rules[i][1], rules[j][1]):
                conf.add((i, j))
    return conf, len(seen)


def run(seed):
    rng = random.Random(seed)
    U = requests(("role", "region"))
    tool_rules = [(rng.choice(["permit", "permit", "deny"]), rand_cond(rng, TOOL_ATTRS)) for _ in range(8)]
    data_rules = [(rng.choice(["permit", "permit", "deny"]), rand_cond(rng, DATA_ATTRS, ["svc"])) for _ in range(8)]
    tools = {t: dict(reads=rng.choice(DOM["res"]), binding="obo") for t in DOM["tool"]}

    # Lemma 1 / 2: incidence conditions equal brute force over all requests
    Q = requests(TOOL_ATTRS)
    for (_, c1), (_, c2) in combinations(tool_rules, 2):
        assert overlap(c1, c2) == any(matches(c1, q) and matches(c2, q) for q in Q)
        assert subsumed(c1, c2) == all(matches(c2, q) for q in Q if matches(c1, q))

    # Algorithm 1 equals all-pairs
    conf, _ = pair_anomalies(tool_rules)
    iconf, _ = indexed_conflicts(tool_rules)
    assert conf == iconf

    # Theorem 1 (duality), restricted to tools whose service account is permitted on reads(t)
    ok = {t for t, d in tools.items() if decide(data_rules, {**SVC, "res": d["reads"]}) == "permit"}
    g_obo, a_obo = cross_layer(tool_rules, data_rules, tools, U)
    svc_tools = {t: dict(d, binding="svc" if t in ok else "obo") for t, d in tools.items()}
    g_svc, a_svc = cross_layer(tool_rules, data_rules, svc_tools, U)
    assert not a_obo
    assert {k for k in g_obo if k[2] in ok} == {k for k in a_svc if k[2] in ok}
    assert not {k for k in g_svc if k[2] in ok}

    # Theorem 2 (drift locality): changing one descriptor leaves other tools' findings unchanged
    t = rng.choice(DOM["tool"])
    drifted = {x: dict(d) for x, d in svc_tools.items()}
    drifted[t]["reads"] = rng.choice(DOM["res"])
    g2, a2 = cross_layer(tool_rules, data_rules, drifted, U)
    assert {k for k in g2 if k[2] != t} == {k for k in g_svc if k[2] != t}
    assert {k for k in a2 if k[2] != t} == {k for k in a_svc if k[2] != t}

    # Theorem 4 (potential covers confirmed; equal when the directory populates every class)
    directory = rng.sample(U, rng.randint(1, len(U)))
    gc, ac = cross_layer(tool_rules, data_rules, svc_tools, directory)
    assert gc <= g_svc and ac <= a_svc
    assert cross_layer(tool_rules, data_rules, svc_tools, U) == (g_svc, a_svc)

    # Theorem 6 (incremental maintenance equals recomputation)
    rules = list(tool_rules)
    conf, red = pair_anomalies(rules)
    new = (rng.choice(["permit", "deny"]), rand_cond(rng, TOOL_ATTRS))
    n = len(rules)
    for i, (e, c) in enumerate(rules):           # only pairs involving the new rule
        if e != new[0] and overlap(c, new[1]):
            conf.add((i, n))
        if e == new[0]:
            if subsumed(c, new[1]):
                red.add((i, n))
            elif subsumed(new[1], c):
                red.add((n, i))
    rules.append(new)
    assert (conf, red) == pair_anomalies(rules)
    k = rng.randrange(len(rules))                # removal: drop pairs involving k (ids kept stable)
    conf_k = {p for p in conf if k not in p}
    red_k = {p for p in red if k not in p}
    full_c, full_r = pair_anomalies(rules)
    assert conf_k == {p for p in full_c if k not in p} and red_k == {p for p in full_r if k not in p}


def dnf_reduction(seed, n=4):
    """Theorem 5: a DNF is a tautology iff its terms, read as permit boxes, leave no gap."""
    rng = random.Random(seed)
    terms = []
    for _ in range(rng.randint(1, 6)):
        vs = rng.sample(range(n), rng.randint(1, 3))
        terms.append({v: rng.random() < 0.5 for v in vs})
    points = list(product([False, True], repeat=n))
    taut = all(any(all(p[v] == b for v, b in t.items()) for t in terms) for p in points)
    rules = [("permit", {f"x{v}": {b} for v, b in t.items()}) for t in terms]
    no_gap = all(decide(rules, {f"x{i}": p[i] for i in range(n)}) == "permit" for p in points)
    assert taut == no_gap
    return taut


if __name__ == "__main__":
    for s in range(2000):
        run(s)
    tauts = sum(dnf_reduction(s) for s in range(2000))
    print(f"2000 random policy instances and 2000 DNF instances passed ({tauts} tautologies)")
