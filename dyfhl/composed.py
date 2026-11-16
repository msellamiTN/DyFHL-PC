"""Exact analysis of composed policies (v4 model).

Layers: tool (who may call which tool), data (which identity may access which resource),
usage (purpose, sink, region, retention, obligations). Rules are boxes over role-qualified
attributes; an absent attribute is the wildcard. See claude/DyFHL-PA-v4-proofs-and-algorithms.
"""
from dataclasses import dataclass, field
from itertools import combinations

ALGOS = ("deny-overrides", "permit-overrides", "first-applicable")


@dataclass(frozen=True)
class Rule:
    layer: str
    owner: str
    effect: str                      # "permit" | "deny"
    cond: tuple                      # sorted tuple of (attribute, frozenset(values))
    obl: tuple = ()                  # sorted tuple of (name, value)

    @property
    def c(self):
        return dict(self.cond)


def rule(layer, owner, effect, cond, obl=None, domains=None):
    """Build a rule in normal form: a full-domain value set is the wildcard."""
    c = {k: frozenset(v) for k, v in cond.items()}
    if domains:
        c = {k: v for k, v in c.items() if k not in domains or v != frozenset(domains[k])}
    return Rule(layer, owner, effect, tuple(sorted(c.items())), tuple(sorted((obl or {}).items())))


def matches(r, q):
    return all(q.get(k) in v for k, v in r.cond)


def overlap(r1, r2):                 # Lemma 1
    c1, c2 = r1.c, r2.c
    return all(c1[k] & c2[k] for k in c1.keys() & c2.keys())


def subsumed(r1, r2):                # Lemma 2: M(r1) inside M(r2)
    c1 = r1.c
    return all(k in c1 and c1[k] <= v for k, v in r2.cond)


def obligations_conflict(r1, r2):
    o1, o2 = dict(r1.obl), dict(r2.obl)
    for a, b in ((o1, o2), (o2, o1)):
        lo, hi = a.get("retain_min_days"), b.get("retain_max_days")
        if lo is not None and hi is not None and lo > hi:
            return True
    return False


def decide(rules, q, algo="deny-overrides"):
    hit = [r.effect for r in rules if matches(r, q)]
    if not hit:
        return "deny"
    if algo == "deny-overrides":
        return "deny" if "deny" in hit else "permit"
    if algo == "permit-overrides":
        return "permit" if "permit" in hit else "deny"
    if algo == "first-applicable":
        return hit[0]
    raise ValueError(algo)


def pair_anomalies(rules, pivot=None, pivot_domain=()):
    """Types 1 and 2 inside one layer. With a pivot attribute, only rules sharing a pivot value
    are compared (Algorithm 1). Returns (conflicts, redundancies, comparisons)."""
    n = len(rules)
    if pivot is None:
        pairs = combinations(range(n), 2)
    else:
        buckets = {}
        for i, r in enumerate(rules):
            for v in r.c.get(pivot, pivot_domain):
                buckets.setdefault(v, []).append(i)
        pairs = sorted({p for ids in buckets.values() for p in combinations(ids, 2)})
    conf, red, k = set(), set(), 0
    for i, j in pairs:
        k += 1
        a, b = rules[i], rules[j]
        if a.layer != b.layer:
            continue
        if overlap(a, b) and (a.effect != b.effect or obligations_conflict(a, b)):
            conf.add((i, j))
        if a.effect == b.effect and a.obl == b.obl:
            if subsumed(a, b):
                red.add((i, j))
            elif subsumed(b, a):
                red.add((j, i))
    return conf, red, k


@dataclass(frozen=True)
class Tool:
    name: str
    reads: frozenset                 # resources the tool reads
    binding: str = "obo"             # "obo" | "svc"
    purpose: str = ""
    retention: int = 0
    sinks: frozenset = field(default_factory=frozenset)   # of (org, region)


def cross_layer(tool_rules, data_rules, tools, population, svc, algo="deny-overrides",
                key=lambda u: tuple(sorted(u.items()))):
    """Types 4 (gap) and 5 (one-step amplification) for a population of user attribute vectors.
    svc is the attribute vector of the service identity used under the 'svc' binding."""
    gap, amp = set(), set()
    for u in population:
        for t in tools:
            if decide(tool_rules, {**u, "tool": t.name}, algo) != "permit":
                continue
            ident = u if t.binding == "obo" else svc
            for r in t.reads:
                if decide(data_rules, {**ident, "res": r}, algo) != "permit":
                    gap.add((key(u), t.name, r))
                elif decide(data_rules, {**u, "res": r}, algo) != "permit":
                    amp.add((key(u), t.name, r))
    return gap, amp


def usage_violations(usage_rules, tools, retain_max=None, algo="deny-overrides"):
    """Type 6: purpose, sink, region and retention against the owner's usage layer."""
    out = set()
    for t in tools:
        for r in t.reads:
            for org, region in t.sinks:
                q = {"res": r, "purpose": t.purpose, "sink.org": org, "sink.region": region}
                if decide(usage_rules, q, algo) != "permit":
                    out.add((t.name, r, "sink", org, region))
            mx = (retain_max or {}).get(r)
            if mx is not None and t.retention > mx:
                out.add((t.name, r, "retention", t.retention, mx))
    return out


def drift_report(usage_rules, approved, declared, observed, retain_max=None):
    """Type 7 for one tool: definition drift, its impact, and behavioural drift."""
    v0 = usage_violations(usage_rules, [approved], retain_max)
    v1 = usage_violations(usage_rules, [declared], retain_max)
    vh = usage_violations(usage_rules, [observed], retain_max)
    return dict(definition_drift=approved != declared, impact=v1 - v0,
                behavioural_drift=not (observed.sinks <= declared.sinks and observed.reads <= declared.reads), behavioural_findings=vh - v1)


def intended_authority(perms, chain):
    eff = set(perms[chain[0]])
    for p in chain[1:]:
        eff &= perms[p]
    return eff


def chain_amplification(perms, chain, exercised=None):
    """Type 5 along a delegation chain; by default the last principal exercises its own rights."""
    x = set(perms[chain[-1]]) if exercised is None else set(exercised)
    return x - intended_authority(perms, chain)
