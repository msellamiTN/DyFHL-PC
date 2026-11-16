"""Federated dynamic policy hypergraph: several organisations, three layers (tool, data, usage), tool
descriptions with approved and current versions, and change events with exact incremental admission.

Vertices are attribute values; a rule is a hyperedge given by one value set per constrained attribute
(None = wildcard). Subjects are classes (org, role, dept) plus one service identity per organisation
(role = ROLES, dept = DEPTS). Resources are owned by one organisation, or co-owned by all (shared scope).
Decisions are deny-overrides with default deny inside a layer; along a chain the tool owner and the data
owner must both permit (composition by conjunction).

Findings (as in the paper):
  conf  (j, k)            rules of different owners, same layer, overlapping, opposite effects
  red   (j, k)            rule j of one owner subsumed by rule k of another owner, same effect
  gap   (u, t, r)         u may call t, t reads r, the identity used for r is denied
  amp   (u, t, r)         u may call t, t reads r with its service identity, which is permitted, u is not
  use   (t, r, kind, x)   declared purpose/sink/retention of t not permitted by r's owners
  drift (t, r, kind, x)   usage findings of the current description absent from the approved one
"""
from dataclasses import dataclass, replace
from itertools import combinations
import numpy as np

ROLES, DEPTS, PURPOSES, REGIONS = 8, 5, 4, 3
RETS = (7, 30, 90, 365)
KEY = {"data": "res", "tool": "tool", "usage": "res"}
SUBJ = ("org", "role", "dept")


@dataclass(frozen=True)
class Rule:
    id: int
    layer: str
    owner: int
    effect: str
    cond: tuple          # tuple of (attr, frozenset | None)

    @property
    def c(self):
        return dict(self.cond)


@dataclass(frozen=True)
class Desc:
    reads: frozenset
    binding: str
    purpose: int
    retention: int
    sinks: frozenset     # of (org, region)


def overlap(a, b):
    ca, cb = a.c, b.c
    return all(ca[k] is None or cb[k] is None or (ca[k] & cb[k]) for k in ca)


def subsumed(a, b):
    ca, cb = a.c, b.c
    return all(cb[k] is None or (ca[k] is not None and ca[k] <= cb[k]) for k in ca)


class Federation:
    def __init__(self, K=4, nres=20, nshared=10, ntool=10, rules_per_org=60, p_cross=0.4, seed=0):
        self.rng = np.random.default_rng(seed)
        self.K, self.nid, self.p_cross = K, 0, p_cross
        subj = [(o, r, d) for o in range(K) for r in range(ROLES) for d in range(DEPTS)]
        subj += [(o, ROLES, DEPTS) for o in range(K)]
        self.so = np.array(subj)
        self.H = K * ROLES * DEPTS
        self.dom = {"org": K, "role": ROLES + 1, "dept": DEPTS + 1, "purpose": PURPOSES, "sorg": K, "sreg": REGIONS}
        self.res_owner = np.array([k for k in range(K) for _ in range(nres)] + [-1] * nshared)
        self.NR = len(self.res_owner)
        self.dom["res"] = self.NR
        self.host = np.array([k for k in range(K) for _ in range(ntool)])
        self.NT = len(self.host)
        self.dom["tool"] = self.NT
        self.retain_max = self.rng.choice(RETS, self.NR)
        self.d1 = {t: self._desc(t) for t in range(self.NT)}
        self.d0 = dict(self.d1)
        self.rules = {}
        for k in range(K):
            for layer, n in (("data", rules_per_org), ("tool", rules_per_org // 2), ("usage", rules_per_org // 2)):
                for _ in range(n):
                    r = self.random_rule(layer, k)
                    self.rules[r.id] = r
        self.full()

    # ------------------------------------------------------------------ generation
    def owners(self, r):
        return list(range(self.K)) if self.res_owner[r] < 0 else [int(self.res_owner[r])]

    def own_res(self, k):
        return np.flatnonzero((self.res_owner == k) | (self.res_owner < 0))

    def _set(self, attr, lo=1, hi=3, pool=None):
        pool = np.arange(self.dom[attr]) if pool is None else pool
        s = frozenset(int(x) for x in self.rng.choice(pool, min(len(pool), self.rng.integers(lo, hi + 1)), replace=False))
        return None if len(s) == self.dom[attr] else s

    def random_rule(self, layer, k):
        g = self.rng
        if layer in ("data", "tool"):
            c = {a: (self._set(a) if g.random() < 0.6 else None) for a in SUBJ}
            if layer == "data" and g.random() < 0.15:              # grant service identities
                c["role"], c["dept"] = frozenset([ROLES]), None
            if layer == "data":
                own, shared = np.flatnonzero(self.res_owner == k), np.flatnonzero(self.res_owner < 0)
                c["res"] = self._set("res", 1, 3, shared if len(shared) and g.random() < 0.35 else own)
            else:
                c["tool"] = self._set("tool", 1, 3, np.flatnonzero(self.host == k))
        else:
            pool = self.own_res(k)
            c = {"res": self._set("res", 1, 3, pool), "purpose": self._set("purpose", 1, 2) if g.random() < .7 else None,
                 "sorg": self._set("sorg", 1, 2) if g.random() < .6 else None,
                 "sreg": self._set("sreg", 1, 2) if g.random() < .6 else None}
        eff = "permit" if g.random() < 0.8 else "deny"
        self.nid += 1
        return Rule(self.nid, layer, k, eff, tuple(sorted(c.items())))

    def _desc(self, t):
        g, h = self.rng, int(self.host[t])
        reads = set()
        for _ in range(g.integers(1, 4)):
            pool = self.own_res(h) if g.random() > self.p_cross else np.arange(self.NR)
            reads.add(int(g.choice(pool)))
        sinks = frozenset((int(g.integers(self.K)) if g.random() < .5 else h, int(g.integers(REGIONS)))
                          for _ in range(g.integers(1, 3)))
        return Desc(frozenset(reads), "obo" if g.random() < .6 else "svc", int(g.integers(PURPOSES)),
                    int(g.choice(RETS)), sinks)

    def random_event(self):
        g = self.rng
        x = g.random()
        k = int(g.integers(self.K))
        ids = {L: [i for i, r in self.rules.items() if r.layer == L] for L in ("data", "tool", "usage")}
        for L, lo, hi in (("data", 0, .30), ("tool", .30, .45), ("usage", .45, .53)):
            if lo <= x < hi:
                return ("add", self.random_rule(L, k))
        for L, lo, hi in (("data", .53, .65), ("tool", .65, .71), ("usage", .71, .75)):
            if lo <= x < hi and ids[L]:
                return ("del", int(g.choice(ids[L])))
        t = int(g.integers(self.NT))
        if x < .93:
            d = self._desc(t)
            old = self.d1[t]
            if g.random() < .5:      # small edit: keep reads, change sinks or retention
                d = replace(old, sinks=d.sinks, retention=d.retention)
            return ("redescribe", t, d)
        return ("approve", t)

    # ------------------------------------------------------------------ decisions
    def subj_mask(self, r):
        c, m = r.c, np.ones(len(self.so), bool)
        for i, a in enumerate(SUBJ):
            if c[a] is not None:
                m &= np.isin(self.so[:, i], list(c[a]))
        return m

    def _row(self, layer, key):
        P = np.zeros(len(self.so), bool)
        N = np.zeros(len(self.so), bool)
        for j in self.idx[layer].get(key, ()):
            r = self.rules[j]
            (P if r.effect == "permit" else N)[:] |= self.subj_mask(r)
        return P & ~N

    def usage_ok(self, r, purpose, sorg, sreg):
        hit = []
        for j in self.idx["usage"].get(r, ()):
            c = self.rules[j].c
            if (c["purpose"] is None or purpose in c["purpose"]) and (c["sorg"] is None or sorg in c["sorg"]) \
                    and (c["sreg"] is None or sreg in c["sreg"]):
                hit.append(self.rules[j].effect)
        return bool(hit) and "deny" not in hit

    def usage_findings(self, t, d):
        out = set()
        for r in d.reads:
            for o, g in d.sinks:
                if not self.usage_ok(r, d.purpose, o, g):
                    out.add((t, r, "sink", (o, g)))
            if d.retention > self.retain_max[r]:
                out.add((t, r, "ret", d.retention))
        return out

    def cross_cell(self, t, r):
        d, H = self.d1[t], self.H
        Ut, Dr = self.U[t][:H], self.D[r]
        if d.binding == "obo":
            return set(np.flatnonzero(Ut & ~Dr[:H]).tolist()), set()
        s = Dr[H + int(self.host[t])]
        gap = set(np.flatnonzero(Ut).tolist()) if not s else set()
        amp = set(np.flatnonzero(Ut & ~Dr[:H]).tolist()) if s else set()
        return gap, amp

    # ------------------------------------------------------------------ full recomputation
    def build_index(self):
        self.idx = {L: {} for L in KEY}
        for j, r in self.rules.items():
            for v in r.c[KEY[r.layer]]:
                self.idx[r.layer].setdefault(v, set()).add(j)
        self.readers = {}
        for t, d in self.d1.items():
            for r in d.reads:
                self.readers.setdefault(r, set()).add(t)

    def pairs_of(self, j, cand):
        a, conf, red = self.rules[j], set(), set()
        for k in cand:
            b = self.rules[k]
            if k == j or b.owner == a.owner or b.layer != a.layer or not overlap(a, b):
                continue
            if a.effect != b.effect:
                conf.add((min(j, k), max(j, k)))
            else:
                if subsumed(a, b):
                    red.add((j, k))
                if subsumed(b, a):
                    red.add((k, j))
        return conf, red

    def full(self):
        self.build_index()
        self.D = {r: self._row("data", r) for r in range(self.NR)}
        self.U = {t: self._row("tool", t) for t in range(self.NT)}
        self.conf, self.red = set(), set()
        for L in KEY:
            for v, ids in self.idx[L].items():
                for j, k in combinations(sorted(ids), 2):
                    c, rd = self.pairs_of(j, [k])
                    self.conf |= c; self.red |= rd
        self.cross = {(t, r): self.cross_cell(t, r) for t, d in self.d1.items() for r in d.reads}
        self.use = {t: self.usage_findings(t, self.d1[t]) for t in range(self.NT)}
        self.drift = {t: self.use[t] - self.usage_findings(t, self.d0[t]) for t in range(self.NT)}

    def snapshot(self):
        gap = {(u, t, r) for (t, r), (g, a) in self.cross.items() for u in g}
        amp = {(u, t, r) for (t, r), (g, a) in self.cross.items() for u in a}
        return dict(conf=set(self.conf), red=set(self.red), gap=gap, amp=amp,
                    use=set().union(*self.use.values()), drift=set().union(*self.drift.values()))

    # ------------------------------------------------------------------ incremental admission
    def apply(self, ev, diff=True):
        """Apply one event and update every finding inside its region only. Returns the region size
        (cells re-evaluated), the findings that appeared (if diff) and the cross-layer cells touched."""
        before = self.snapshot() if diff else None
        kind, cells, tools_usage, cross_keys = ev[0], 0, set(), set()
        if kind in ("add", "del"):
            r = ev[1] if kind == "add" else self.rules[ev[1]]
            L, keys = r.layer, r.c[KEY[r.layer]]
            if kind == "add":
                self.rules[r.id] = r
                for v in keys:
                    self.idx[L].setdefault(v, set()).add(r.id)
                cand = set().union(*[self.idx[L][v] for v in keys])
                c, rd = self.pairs_of(r.id, cand)
                self.conf |= c; self.red |= rd
                cells += len(cand)
            else:
                del self.rules[r.id]
                for v in keys:
                    self.idx[L][v].discard(r.id)
                self.conf = {p for p in self.conf if r.id not in p}
                self.red = {p for p in self.red if r.id not in p}
            if L == "data":
                for v in keys:
                    self.D[v] = self._row("data", v)
                    cross_keys |= {(t, v) for t in self.readers.get(v, ())}
            elif L == "tool":
                for v in keys:
                    self.U[v] = self._row("tool", v)
                    cross_keys |= {(v, x) for x in self.d1[v].reads}
            else:
                for v in keys:
                    tools_usage |= self.readers.get(v, set())
        elif kind == "redescribe":
            t, d = ev[1], ev[2]
            old = self.d1[t]
            for x in old.reads:
                self.readers[x].discard(t)
                self.cross.pop((t, x), None)
            self.d1[t] = d
            for x in d.reads:
                self.readers.setdefault(x, set()).add(t)
            cross_keys |= {(t, x) for x in d.reads}
            tools_usage.add(t)
        elif kind == "approve":
            self.d0[ev[1]] = self.d1[ev[1]]
            tools_usage.add(ev[1])
        for key in cross_keys:
            self.cross[key] = self.cross_cell(*key)
            cells += self.H
        for t in tools_usage:
            self.use[t] = self.usage_findings(t, self.d1[t])
            self.drift[t] = self.use[t] - self.usage_findings(t, self.d0[t])
            cells += 2 * len(self.d1[t].reads) * (len(self.d1[t].sinks) + 1)
        if not diff:
            return cells, None, cross_keys
        after = self.snapshot()
        return cells, {k: after[k] - before[k] for k in after}, cross_keys

    def total_cells(self):
        pairs = sum(len(ids) * (len(ids) - 1) // 2 for L in KEY for ids in self.idx[L].values())
        cross = sum(len(d.reads) for d in self.d1.values()) * self.H
        use = sum(2 * len(d.reads) * (len(d.sinks) + 1) for d in self.d1.values())
        return pairs + cross + use

    # ------------------------------------------------------------------ encoding for the secure pairwise check
    def incidence(self, ids, layer):
        attrs = sorted(self.rules[ids[0]].c) if ids else []
        slices, s = [], 0
        for a in attrs:
            slices.append((s, s + self.dom[a])); s += self.dom[a]
        X = np.zeros((len(ids), s), np.int64)
        for i, j in enumerate(ids):
            c = self.rules[j].c
            for a, (lo, hi) in zip(attrs, slices):
                if c[a] is None:
                    X[i, lo:hi] = 1
                else:
                    X[i, [lo + v for v in c[a]]] = 1
        eff = np.array([1 if self.rules[j].effect == "permit" else 0 for j in ids], np.int64)
        return X, eff, slices
