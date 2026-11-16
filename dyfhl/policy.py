"""Policy model, rule sources, multi-domain world generator and exact oracle.

A rule is a conjunction over attributes of value sets (disjunction inside an
attribute); an attribute that is not constrained is a wildcard. Rules live in a
domain and in a collaboration scope shared by several domains.
"""
from __future__ import annotations
import csv, os, re
from dataclasses import dataclass, field
import numpy as np

NONE = "_none"


# --------------------------------------------------------------------------- sources
class Source:
    """A rule/request generator over a fixed attribute vocabulary."""
    name = "base"
    attrs: list
    values: dict            # attr -> list of values
    id_attrs: list          # attributes whose values denote entities that can be decommissioned

    def setup(self, K, rng):
        pass

    def is_id(self, a, v):
        return a in self.id_attrs and v != NONE

    def sample_rule(self, rng, d):
        raise NotImplementedError

    def sample_request(self, rng, d):
        raise NotImplementedError


class AgenticSource(Source):
    """Synthetic multi-domain agentic environment: agents, MCP servers, tools,
    applications, legacy systems and humans acting on heterogeneous resources."""
    name = "agentic"

    def __init__(self, n_tools=60):
        self.values = {
            "subj_type": ["agent", "mcp_server", "tool", "application", "legacy", "human"],
            "role": [f"role{i}" for i in range(16)],
            "clearance": ["public", "internal", "confidential", "secret"],
            "res_type": ["dataset", "api", "model", "prompt", "file", "database", "queue", "secret_store"],
            "sensitivity": ["low", "medium", "high", "critical"],
            "action": ["read", "write", "invoke", "delegate", "list", "delete", "train", "export"],
            "trust_ctx": ["verified", "attested", "unverified", "quarantined"],
            "tool_id": [f"tool{i}" for i in range(n_tools)],
        }
        self.attrs = list(self.values)
        self.id_attrs = ["tool_id"]

    def setup(self, K, rng):
        # non-IID: every domain has its own Dirichlet preference over the values of each attribute
        self.pref = [{a: rng.dirichlet(np.full(len(v), 1.5)) for a, v in self.values.items()} for _ in range(K)]

    def sample_rule(self, rng, d):
        k = rng.integers(5, 8)
        cand = [a for a in self.attrs if a != "tool_id"]
        chosen = list(rng.choice(cand, size=k, replace=False))
        if rng.random() < 0.25:
            chosen.append("tool_id")
        conds = {}
        for a in chosen:
            m = 1 if rng.random() < 0.85 else 2
            idx = rng.choice(len(self.values[a]), size=m, replace=False, p=self.pref[d][a])
            conds[a] = {self.values[a][i] for i in idx}
        return conds, int(rng.random() < 0.3)

    def sample_request(self, rng, d):
        return {a: self.values[a][rng.choice(len(v), p=self.pref[d][a])] for a, v in self.values.items()}


class TableSource(Source):
    """Rules and requests derived from real subject/resource attribute tables."""

    def __init__(self, name, apps, id_attrs, deny_p=0.25, ks=(2, 5)):
        self.ks = ks
        # apps: dict app -> dict(subjects=[{attr:[vals]}], resources=[...], actions=[...], effects=None|array)
        self.name, self.apps, self.deny_p = name, apps, deny_p
        vals = {"app": list(apps), "action": []}
        for app, t in apps.items():
            for a in t["actions"]:
                if a not in vals["action"]:
                    vals["action"].append(a)
            for row in t["subjects"] + t["resources"]:
                for a, vs in row.items():
                    vals.setdefault(a, [NONE])
                    for v in vs:
                        if v not in vals[a]:
                            vals[a].append(v)
        self.values = vals
        self.attrs = list(vals)
        self.id_prefix = tuple(id_attrs)
        self.id_attrs = [a for a in vals if any(self.is_id(a, v) for v in vals[a])]

    def is_id(self, a, v):
        return v != NONE and (a.startswith(self.id_prefix) or str(v).startswith(self.id_prefix))

    def setup(self, K, rng):
        self.sub = []
        for _ in range(K):
            view = {}
            for app, t in self.apps.items():
                ns, nr = len(t["subjects"]), len(t["resources"])
                view[app] = (rng.choice(ns, size=max(2, int(0.6 * ns)), replace=False),
                             rng.choice(nr, size=max(2, int(0.6 * nr)), replace=False))
            self.sub.append(view)
        self.app_names = list(self.apps)

    def _pick(self, rng, d):
        app = self.app_names[rng.integers(len(self.app_names))]
        t = self.apps[app]
        si, ri = self.sub[d][app]
        return app, t, t["subjects"][rng.choice(si)], t["resources"][rng.choice(ri)]

    def sample_rule(self, rng, d):
        app, t, s, r = self._pick(rng, d)
        conds = {"app": {app}} if len(self.apps) > 1 else {}
        sa = [a for a in s if s[a]]
        ra = [a for a in r if r[a]]
        ks = min(len(sa), rng.integers(*self.ks))
        kr = min(len(ra), rng.integers(1, 4))
        for a in list(rng.choice(sa, size=ks, replace=False)) + list(rng.choice(ra, size=kr, replace=False)):
            row = s if a in s else r
            vs = {row[a][rng.integers(len(row[a]))]}
            if rng.random() < 0.1:
                v0 = next(iter(vs))
                pool = [v for v in self.values[a] if v != NONE and str(v).split("=")[0] == str(v0).split("=")[0]]
                vs.add(pool[rng.integers(len(pool))])
            conds[a] = vs
        if len(t["actions"]) > 1:
            m = 1 if rng.random() < 0.6 else 2
            conds["action"] = set(rng.choice(t["actions"], size=min(m, len(t["actions"])), replace=False))
        if len(conds) < 2 and ra:
            a = ra[rng.integers(len(ra))]
            conds[a] = {r[a][0]}
        return conds, int(rng.random() < self.deny_p)

    def sample_request(self, rng, d):
        app, t, s, r = self._pick(rng, d)
        req = {a: NONE for a in self.attrs}
        req["app"] = app
        req["action"] = t["actions"][rng.integers(len(t["actions"]))]
        for row in (s, r):
            for a, vs in row.items():
                if vs:
                    req[a] = vs[rng.integers(len(vs))]
        return req


def _parse_abac(path, prefix):
    subs, ress, actions = [], [], []
    pat = re.compile(r"(\w+)\s*=\s*(\{[^}]*\}|[^,)]+)")
    for line in open(path):
        line = line.strip()
        for kind, store, tag in (("userAttrib", subs, "u"), ("resourceAttrib", ress, "r")):
            if line.startswith(kind + "("):
                body = line[len(kind) + 1:line.rindex(")")]
                body = body.split(",", 1)[1] if "," in body else ""
                row = {}
                for a, v in pat.findall(body):
                    v = v.strip()
                    row[f"{prefix}.{tag}.{a}"] = v.strip("{}").split() if v.startswith("{") else [v]
                store.append(row)
        if line.startswith("rule("):
            parts = line[5:line.rindex(")")].split(";")
            if len(parts) >= 3:
                for a in parts[2].strip().strip("{}").split():
                    if a not in actions:
                        actions.append(a)
    return dict(subjects=subs, resources=ress, actions=actions or ["access"])


def _slotify(apps):
    """Map each application's attributes onto shared slots (u0, u1, ..., r0, r1, ...). Values keep
    their application and attribute name, so semantics are unchanged while the attribute count stays small."""
    for app, t in apps.items():
        for tag, key in (("u", "subjects"), ("r", "resources")):
            names = sorted({a for row in t[key] for a in row})
            slot = {a: f"{tag}{i}" for i, a in enumerate(names)}
            t[key] = [{slot[a]: [f"{a}={v}" for v in vs] for a, vs in row.items()} for row in t[key]]
    return apps


def xu_stoller_source(root):
    cs = {"hc": "healthcare", "un": "university", "pm": "project-management", "ov": "online-video"}
    apps = _slotify({k: _parse_abac(os.path.join(root, v + ".abac"), k) for k, v in cs.items()})
    return TableSource("xu_stoller", apps, ["hc.r.patient=", "un.r.crs=", "pm.r.project="])


def amazon_source(path, top=25, max_rows=8000, seed=0):
    rows = list(csv.DictReader(open(path)))
    rng = np.random.default_rng(seed)
    cols = [c for c in rows[0] if c != "target"]
    keep = {}
    for c in cols:
        u, n = np.unique([r[c] for r in rows], return_counts=True)
        keep[c] = set(u[np.argsort(-n)[:top]])
    deny = [r for r in rows if r["target"] == "0"]
    perm = [r for r in rows if r["target"] == "1"]
    sel = [deny[i] for i in rng.choice(len(deny), size=min(len(deny), max_rows // 4), replace=False)] + \
          [perm[i] for i in rng.choice(len(perm), size=max_rows - max_rows // 4, replace=False)]
    subs, ress = [], []
    for r in sel:
        subs.append({"az.u." + c: ([r[c]] if r[c] in keep[c] else []) for c in cols if c != "RESOURCE"})
        ress.append({"az.r.RESOURCE": [r["RESOURCE"]] if r["RESOURCE"] in keep["RESOURCE"] else []})
    ress = [r for r in ress if r["az.r.RESOURCE"]] or ress
    apps = {"az": dict(subjects=subs, resources=ress, actions=["access"])}
    return TableSource("amazon", apps, ["az.u.MGR_ID"], deny_p=0.25, ks=(3, 6))


def get_source(name, data_root="data/raw"):
    if name == "agentic":
        return AgenticSource()
    if name == "xu_stoller":
        return xu_stoller_source(os.path.join(data_root, "xu-stoller"))
    if name == "amazon":
        return amazon_source(os.path.join(data_root, "amazon_employee_access.csv"))
    raise ValueError(name)


# --------------------------------------------------------------------------- oracle
def pair_matrices(H, Wm, slices):
    """Exact overlap and subsumption between all rule pairs.
    overlap[i,j]: some request matches both. sub[i,j]: every request matching i matches j."""
    n = H.shape[0]
    ov = np.ones((n, n), bool)
    sub = np.ones((n, n), bool)
    Hf = H.astype(np.float32)
    for a, (s, e) in enumerate(slices):
        R = Hf[:, s:e]
        inter = R @ R.T
        w = Wm[:, a].astype(bool)
        ov &= (inter > 0) | w[:, None] | w[None, :]
        size = R.sum(1)
        sub &= w[None, :] | (~w[:, None] & (inter >= size[:, None] - 1e-6))
    return ov, sub


def cover_matrix(Q, H, Wm):
    """cov[q,j]: rule j matches request q (Q is one-hot with one value per attribute)."""
    A = Wm.shape[1]
    hit = Q.astype(np.float32) @ H.astype(np.float32).T + Wm.sum(1)[None, :]
    return hit >= A - 1e-6


# --------------------------------------------------------------------------- world
@dataclass
class World:
    source: Source
    K: int = 6
    S: int = 6
    n_rules: int = 400          # shared-scope rules per domain
    n_queries: int = 250        # candidate requests per domain
    p_conf: float = 0.10
    p_red: float = 0.10
    p_irr: float = 0.08
    p_dead: float = 0.15
    seed: int = 0
    rules: list = field(default_factory=list)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        src = self.source
        src.setup(self.K, self.rng)
        self.attrs = src.attrs
        self.A = len(self.attrs)
        self.slices, self.vidx, off = [], {}, 0
        for a in self.attrs:
            for i, v in enumerate(src.values[a]):
                self.vidx[(a, v)] = off + i
            self.slices.append((off, off + len(src.values[a])))
            off += len(src.values[a])
        self.V = off
        self.id_mask = np.zeros(self.V, bool)
        for (a, v), k in self.vidx.items():
            self.id_mask[k] = src.is_id(a, v)
        self.owner = np.where(self.id_mask, self.rng.integers(0, self.K, self.V), -1)
        self.alive = np.where(self.id_mask, self.rng.random(self.V) > self.p_dead, True)
        self.members = []
        for s in range(self.S):
            m = self.rng.integers(2, min(4, self.K) + 1)
            self.members.append(sorted(self.rng.choice(self.K, size=m, replace=False).tolist()))
        for d in range(self.K):  # every domain takes part in at least one scope
            if not any(d in m for m in self.members):
                self.members[self.rng.integers(self.S)].append(d)
        self.dom_scopes = [[s for s in range(self.S) if d in self.members[s]] for d in range(self.K)]
        self._uid = 0
        self.t = 0
        for d in range(self.K):
            for _ in range(self.n_rules):
                self.rules.append(self._new_rule(d))
        self._inject(range(len(self.rules)))
        self.queries = []
        for d in range(self.K):
            for _ in range(self.n_queries):
                s = self.dom_scopes[d][self.rng.integers(len(self.dom_scopes[d]))]
                self.queries.append(dict(dom=d, scope=s, req=self._request(d, s)))

    def _request(self, d, s):
        """A request in scope s; half of them target the region addressed by some existing rule."""
        req = self.source.sample_request(self.rng, d)
        if self.rng.random() < 0.6:
            cand = [r for r in self.rules if r["scope"] == s]
            r = cand[self.rng.integers(len(cand))]
            for a, vs in r["conds"].items():
                req[a] = sorted(vs)[self.rng.integers(len(vs))]
        return req

    # -- rule construction
    def _new_rule(self, d, conds=None, eff=None, scope=None):
        if conds is None:
            conds, eff = self.source.sample_rule(self.rng, d)
        if scope is None:
            scope = self.dom_scopes[d][self.rng.integers(len(self.dom_scopes[d]))]
        self._uid += 1
        return dict(uid=self._uid, dom=d, scope=scope, conds={a: set(v) for a, v in conds.items()}, eff=eff, t=self.t)

    def _inject(self, idxs):
        """Overwrite a fraction of the given rules with variants of foreign rules
        (conflict / redundancy) or with references to decommissioned foreign entities."""
        rng = self.rng
        by_scope = {}
        for i, r in enumerate(self.rules):
            by_scope.setdefault(r["scope"], []).append(i)
        for i in idxs:
            r = self.rules[i]
            u = rng.random()
            if u < self.p_conf + self.p_red:
                cand = [j for j in by_scope[r["scope"]] if self.rules[j]["dom"] != r["dom"]]
                if not cand:
                    continue
                o = self.rules[cand[rng.integers(len(cand))]]
                conds = {a: set(v) for a, v in o["conds"].items()}
                if u < self.p_conf:      # overlapping rule with the opposite effect
                    v = rng.random()
                    if v < 0.5:
                        self._specialise(conds)
                    eff = 1 - o["eff"]
                else:                     # rule subsumed by a foreign rule with the same effect
                    if rng.random() < 0.7:
                        self._specialise(conds)
                    eff = o["eff"]
                self.rules[i] = self._new_rule(r["dom"], conds, eff, r["scope"])
            elif u < self.p_conf + self.p_red + self.p_irr and self.source.id_attrs:
                a = self.source.id_attrs[rng.integers(len(self.source.id_attrs))]
                s, e = self.slices[self.attrs.index(a)]
                dead = [k for k in range(s, e) if self.id_mask[k] and not self.alive[k] and self.owner[k] != r["dom"]]
                if dead:
                    vals = self.source.values[a]
                    pick = rng.choice(dead, size=min(len(dead), 1 + int(rng.random() < 0.3)), replace=False)
                    conds = {b: set(v) for b, v in r["conds"].items()}
                    conds[a] = {vals[k - s] for k in pick}
                    self.rules[i] = self._new_rule(r["dom"], conds, r["eff"], r["scope"])

    def _specialise(self, conds):
        rng = self.rng
        free = [a for a in self.attrs if a not in conds and a not in self.source.id_attrs and a not in ("app", "action")]
        if "app" in conds:   # stay inside the application's own attribute namespace
            free = []
        shrink = [a for a in conds if len(conds[a]) > 1]
        if shrink and rng.random() < 0.4:
            a = shrink[rng.integers(len(shrink))]
            conds[a] = {sorted(conds[a])[rng.integers(len(conds[a]))]}
        elif free:
            a = free[rng.integers(len(free))]
            pool = [v for v in self.source.values[a] if v != NONE]
            conds[a] = {pool[rng.integers(len(pool))]}

    # -- dynamics
    def step(self, churn=0.1, id_flip=0.05):
        """One time step of policy churn: a fraction of rules is replaced by new ones and
        entities are decommissioned or reinstated. Returns the uids removed and added."""
        self.t += 1
        rng = self.rng
        n = len(self.rules)
        idx = rng.choice(n, size=int(churn * n), replace=False)
        removed = [self.rules[i]["uid"] for i in idx]
        for i in idx:
            self.rules[i] = self._new_rule(self.rules[i]["dom"])
        self._inject(idx)
        flip = self.id_mask & (rng.random(self.V) < id_flip)
        self.alive = np.where(flip, ~self.alive, self.alive)
        return removed, [self.rules[i]["uid"] for i in idx], np.flatnonzero(flip)

    # -- arrays
    def sample_requests(self, d, m):
        qs = []
        for _ in range(m):
            s = self.dom_scopes[d][self.rng.integers(len(self.dom_scopes[d]))]
            qs.append(dict(dom=d, scope=s, req=self._request(d, s)))
        return self.encode_queries(qs)

    def encode_rules(self, rules=None):
        rules = self.rules if rules is None else rules
        n = len(rules)
        H = np.zeros((n, self.V), np.uint8)
        Wm = np.ones((n, self.A), np.uint8)
        for i, r in enumerate(rules):
            for a, vs in r["conds"].items():
                Wm[i, self.attrs.index(a)] = 0
                for v in vs:
                    H[i, self.vidx[(a, v)]] = 1
        g = lambda k: np.array([r[k] for r in rules], np.int64)
        return dict(H=H, Wm=Wm, eff=g("eff"), dom=g("dom"), scope=g("scope"), uid=g("uid"), t=g("t"))

    def encode_queries(self, queries=None):
        queries = self.queries if queries is None else queries
        Q = np.zeros((len(queries), self.V), np.uint8)
        for i, q in enumerate(queries):
            for a, v in q["req"].items():
                Q[i, self.vidx[(a, v)]] = 1
        g = lambda k: np.array([q[k] for q in queries], np.int64)
        return dict(H=Q, dom=g("dom"), scope=g("scope"))

    def irrelevancy(self, R, local=False):
        """Cross-domain irrelevancy: some constrained entity attribute refers only to
        decommissioned entities, at least one of them owned by another domain."""
        out = np.zeros(len(R["eff"]), bool)
        dead = self.id_mask & ~self.alive
        for a in self.source.id_attrs:
            s, e = self.slices[self.attrs.index(a)]
            Ha = R["H"][:, s:e].astype(bool)
            con = (Ha & self.id_mask[s:e][None, :]).any(1) & ~(Ha & ~self.id_mask[s:e][None, :]).any(1)
            all_dead = ~(Ha & ~dead[s:e][None, :]).any(1)
            foreign = (Ha & (self.owner[s:e][None, :] != R["dom"][:, None]) & self.id_mask[s:e][None, :]).any(1)
            out |= con & all_dead & foreign
        return out

    def labels(self):
        """Exact global labels (requires pooling every domain's rules; used for evaluation only).
        Returns rule labels [n,3] = (cross-conflict, cross-redundancy, cross-irrelevancy),
        the encoded rules, the locally-uncovered queries and their cross-domain gap label."""
        R = self.encode_rules()
        Qall = self.encode_queries()
        n = len(R["eff"])
        Y = np.zeros((n, 3), bool)
        keepq = np.zeros(len(Qall["dom"]), bool)
        gap = np.zeros(len(Qall["dom"]), bool)
        for s in range(self.S):
            idx = np.flatnonzero(R["scope"] == s)
            if len(idx) == 0:
                continue
            ov, sub = pair_matrices(R["H"][idx], R["Wm"][idx], self.slices)
            diff_dom = R["dom"][idx][:, None] != R["dom"][idx][None, :]
            diff_eff = R["eff"][idx][:, None] != R["eff"][idx][None, :]
            Y[idx, 0] = (ov & diff_dom & diff_eff).any(1)
            Y[idx, 1] = (sub & diff_dom & ~diff_eff).any(1)
            qi = np.flatnonzero(Qall["scope"] == s)
            if len(qi):
                cov = cover_matrix(Qall["H"][qi], R["H"][idx], R["Wm"][idx])
                own = R["dom"][idx][None, :] == Qall["dom"][qi][:, None]
                keepq[qi] = ~(cov & own).any(1)
                gap[qi] = ~cov.any(1)
        Y[:, 2] = self.irrelevancy(R)
        Q = {k: v[keepq] for k, v in Qall.items()}
        return R, Y, Q, gap[keepq]
