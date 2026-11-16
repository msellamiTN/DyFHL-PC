"""Executable check of the v4 model extensions (draft 2):
usage layer (purpose, sink organisation, region, retention), obligation conflicts,
declared-vs-observed drift, delegation chains, confirmed vs potential findings.
Complements formal_model_check.py; the original four anomaly types are unchanged.
"""
from itertools import product


def matches(cond, q):
    return all(q.get(k) in v for k, v in cond.items())


def decide(rules, q):  # deny-overrides, default deny
    hit = [e for e, c in rules if matches(c, q)]
    return "deny" if "deny" in hit or not hit else "permit"


# ---- usage layer, owned by the data owner (organisation B)
USAGE = [
    ("permit", {"res.class": {"sales"}, "use.purpose": {"reporting"},
                "sink.org": {"A", "B"}, "sink.region": {"eu"}}),
    ("deny", {"res.class": {"sales", "hr"}, "sink.org": {"third_party"}}),
]
# obligations attached by each side to the same resource class
OBLIG_B = {"sales": {"retain_max_days": 30, "log": True}}      # data owner
OBLIG_A = {"sales": {"retain_min_days": 365}}                   # consumer's audit rule

TOOLS = {
    "query_sales": dict(reads="sales", purpose="reporting", retention=7,
                        declared={("A", "eu")}, observed={("A", "eu")}),
    "export_report": dict(reads="sales", purpose="reporting", retention=400,
                          declared={("A", "eu")}, observed={("A", "eu"), ("A", "us")}),
    "train_model": dict(reads="sales", purpose="model_training", retention=7,
                        declared={("A", "eu")}, observed={("A", "eu")}),
}


def usage_violations(tools, source):
    out = set()
    for t, d in tools.items():
        for org, region in d[source]:
            q = {"res.class": d["reads"], "use.purpose": d["purpose"],
                 "sink.org": org, "sink.region": region}
            if decide(USAGE, q) != "permit":
                out.add((t, d["purpose"], org, region))
        mx = OBLIG_B.get(d["reads"], {}).get("retain_max_days")
        if mx is not None and d["retention"] > mx:
            out.add((t, "retention", d["retention"], mx))
    return out


declared = usage_violations(TOOLS, "declared")
observed = usage_violations(TOOLS, "observed")
print("usage violations (declared):", sorted(map(str, declared)))
print("behavioural drift (observed only):", sorted(map(str, observed - declared)))
assert ("train_model", "model_training", "A", "eu") in declared        # purpose mismatch
assert ("export_report", "retention", 400, 30) in declared             # retention
assert observed - declared == {("export_report", "reporting", "A", "us")}  # sleeper drift, region
assert declared <= observed            # observation can only add findings when observed ⊇ declared

# ---- obligation conflict across organisations (extension of type 1)
def obligation_conflicts(a, b):
    out = set()
    for res in a.keys() & b.keys():
        lo, hi = a[res].get("retain_min_days"), b[res].get("retain_max_days")
        if lo is not None and hi is not None and lo > hi:
            out.add((res, f"min {lo} > max {hi}"))
    return out


oc = obligation_conflicts(OBLIG_A, OBLIG_B)
print("obligation conflicts:", sorted(oc))
assert oc == {("sales", "min 365 > max 30")}

# ---- delegation chains (transitive form of type 5)
PERM = {"u_business": {"sales"}, "agent_planner": {"sales", "hr"}, "agent_exec": {"sales", "hr", "finance"}}


def effective(chain):                       # intended semantics: authority only shrinks
    eff = set(PERM[chain[0]])
    for p in chain[1:]:
        eff &= PERM[p]
    return eff


def exercised(chain):                       # what happens if the last agent uses its own rights
    return set(PERM[chain[-1]])


chain = ["u_business", "agent_planner", "agent_exec"]
for k in range(1, len(chain) + 1):          # attenuation: monotone non-increasing along the chain
    assert effective(chain[:k]) <= effective(chain[:max(k - 1, 1)])
amp = exercised(chain) - effective(chain)
print("chain amplification:", sorted(amp))
assert amp == {"hr", "finance"} and effective(chain) <= PERM[chain[0]]

# ---- confirmed vs potential (open point 1)
DATA = [("permit", {"id.role": {"analyst"}, "id.region": {"eu", "us"}}),
        ("permit", {"id.role": {"business"}, "id.region": {"eu"}})]
TOOL = [("permit", {"id.role": {"analyst", "business"}})]
classes = [{"id.role": r, "id.region": g} for r, g in product(("analyst", "business"), ("eu", "us"))]
directory = [{"id.role": "analyst", "id.region": "eu"}, {"id.role": "business", "id.region": "eu"}]


def uncovered(population):
    return {tuple(sorted(q.items())) for q in population
            if decide(TOOL, q) == "permit" and decide(DATA, q) != "permit"}


potential, confirmed = uncovered(classes), uncovered(directory)
print("potential:", len(potential), "confirmed:", len(confirmed))
assert confirmed <= potential               # class-based check never misses a confirmed case
assert len(potential) == 1 and len(confirmed) == 0   # business/us: possible, nobody there today
print("all checks passed")
