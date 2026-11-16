"""Executable check of the v4 formal model (seven anomaly types) on a toy
case modelled on an enterprise agent platform over a cloud data platform.

Rule = (layer, owner, effect, cond) ; cond maps "role.attr" -> set of values, absent = wildcard.
Layers: tool (who may call which tool through which agent),
        data (which identity may read which resource),
        flow (which resource may go to which sink).
"""
from itertools import product

STAR = None


def matches(cond, q):
    return all(q.get(k) in v for k, v in cond.items())


def overlap(c1, c2):
    return all(c1[k] & c2[k] for k in c1.keys() & c2.keys())


def subsumed(c1, c2):  # c1 inside c2
    return all(k in c1 and c1[k] <= c2[k] for k in c2)


def R(layer, owner, effect, **cond):
    return dict(layer=layer, owner=owner, effect=effect,
                cond={k.replace("_", ".", 1): set(v) for k, v in cond.items()})


def build(binding):
    rules = [
        # tool layer, organisation A (agent platform)
        R("tool", "A", "permit", usr_role={"analyst", "business"}, tool_name={"query_sales"}),
        R("tool", "A", "permit", usr_role={"analyst"}, tool_name={"query_sales"}),          # redundant
        R("tool", "A", "permit", usr_role={"business"}, tool_name={"export_report"}),
        R("tool", "A", "deny", usr_role={"business"}, tool_name={"export_report"}, usr_region={"eu"}),  # conflict
        R("tool", "A", "permit", usr_role={"auditor"}, tool_name={"query_hr"}),             # never used
        # data layer, organisation B (data platform)
        R("data", "B", "permit", id_role={"analyst"}, res_class={"sales"}),
        R("data", "B", "permit", id_role={"app_account"}, res_class={"sales", "hr"}),
        # flow layer, organisation B (data owner)
        R("flow", "B", "permit", res_class={"sales"}, sink_org={"B", "A"}),
        R("flow", "B", "deny", res_class={"sales", "hr"}, sink_org={"third_party"}),
    ]
    tools = {
        "query_sales": dict(reads="sales", sinks={"A"}, binding=binding),
        "export_report": dict(reads="sales", sinks={"A"}, binding=binding),
        "query_hr": dict(reads="hr", sinks={"A"}, binding=binding),
    }
    return rules, tools


def decide(rules, layer, q):  # deny-overrides, default deny
    hit = [r["effect"] for r in rules if r["layer"] == layer and matches(r["cond"], q)]
    return "deny" if "deny" in hit or not hit else "permit"


def analyse(rules, tools, log, roles=("analyst", "business", "auditor"), regions=("eu", "us")):
    out = {k: set() for k in ("conflict", "redundancy", "irrelevancy", "gap",
                              "amplification", "flow_violation")}
    for i, a in enumerate(rules):
        for j, b in enumerate(rules):
            if i >= j or a["layer"] != b["layer"]:
                continue
            if a["effect"] != b["effect"] and overlap(a["cond"], b["cond"]):
                out["conflict"].add((i, j))
            if a["effect"] == b["effect"]:
                if subsumed(a["cond"], b["cond"]):
                    out["redundancy"].add((i, j))
                elif subsumed(b["cond"], a["cond"]):
                    out["redundancy"].add((j, i))
    for i, r in enumerate(rules):                       # behaviour-dependent
        if r["layer"] == "tool" and not any(matches(r["cond"], q) for q in log):
            out["irrelevancy"].add(i)
    for role, region, (t, spec) in product(roles, regions, tools.items()):
        q = {"usr.role": role, "usr.region": region, "tool.name": t}
        if decide(rules, "tool", q) != "permit":
            continue
        res = spec["reads"]
        user_ok = decide(rules, "data", {"id.role": role, "res.class": res}) == "permit"
        ident = role if spec["binding"] == "obo" else "app_account"
        eff_ok = decide(rules, "data", {"id.role": ident, "res.class": res}) == "permit"
        if not eff_ok:
            out["gap"].add((role, t, res))               # allowed at tool layer, blocked below
        if eff_ok and not user_ok:
            out["amplification"].add((role, t, res))     # deputy can, user cannot
        if eff_ok:
            for s in spec["sinks"]:
                if decide(rules, "flow", {"res.class": res, "sink.org": s}) != "permit":
                    out["flow_violation"].add((t, res, s))
    return out


log = [{"usr.role": "analyst", "usr.region": "eu", "tool.name": "query_sales"},
       {"usr.role": "business", "usr.region": "us", "tool.name": "query_sales"},
       {"usr.role": "business", "usr.region": "us", "tool.name": "export_report"}]

res = {}
for b in ("obo", "svc"):
    rules, tools = build(b)
    res[b] = analyse(rules, tools, log)
    print(f"--- binding={b}")
    for k, v in res[b].items():
        print(f"{k:15s} {sorted(v)}")

# Proposition (gap-amplification duality): switching a tool from on-behalf-of to a
# service account that holds the permission turns every data-layer gap into an amplification.
assert res["obo"]["gap"] and not res["obo"]["amplification"]
assert res["obo"]["gap"] == res["svc"]["amplification"], "duality violated"
assert not res["svc"]["gap"]

# Drift: approved vs deployed tool descriptor; impact = change in the anomaly set.
rules, tools = build("svc")
before = analyse(rules, tools, log)
tools["export_report"]["sinks"] = {"A", "third_party"}   # silent change after approval
after = analyse(rules, tools, log)
impact = {k: after[k] - before[k] for k in after if after[k] - before[k]}
print("--- drift impact", impact)
assert impact == {"flow_violation": {("export_report", "sales", "third_party")}}
print("all checks passed")
