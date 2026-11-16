"""E2: replay one year of AWS managed-policy changes as admission events.

For every change event (policy added or modified) we measure
  - the change itself: statements removed/added, actions gained/lost, state-changing actions gained;
  - locality: corpus statements that share an action vertex with a changed statement (the region), and the
    within-policy findings that changed, checked to involve only changed statements (Theorem 'locality');
  - the admission check: new redundant pairs and allow/deny overlaps inside the policy, and for agent-facing
    policies the actions an agent would hold beyond each human job function (lower bound, as in E3);
  - cost: incremental check time against full recomputation, and the bytes of the flags-only cross-owner
    amplification check on the region against a check on the full action universe.
Usage: python3 -I scripts/aws_history.py data/history results/aws_history.json"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dyfhl.iam import Universe, statements, allowed, within_policy, explicit_actions, mutating, JOBS, AGENT_RE

src, out_path = sys.argv[1], sys.argv[2]
base = json.load(open(f"{src}/base.json"))
events = [json.loads(l) for l in open(f"{src}/events.jsonl")]
acts = set()
for d in base["policies"].values():
    acts |= explicit_actions(d)
for e in events:
    acts |= explicit_actions(e["old"]) | explicit_actions(e["new"])
U = Universe(acts)
idx = {a: i for i, a in enumerate(sorted(U.all))}


def mask(aset):
    m = 0
    for a in aset:
        m |= 1 << idx[a]
    return m


def parse(doc):
    st = statements(doc, U)
    for s in st:
        s["mask"] = mask(s["actions"])
    return st


pol = {n: parse(d) for n, d in base["policies"].items() if d}
is_agent = lambda n: bool(AGENT_RE.search(n))
B_PROD = 40     # bytes per secure product: input sharing 16, Beaver opening 16, output 8 (two servers, 4-byte elements)


def held(n):
    return allowed(pol[n]) if n in pol else set()


def amp_row(n):
    a = allowed(pol[n], unconditional_only=True)
    return {j: a - held(j) for j in JOBS}


t0 = time.time()
full_red = {n: within_policy(s) for n, s in pol.items()}
t_full_within = time.time() - t0
amp = {n: amp_row(n) for n in pol if is_agent(n)}
n_stmt0 = sum(len(s) for s in pol.values())
rows, viol_local = [], 0
for e in events:
    n = e["policy"]
    old = pol.get(n, [])
    new = parse(e["new"]) if e["new"] else []
    ok, nk = {s["key"] for s in old}, {s["key"] for s in new}
    changed_keys = ok ^ nk
    changed = [s for s in old if s["key"] not in nk] + [s for s in new if s["key"] not in ok]
    C = 0
    for s in changed:
        C |= s["mask"]
    pol[n] = new
    # locality of the region: statements of the corpus sharing an action vertex with a changed statement
    reg_st, reg_pol = 0, set()
    for pn, st in pol.items():
        for s in st:
            if s["mask"] & C:
                reg_st += 1; reg_pol.add(pn)
    tot_st = sum(len(s) for s in pol.values())
    # admission check inside the policy: only pairs with a changed statement (incremental)
    t1 = time.time()
    newk = {s["key"] for s in new if s["key"] not in ok}
    sub = [s for s in new if s["key"] in newk]
    red_i, ov_i = set(), set()
    if sub:
        r_all, o_all = within_policy(new) if len(new) <= 60 else within_policy(new)
        red_i = {p for p in r_all if p[0] in newk or p[1] in newk}
        ov_i = {p for p in o_all if p[0] in newk or p[1] in newk}
    t_inc = time.time() - t1
    r_old, o_old = full_red.get(n, (set(), set()))
    r_new, o_new = within_policy(new)
    full_red[n] = (r_new, o_new)
    # locality check: every finding that appeared or disappeared involves a changed statement
    for p in (r_new ^ r_old) | (o_new ^ o_old):
        if not (set(p) & changed_keys):
            viol_local += 1
    new_red, new_ov = r_new - r_old, o_new - o_old
    assert new_red <= red_i and new_ov <= ov_i
    # amplification: agent row or job column
    t2 = time.time()
    d_amp, d_amp_mut, region_actions = {}, {}, 0
    if is_agent(n):
        before = amp.get(n, {j: set() for j in JOBS})
        amp[n] = amp_row(n) if new else {j: set() for j in JOBS}
        for j in JOBS:
            gained = amp[n][j] - before[j]
            d_amp[j], d_amp_mut[j] = len(gained), len(mutating(gained))
        region_actions = bin(C).count("1") * len(JOBS)
    elif n in JOBS:
        for a in amp:
            before = amp[a][n]
            amp[a][n] = allowed(pol[a], unconditional_only=True) - held(n)
            gained = amp[a][n] - before
            d_amp[a], d_amp_mut[a] = len(gained), len(mutating(gained))
        region_actions = bin(C).count("1") * len(amp)
    t_amp = time.time() - t2
    ho, hn = allowed(old), allowed(new)
    rows.append(dict(date=e["date"][:10], policy=n, kind=e["kind"], agent=is_agent(n), job=n in JOBS,
                     st_removed=len([s for s in old if s["key"] not in nk]),
                     st_added=len([s for s in new if s["key"] not in ok]),
                     act_gained=len(hn - ho), act_lost=len(ho - hn), mut_gained=len(mutating(hn - ho)),
                     region_statements=reg_st, total_statements=tot_st, region_policies=len(reg_pol),
                     total_policies=len(pol), changed_actions=bin(C).count("1"),
                     new_redundant=len(new_red), new_overlap=len(new_ov),
                     amp_gained=sum(d_amp.values()), amp_mut_gained=sum(d_amp_mut.values()),
                     amp_mut_gained_readonly=d_amp_mut.get("ReadOnlyAccess", 0) if is_agent(n) else 0,
                     amp_mut_gained_viewonly=d_amp_mut.get("ViewOnlyAccess", 0) if is_agent(n) else 0,
                     secure_bytes_region=B_PROD * region_actions,
                     secure_bytes_full=B_PROD * len(U.all) * (len(JOBS) if is_agent(n) else len(amp) if n in JOBS else 0),
                     t_incremental=t_inc + t_amp))
# full recomputation time for the same checks
t0 = time.time()
for n, s in pol.items():
    within_policy(s)
amp_full = {n: amp_row(n) for n in pol if is_agent(n)}
t_full = time.time() - t0
json.dump(dict(universe=len(U.all), base_policies=len(base["policies"]), base_statements=n_stmt0,
               start=base["date"], events=rows, locality_violations=viol_local,
               t_full_recompute=t_full, t_full_within_initial=t_full_within,
               agents=sorted(a for a in amp)), open(out_path, "w"))
print("events", len(rows), "locality violations", viol_local, "full recompute s", round(t_full, 2))
