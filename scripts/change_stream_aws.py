"""Replay of a real change stream: every commit that changes an AWS managed IAM policy in the public
tracking repository is one change event. For each event we measure
  (1) the affected region: statements and policies that share an action vertex with a changed statement;
  (2) what an admission check finds: actions an agent policy gains that a human job function does not hold
      (privilege amplification, lower bound as in case_agent_vs_human.py) and new redundant or conflicting
      statement pairs inside the changed policy;
  (3) the cost of the check on the region against a full re-analysis.
The corpus is untrusted data: files are only parsed as JSON.

Usage: python3 -I scripts/change_stream_aws.py <git repo> <out.json>
"""
import json, subprocess, sys, time, fnmatch, re, collections, statistics as S

REPO, OUT = sys.argv[1], sys.argv[2]
JOBS = ["ViewOnlyAccess", "ReadOnlyAccess", "SupportUser", "Billing", "SecurityAudit", "DataScientist",
        "DatabaseAdministrator", "NetworkAdministrator", "SystemAdministrator", "PowerUserAccess",
        "AdministratorAccess"]
READ = re.compile(r"^(get|list|describe|view|search|lookup|batchget|query|scan|select|read|check|"
                  r"validate|estimate|preview|simulate|test|detect|count|is|has|show|retrieve|discover)", re.I)
IS_AGENT = re.compile(r"agent", re.I)


def git(*a):
    return subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True, check=True).stdout


def as_list(x):
    return x if isinstance(x, list) else [x]


def parse(text):
    try:
        doc = json.loads(text)
        doc = doc.get("PolicyVersion", doc).get("Document", doc)
        return [s for s in as_list(doc["Statement"]) if isinstance(s, dict)]
    except Exception:
        return None


commits = git("log", "--reverse", "--format=%H %ad", "--date=short", "--", "policies").split("\n")
commits = [c.split() for c in commits if c]
base, events_c = commits[0][0], commits[1:]
names = [n for n in git("ls-tree", "--name-only", f"{base}:policies").split("\n") if n]
state_raw = {}
for n in names:
    st = parse(git("show", f"{base}:policies/{n}"))
    if st is not None:
        state_raw[n] = st
changes = []
for h, d in events_c:
    for line in git("diff-tree", "--no-commit-id", "--name-status", "-r", h, "--", "policies").split("\n"):
        if not line:
            continue
        kind, path = line.split("\t")[0][0], line.split("\t")[-1]
        new = parse(git("show", f"{h}:{path}")) if kind != "D" else []
        if new is not None:
            changes.append((h, d, kind, path.split("/", 1)[1], new))

# ---- vocabulary over the whole history, actions as bit positions
vocab = set()
for sts in list(state_raw.values()) + [c[4] for c in changes]:
    for s in sts:
        for a in as_list(s.get("Action", s.get("NotAction", []))):
            if isinstance(a, str) and "*" not in a and "?" not in a and ":" in a:
                vocab.add(a.lower())
vocab = sorted(vocab)
pos = {a: i for i, a in enumerate(vocab)}
by_svc = collections.defaultdict(list)
for a in vocab:
    by_svc[a.split(":")[0]].append(a)
FULL = (1 << len(vocab)) - 1
MUT = sum(1 << pos[a] for a in vocab if not READ.match(a.split(":")[1]))
cache = {}


def mask(pat):
    pat = pat.lower()
    if pat not in cache:
        if pat == "*":
            m = FULL
        elif "*" not in pat and "?" not in pat:
            m = 1 << pos[pat] if pat in pos else 0
        else:
            svc = pat.split(":")[0]
            cand = by_svc.get(svc, []) if "*" not in svc and "?" not in svc else vocab
            m = sum(1 << pos[a] for a in cand if fnmatch.fnmatchcase(a, pat))
        cache[pat] = m
    return cache[pat]


def norm(sts):
    """Statements as (key, effect, action mask, conditional, resources, equality-condition sets or None)."""
    out = []
    for s in sts:
        neg = "NotAction" in s
        m = 0
        for a in as_list(s.get("NotAction" if neg else "Action", [])):
            if isinstance(a, str):
                m |= mask(a)
        if neg:
            m = FULL & ~m
        cond = s.get("Condition")
        cs = None
        if not cond:
            cs = {}
        elif all(op.split(":")[-1].replace("IfExists", "") in ("StringEquals", "StringEqualsIgnoreCase", "ArnEquals", "Bool")
                 for op in cond):
            cs = {(op, k.lower()): frozenset(str(x) for x in as_list(v)) for op, kv in cond.items() for k, v in kv.items()}
        ress = tuple(sorted(r for r in as_list(s.get("Resource", s.get("NotResource", "*"))) if isinstance(r, str)))
        fit = cs is not None and not neg and "NotResource" not in s
        out.append(dict(key=json.dumps({k: v for k, v in s.items() if k != "Sid"}, sort_keys=True), eff=s.get("Effect"),
                        m=m, cond=bool(cond), res=ress, cs=cs, fit=fit))
    return out


def held(st):          # unconditional Allow minus anything a Deny mentions (agent side, lower bound)
    a = d = 0
    for s in st:
        if s["eff"] == "Allow" and not s["cond"]:
            a |= s["m"]
        elif s["eff"] == "Deny":
            d |= s["m"]
    return a & ~d


def grant_any(st):     # any Allow (person side, upper bound)
    a = 0
    for s in st:
        if s["eff"] == "Allow":
            a |= s["m"]
    return a


def res_cover(r1, r2):
    return all(any(y == "*" or x == y or ("*" not in x and "?" not in x and fnmatch.fnmatchcase(x, y)) for y in r2) for x in r1)


def res_overlap(r1, r2):
    return any(x == "*" or y == "*" or x == y or fnmatch.fnmatchcase(x, y) or fnmatch.fnmatchcase(y, x) for x in r1 for y in r2)


def pair_findings(st, only=None):
    """Redundant and conflicting ordered/unordered pairs among fitting statements; if `only` is given, pairs
    that involve at least one statement whose key is in `only`."""
    red, conf, cmp_ = set(), set(), 0
    f = [s for s in st if s["fit"] and s["m"]]
    for i, a in enumerate(f):
        for j, b in enumerate(f):
            if i == j or (only is not None and a["key"] not in only and b["key"] not in only):
                continue
            cmp_ += 1
            if a["eff"] == b["eff"] and a["m"] & ~b["m"] == 0 and res_cover(a["res"], b["res"]) and \
                    all(k in a["cs"] and a["cs"][k] <= v for k, v in b["cs"].items()):
                red.add((a["key"], b["key"]))
            if i < j and a["eff"] != b["eff"] and a["m"] & b["m"] and res_overlap(a["res"], b["res"]) and \
                    all(a["cs"][k] & b["cs"][k] for k in a["cs"].keys() & b["cs"].keys()):
                conf.add((a["key"], b["key"]))
    return red, conf, cmp_


state = {n: norm(s) for n, s in state_raw.items()}
bits = lambda x: bin(x).count("1")


def full_analysis():
    t = time.perf_counter()
    cmp_ = 0
    for n, st in state.items():
        cmp_ += pair_findings(st)[2]
    jobs = {j: grant_any(state[j]) for j in JOBS if j in state}
    for n, st in state.items():
        if IS_AGENT.search(n):
            h = held(st)
            for j, g in jobs.items():
                _ = h & ~g
    return time.perf_counter() - t, cmp_


ev, full_t = [], []
for k, (h, d, kind, name, new_raw) in enumerate(changes):
    if k % 60 == 0:
        full_t.append(full_analysis())
    old = state.get(name, [])
    new = norm(new_raw)
    t0 = time.perf_counter()
    ok, nk = {s["key"] for s in old}, {s["key"] for s in new}
    added = [s for s in new if s["key"] not in ok]
    removed = [s for s in old if s["key"] not in nk]
    delta = 0
    for s in added + removed:
        delta |= s["m"]
    # affected region: statements and policies sharing an action vertex with a changed statement
    reg_st = reg_pol = tot_st = 0
    for n, st in state.items():
        if n == name:
            continue
        c = sum(1 for s in st if s["m"] & delta)
        reg_st += c
        reg_pol += c > 0
        tot_st += len(st)
    # admission check on the region
    r0, c0, _ = pair_findings(old, {s["key"] for s in removed}) if old else (set(), set(), 0)
    r1, c1, cmp_inc = pair_findings(new, {s["key"] for s in added})
    amp_new = amp_mut = amp_lost = 0
    is_agent, is_job = bool(IS_AGENT.search(name)), name in JOBS
    if is_agent and "ReadOnlyAccess" in state:
        g = grant_any(state["ReadOnlyAccess"])
        a_old, a_new = held(old) & ~g, held(new) & ~g
        amp_new, amp_mut, amp_lost = bits(a_new & ~a_old), bits(a_new & ~a_old & MUT), bits(a_old & ~a_new)
    elif is_job:
        g_old, g_new = grant_any(old), grant_any(new)
        for n, st in state.items():
            if IS_AGENT.search(n):
                hh = held(st)
                amp_new += bits((hh & ~g_new) & ~(hh & ~g_old))
                amp_lost += bits((hh & ~g_old) & ~(hh & ~g_new))
    t_inc = time.perf_counter() - t0
    wide = any(s["m"] == FULL or bits(s["m"]) > 0.25 * len(vocab) for s in added + removed)
    ev.append(dict(commit=h[:8], date=d, kind=kind, policy=name, agent=is_agent, job=is_job,
                   st_added=len(added), st_removed=len(removed), delta_actions=bits(delta),
                   granted_new=bits(held(new) & ~held(old)), granted_removed=bits(held(old) & ~held(new)),
                   region_statements=reg_st, total_statements=tot_st, region_policies=reg_pol,
                   total_policies=len(state) - (name in state), wide=wide,
                   new_redundant=len(r1 - r0), new_conflict=len(c1 - c0), comparisons=cmp_inc,
                   amp_new=amp_new, amp_new_mutating=amp_mut, amp_lost=amp_lost, seconds=t_inc))
    if kind == "D":
        state.pop(name, None)
    else:
        state[name] = new
full_t.append(full_analysis())

frac_s = [e["region_statements"] / e["total_statements"] for e in ev]
frac_p = [e["region_policies"] / e["total_policies"] for e in ev]
q = lambda v, p: sorted(v)[min(len(v) - 1, int(p * len(v)))]
ag = [e for e in ev if e["agent"]]
summ = dict(
    commits=len(events_c), events=len(ev), first=events_c[0][1], last=events_c[-1][1],
    kinds=dict(collections.Counter(e["kind"] for e in ev)), vocabulary=len(vocab),
    policies_final=len(state), statements_final=sum(len(v) for v in state.values()),
    events_no_statement_change=sum(e["st_added"] + e["st_removed"] == 0 for e in ev),
    region_statements_frac=dict(median=S.median(frac_s), p75=q(frac_s, .75), p90=q(frac_s, .9), max=max(frac_s),
                                mean=S.mean(frac_s)),
    region_policies_frac=dict(median=S.median(frac_p), p75=q(frac_p, .75), p90=q(frac_p, .9), max=max(frac_p),
                              mean=S.mean(frac_p)),
    events_region_below_1pct=sum(f < .01 for f in frac_s), events_region_below_10pct=sum(f < .10 for f in frac_s),
    events_wide=sum(e["wide"] for e in ev),
    events_granting_new_actions=sum(e["granted_new"] > 0 for e in ev),
    agent_events=len(ag), agent_policies_changed=len({e["policy"] for e in ag}),
    agent_events_with_new_amplification=sum(e["amp_new"] > 0 for e in ag),
    agent_events_with_new_mutating_amplification=sum(e["amp_new_mutating"] > 0 for e in ag),
    agent_new_amplified_actions=sum(e["amp_new"] for e in ag),
    agent_new_amplified_mutating=sum(e["amp_new_mutating"] for e in ag),
    agent_amplified_actions_removed=sum(e["amp_lost"] for e in ag),
    job_events=sum(e["job"] for e in ev),
    events_with_new_redundancy=sum(e["new_redundant"] > 0 for e in ev), new_redundant_pairs=sum(e["new_redundant"] for e in ev),
    events_with_new_conflict=sum(e["new_conflict"] > 0 for e in ev), new_conflict_pairs=sum(e["new_conflict"] for e in ev),
    incremental_seconds=dict(median=S.median(e["seconds"] for e in ev), p90=q([e["seconds"] for e in ev], .9),
                             max=max(e["seconds"] for e in ev)),
    full_seconds=dict(median=S.median(t for t, _ in full_t), samples=len(full_t)),
    incremental_comparisons_median=S.median(e["comparisons"] for e in ev),
    full_comparisons_median=S.median(c for _, c in full_t))
top = sorted(ag, key=lambda e: -e["amp_new_mutating"])[:8]
summ["largest_agent_events"] = [{k: e[k] for k in ("date", "policy", "st_added", "amp_new", "amp_new_mutating", "region_policies")} for e in top]
json.dump(dict(summary=summ, events=ev), open(OUT, "w"), indent=1)
print(json.dumps(summ, indent=1))
