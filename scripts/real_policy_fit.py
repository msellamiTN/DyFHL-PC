"""RQ3 (expressiveness) on real policies: how much of a public corpus of AWS managed IAM
policies fits the attribute-set rule model, and what the exact checks find on the fitting part.

Usage: python3 -I scripts/real_policy_fit.py <dir with policy JSON files> [out.json]
The corpus is untrusted data: it is only parsed as JSON.
"""
import json, os, sys, fnmatch, collections, itertools


def as_list(x):
    return x if isinstance(x, list) else [x]


def load(root):
    out = []
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if not os.path.isfile(p):
            continue
        try:
            doc = json.load(open(p, encoding="utf-8"))
            doc = doc.get("PolicyVersion", doc).get("Document", doc)
            out.append((name, as_list(doc["Statement"])))
        except Exception:
            continue
    return out


def kind(pat):
    if pat == "*":
        return "all"
    if "*" not in pat and "?" not in pat:
        return "exact"
    body = pat.split(":", 1)[-1]
    if body == "*":
        return "service_all"
    if body.endswith("*") and "*" not in body[:-1] and "?" not in body:
        return "prefix"
    return "pattern"


SET_OPS = ("StringEquals", "StringEqualsIgnoreCase", "ArnEquals", "Bool")


def cond_class(cond):
    """'set' if every operator is an equality on literal values (reducible to an attribute set)."""
    for op in cond:
        base = op.split(":")[-1].replace("IfExists", "")
        if base not in SET_OPS:
            return "other"
    return "set"


def main(root, out_path=None):
    pols = load(root)
    st = [(n, s) for n, ss in pols for s in ss if isinstance(s, dict)]
    c = collections.Counter()
    ops = collections.Counter()
    for _, s in st:
        c["statements"] += 1
        c["effect_" + s.get("Effect", "?")] += 1
        neg = "NotAction" in s or "NotResource" in s or "NotPrincipal" in s
        cond = s.get("Condition")
        acts = [a for a in as_list(s.get("Action", s.get("NotAction", []))) if isinstance(a, str)]
        ress = [r for r in as_list(s.get("Resource", s.get("NotResource", ["*"]))) if isinstance(r, str)]
        ak = {kind(a) for a in acts}
        rk = {kind(r) for r in ress}
        c["has_condition"] += bool(cond)
        c["has_negation"] += neg
        c["action_pattern_general"] += "pattern" in ak
        c["resource_specific"] += rk != {"all"}
        if cond:
            for op in cond:
                ops[op.split(":")[-1].replace("IfExists", "")] += 1
        if neg:
            level = "negation"
        elif cond and cond_class(cond) == "other":
            level = "condition_non_set"
        elif cond:
            level = "condition_set"
        elif "pattern" in ak or "pattern" in rk or "prefix" in ak or "prefix" in rk:
            level = "pattern_reducible"
        else:
            level = "direct"
        c["level_" + level] += 1
        s["_level"] = level

    # ---- exact checks on unconditional, non-negated statements (levels direct + pattern_reducible)
    vocab = sorted({a.lower() for _, s in st for a in as_list(s.get("Action", [])) if isinstance(a, str) and kind(a) == "exact"})
    by_service = collections.defaultdict(list)
    for a in vocab:
        by_service[a.split(":")[0]].append(a)
    cache = {}

    def expand(pat):
        pat = pat.lower()
        if pat not in cache:
            if pat == "*":
                cache[pat] = frozenset(vocab)
            elif "*" not in pat and "?" not in pat:
                cache[pat] = frozenset([pat])
            else:
                svc = pat.split(":")[0]
                cand = by_service.get(svc, []) if "*" not in svc else vocab
                cache[pat] = frozenset(a for a in cand if fnmatch.fnmatchcase(a, pat))
        return cache[pat]

    def res_cover(r1, r2):      # every resource matching r1 matches r2 (syntactic, sound)
        return r2 == "*" or r1 == r2 or ("*" not in r1 and "?" not in r1 and fnmatch.fnmatchcase(r1, r2))

    def res_overlap(r1, r2):    # sound over-approximation
        return r1 == "*" or r2 == "*" or r1 == r2 or fnmatch.fnmatchcase(r1, r2) or fnmatch.fnmatchcase(r2, r1)

    def cond_sets(cond):
        """Equality conditions as attribute sets. The attribute is (operator, key), so two rules
        are compared on a condition only when they use the same operator on the same key."""
        out = {}
        for op, kv in (cond or {}).items():
            for k, v in kv.items():
                out[(op, k.lower())] = frozenset(str(x) for x in as_list(v))
        return out

    def c_sub(c1, c2):          # condition of rule 1 at least as strict as rule 2 (Lemma 2)
        return all(k in c1 and c1[k] <= v for k, v in c2.items())

    def c_overlap(c1, c2):      # Lemma 1 on shared condition attributes
        return all(c1[k] & c2[k] for k in c1.keys() & c2.keys())

    rules = collections.defaultdict(list)
    n_cond = 0
    for n, s in st:
        if s["_level"] in ("direct", "pattern_reducible", "condition_set") and "Action" in s:
            acts = frozenset().union(*[expand(a) for a in as_list(s["Action"]) if isinstance(a, str)])
            ress = [r for r in as_list(s.get("Resource", "*")) if isinstance(r, str)]
            if acts:
                cs = cond_sets(s.get("Condition"))
                n_cond += bool(cs)
                rules[n].append((s.get("Effect"), acts, ress, cs))
    red = conf = pol_red = pol_conf = red_c = conf_c = 0
    for n, rs in rules.items():
        r_here = c_here = 0
        for (e1, a1, r1, c1), (e2, a2, r2, c2) in itertools.permutations(rs, 2):
            if e1 == e2 and a1 <= a2 and all(any(res_cover(x, y) for y in r2) for x in r1) and c_sub(c1, c2):
                r_here += 1; red_c += bool(c1 or c2)
        for (e1, a1, r1, c1), (e2, a2, r2, c2) in itertools.combinations(rs, 2):
            if e1 != e2 and a1 & a2 and any(res_overlap(x, y) for x in r1 for y in r2) and c_overlap(c1, c2):
                c_here += 1; conf_c += bool(c1 or c2)
        red += r_here; conf += c_here; pol_red += r_here > 0; pol_conf += c_here > 0
    rules_u = {n: [(e, a, r) for e, a, r, c in rs if not c] for n, rs in rules.items()}
    denies = [(n, a, r) for n, rs in rules_u.items() for e, a, r in rs if e == "Deny"]
    allows = [(n, a, r) for n, rs in rules_u.items() for e, a, r in rs if e == "Allow"]
    cross = set()
    for nd, ad, rd in denies:
        for na, aa, ra in allows:
            if na != nd and ad & aa and any(res_overlap(x, y) for x in rd for y in ra):
                cross.add((nd, na))

    res = dict(policies=len(pols), counts=dict(c), condition_operators=dict(ops.most_common(12)),
               action_vocabulary=len(vocab), analysed_policies=len(rules),
               analysed_statements=sum(len(v) for v in rules.values()),
               analysed_with_condition=n_cond, intra_redundant_pairs=red, redundant_pairs_involving_condition=red_c, policies_with_redundancy=pol_red,
               intra_conflict_pairs=conf, conflict_pairs_involving_condition=conf_c, policies_with_conflict=pol_conf,
               unconditional_deny_statements=len(denies),
               cross_policy_deny_allow_pairs=len(cross),
               policies_hit_by_some_deny=len({na for _, na in cross}))
    if out_path:
        json.dump(res, open(out_path, "w"), indent=1)
    n = c["statements"]
    print(f"policies {len(pols)}  statements {n}")
    for k in ("direct", "pattern_reducible", "condition_set", "condition_non_set", "negation"):
        print(f"  level {k:18s} {c['level_'+k]:6d}  {100*c['level_'+k]/n:5.1f}%")
    for k in ("effect_Allow", "effect_Deny", "has_condition", "has_negation", "action_pattern_general", "resource_specific"):
        print(f"  {k:24s} {c[k]:6d}  {100*c[k]/n:5.1f}%")
    print("  condition operators:", dict(ops.most_common(8)))
    for k in ("action_vocabulary", "analysed_policies", "analysed_statements", "analysed_with_condition", "intra_redundant_pairs", "redundant_pairs_involving_condition",
              "policies_with_redundancy", "intra_conflict_pairs", "conflict_pairs_involving_condition", "policies_with_conflict",
              "unconditional_deny_statements", "cross_policy_deny_allow_pairs", "policies_hit_by_some_deny"):
        print(f"  {k:32s} {res[k]}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
