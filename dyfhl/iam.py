"""AWS IAM policy documents as attribute-set rules (hyperedges over action vertices).

A statement becomes a rule whose action attribute is the set of concrete actions it names, after expanding
patterns against an action universe collected from the corpus itself (so expansion is a lower bound for
wildcards that name actions never written explicitly in any policy). Resources and conditions are kept as
canonical strings; they are used only for the within-policy checks, never to widen an action set."""
import fnmatch, json, re

READ = re.compile(r"^(get|list|describe|view|search|lookup|batchget|query|scan|select|read|check|"
                  r"validate|estimate|preview|simulate|test|detect|count|is|has|show|retrieve|discover)", re.I)
JOBS = ["ViewOnlyAccess", "ReadOnlyAccess", "SupportUser", "Billing", "SecurityAudit", "DataScientist",
        "DatabaseAdministrator", "NetworkAdministrator", "SystemAdministrator", "PowerUserAccess",
        "AdministratorAccess"]
AGENT_RE = re.compile(r"(Bedrock|AIDevOps|SecurityAgent|FinOpsAgent|Transform\w*Agent|WellArchitectedAgent|"
                      r"AgentRegistry|AgenticCommerce|AmazonQDeveloper|AmazonQFull|QBusiness|QuickSuite|"
                      r"SageMakerCanvasAI)")


def as_list(x):
    return [] if x is None else (x if isinstance(x, list) else [x])


def explicit_actions(doc):
    out = set()
    for s in as_list((doc or {}).get("Statement")):
        for a in as_list(s.get("Action")) + as_list(s.get("NotAction")):
            if "*" not in a and "?" not in a:
                out.add(a.lower())
    return out


class Universe:
    def __init__(self, actions):
        self.all = frozenset(actions)
        self.by_svc = {}
        for a in self.all:
            self.by_svc.setdefault(a.split(":")[0], set()).add(a)
        self.cache = {}

    def expand(self, pat):
        pat = pat.lower()
        if pat in self.cache:
            return self.cache[pat]
        if pat == "*":
            r = self.all
        elif "*" not in pat and "?" not in pat:
            r = frozenset([pat])
        else:
            svc = pat.split(":")[0]
            pool = self.by_svc.get(svc, ()) if "*" not in svc and "?" not in svc else self.all
            r = frozenset(a for a in pool if fnmatch.fnmatchcase(a, pat))
        self.cache[pat] = r
        return r


def statements(doc, U):
    """List of dicts: effect, actions (frozenset), res (tuple of patterns), cond (canonical string),
    conditional (bool), key (canonical JSON of the statement)."""
    out = []
    for s in as_list((doc or {}).get("Statement")):
        if "Action" in s:
            acts = frozenset().union(*[U.expand(a) for a in as_list(s["Action"])]) if as_list(s["Action"]) else frozenset()
        else:
            acts = U.all - frozenset().union(*[U.expand(a) for a in as_list(s.get("NotAction"))])
        res = tuple(sorted(as_list(s.get("Resource")) or ["*"])) if "Resource" in s else ("*",)
        if "NotResource" in s:
            res = ("<not>",) + tuple(sorted(as_list(s["NotResource"])))
        cond = s.get("Condition") or {}
        out.append(dict(effect=s.get("Effect", "Allow"), actions=acts, res=res,
                        cond=json.dumps(cond, sort_keys=True), conditional=bool(cond),
                        key=json.dumps(s, sort_keys=True)))
    return out


def allowed(stmts, unconditional_only=False):
    """Actions held: union of Allow statements minus actions of unconditional Deny statements on every resource."""
    a = set()
    for s in stmts:
        if s["effect"] == "Allow" and (not unconditional_only or (not s["conditional"] and s["res"] == ("*",))):
            a |= s["actions"]
    for s in stmts:
        if s["effect"] == "Deny" and not s["conditional"] and s["res"] == ("*",):
            a -= s["actions"]
    return a


def _res_cover(r1, r2):
    if "<not>" in r1 or "<not>" in r2:
        return r1 == r2
    return all(any(fnmatch.fnmatchcase(x, y) for y in r2) for x in r1)


def _res_overlap(r1, r2):
    if "<not>" in r1 or "<not>" in r2:
        return True
    return any(fnmatch.fnmatchcase(x, y) or fnmatch.fnmatchcase(y, x) for x in r1 for y in r2)


def _cond_items(c):
    d = json.loads(c)
    return {(op, k, json.dumps(v, sort_keys=True)) for op, kv in d.items() for k, v in kv.items()}


def within_policy(stmts):
    """Redundant pairs (i contained in j, same effect) and allow/deny overlaps inside one policy.
    Sound for redundancy (syntactic containment); overlap is an over-approximation on resources."""
    red, ov = set(), set()

    def contained(a, b):
        return a["effect"] == b["effect"] and a["actions"] and a["actions"] <= b["actions"] \
            and _res_cover(a["res"], b["res"]) and _cond_items(b["cond"]) <= _cond_items(a["cond"])
    for i, a in enumerate(stmts):
        for j, b in enumerate(stmts):
            if i == j:
                continue
            if contained(a, b) and (not contained(b, a) or i > j):
                red.add((a["key"], b["key"]))
            if i < j and a["effect"] != b["effect"] and a["actions"] & b["actions"] and _res_overlap(a["res"], b["res"]):
                ov.add(tuple(sorted((a["key"], b["key"]))))
    return red, ov


def mutating(actions):
    return {a for a in actions if not READ.match(a.split(":", 1)[1])}
