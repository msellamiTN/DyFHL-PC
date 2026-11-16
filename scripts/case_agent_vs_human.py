"""Case study on real policies: privilege amplification between human job functions and
AI-agent service policies in the AWS managed IAM policy corpus.

Question: if a person holding job-function policy J uses an agent that runs under service
policy A, which actions does the agent hold that the person does not?
By Theorem 1 the same set is the access gap if the agent ran with the person's identity.

The count is a sound LOWER bound at action level:
  agent side  = actions allowed by UNCONDITIONAL Allow statements, minus anything any Deny mentions;
  person side = actions matched by ANY Allow statement (conditional or resource-specific counted as held),
                Deny statements ignored.
Assumption (the only constructed part): every holder of J may invoke the agent.

Usage: python3 -I scripts/case_agent_vs_human.py <policy dir> [out.json]
"""
import json, os, sys, fnmatch, re

JOBS = ["ViewOnlyAccess", "ReadOnlyAccess", "SupportUser", "Billing", "SecurityAudit", "DataScientist",
        "DatabaseAdministrator", "NetworkAdministrator", "SystemAdministrator", "PowerUserAccess",
        "AdministratorAccess"]
AGENTS = ["AIDevOpsAgentAccessPolicy", "AIDevOpsAgentActionsPolicy", "AWSSecurityAgentServiceRolePolicy",
          "AWSSecurityAgentContinuousPentestPolicy", "AWSFinOpsAgentServiceRolePolicy", "FinOpsAgentAgentPolicy",
          "AWSTransformLandingZoneAgentPolicy", "AWSTransformNetworkMigrationAgentPolicy",
          "AWSTransformServerMigrationAgentPolicy", "AWSTransformSecurityAgentExecutorAccess",
          "AWSWellArchitectedAgentOrganizationsServiceRolePolicy",
          "AWSWellArchitectedAgentResourceScanningServiceRolePolicy", "WellArchitectedAgentResourceScanning",
          "AWSMarketplaceAgenticCommerceServiceRolePolicy", "SageMakerStudioBedrockAgentServiceRolePolicy"]
READ = re.compile(r"^(get|list|describe|view|search|lookup|batchget|query|scan|select|read|check|"
                  r"validate|estimate|preview|simulate|test|detect|count|is|has|show|retrieve|discover)", re.I)


def as_list(x):
    return x if isinstance(x, list) else [x]


def statements(root, name):
    doc = json.load(open(os.path.join(root, name), encoding="utf-8"))
    doc = doc.get("PolicyVersion", doc).get("Document", doc)
    out = []
    for s in as_list(doc["Statement"]):
        neg = "NotAction" in s
        pats = [a.lower() for a in as_list(s.get("NotAction" if neg else "Action", [])) if isinstance(a, str)]
        out.append(dict(effect=s.get("Effect"), neg=neg, pats=pats, cond=bool(s.get("Condition"))))
    return out


def hit(st, action):
    m = any(fnmatch.fnmatchcase(action, p) for p in st["pats"])
    return (not m) if st["neg"] else m


def main(root, out_path=None):
    vocab = set()
    for name in os.listdir(root):
        try:
            for s in statements(root, name):
                vocab.update(p for p in s["pats"] if "*" not in p and "?" not in p and ":" in p)
        except Exception:
            pass
    vocab = sorted(vocab)
    jobs = {j: [s for s in statements(root, j) if s["effect"] == "Allow"] for j in JOBS}
    res = {"vocabulary": len(vocab), "agents": {}}
    print(f"action vocabulary {len(vocab)}")
    print(f"{'agent policy':58s} {'held':>5s} " + " ".join(f"{j[:9]:>9s}" for j in JOBS))
    for a in AGENTS:
        st = statements(root, a)
        allow = [s for s in st if s["effect"] == "Allow" and not s["cond"]]
        deny = [s for s in st if s["effect"] == "Deny"]
        held = [x for x in vocab if any(hit(s, x) for s in allow) and not any(hit(s, x) for s in deny)]
        row = {"statements": len(st), "conditional_allow": sum(s["effect"] == "Allow" and s["cond"] for s in st),
               "held": len(held), "held_mutating": sum(not READ.match(x.split(":")[1]) for x in held), "amp": {}}
        for j, js in jobs.items():
            amp = [x for x in held if not any(hit(s, x) for s in js)]
            row["amp"][j] = {"n": len(amp), "mutating": sum(not READ.match(x.split(":")[1]) for x in amp),
                             "examples": amp[:5]}
        res["agents"][a] = row
        print(f"{a:58s} {len(held):5d} " + " ".join(f"{row['amp'][j]['n']:9d}" for j in JOBS))
    tot = {j: sum(r["amp"][j]["n"] for r in res["agents"].values()) for j in JOBS}
    mut = {j: sum(r["amp"][j]["mutating"] for r in res["agents"].values()) for j in JOBS}
    nz = {j: sum(r["amp"][j]["n"] > 0 for r in res["agents"].values()) for j in JOBS}
    held = sum(r["held"] for r in res["agents"].values())
    res["totals"] = dict(held=held, amplified=tot, amplified_mutating=mut, agents_with_amplification=nz)
    print(f"{'TOTAL':58s} {held:5d} " + " ".join(f"{tot[j]:9d}" for j in JOBS))
    print(f"{'of which mutating (heuristic)':58s} {'':5s} " + " ".join(f"{mut[j]:9d}" for j in JOBS))
    print(f"{'agent policies with any amplification (of ' + str(len(AGENTS)) + ')':58s} {'':5s} " + " ".join(f"{nz[j]:9d}" for j in JOBS))
    assert tot["AdministratorAccess"] == 0, "sanity: nothing can exceed full administrator access"
    if out_path:
        json.dump(res, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
