"""E3: what tool descriptions in the wild let an owner declare (MCP Atlas Census, CC0).
Classes come from the input schema, not from wording:
  open-ended         a string parameter that carries code, SQL, a shell command, an expression or a URL
                     chosen at call time (the resources touched cannot be declared in advance);
  resource-addressed a parameter that names a resource (identifier, path, table, bucket, repository, ...);
  other              neither (free-text search, fixed computations, no parameters).
Usage: python3 -I scripts/census_tools.py <tools.gz> out.json"""
import gzip, json, re, sys, collections, random
OPEN = {"sql", "code", "command", "cmd", "script", "shell", "expression", "statement", "javascript", "python",
        "url", "uri", "endpoint", "query_sql", "sql_query", "raw_sql", "js", "bash", "program", "source_code"}
RES = re.compile(r"(_id$|^id$|^ids$|_ids$|path|file|table|database|bucket|repo|resource|document|channel|"
                 r"project|account|folder|key$|_key$|record|sheet|issue|ticket|page_id|user_id|email)", re.I)
cnt, srv = collections.Counter(), collections.defaultdict(set)
ex = collections.defaultdict(list)
random.seed(1)
with gzip.open(sys.argv[1], "rt") as f:
    for ln in f:
        r = json.loads(ln)
        props = ((r.get("input_schema") or {}).get("properties") or {})
        names = {k.lower() for k, v in props.items() if isinstance(v, dict) and v.get("type") in ("string", None)}
        allnames = {k.lower() for k in props}
        c = "open-ended" if names & OPEN else "resource-addressed" if any(RES.search(k) for k in allnames) else "other"
        cnt[c] += 1; srv[r.get("server_id")].add(c)
        if random.random() < 0.002 and len(ex[c]) < 30:
            ex[c].append((r.get("tool_name"), sorted(allnames)[:6]))
out = dict(tools=sum(cnt.values()), servers=len(srv), classes=dict(cnt),
           servers_with_open=sum("open-ended" in s for s in srv.values()),
           servers_only_declarable=sum("open-ended" not in s for s in srv.values()), examples=ex)
json.dump(out, open(sys.argv[2], "w"), indent=1)
print({k: v for k, v in out.items() if k != "examples"})
for x in ex["open-ended"][:20]:
    print(x)
