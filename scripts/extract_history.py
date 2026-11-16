"""Extract change events from the AWS managed-policy history (MAMIP git repository).
Output: data/history/events.jsonl, one line per (commit, policy) change with the old and new document,
and data/history/base.json with every policy at the oldest commit of the fetched history."""
import json, subprocess, sys
repo, out = sys.argv[1], sys.argv[2]
def git(*a):
    return subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True).stdout
log = git("log", "--reverse", "--format=@%H %aI", "--name-status", "--", "policies").splitlines()
commits, cur = [], None
for ln in log:
    if ln.startswith("@"):
        h, d = ln[1:].split(" ", 1); cur = dict(h=h, date=d, ch=[]); commits.append(cur)
    elif ln.strip():
        st, f = ln.split("\t", 1); cur["ch"].append((st[0], f))
def doc(h, f):
    try:
        return json.loads(git("show", f"{h}:{f}"))["PolicyVersion"]["Document"]
    except Exception:
        return None
base_commit = commits[0]["h"]
base = {f.split("/")[-1]: doc(base_commit, f) for f in git("ls-tree", "--name-only", base_commit, "policies/").split()}
json.dump(dict(commit=base_commit, date=commits[0]["date"], policies=base), open(f"{out}/base.json", "w"))
n = 0
with open(f"{out}/events.jsonl", "w") as fo:
    prev = commits[0]["h"]
    for c in commits[1:]:
        for st, f in c["ch"]:
            name = f.split("/")[-1]
            old = doc(prev, f) if st != "A" else None
            new = doc(c["h"], f) if st != "D" else None
            if old == new:
                continue
            fo.write(json.dumps(dict(date=c["date"], commit=c["h"], policy=name, kind=st, old=old, new=new)) + "\n"); n += 1
        prev = c["h"]
print("base policies", len(base), "events", n, "first", commits[0]["date"], "last", commits[-1]["date"])
