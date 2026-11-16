"""Kill tests 1-3 (first pass) on real access requests (Amazon employee access data).

Question: can an organisation forecast which (user class, resource) requests will occur,
better with federation than alone?  Demand forecasting is the common core of the
irrelevancy (type 3) and demand-gap (type 4) anomalies.

Organisations = the 8 largest ROLE_ROLLUP_2 units. Each keeps its request log private.
No timestamps exist in this data, so past/future is a random 50/50 split per organisation.
Test negatives are popularity-matched (drawn from the global resource marginal), so resource
popularity alone scores about 0.5.

Regimes: local / federated (FedAvg) / pooled.  Models: co-occurrence counts, hashed-cross
logistic regression (tabular), hyperedge scorer without and with one hypergraph propagation.
"""
import json, sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import SGDClassifier
from scipy.sparse import csr_matrix

UA = ["MGR_ID", "ROLE_DEPTNAME", "ROLE_TITLE", "ROLE_FAMILY_DESC", "ROLE_FAMILY"]
K, D, ROUNDS, NEG = 8, 32, 30, 4
torch.set_num_threads(2)


def load():
    d = pd.read_csv("data/raw/amazon_employee_access.csv")
    orgs = d["ROLE_ROLLUP_2"].value_counts().index[:K]
    d = d[d["ROLE_ROLLUP_2"].isin(orgs)].reset_index(drop=True)
    vid = {}
    for c in UA + ["RESOURCE"]:
        for v in d[c].unique():
            vid[(c, v)] = len(vid)
    X = np.stack([[vid[(c, v)] for v in d[c]] for c in UA], 1)      # user-attribute vertices
    R = np.array([vid[("RESOURCE", v)] for v in d["RESOURCE"]])     # resource vertex
    org = pd.Categorical(d["ROLE_ROLLUP_2"], categories=orgs).codes
    return X, R, org, len(vid)


def make_split(X, R, org, rng):
    res_all, cnt = np.unique(R, return_counts=True)
    p_glob = cnt / cnt.sum()
    out = []
    for k in range(K):
        idx = rng.permutation(np.where(org == k)[0])
        tr, te = idx[: len(idx) // 2], idx[len(idx) // 2:]
        pos = {(tuple(X[i]), R[i]) for i in idx}
        Xte, Rte = X[te], R[te]
        neg_r = rng.choice(res_all, size=len(te), p=p_glob)
        keep = np.array([(tuple(x), r) not in pos for x, r in zip(Xte, neg_r)])
        Xe = np.concatenate([Xte, Xte[keep]]); Re = np.concatenate([Rte, neg_r[keep]])
        ye = np.concatenate([np.ones(len(te)), np.zeros(keep.sum())])
        seen = np.isin(Re, np.unique(R[tr]))
        out.append(dict(Xtr=X[tr], Rtr=R[tr], Xe=Xe, Re=Re, ye=ye, seen=seen))
    return out


def train_negs(c, rng):
    r = rng.choice(c["Rtr"], size=len(c["Rtr"]) * NEG)             # local marginal only
    return np.repeat(c["Xtr"], NEG, 0), r


# ---------------------------------------------------------------- count baseline
def count_scores(trains, c):
    tab = {}
    for t in trains:
        for x, r in zip(t["Xtr"], t["Rtr"]):
            for a in x:
                tab[(a, r)] = tab.get((a, r), 0) + 1
    return np.array([sum(np.log1p(tab.get((a, r), 0)) for a in x) for x, r in zip(c["Xe"], c["Re"])])


# ---------------------------------------------------------------- tabular: hashed crosses
HB = 2 ** 18


def crosses(X, R):
    rows = np.repeat(np.arange(len(R)), X.shape[1] * 2)
    cross = (X * 1000003 + R[:, None] * 7919) % HB
    cols = np.concatenate([cross, X % HB], 1).ravel()
    return csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(R), HB))


def lr_scores(trains, c, rng):
    Xs, Rs, ys = [], [], []
    for t in trains:
        xn, rn = train_negs(t, rng)
        Xs += [t["Xtr"], xn]; Rs += [t["Rtr"], rn]; ys += [np.ones(len(t["Rtr"])), np.zeros(len(rn))]
    m = SGDClassifier(loss="log_loss", alpha=1e-6, max_iter=30, tol=None, random_state=0)
    m.fit(crosses(np.concatenate(Xs), np.concatenate(Rs)), np.concatenate(ys))
    return m.decision_function(crosses(c["Xe"], c["Re"]))


# ---------------------------------------------------------------- hyperedge scorer
class Scorer(torch.nn.Module):
    def __init__(s, n, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        s.E = torch.nn.Parameter(torch.randn(n, D, generator=g) * 0.1)
        s.b = torch.nn.Parameter(torch.full((n,), -3.0))  # rare-by-default prior, same in every regime

    def emb(s, P):
        return s.E if P is None else s.E + torch.sparse.mm(P, s.E)

    def forward(s, X, R, P=None):
        Z = s.emb(P)
        return (Z[X].sum(1) * Z[R]).sum(-1) + s.b[R]


def propagation(c, n):
    """One vertex->hyperedge->vertex mean step over the client's own past requests."""
    V = np.concatenate([c["Xtr"], c["Rtr"][:, None]], 1)
    m, w = V.shape
    H = csr_matrix((np.ones(m * w, np.float32), (np.repeat(np.arange(m), w), V.ravel())), shape=(m, n))
    deg = np.asarray(H.sum(0)).ravel(); deg[deg == 0] = 1
    P = (H.T @ H).multiply(1.0 / (w * deg[:, None])).tocoo()
    return torch.sparse_coo_tensor(np.vstack([P.row, P.col]), P.data.astype(np.float32), (n, n)).coalesce()


def fit(model, c, rng, epochs, P=None):
    opt = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=1e-6)
    for _ in range(epochs):
        xn, rn = train_negs(c, rng)
        X = torch.tensor(np.concatenate([c["Xtr"], xn])); R = torch.tensor(np.concatenate([c["Rtr"], rn]))
        y = torch.cat([torch.ones(len(c["Rtr"])), torch.zeros(len(rn))])
        for b in torch.randperm(len(y)).split(2048):
            opt.zero_grad()
            torch.nn.functional.binary_cross_entropy_with_logits(model(X[b], R[b], P), y[b]).backward()
            opt.step()


def score(model, c, P=None):
    with torch.no_grad():
        return model(torch.tensor(c["Xe"]), torch.tensor(c["Re"]), P).numpy()


def touched(c, n):
    w = np.zeros(n, np.float32)
    np.add.at(w, np.concatenate([c["Xtr"].ravel(), c["Rtr"]]), 1)
    return torch.tensor(w)


def fedavg(clients, n, seed, rng, props):
    glob = Scorer(n, seed)
    W = torch.stack([touched(c, n) for c in clients])               # per-vertex weights
    Wn = W / W.sum(0).clamp(min=1)
    for _ in range(ROUNDS):
        Es, bs = [], []
        for c, P in zip(clients, props):
            m = Scorer(n, seed); m.load_state_dict(glob.state_dict())
            fit(m, c, rng, 1, P)
            Es.append(m.E.data); bs.append(m.b.data)
        any_t = (W.sum(0) > 0)
        newE = (torch.stack(Es) * Wn[:, :, None]).sum(0); newb = (torch.stack(bs) * Wn).sum(0)
        glob.E.data = torch.where(any_t[:, None], newE, glob.E.data)
        glob.b.data = torch.where(any_t, newb, glob.b.data)
    return glob


def metrics(c, s):
    out = dict(auroc=roc_auc_score(c["ye"], s), auprc=average_precision_score(c["ye"], s))
    for name, mask in (("seen", c["seen"]), ("unseen", ~c["seen"])):
        y = c["ye"][mask]
        out["auroc_" + name] = roc_auc_score(y, s[mask]) if 0 < y.sum() < len(y) else float("nan")
    return out


def run(seed, X, R, org, n):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    cl = make_split(X, R, org, rng)
    pooled = dict(Xtr=np.concatenate([c["Xtr"] for c in cl]), Rtr=np.concatenate([c["Rtr"] for c in cl]))
    res = {}

    def add(name, scores):
        res[name] = [metrics(c, s) for c, s in zip(cl, scores)]

    add("count/local", [count_scores([c], c) for c in cl])
    add("count/pooled", [count_scores(cl, c) for c in cl])
    add("tabular/local", [lr_scores([c], c, rng) for c in cl])
    add("tabular/pooled", [lr_scores(cl, c, rng) for c in cl])
    for tag, use_p in (("hyperedge", False), ("hypergraph", True)):
        props = [propagation(c, n) if use_p else None for c in cl]
        loc = []
        for c, P in zip(cl, props):
            m = Scorer(n, seed); fit(m, c, rng, ROUNDS, P); loc.append(score(m, c, P))
        add(tag + "/local", loc)
        g = fedavg(cl, n, seed, rng, props)
        add(tag + "/federated", [score(g, c, P) for c, P in zip(cl, props)])
        Pp = propagation(pooled, n) if use_p else None
        m = Scorer(n, seed); fit(m, pooled, rng, ROUNDS, Pp)
        add(tag + "/pooled", [score(m, c, Pp) for c in cl])
    return res


if __name__ == "__main__":
    seeds = [int(a) for a in sys.argv[1:]] or [0]
    X, R, org, n = load()
    for s in seeds:
        t = time.time()
        r = run(s, X, R, org, n)
        json.dump(r, open(f"results/kill_demand_seed{s}.json", "w"))
        print(f"seed {s} done in {time.time()-t:.0f}s", flush=True)
        for k, v in r.items():
            print(f"  {k:22s} AUROC {np.mean([m['auroc'] for m in v]):.3f}  AUPRC {np.mean([m['auprc'] for m in v]):.3f}"
                  f"  seen {np.nanmean([m['auroc_seen'] for m in v]):.3f}  unseen {np.nanmean([m['auroc_unseen'] for m in v]):.3f}", flush=True)
