"""Run every experiment reported in the paper and write results/*.json.
Usage: python scripts/run_experiments.py [main ablation sketch noise scale stream] [--quick]"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from dyfhl.policy import World, get_source
from dyfhl.model import Config, TASKS
from dyfhl.fed import run, make_clients, evaluate, exchange
from dyfhl.baselines import VARIANTS, rf_baseline
from dyfhl.dynamic import run_stream

torch.set_num_threads(int(os.environ.get('DYFHL_THREADS', 2)))
QUICK = "--quick" in sys.argv
WHAT = [a for a in sys.argv[1:] if not a.startswith("--")] or ["main", "ablation", "sketch", "noise", "scale", "stream"]
DATASETS = ["agentic", "xu_stoller", "amazon"]
SEEDS = [0, 1] if QUICK else [0, 1, 2, 3, 4]
SEEDS3 = SEEDS[:3]
OUT = "results"
os.makedirs(OUT, exist_ok=True)


def slim(r):
    return {k: v for k, v in r.items()}


def save(name, rows):
    json.dump(rows, open(f"{OUT}/{name}.json", "w"), indent=1)
    print("saved", name, len(rows), flush=True)


def world(ds, seed, **kw):
    return World(get_source(ds), seed=seed, **kw)


def resume(name, keys, complete):
    """Rows of an interrupted run whose group (by keys) is complete; returns (rows, done-set)."""
    p = f"{OUT}/{name}.json"
    if not os.path.exists(p) or "--fresh" in sys.argv:
        return [], set()
    rows = json.load(open(p))
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    done = {g for g, rs in groups.items() if complete(rs)}
    print("resuming", name, "with", len(done), "complete groups", flush=True)
    return [r for g in done for r in groups[g]], done


def main():
    n_methods = 3 + len(VARIANTS)
    rows, done = resume("main", ("dataset", "seed"), lambda rs: len({r["method"] for r in rs}) == n_methods)
    for ds in DATASETS:
        for seed in SEEDS:
            if (ds, seed) in done:
                continue
            w = world(ds, seed); lab = w.labels()
            def add(method, r):
                rows.append(dict(dataset=ds, seed=seed, method=method, **slim(r)))
                print(ds, seed, method, round(r["macro_f1"], 3), round(r["macro_ap"], 3), flush=True)
            add("RF (20% audited labels)", rf_baseline(w, lab, seed, context=False))
            add("RF + count context (20% audited labels)", rf_baseline(w, lab, seed, context=True))
            for name, ov in VARIANTS.items():
                add(name, run(w, Config(seed=seed, **ov), labels=lab))
            add("DyFHL-PA-R", run(w, Config(seed=seed, h=0), labels=lab, rule=True))
            save("main", rows)


def load_rows(name):
    p = f"{OUT}/{name}.json"
    return json.load(open(p)) if os.path.exists(p) else []


def ablation():
    """Row-level resumable: every finished (dataset, seed, group, variant) is saved at once."""
    struct = {"full": {}, "- scope partition": dict(scope_part=False), "- conjunctive pooling": dict(conj=False),
              "- wildcard vertices": dict(wild=False), "- signature sketch (counts)": dict(sketch=False)}
    learned = {"full (learned decoder)": {}, "+ learned message channel (h=16)": dict(h=16),
               "+ hyperedge priors": dict(priors=True), "- FedAvg (local training)": dict(fedavg=False),
               "- pseudo-anomaly augmentation": dict(aug=0.0), "+ density matching (recomb=1.5)": dict(recomb=1.5)}
    rows = load_rows("ablation")
    have = {(r["dataset"], r["seed"], r["group"], r["variant"]) for r in rows}
    for ds in DATASETS:
        for seed in SEEDS3:
            todo = [("structure", n, o) for n, o in struct.items()] + \
                   ([] if ds == "xu_stoller" else [("learning", n, o) for n, o in learned.items()])
            todo = [t for t in todo if (ds, seed, t[0], t[1]) not in have]
            if not todo:
                continue
            w = world(ds, seed); lab = w.labels()
            for group, name, ov in todo:
                if group == "structure":
                    r = run(w, Config(seed=seed, h=0, **ov), labels=lab, rule=True)
                else:
                    r = run(w, Config(seed=seed, **ov), labels=lab)
                rows.append(dict(dataset=ds, seed=seed, group=group, variant=name, **slim(r)))
                save("ablation", rows)
                print(ds, seed, group, name, round(r["macro_f1"], 3), flush=True)


def sketch():
    rows = []
    for ds in DATASETS:
        for seed in SEEDS3:
            w = world(ds, seed); lab = w.labels()
            for B in [32, 64, 128, 256, 512, 1024, 2048, 4096, 16384]:
                r = run(w, Config(seed=seed, h=0, B=B), labels=lab, rule=True)
                rows.append(dict(dataset=ds, seed=seed, B=B, decoder="rule", **slim(r)))
                if B <= 1024 and ds != "xu_stoller":
                    r = run(w, Config(seed=seed, B=B), labels=lab)
                    rows.append(dict(dataset=ds, seed=seed, B=B, decoder="learned", **slim(r)))
                print(ds, seed, B, flush=True)
            save("sketch", rows)


def noise():
    """Row-level resumable."""
    out = os.environ.get("NOISE_OUT", "noise")                 # separate files let seeds run in parallel
    only = [int(x) for x in os.environ.get("NOISE_SEEDS", "").split(",") if x != ""]
    rows = [r for r in load_rows(out) if r.get("salted")]
    have = {(r["dataset"], r["seed"], r["sigma"], r["decoder"]) for r in rows}
    if out != "noise":
        have |= {(r["dataset"], r["seed"], r["sigma"], r["decoder"]) for r in load_rows("noise") if r.get("salted")}
    for ds in ["agentic", "amazon"]:
        for seed in (only or SEEDS3):
            todo = [(sg, dec) for sg in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5] for dec in ("rule", "learned (noise-aware)")
                    if (ds, seed, sg, dec) not in have]
            if not todo:
                continue
            w = world(ds, seed); lab = w.labels()
            for sg, dec in todo:
                if dec == "rule":
                    r = run(w, Config(seed=seed, h=0, sigma=sg), labels=lab, rule=True)
                else:
                    r = run(w, Config(seed=seed, sigma=sg), labels=lab)
                rows.append(dict(dataset=ds, seed=seed, sigma=sg, decoder=dec, A=w.A, salted=True, **slim(r)))
                save(out, rows)
                print(ds, seed, sg, dec, round(r["macro_f1"], 3), flush=True)


def scale():
    rows = []
    for seed in SEEDS3:
        for n in [100, 200, 400, 800, 1600]:
            w = world("agentic", seed, n_rules=n)
            t0 = time.perf_counter(); lab = w.labels(); t_or = time.perf_counter() - t0
            for B in [512, 4 * n]:
                cfg = Config(seed=seed, h=0, B=B)
                clients, _ = make_clients(w, cfg, lab)
                from dyfhl.model import DyFHL
                m = DyFHL(clients[0].geo, cfg)
                t0 = time.perf_counter(); r, _ = evaluate(clients, m, lab, cfg, rule=True); t_inf = time.perf_counter() - t0
                u, _ = clients[0].upload(m)
                rows.append(dict(seed=seed, n_rules=n, K=w.K, B=B, B_mode="fixed" if B == 512 else "scaled",
                                 oracle_s=t_or, dyfhl_s=t_inf, macro_f1=r["macro_f1"],
                                 nnz_bytes=int((u != 0).sum()) * 8, **{t: r[t]["f1"] for t in TASKS}))
            print("scale n", seed, n, flush=True)
        for K in [3, 6, 9, 12]:
            w = world("agentic", seed, K=K, S=K)
            lab = w.labels()
            r = run(w, Config(seed=seed, h=0), labels=lab, rule=True)
            rows.append(dict(seed=seed, n_rules=400, K=K, B=512, B_mode="domains", macro_f1=r["macro_f1"],
                             **{t: r[t]["f1"] for t in TASKS}))
            print("scale K", seed, K, flush=True)
        save("scale", rows)


def stream():
    rows, done = resume("stream", ("dataset", "seed", "churn"), lambda rs: len(rs) == 11)
    for ds in ["agentic", "amazon"]:
        for seed in SEEDS3:
            for churn in [0.05, 0.1, 0.2]:
                if (ds, seed, churn) in done:
                    continue
                w = world(ds, seed)
                for r in run_stream(w, Config(seed=seed, h=0), T=10, churn=churn):
                    rows.append(dict(dataset=ds, seed=seed, churn=churn, **r))
                print("stream", ds, seed, churn, flush=True)
            save("stream", rows)


def main_b4096():
    """Rule decoder at B = 4096 on the worlds of the main comparison (added to main.json)."""
    rows = [r for r in json.load(open(f"{OUT}/main.json")) if r["method"] != "DyFHL-PA-R (B=4096)"]
    for ds in DATASETS:
        for seed in SEEDS:
            w = world(ds, seed); lab = w.labels()
            r = run(w, Config(seed=seed, h=0, B=4096), labels=lab, rule=True)
            rows.append(dict(dataset=ds, seed=seed, method="DyFHL-PA-R (B=4096)", **slim(r)))
            print(ds, seed, "R4096", round(r["macro_f1"], 3), flush=True)
    save("main", rows)


def natural():
    """No injected anomalies: every label is a naturally occurring one."""
    rows = []
    for ds in DATASETS:
        for seed in SEEDS3:
            w = world(ds, seed, p_conf=0.0, p_red=0.0, p_irr=0.0); lab = w.labels()
            for B in [512, 4096, 16384]:
                r = run(w, Config(seed=seed, h=0, B=B), labels=lab, rule=True)
                rows.append(dict(dataset=ds, seed=seed, B=B, decoder="rule", **slim(r)))
            print("natural", ds, seed, flush=True)
            save("natural", rows)


if __name__ == "__main__":
    t = time.time()
    for name in WHAT:
        globals()[name]()
        print("done", name, round(time.time() - t), "s", flush=True)
