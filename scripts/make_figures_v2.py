"""Data figures and generated numbers for the change-admission paper (paper2/).
Reads results/aws_history.json, results/federation.json, results/kill_demand_seed*.json, results/census_tools.json,
results/secure_exact.json and writes paper2/figures/*.pdf and paper2/generated/numbers.tex."""
import json, glob, os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from dyfhl import figstyle as fs

fs.apply()
F, G = "paper/figures", "paper/generated"
os.makedirs(F, exist_ok=True); os.makedirs(G, exist_ok=True)
C = fs.OI
N = {}

# ------------------------------------------------------------------ AWS history
aws = json.load(open("results/aws_history.json"))
R = aws["events"]
fr = np.array([r["region_statements"] / r["total_statements"] for r in R])
fp = np.array([r["region_policies"] / r["total_policies"] for r in R])
held = np.array([r["new_redundant"] > 0 or r["new_overlap"] > 0 or r["amp_mut_gained"] > 0 for r in R])
mut = np.array([r["mut_gained"] > 0 for r in R])
ag = [r for r in R if r["agent"]]
N.update(awsEvents=len(R), awsMod=sum(r["kind"] == "M" for r in R), awsAdd=sum(r["kind"] == "A" for r in R),
         awsBasePol=aws["base_policies"], awsBaseSt=aws["base_statements"], awsUniverse=aws["universe"],
         awsGain=sum(r["act_gained"] > 0 for r in R), awsLose=sum(r["act_lost"] > 0 for r in R),
         awsMut=int(mut.sum()), awsRegMed=f"{100 * np.median(fr):.1f}", awsRegMean=f"{100 * fr.mean():.1f}",
         awsRegNinety=f"{100 * np.percentile(fr, 90):.1f}", awsRegMax=f"{100 * fr.max():.1f}",
         awsRegUnderOne=f"{100 * (fr < .01).mean():.0f}", awsRegUnderFive=f"{100 * (fr < .05).mean():.0f}",
         awsPolMed=f"{100 * np.median(fp):.1f}", awsPolNinety=f"{100 * np.percentile(fp, 90):.1f}",
         awsLocViol=aws["locality_violations"],
         awsRedEv=sum(r["new_redundant"] > 0 for r in R), awsRedPairs=sum(r["new_redundant"] for r in R),
         awsOvEv=sum(r["new_overlap"] > 0 for r in R), awsOvPairs=sum(r["new_overlap"] for r in R),
         awsAgentEv=len(ag), awsJobEv=sum(r["job"] for r in R), awsAgentPol=len(aws["agents"]),
         awsAmpEv=sum(r["amp_gained"] > 0 for r in ag),
         awsAmpROEv=sum(r["amp_mut_gained_readonly"] > 0 for r in ag),
         awsAmpROAct=sum(r["amp_mut_gained_readonly"] for r in ag),
         awsAmpVOEv=sum(r["amp_mut_gained_viewonly"] > 0 for r in ag),
         awsHeld=int(held.sum()), awsHeldPct=f"{100 * held.mean():.1f}",
         awsChangedMed=f"{np.median([r['changed_actions'] for r in R]):.0f}",
         awsIncMs=f"{1e3 * np.median([r['t_incremental'] for r in R]):.2f}",
         awsIncNinetyNine=f"{1e3 * np.percentile([r['t_incremental'] for r in R], 99):.1f}",
         awsFull=f"{aws['t_full_recompute']:.2f}")
sb = np.array([r["secure_bytes_region"] for r in R if r["agent"] or r["job"]])
sf = np.array([r["secure_bytes_full"] for r in R if r["agent"] or r["job"]])
N.update(awsSecKBMed=f"{np.median(sb) / 1e3:.1f}", awsSecKBNinety=f"{np.percentile(sb, 90) / 1e3:.0f}",
         awsSecMBMax=f"{sb.max() / 1e6:.1f}", awsSecFullMBMed=f"{np.median(sf) / 1e6:.1f}")
top = max(ag, key=lambda r: r["amp_mut_gained_readonly"])
N.update(awsTopPol=top["policy"], awsTopDate=top["date"], awsTopAct=top["amp_mut_gained_readonly"])

fig, ax = plt.subplots(1, 3, figsize=fs.figsize("double", 52), constrained_layout=True)
for x, lab, col in ((fr, "statements", C["blue"]), (fp, "policies", C["vermillion"])):
    xs = np.sort(np.maximum(x, 1e-4)); ax[0].step(xs, np.arange(1, len(xs) + 1) / len(xs), where="post", color=col, label=lab)
ax[0].set_xscale("log"); ax[0].set_xlabel("share of corpus in the region"); ax[0].set_ylabel("share of change events")
ax[0].legend(loc="lower right"); ax[0].set_title("(a) Region of a change", loc="left")
months = sorted({r["date"][:7] for r in R})
cnt = lambda pred: [sum(1 for r, h, m in zip(R, held, mut) if r["date"][:7] == mo and pred(r, h, m)) for mo in months]
xi = np.arange(len(months))
ax[1].bar(xi, cnt(lambda r, h, m: True), color=C["lightgrey"], label="all changes")
ax[1].bar(xi, cnt(lambda r, h, m: m), color=C["sky"], label="add state-changing actions")
ax[1].bar(xi, cnt(lambda r, h, m: h), color=C["vermillion"], width=0.45, label="new finding")
ax[1].set_xticks(xi[::2]); ax[1].set_xticklabels([m[2:] for m in months][::2], rotation=45, ha="right")
ax[1].set_ylabel("change events"); ax[1].legend(loc="upper left", fontsize=6.5); ax[1].set_title("(b) Changes per month", loc="left")
agd = sorted(ag, key=lambda r: r["date"])
cum = np.cumsum([r["amp_mut_gained_readonly"] for r in agd])
days = np.array([np.datetime64(r["date"]) for r in agd])
ax[2].step(days, np.maximum(cum, 0.8), where="post", color=C["vermillion"])
ax[2].set_yscale("log"); ax[2].set_ylabel("state-changing actions")
ax[2].set_title("(c) Agent actions beyond read-only", loc="left")
ax[2].tick_params(axis="x", rotation=45)
import matplotlib.dates as md
ax[2].xaxis.set_major_formatter(md.DateFormatter("%y-%m"))
fs.save(fig, f"{F}/fig_aws", submission_dir="paper/figures_submission"); plt.close(fig)

# ------------------------------------------------------------------ federation
fed = json.load(open("results/federation.json"))
e1 = fed["e1"]
N.update(eOneEvents=sum(x["events"] for x in e1), eOneMism=sum(x["mismatches"] for x in e1),
         eOneSP=sum(x["secure_pairwise"] for x in e1), eOneSPok=sum(x["secure_pairwise_agree"] for x in e1),
         eOneSC=sum(x["secure_cross"] for x in e1), eOneSCok=sum(x["secure_cross_agree"] for x in e1))
H = fed["held"]
by = collections.defaultdict(list)
for h in H:
    by[h["kind"]].append(h)
FT = ("conf", "red", "gap", "amp", "use", "drift")
order = ("add-data", "add-tool", "add-usage", "del", "redescribe", "approve")
with open(f"{G}/tab_held.tex", "w") as f:
    for k in order:
        v = by[k]
        cells = [f"{100 * np.mean([x[t] > 0 for x in v]):.0f}" for t in FT]
        anyp = 100 * np.mean([any(x[t] > 0 for t in FT) for x in v])
        f.write(f"{k.replace('add-', 'add ').replace('del', 'delete rule').replace('redescribe', 'redescribe tool').replace('approve', 'approve tool')} & {len(v)} & " + " & ".join(cells) + f" & {anyp:.0f}\\\\\n")
N["heldAll"] = f"{100 * np.mean([any(x[t] > 0 for t in FT) for x in H]):.0f}"
g = collections.defaultdict(list)
for r in fed["e4"]:
    g[(r["K"], r["rules_per_org"])].append(r)
agg = {k: dict(rules=np.mean([x["rules"] for x in v]), full=np.mean([x["t_full"] for x in v]),
               med=np.mean([x["t_inc_median"] for x in v]), p99=np.mean([x["t_inc_p99"] for x in v]),
               mean=np.mean([x["t_inc_mean"] for x in v]), reg=np.mean([x["region_frac_median"] for x in v]),
               cells=np.mean([x["total_cells"] for x in v])) for k, v in g.items()}
with open(f"{G}/tab_scale.tex", "w") as f:
    for k in [(2, 200), (4, 200), (8, 200), (16, 200), (4, 50), (4, 800), (4, 3200)]:
        a = agg[k]
        f.write(f"{k[0]} & {k[1]} & {a['rules']:,.0f} & {a['cells']:.1e} & {a['full']:.2f} & {1e3 * a['med']:.2f} & "
                f"{1e3 * a['p99']:.1f} & {a['full'] / a['mean']:,.0f} & {100 * a['reg']:.3f}\\\\\n".replace("e+0", "e").replace("e+", "e"))
big = agg[(4, 3200)]
N.update(scBigRules=f"{big['rules']:,.0f}", scBigFull=f"{big['full']:.1f}", scBigMed=f"{1e3 * big['med']:.1f}",
         scBigPnn=f"{1e3 * big['p99']:.0f}", scBigSpeed=f"{big['full'] / big['mean']:,.0f}",
         scBigReg=f"{100 * big['reg']:.4f}")
fig, ax = plt.subplots(1, 2, figsize=fs.figsize("onehalf", 52), constrained_layout=True)
for i, (keys, xl) in enumerate(((([(4, 50), (4, 200), (4, 800), (4, 3200)]), "rules in the federation (4 organisations)"),
                                (([(2, 200), (4, 200), (8, 200), (16, 200)]), "rules (200 per organisation per layer)"))):
    x = [agg[k]["rules"] for k in keys]
    ax[i].plot(x, [agg[k]["full"] for k in keys], "o-", color=C["grey"], label="full recomputation")
    ax[i].plot(x, [agg[k]["p99"] for k in keys], "s--", color=C["vermillion"], label="admission, 99th pct.")
    ax[i].plot(x, [agg[k]["med"] for k in keys], "s-", color=C["blue"], label="admission, median")
    ax[i].set_xscale("log"); ax[i].set_yscale("log"); ax[i].set_xlabel(xl); ax[i].set_ylabel("seconds")
    for k, xx in zip(keys, x):
        if i == 1:
            ax[i].annotate(f"K={k[0]}", (xx, agg[k]["full"]), textcoords="offset points", xytext=(-4, 5), fontsize=6.5)
ax[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax[0].legend(loc="lower right"); ax[0].set_title("(a) Policy size", loc="left"); ax[1].set_title("(b) Organisations", loc="left")
fs.save(fig, f"{F}/fig_scaling", submission_dir="paper/figures_submission"); plt.close(fig)

with open(f"{G}/tab_secure.tex", "w") as f:
    for r in fed["e5"]:
        f.write(f"{r['K']} & {r['rules']:,} & {r['full_MB']:.1f} & {r['full_rounds']:,} & {1e3 * r['event_MB_median']:.1f} & "
                f"{1e3 * r['event_MB_mean']:.0f} & {1e3 * r['event_MB_p95']:.0f} & {r['event_rounds_median']:.0f} & "
                f"{r['full_MB'] / r['event_MB_mean']:,.0f}\\\\\n")
e5 = {(r["K"], r["rules_per_org"]): r for r in fed["e5"]}
N.update(secMidFull=f"{e5[(4, 200)]['full_MB']:.1f}", secMidMean=f"{1e3 * e5[(4, 200)]['event_MB_mean']:.0f}",
         secMidMed=f"{1e3 * e5[(4, 200)]['event_MB_median']:.1f}", secBigFull=f"{e5[(8, 200)]['full_MB']:.0f}",
         secBigMean=f"{1e3 * e5[(8, 200)]['event_MB_mean']:.0f}", secBigPnf=f"{1e3 * e5[(8, 200)]['event_MB_p95']:.0f}")
e6 = fed["e6"]
N.update(leakClasses=e6[0]["classes"], leakDenied=f"{np.mean([x['denied'] for x in e6]):.0f}",
         leakProbes=f"{np.mean([x['existence_probes'] for x in e6]):.0f}",
         leakEntropy=f"{np.mean([x['entropy_bits'] for x in e6]):.0f}", leakRes=len(e6))

# ------------------------------------------------------------------ static comparison on the earlier benchmarks
sx = json.load(open("results/secure_exact.json"))
rt = json.load(open("results/revision_tops.json"))
DS = {"agentic": "Agentic", "xu_stoller": "XS-ABAC", "amazon": "AMZ-Access"}
with open(f"{G}/tab_static.tex", "w") as f:
    for ds, nm in DS.items():
        xs = [x for x in sx if x["dataset"] == ds]
        mb = np.mean([x["online_MB"] for x in xs]); sec = np.mean([x["seconds_per_domain"] for x in xs])
        fp = {r["B"]: r["fpr"] for r in rt["fp"] if r["dataset"] == ds}
        mp = {r["B"]: r["mults_per_domain"] * 16 / 1e6 for r in rt["mpc"] if r["dataset"] == ds}
        f.write(f"{nm} & {fp[512]:.3f} & {mp[512]:.0f} & {fp[4096]:.3f} & {mp[4096]:.0f} & 0 & {mb:.1f} & {sec:.2f}\\\\\n")
N["exactRuns"] = len(sx)

# ------------------------------------------------------------------ learned ranking
acc = collections.defaultdict(lambda: collections.defaultdict(list))
for fn in sorted(glob.glob("results/kill_demand_seed*.json")):
    d = json.load(open(fn))
    for k, v in d.items():
        if isinstance(v, list):
            for m in ("auroc", "auroc_seen", "auroc_unseen", "auprc"):
                acc[k][m].append(np.mean([x[m] for x in v]))
with open(f"{G}/tab_rank.tex", "w") as f:
    for model, nm in (("count", "Exercise counts"), ("tabular", "Tabular (logistic)"), ("hyperedge", "Hyperedge model"),
                      ("hypergraph", "Hyperedge + propagation")):
        cells = []
        for reg in ("local", "federated", "pooled"):
            k = f"{model}/{reg}"
            if k in acc:
                cells.append(f"{np.mean(acc[k]['auroc']):.3f} & {np.mean(acc[k]['auroc_unseen']):.3f}")
            else:
                cells.append("-- & --")
        f.write(f"{nm} & " + " & ".join(cells) + "\\\\\n")
m = lambda k, s="auroc": np.mean(acc[k][s])
N.update(rkFed=f"{m('hyperedge/federated'):.3f}", rkCount=f"{m('count/local'):.3f}", rkPool=f"{m('hyperedge/pooled'):.3f}",
         rkFedNew=f"{m('hyperedge/federated', 'auroc_unseen'):.3f}", rkFedSeen=f"{m('hyperedge/federated', 'auroc_seen'):.3f}",
         rkCountSeen=f"{m('count/local', 'auroc_seen'):.3f}", rkSeeds=len(acc["count/local"]["auroc"]),
         rkProp=f"{m('hypergraph/federated'):.3f}")
fig, ax = plt.subplots(figsize=fs.figsize("single", 55), constrained_layout=True)
labs = [("count/local", "counts\nlocal"), ("tabular/pooled", "tabular\npooled"), ("hyperedge/local", "hyperedge\nlocal"),
        ("hyperedge/federated", "hyperedge\nfederated"), ("hyperedge/pooled", "hyperedge\npooled")]
x = np.arange(len(labs))
ax.bar(x - .2, [m(k, "auroc_seen") for k, _ in labs], .4, color=C["blue"], label="known resources")
ax.bar(x + .2, [m(k, "auroc_unseen") for k, _ in labs], .4, color=C["orange"], label="resources new to the organisation")
ax.axhline(.5, color=C["grey"], lw=.5, ls=":")
ax.set_xticks(x); ax.set_xticklabels([l for _, l in labs], fontsize=6.5); ax.set_ylim(.4, .85); ax.set_ylabel("AUROC")
ax.legend(loc="upper left", fontsize=6.5)
fs.save(fig, f"{F}/fig_rank", submission_dir="paper/figures_submission"); plt.close(fig)

# ------------------------------------------------------------------ census
cs = json.load(open("results/census_tools.json"))
N.update(csTools=f"{cs['tools']:,}", csServers=f"{cs['servers']:,}", csOpen=f"{cs['classes']['open-ended']:,}",
         csOpenPct=f"{100 * cs['classes']['open-ended'] / cs['tools']:.1f}", csRes=f"{cs['classes']['resource-addressed']:,}",
         csResPct=f"{100 * cs['classes']['resource-addressed'] / cs['tools']:.1f}",
         csSrvOpen=f"{cs['servers_with_open']:,}", csSrvOpenPct=f"{100 * cs['servers_with_open'] / cs['servers']:.0f}")

N = {k: (f"{v:,}" if isinstance(v, (int, np.integer)) and v >= 1000 else v) for k, v in N.items()}
with open(f"{G}/numbers.tex", "w") as f:
    for k, v in N.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(json.dumps(N, indent=0, default=str)[:3000])
