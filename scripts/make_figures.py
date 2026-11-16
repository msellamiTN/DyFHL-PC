"""Turn results/*.json into the paper's tables (LaTeX), figures (PDF) and number macros.
Every number in the manuscript is produced here from the result files."""
import json, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

RES, OUT = "results", "paper/generated"
os.makedirs(OUT, exist_ok=True)
TASKS = ["x-conflict", "x-redundancy", "x-irrelevancy", "x-gap"]
DS = {"agentic": "Agentic", "xu_stoller": "XS-ABAC", "amazon": "AMZ-Access"}
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dyfhl import figstyle as FS
FS.apply()

C = FS.METHOD
DSC = {"agentic": FS.OI["blue"], "xu_stoller": FS.OI["orange"], "amazon": FS.OI["green"]}
DSM = {"agentic": "o", "xu_stoller": "s", "amazon": "^"}
macros = {}


def load(name):
    p = f"{RES}/{name}.json"
    if not os.path.exists(p):
        return None
    rows = json.load(open(p))
    for r in rows:
        for t in TASKS:
            if isinstance(r.get(t), dict):
                r[t + "_f1"], r[t + "_ap"], r[t + "_prec"], r[t + "_rec"] = r[t]["f1"], r[t]["ap"], r[t]["prec"], r[t]["rec"]
                r[t + "_pos"], r[t + "_n"] = r[t]["pos"], r[t]["n"]
                del r[t]
            elif t in r:
                r[t + "_f1"] = r.pop(t)
        # a task without any positive instance in a run has no defined F1: exclude it from the averages
        if all((t + "_pos") in r for t in TASKS):
            for t in TASKS:
                if r[t + "_pos"] == 0:
                    r[t + "_f1"] = r[t + "_ap"] = r[t + "_prec"] = r[t + "_rec"] = float("nan")
            r["macro_f1"] = float(np.nanmean([r[t + "_f1"] for t in TASKS]))
            r["macro_ap"] = float(np.nanmean([r[t + "_ap"] for t in TASKS]))
    return pd.DataFrame(rows)


def ms(x, d=3):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return "--", "--"
    return f"{x.mean():.{d}f}", f"{x.std(ddof=1) if len(x) > 1 else 0:.{d}f}"


def cell(x, bold=False):
    m, s = ms(x)
    body = f"{m}\\,{{\\scriptsize$\\pm${s}}}"
    return f"\\textbf{{{m}}}\\,{{\\scriptsize$\\pm${s}}}" if bold else body


def mac(name, val):
    macros[name] = val


# ----------------------------------------------------------------------------- main table
def main_table():
    df = load("main")
    if df is None:
        return
    order = ["RF (20% audited labels)", "RF + count context (20% audited labels)", "Local-only", "FedGCN-clique",
             "FedHGNN-count", "DyFHL-PA-L", "DyFHL-PA-R", "DyFHL-PA-R (B=4096)"]
    short = {"RF (20% audited labels)": "RF, local features$^\\dagger$", "RF + count context (20% audited labels)":
             "RF + count context$^\\dagger$", "Local-only": "Local-only", "FedGCN-clique": "FedGCN (clique exp.)",
             "FedHGNN-count": "FedHGNN (count completion)", "DyFHL-PA-L": "\\textbf{DyFHL-PA-L} (learned decoder)",
             "DyFHL-PA-R": "\\textbf{DyFHL-PA-R} (rule decoder)", "DyFHL-PA-R (B=4096)": "\\textbf{DyFHL-PA-R}, $B=4096$"}
    cols = [t + "_f1" for t in TASKS] + ["macro_f1", "macro_ap"]
    lines = []
    for ds, dname in DS.items():
        sub = df[df.dataset == ds]
        if sub.empty:
            continue
        nseeds = sub.seed.nunique()
        best = {c: max(sub[sub.method == m][c].mean() for m in order if (sub.method == m).any()) for c in cols}
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{dname}}} ({nseeds} seeds)}}\\\\")
        for m in order:
            x = sub[sub.method == m]
            if x.empty:
                continue
            lines.append(short[m] + " & " + " & ".join(cell(x[c], abs(x[c].mean() - best[c]) < 1e-9) for c in cols) + "\\\\")
        lines.append("\\midrule")
        # numbers for the text
        key = ds.replace("_", "")
        for m, tag in [("DyFHL-PA-R", "R"), ("DyFHL-PA-R (B=4096)", "Rbig"), ("DyFHL-PA-L", "L"), ("FedHGNN-count", "H"), ("Local-only", "Loc"),
                       ("RF + count context (20% audited labels)", "RFc"), ("FedGCN-clique", "G")]:
            x = sub[sub.method == m]
            if not x.empty:
                mac(f"main{key}{tag}", ms(x.macro_f1)[0]); mac(f"main{key}{tag}ap", ms(x.macro_ap)[0])
                for t in TASKS:
                    mac(f"main{key}{tag}{t[2:5]}", ms(x[t + '_f1'])[0])
                    mac(f"main{key}{tag}{t[2:5]}rec", ms(x[t + '_rec'])[0]); mac(f"main{key}{tag}{t[2:5]}prec", ms(x[t + '_prec'])[0])
        # paired test: rule decoder vs best baseline
        base = [m for m in order if not m.startswith("DyFHL") and (sub.method == m).any()]
        bb = max(base, key=lambda m: sub[sub.method == m].macro_f1.mean())
        a = sub[sub.method == "DyFHL-PA-R"].sort_values("seed").macro_f1.values
        b = sub[sub.method == bb].sort_values("seed").macro_f1.values
        mac(f"main{key}best", f"{b.mean():.3f}")
        if len(a) == len(b) and len(a) > 2:
            p = stats.ttest_rel(a, b).pvalue
            mac(f"main{key}p", (f"{p:.2f}" if p >= 0.01 else f"{p:.1e}".replace("e-0", "\\times10^{-").replace("e-", "\\times10^{-") + "}"))
            mac(f"main{key}gain", f"{a.mean() - b.mean():+.3f}")
            print(key, "best baseline:", bb)
        a3 = sub[sub.method == "DyFHL-PA-R (B=4096)"].sort_values("seed").macro_f1.values
        if len(a3) == len(b) and len(b) > 2:
            p3 = stats.ttest_rel(a3, b).pvalue
            mac(f"main{key}bigp", (f"{p3:.2f}" if p3 >= 0.01 else f"{p3:.1e}".replace("e-0", "\\times10^{-").replace("e-", "\\times10^{-") + "}"))
            mac(f"main{key}biggain", f"{a3.mean() - b.mean():+.3f}")
        a2 = sub[sub.method == "DyFHL-PA-L"].sort_values("seed").macro_f1.values
        if len(a2) == len(a) and len(a) > 2:
            mac(f"main{key}pRL", f"{stats.ttest_rel(a, a2).pvalue:.3f}")
        st = sub[sub.method == "DyFHL-PA-R"]
        for t in TASKS:
            mac(f"prev{key}{t[2:5]}", f"{100 * st[t + '_pos'].mean():.1f}")
        mac(f"n{key}rules", f"{int(st['x-conflict_n'].mean())}"); mac(f"n{key}req", f"{int(st['x-gap_n'].mean())}")
        mac(f"cost{key}ctx", f"{st.context_nnz_bytes.mean() / 1024:.0f}")
        ll = sub[sub.method == "DyFHL-PA-L"]
        if not ll.empty:
            mac(f"cost{key}model", f"{ll.model_bytes.mean() / 1024:.0f}"); mac(f"time{key}L", f"{ll.train_s.mean():.0f}")
            mac(f"time{key}R", f"{st.infer_s.mean():.1f}")
    open(f"{OUT}/tab_main.tex", "w").write("\n".join(lines[:-1]) + "\n")


# ----------------------------------------------------------------------------- ablations
def ablation_tables():
    df = load("ablation")
    if df is None:
        return
    for group, fn in [("structure", "tab_abl_structure.tex"), ("learning", "tab_abl_learning.tex")]:
        sub = df[df.group == group]
        dss = [d for d in DS if (sub.dataset == d).any()]
        lines = []
        for v in sub.variant.unique():
            row = [v.replace("- ", "$-$ ").replace("+ ", "$+$ ").replace("%", "\\%")]
            for d in dss:
                x = sub[(sub.variant == v) & (sub.dataset == d)]
                row.append(cell(x.macro_f1) if len(x) else "--")
            x = sub[sub.variant == v]
            row += [f"{x[t + '_f1'].mean():.3f}" for t in TASKS]
            lines.append(" & ".join(row) + "\\\\")
            tag = "".join(ch for ch in v if ch.isalpha())[:14]
            mac(f"abl{group[:3]}{tag}", f"{x.macro_f1.mean():.3f}")
            for d in dss:
                xx = x[x.dataset == d]
                mac(f"abl{group[:3]}{tag}{d.replace('_', '')}", f"{xx.macro_f1.mean():.3f}")
            for t in TASKS:
                mac(f"abl{group[:3]}{tag}{t[2:5]}", f"{x[t + '_f1'].mean():.3f}")
        open(f"{OUT}/{fn}", "w").write("\n".join(lines) + "\n")
        mac(f"abl{group[:3]}ncols", str(len(dss)))


# ----------------------------------------------------------------------------- plotting helpers
def band(ax, x, m, sd, color, marker=None, ls="-", label=None, z=3):
    x, m, sd = np.asarray(x, float), np.asarray(m, float), np.nan_to_num(np.asarray(sd, float))
    ax.fill_between(x, m - sd, m + sd, color=color, alpha=0.18, lw=0, zorder=z - 1)
    return ax.plot(x, m, color=color, marker=marker, ls=ls, label=label, zorder=z, markeredgecolor="white")[0]


def panel(ax, label):
    ax.set_title(label, loc="left", fontsize=FS.BASE_PT, pad=2.5)
    ax.grid(True, axis="y", alpha=0.6)


def log2_axis(ax, ticks=(32, 128, 512, 2048, 8192)):
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(ticks)); ax.set_xticklabels([str(t) for t in ticks]); ax.minorticks_off()


def f1_axis(ax, label=True):
    ax.set_ylim(0, 1.03); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    if label:
        ax.set_ylabel("macro-F1")


# ----------------------------------------------------------------------------- sketch size
def sketch_fig():
    df = load("sketch")
    if df is None:
        return
    dss = [d for d in ["agentic", "xu_stoller", "amazon"] if (df.dataset == d).any()]
    fig, axes = plt.subplots(1, len(dss), figsize=FS.figsize("onehalf", 52), sharey=True, layout="constrained")
    for k, (ax, ds) in enumerate(zip(axes, dss)):
        for dec, col, mk, lab in [("rule", C["rule"], "o", "DyFHL-PA-R (rule decoder)"),
                                  ("learned", C["learned"], "s", "DyFHL-PA-L (learned decoder)")]:
            g = df[(df.dataset == ds) & (df.decoder == dec)].groupby("B").macro_f1.agg(["mean", "std"])
            if len(g):
                band(ax, g.index, g["mean"], g["std"], col, mk, label=lab)
        log2_axis(ax); f1_axis(ax, k == 0); ax.set_xlabel("sketch bins $B$")
        panel(ax, f"({'abc'[k]}) {DS[ds]}")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=2)
    FS.save(fig, f"{OUT}/fig_sketch"); plt.close(fig)
    for ds in ["agentic", "amazon", "xu_stoller"]:
        for dec in ["rule", "learned"]:
            g = df[(df.dataset == ds) & (df.decoder == dec)].groupby("B").macro_f1.mean()
            for B in g.index:
                mac(f"sk{ds}{dec}{'B'}{num(B)}", f"{g[B]:.3f}")
        g = df[(df.dataset == ds) & (df.decoder == "rule")].groupby("B")
        for B in g.context_nnz_bytes.mean().index:
            mac(f"sk{ds}bytesB{num(B)}", f"{g.context_nnz_bytes.mean()[B] / 1024:.0f}")
            mac(f"sk{ds}densekB{num(B)}", f"{g.context_bytes.mean()[B] / 1024:.0f}")
        for t in TASKS:
            gg = df[(df.dataset == ds) & (df.decoder == "rule")].groupby("B")
            for B in [512, 4096, 16384]:
                if B in gg[t + "_prec"].mean().index:
                    mac(f"sk{ds}{t[2:5]}fB{num(B)}", f"{gg[t + '_f1'].mean()[B]:.3f}")
                    mac(f"sk{ds}{t[2:5]}precB{num(B)}", f"{gg[t + '_prec'].mean()[B]:.3f}")
                    mac(f"sk{ds}{t[2:5]}recB{num(B)}", f"{gg[t + '_rec'].mean()[B]:.3f}")


def num(n):
    words = {32: "xxxii", 64: "lxiv", 128: "cxxviii", 256: "cclvi", 512: "dxii", 1024: "mxxiv", 2048: "mmxlviii",
             4096: "mmmmxcvi", 16384: "sixteenk"}
    return words.get(int(n), str(n))


# ----------------------------------------------------------------------------- exposure (privacy) vs utility
def leakage_fig():
    lk, sk = load("leakage"), load("sketch")
    if lk is None or sk is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=FS.figsize("onehalf", 52), layout="constrained")
    ax = axes[0]
    for ds in ["agentic", "xu_stoller", "amazon"]:
        g = lk[lk.dataset == ds].groupby("B")[["exposed", "predicted"]].agg(["mean", "std"])
        band(ax, g.index, g[("exposed", "mean")], g[("exposed", "std")], DSC[ds], DSM[ds], label=DS[ds])
    g = lk.groupby("B").predicted.mean()
    ax.plot(g.index, g.values, color="black", ls=(0, (2, 1.5)), lw=0.8, label="uniform-hash prediction", zorder=4)
    log2_axis(ax); ax.set_ylim(0, 1.03); ax.set_xlabel("sketch bins $B$"); ax.set_ylabel("foreign rules exposed")
    panel(ax, "(a) rules alone in their bin")
    ax.legend(loc="lower right", handlelength=1.6)
    ax = axes[1]
    m = sk[sk.decoder == "rule"].groupby(["dataset", "B"]).macro_f1.mean()
    e = lk.groupby(["dataset", "B"]).exposed.mean()
    for ds in ["agentic", "xu_stoller", "amazon"]:
        if ds not in m.index.get_level_values(0) or ds not in e.index.get_level_values(0):
            continue
        Bs = sorted(set(m.loc[ds].index) & set(e.loc[ds].index))
        x, y = [e.loc[(ds, B)] for B in Bs], [m.loc[(ds, B)] for B in Bs]
        ax.plot(x, y, color=DSC[ds], marker=DSM[ds], markeredgecolor="white", label=DS[ds])
    ax.set_xlim(0, 1.03); f1_axis(ax); ax.set_xlabel("foreign rules exposed")
    panel(ax, "(b) accuracy against exposure")
    FS.save(fig, f"{OUT}/fig_leakage"); plt.close(fig)
    for ds in ["agentic", "xu_stoller", "amazon"]:
        for B in sorted(lk.B.unique()):
            mac(f"lk{ds}B{num(B)}", f"{100 * e.loc[(ds, B)]:.0f}")
    for B in sorted(lk.B.unique()):
        mac(f"lkallB{num(B)}", f"{100 * lk[lk.B == B].exposed.mean():.0f}")


# ----------------------------------------------------------------------------- noise
def noise_fig():
    df = load("noise")
    if df is None:
        return
    delta = 1e-5
    style = {"rule": (C["rule"], "o", "DyFHL-PA-R (rule decoder)"),
             "learned (noise-aware)": (C["learned"], "s", "DyFHL-PA-L, trained with the same noise")}
    fig, axes = plt.subplots(1, 2, figsize=FS.figsize("onehalf", 56), sharey=True, layout="constrained")
    for k, (ax, ds) in enumerate(zip(axes, ["agentic", "amazon"])):
        for dec, (col, mk, lab) in style.items():
            g = df[(df.dataset == ds) & (df.decoder == dec)].groupby("sigma").macro_f1.agg(["mean", "std"])
            if len(g):
                band(ax, g.index, g["mean"], g["std"], col, mk, label=lab)
        f1_axis(ax, k == 0); ax.set_xlabel("noise standard deviation")
        ax.set_xlim(-0.01, 0.51)
        A = int(df[df.dataset == ds].A.iloc[0])
        top = ax.secondary_xaxis("top")
        ts = [0.1, 0.2, 0.3, 0.5]
        top.set_ticks(ts); top.set_xticklabels([f"{np.sqrt(2 * A) * np.sqrt(2 * np.log(1.25 / delta)) / t:.0f}" for t in ts])
        top.set_xlabel("rule-level $\\varepsilon$ per release", fontsize=FS.SMALL_PT, labelpad=2)
        top.tick_params(labelsize=FS.SMALL_PT, length=2)
        ax.text(0.98, 0.95, f"({'ab'[k]}) {DS[ds]}", transform=ax.transAxes, ha="right", va="top", fontsize=FS.BASE_PT)
        ax.grid(True, axis="y", alpha=0.6)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=2)
    FS.save(fig, f"{OUT}/fig_noise"); plt.close(fig)
    lines = []
    for sg in sorted(df.sigma.unique()):
        row = [f"{sg:g}"]
        for ds in ["agentic", "amazon"]:
            A = int(df[df.dataset == ds].A.iloc[0])
            eps = "$\\infty$" if sg == 0 else f"{np.sqrt(2 * A) * np.sqrt(2 * np.log(1.25 / delta)) / sg:.0f}"
            row.append(eps)
            for dec in ["rule", "learned (noise-aware)"]:
                x = df[(df.dataset == ds) & (df.decoder == dec) & (df.sigma == sg)]
                row.append(cell(x.macro_f1))
                mac(f"nz{ds}{'R' if dec == 'rule' else 'L'}s{str(sg).replace('.', 'p')}", ms(x.macro_f1)[0])
        lines.append(" & ".join(row) + "\\\\")
    open(f"{OUT}/tab_noise.tex", "w").write("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------- scalability
def scale_fig():
    df = load("scale")
    if df is None:
        return
    a = df[df.B_mode.isin(["fixed", "scaled"])].copy()
    a["total"] = a.n_rules * a.K
    fig, axes = plt.subplots(1, 3, figsize=FS.figsize("onehalf", 54), layout="constrained")
    g = a[a.B_mode == "fixed"].groupby("total")
    ax = axes[0]
    ax.plot(g.oracle_s.mean().index, g.oracle_s.mean(), color=C["base"], marker="^", markeredgecolor="white",
            label="pooled exact oracle")
    sty = [("fixed", C["rule"], "o", "DyFHL-PA-R, B = 512"), ("scaled", C["g"], "s", "DyFHL-PA-R, B = 4 × rules per domain")]
    for mode, col, mk, lab in sty:
        g = a[a.B_mode == mode].groupby("total")
        ax.plot(g.dyfhl_s.mean().index, g.dyfhl_s.mean(), color=col, marker=mk, markeredgecolor="white", label=lab)
        band(axes[1], g.macro_f1.mean().index, g.macro_f1.mean(), g.macro_f1.std(), col, mk)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("rules in shared scopes"); ax.set_ylabel("wall-clock time (s)")
    yt = [t for t in (0.001, 0.01, 0.1, 1, 10, 100) if a[["oracle_s", "dyfhl_s"]].min().min() / 3 <= t <= a[["oracle_s", "dyfhl_s"]].max().max() * 3]
    ax.set_yticks(yt); ax.set_yticklabels([f"{t:g}" for t in yt]); ax.minorticks_off()
    panel(ax, "(a) analysis time"); ax.grid(True, which="major", axis="y", alpha=0.6)
    axes[1].set_xscale("log"); axes[1].set_xlabel("rules in shared scopes"); f1_axis(axes[1]); panel(axes[1], "(b) accuracy vs. rules")
    for axx in axes[:2]:
        axx.set_xticks([600, 1200, 2400, 4800, 9600]); axx.set_xticklabels(["600", "1.2k", "2.4k", "4.8k", "9.6k"]); axx.minorticks_off()
    k = df[df.B_mode == "domains"].groupby("K").macro_f1.agg(["mean", "std"])
    band(axes[2], k.index, k["mean"], k["std"], C["rule"], "o")
    axes[2].set_xticks(list(k.index)); axes[2].set_xlabel("domains $K$"); f1_axis(axes[2]); panel(axes[2], "(c) accuracy vs. domains")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=3)
    FS.save(fig, f"{OUT}/fig_scale"); plt.close(fig)
    for mode in ["fixed", "scaled"]:
        g = a[a.B_mode == mode].groupby("total")
        for tot in g.macro_f1.mean().index:
            mac(f"sc{mode}f{num2(tot)}", f"{g.macro_f1.mean()[tot]:.3f}")
            mac(f"sc{mode}t{num2(tot)}", f"{g.dyfhl_s.mean()[tot]:.2f}")
            mac(f"sc{mode}o{num2(tot)}", f"{g.oracle_s.mean()[tot]:.2f}")
            mac(f"sc{mode}b{num2(tot)}", f"{g.nnz_bytes.mean()[tot] / 1024:.0f}")
    for K in k.index:
        mac(f"scK{num2(K)}", f"{k['mean'][K]:.3f}")


def num2(n):
    return "".join("abcdefghij"[int(c)] for c in str(int(n)))


# ----------------------------------------------------------------------------- stream
def stream_fig():
    df = load("stream")
    if df is None:
        return
    from matplotlib.lines import Line2D
    LS = {0.05: (0, (1.0, 1.2)), 0.1: (0, (3.5, 1.5)), 0.2: "-"}
    fig, axes = plt.subplots(1, 3, figsize=FS.figsize("onehalf", 56), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1, 0.85]})
    for k, (ax, ds) in enumerate(zip(axes, ["agentic", "amazon"])):
        if not (df.dataset == ds).any():
            continue
        for churn, ls in LS.items():
            g = df[(df.dataset == ds) & (df.churn == churn)].groupby("t")
            ax.plot(g.f1_delta.mean().index, g.f1_delta.mean(), color=C["rule"], ls=ls)
            ax.plot(g.f1_stale.mean().index, g.f1_stale.mean(), color=C["learned"], ls=ls)
        ax.set_xlabel("time step"); ax.set_xticks([0, 2, 4, 6, 8, 10]); f1_axis(ax, k == 0)
        panel(ax, f"({'ab'[k]}) {DS[ds]}")
    d = df[df.t > 0].groupby(["dataset", "churn"])[["bytes_delta", "bytes_full"]].mean()
    x = np.arange(3); wdt = 0.38
    ax = axes[2]
    for i, ds in enumerate(["agentic", "amazon"]):
        if ds not in d.index.get_level_values(0):
            continue
        r = (d.loc[ds].bytes_delta / d.loc[ds].bytes_full).values
        ax.bar(x + (i - .5) * wdt, r, wdt, color=DSC[ds], edgecolor="white", lw=0.4, label=DS[ds])
        for c, v in zip([0.05, 0.1, 0.2], r):
            mac(f"st{ds}ratio{str(c).replace('.', 'p')}", f"{100 * v:.0f}")
    ax.set_xticks(x); ax.set_xticklabels(["5%", "10%", "20%"]); ax.set_xlabel("rules replaced per step")
    ax.set_ylabel("delta size / full upload"); ax.set_ylim(0, 0.5); ax.legend(loc="upper left")
    panel(ax, "(c) upload size")
    hs = [Line2D([0], [0], color=C["rule"], label="delta-maintained context"),
          Line2D([0], [0], color=C["learned"], label="stale context")] + \
         [Line2D([0], [0], color="#555555", ls=ls, label=f"{int(100 * c)}% churn") for c, ls in LS.items()]
    fig.legend(handles=hs, loc="outside upper center", ncol=5, handlelength=2.2, columnspacing=1.2)
    FS.save(fig, f"{OUT}/fig_stream"); plt.close(fig)
    mac("stmaxerr", f"{df.max_abs_err.max():.0e}" if df.max_abs_err.max() > 0 else "0")
    last = df[df.t == df.t.max()]
    for ds in ["agentic", "amazon"]:
        for c in [0.05, 0.1, 0.2]:
            x = last[(last.dataset == ds) & (last.churn == c)]
            if x.empty:
                continue
            mac(f"st{ds}delta{str(c).replace('.', 'p')}", f"{x.f1_delta.mean():.3f}")
            mac(f"st{ds}stale{str(c).replace('.', 'p')}", f"{x.f1_stale.mean():.3f}")
    mac("sttdelta", f"{1000 * df[df.t > 0].time_delta.mean():.1f}"); mac("sttfull", f"{1000 * df[df.t > 0].time_full.mean():.1f}")


# ----------------------------------------------------------------------------- case study and natural anomalies
def case_table():
    p = f"{RES}/case_study.json"
    if not os.path.exists(p):
        return
    d = json.load(open(p))
    yes = lambda b: "\\checkmark" if b else "--"
    lines = []
    for task, tag in (("x-conflict", "conf"), ("x-redundancy", "red")):
        for k, x in enumerate(d.get(task, [])):
            conds = "; ".join(c.split(": ", 1)[1].replace("=", " = ") if ": " in c and c.split(": ")[0] not in ("action", "app")
                              else c.replace(": ", " = ") for c in x["conds"].split("; ") if not c.startswith("app"))
            conds = conds.replace("_", "\\_")
            o, f5, f4 = x["oracle"][task], x["flags_B512"][task], x["flags_B4096"][task]
            lines.append(f"\\#{x['uid']} & $D_{x['dom']}$ & {x['effect']} & \\footnotesize {conds} & {yes(o)} & {yes(f5)} & {yes(f4)}\\\\")
            mac(f"cs{tag}{'ab'[k]}uid", str(x["uid"])); mac(f"cs{tag}{'ab'[k]}dom", str(x["dom"]))
        lines.append("\\midrule")
    open(f"{OUT}/tab_case.tex", "w").write("\n".join(lines[:-1]) + "\n")
    for B in (512, 4096):
        for t, tag in (("x-conflict", "conf"), ("x-redundancy", "red")):
            for m in ("prec", "rec", "f1"):
                mac(f"cs{tag}{m}B{num(B)}", f"{d[f'summary_B{B}'][t][m]:.3f}")
    mac("csnrules", str(d["n_rules_scope"])); mac("csnmembers", str(len(d["scope_members"])))


def natural_table():
    df = load("natural")
    if df is None:
        return
    lines = []
    for ds in ["agentic", "xu_stoller", "amazon"]:
        sub = df[df.dataset == ds]
        if sub.empty:
            continue
        prev = sub[sub.B == 512]
        row = [DS[ds]] + [f"{100 * prev[t + '_pos'].mean():.1f}" for t in TASKS]
        for B in (512, 4096, 16384):
            x = sub[sub.B == B]
            row.append(cell(x.macro_f1))
            mac(f"nat{ds.replace('_', '')}B{num(B)}", ms(x.macro_f1)[0])
        lines.append(" & ".join(row) + "\\\\")
    open(f"{OUT}/tab_natural.tex", "w").write("\n".join(lines) + "\n")


def multihash_macros():
    p = f"{RES}/multihash.json"
    if not os.path.exists(p):
        return
    for r in json.load(open(p)):
        mac(f"mh{r['dataset'].replace('_', '')}B{num(r['B'])}k{'abcde'[r['k']]}", f"{r['macro_f1']:.3f}")


if __name__ == "__main__":
    for f in (main_table, ablation_tables, sketch_fig, leakage_fig, noise_fig, scale_fig, stream_fig, case_table, natural_table, multihash_macros):
        f()
    with open(f"{OUT}/numbers.tex", "w") as fh:
        for k, v in sorted(macros.items()):
            name = "".join(ch if ch.isalpha() else "ABCDEFGHIJ"[int(ch)] if ch.isdigit() else "" for ch in k)
            fh.write(f"\\newcommand{{\\{name}}}{{{v}}}\n")
    # values of runs that have not finished yet print as [pending] instead of breaking the build
    import re
    pat = re.compile(r"\\((?:main(?:agentic|xustoller|amazon)|sk(?:agentic|xustoller|amazon)|nz(?:agentic|amazon)|abl(?:str|lea)"
                     r"|lk(?:agentic|xustoller|amazon|all)|nat(?:agentic|xustoller|amazon)|cs(?:conf|red|n)|sc(?:fixed|scaled|K)"
                     r"|st(?:agentic|amazon|maxerr|tdelta|tfull)|prev(?:agentic|xustoller|amazon)|n(?:agentic|xustoller|amazon)(?:rules|req)"
                     r"|mh(?:agentic|xustoller|amazon))[A-Za-z]*)")
defined = {"".join(ch if ch.isalpha() else "ABCDEFGHIJ"[int(ch)] if ch.isdigit() else "" for ch in k) for k in macros}
used = set()
for f in ("paper/main.tex", "paper/response.tex"):
    if os.path.exists(f):
        used |= set(pat.findall(open(f).read()))
pending = sorted(used - defined)
with open(f"{OUT}/numbers.tex", "a") as fh:
    for name in pending:
        fh.write(f"\\newcommand{{\\{name}}}{{\\textbf{{[pending]}}}}\n")
    print("wrote", len(macros), "macros to", OUT, "; pending:", len(pending), pending[:8])
