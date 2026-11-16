"""Concept figures for paper2: hypergraph of the running example (before / after events with regions),
match-set geometry of conflict and redundancy, incidence rows and inner products, and the hyperedge scorer."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, FancyArrowPatch
from scipy.spatial import ConvexHull
from dyfhl import figstyle as fs

fs.apply()
C = fs.OI
F = "paper/figures"
OWN = {"V": C["orange"], "C": C["blue"], "ev": C["vermillion"]}

# vertices of the running example, grouped by attribute (x = attribute column)
VX = {"clinician": (0, 3), "analyst": (0, 2), "intern": (0, 1), "svc(c_V)": (0, 0),
      "summarise": (2, 3.6), "export": (2, 2.55),
      "ehr": (4, 3), "claims": (4, 2),
      "analytics": (6, 3), "V/EU": (6, 2), "R/US": (6, 1)}
LAB = {"svc(c_V)": r"svc$(c_V)$"}
COLS = [(0, "usr.role"), (2, "tool"), (4, "res"), (6, "purpose / sink")]


def blob(ax, pts, color, ls="-", lw=1.0, alpha=0.16, pad=0.28, label=None, lpos=None, z=1):
    P = np.array([VX[p] for p in pts], float)
    ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    ring = np.concatenate([P + pad * np.c_[np.cos(ang), np.sin(ang)] for P in P[:, None, :].repeat(1, 1)[:, 0, :][:, None]], 0) \
        if False else np.concatenate([p + pad * np.c_[np.cos(ang), np.sin(ang)] for p in P], 0)
    hull = ring[ConvexHull(ring).vertices]
    ax.add_patch(Polygon(hull, closed=True, fc=color, ec=color, alpha=alpha, lw=0, zorder=z))
    ax.add_patch(Polygon(hull, closed=True, fc="none", ec=color, ls=ls, lw=lw, zorder=z + 1))
    if label:
        x, y = lpos if lpos else (hull[:, 0].mean(), hull[:, 1].max() + 0.08)
        ax.text(x, y, label, color=color, fontsize=7.5, ha="center", va="bottom", zorder=5,
                bbox=dict(fc="white", ec="none", pad=0.4, alpha=0.85))


def vertices(ax, hi=()):
    for n, (x, y) in VX.items():
        ax.plot(x, y, "o", ms=4.5, mfc="white" if n not in hi else C["yellow"], mec="black", mew=0.6, zorder=6)
        ax.text(x, y - 0.24, LAB.get(n, n), ha="center", va="top", fontsize=7, zorder=6)
    for x, t in COLS:
        ax.text(x, 4.35, t, ha="center", fontsize=7.5, style="italic", color=C["grey"])
    ax.set_xlim(-0.9, 6.9); ax.set_ylim(-0.75, 4.5); ax.axis("off")


# ---------------------------------------------------------------- Fig A/B: hypergraph before and after the events
fig, ax = plt.subplots(1, 2, figsize=fs.figsize("double", 78), constrained_layout=True)
for k, a in enumerate(ax):
    blob(a, ["clinician", "analyst", "summarise"], OWN["V"], label=r"$e_1$ (tool, $V$)", lpos=(1.0, 3.95))
    blob(a, ["analyst", "export"], OWN["V"], pad=0.2, label=r"$e_2$", lpos=(1.0, 1.55))
    blob(a, ["clinician", "ehr"], OWN["C"], pad=0.2, label=r"$e_3$ (data, $C$)", lpos=(3.2, 3.22))
    blob(a, ["analyst", "claims"], OWN["C"], pad=0.15, label=r"$e_4$", lpos=(3.3, 1.75))
    blob(a, ["claims", "analytics", "V/EU"], OWN["C"], pad=0.22, ls=":", label=r"$e_6$ (usage, $C$)", lpos=(5.6, 1.35))
    blob(a, ["export", "claims", "V/EU"], C["green"], pad=0.30, ls="--", alpha=0.07, label=r"$d_0(\mathtt{export})$", lpos=(4.0, 1.25))
    if k == 1:
        blob(a, ["svc(c_V)", "ehr"], OWN["ev"], pad=0.2, lw=1.5, alpha=0.12, label=r"$\epsilon_1$: $e_5$", lpos=(2.6, 0.55))
        blob(a, ["intern", "analyst", "export"], OWN["ev"], pad=0.12, lw=1.5, alpha=0.10, label=r"$\epsilon_3$: $e_7$", lpos=(0.9, 0.85))
        blob(a, ["export", "claims", "R/US"], OWN["ev"], pad=0.16, lw=1.5, alpha=0.08, label=r"$\epsilon_2$: $d_1(\mathtt{export})$", lpos=(5.0, 0.45))
    vertices(a, hi=("ehr", "summarise", "export", "claims") if k == 1 else ())
ax[0].set_title("(a) State before the events: one hypergraph, two owners", loc="left")
ax[1].set_title("(b) After $\\epsilon_1,\\epsilon_2,\\epsilon_3$: new hyperedges, key vertices highlighted", loc="left")
fs.save(fig, f"{F}/fig_hg_example", submission_dir="paper/figures_submission"); plt.close(fig)

# ---------------------------------------------------------------- Fig C: match sets, overlap and subsumption
fig, ax = plt.subplots(1, 3, figsize=fs.figsize("double", 55), constrained_layout=True)
roles = ["clin.", "analyst", "intern", "svc"]; res = ["ehr", "claims", "billing"]


def grid(a, title):
    for i in range(len(res) + 1):
        a.axvline(i, color=C["lightgrey"], lw=.5)
    for j in range(len(roles) + 1):
        a.axhline(j, color=C["lightgrey"], lw=.5)
    a.set_xticks(np.arange(len(res)) + .5); a.set_xticklabels(res)
    a.set_yticks(np.arange(len(roles)) + .5); a.set_yticklabels(roles)
    a.set_xlim(0, 3); a.set_ylim(0, 4); a.set_title(title, loc="left")
    a.tick_params(length=0); [s.set_visible(False) for s in a.spines.values()]


def box(a, x0, x1, y0, y1, col, lab, ls="-", hatch=None):
    a.add_patch(Rectangle((x0 + .04, y0 + .04), x1 - x0 - .08, y1 - y0 - .08, fc=col, alpha=.25, ec=col, lw=1.2, ls=ls, hatch=hatch))
    a.text(x1 - .1, y1 - .12, lab, color=col, ha="right", va="top", fontsize=7.5)


grid(ax[0], "(a) Conflict: boxes overlap on every attribute")
box(ax[0], 0, 2, 0, 2, C["blue"], "permit ($C$)"); box(ax[0], 1, 3, 1, 3, C["vermillion"], "deny ($R$)", ls="--")
grid(ax[1], "(b) Redundancy: one box inside the other")
box(ax[1], 0, 3, 0, 3, C["blue"], "permit ($C$)"); box(ax[1], 1, 2, 1, 2, C["orange"], "permit ($V$)", ls="--")
grid(ax[2], "(c) Disjoint key vertices: never a finding")
box(ax[2], 0, 1, 0, 3, C["blue"], "$C$ on ehr"); box(ax[2], 2, 3, 1, 4, C["orange"], "$V$ on billing", ls="--")
for a in ax:
    a.set_xlabel("res (key attribute)"); a.set_ylabel("usr.role")
fs.save(fig, f"{F}/fig_matchsets", submission_dir="paper/figures_submission"); plt.close(fig)

# ---------------------------------------------------------------- Fig D: incidence rows and inner products
fig, ax = plt.subplots(figsize=fs.figsize("onehalf", 52), constrained_layout=True)
cols = ["clin.", "analyst", "intern", "svc", "ehr", "claims", "billing"]
rows = {r"$\chi(e)$  event rule of $C$": [1, 1, 0, 0, 1, 0, 0],
        r"$\chi(e'_1)$  rule of $V$": [0, 1, 1, 0, 1, 1, 0],
        r"$\chi(e'_2)$  rule of $V$": [1, 1, 1, 1, 1, 0, 0]}
M = np.array(list(rows.values()))
ax.imshow(M, cmap=matplotlib.colors.ListedColormap(["white", C["sky"]]), aspect="auto", vmin=0, vmax=1)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=8)
ax.plot([3.5, 3.5], [-0.5, 2.5], color="black", lw=1.0)
ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols); ax.set_yticks(range(3)); ax.set_yticklabels(list(rows))
ax.xaxis.tick_top(); ax.tick_params(length=0)
ax.text(1.5, 2.95, "attribute usr.role", ha="center", fontsize=7.5, color=C["grey"])
ax.text(5.0, 2.95, "attribute res (key)", ha="center", fontsize=7.5, color=C["grey"])
txt = ("$e$ vs $e'_1$: role $\\langle\\cdot\\rangle=1$, res $\\langle\\cdot\\rangle=1$; product $\\neq 0$ $\\Rightarrow$ overlap\n"
       "$e$ vs $e'_2$: $|\\chi(e)|-\\langle\\chi(e),\\chi(e'_2)\\rangle = 3-3 = 0$ $\\Rightarrow$ $e$ contained in $e'_2$\n"
       "computed on secret shares: owners see only the resulting flags")
ax.text(-0.5, 3.3, txt, fontsize=7.3, va="top")
ax.set_ylim(4.6, -0.5)
[sp.set_visible(False) for sp in ax.spines.values()]
fs.save(fig, f"{F}/fig_incidence", submission_dir="paper/figures_submission"); plt.close(fig)

# ---------------------------------------------------------------- Fig E: hyperedge scorer for expected use
fig, ax = plt.subplots(figsize=fs.figsize("onehalf", 45), constrained_layout=True)
ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 4)
for y, t in ((3.2, "role = analyst"), (2.2, "dept = finance"), (1.2, "org = $C$")):
    ax.add_patch(Rectangle((0.2, y - .3), 2.0, .6, fc=C["sky"], alpha=.3, ec=C["blue"]))
    ax.text(1.2, y, t, ha="center", va="center", fontsize=7.5)
    ax.add_patch(FancyArrowPatch((2.2, y), (4.0, 2.2), arrowstyle="->", mutation_scale=8, lw=.7))
ax.add_patch(Rectangle((4.0, 1.8), 1.9, .8, fc=C["lightgrey"], ec="black"))
ax.text(4.95, 2.2, r"$\sum_a z_{(a,u_a)}$", ha="center", va="center", fontsize=8)
ax.add_patch(Rectangle((4.0, 0.2), 1.9, .7, fc=C["orange"], alpha=.3, ec=C["orange"]))
ax.text(4.95, .55, r"resource $r$: $z_r$, $b_r$", ha="center", va="center", fontsize=7.5)
ax.add_patch(FancyArrowPatch((5.9, 2.2), (7.0, 1.5), arrowstyle="->", mutation_scale=8, lw=.7))
ax.add_patch(FancyArrowPatch((5.9, .55), (7.0, 1.3), arrowstyle="->", mutation_scale=8, lw=.7))
ax.add_patch(Rectangle((7.0, 1.0), 2.8, .8, fc="white", ec="black"))
ax.text(8.4, 1.4, r"$\hat\lambda(u,r)=\sigma(\langle\cdot,z_r\rangle+b_r)$", ha="center", va="center", fontsize=7.5)
ax.text(5.0, 3.6, "vertex embeddings averaged across organisations (FedAvg); logs never leave", ha="center", fontsize=7.3, color=C["grey"])
fs.save(fig, f"{F}/fig_scorer", submission_dir="paper/figures_submission"); plt.close(fig)
print("ok")
