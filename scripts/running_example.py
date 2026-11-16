"""Running example of the paper: three domains, six rules, two requests.

Every number quoted in the worked examples is computed here with the package's own
aggregation and evidence code and cross-checked against the exact pooled oracle.
Outputs: paper/generated/ex_*.pdf (figures), ex_*.tex (tables) and results/running_example.json."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from dyfhl.model import Config, DyFHL, Geometry
from dyfhl.policy import pair_matrices, cover_matrix

OUT = "paper/generated"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
DOMCOL = {"A": "#0072B2", "B": "#D55E00", "C": "#009E73"}       # colour-blind safe
DOMNAME = {"A": "Data Platform", "B": "Compliance", "C": "ML Engineering"}

ATTRS = {"type": ["agent", "human", "tool"], "res": ["dataset", "model", "api"], "sens": ["low", "high"],
         "act": ["read", "export", "invoke", "delete"], "tool": ["t1", "t7", "t9"]}
ANAMES = list(ATTRS)
OWNER = {"t1": "A", "t7": "A", "t9": "C"}
ALIVE = {"t1": 1, "t7": 0, "t9": 1}
RULES = [  # name, domain, effect (0 permit, 1 deny), conditions
    ("r1", "A", 0, {"type": ["agent"], "res": ["dataset"], "act": ["read"]}),
    ("r2", "A", 1, {"type": ["human"], "res": ["model"], "act": ["delete"]}),
    ("r3", "B", 1, {"sens": ["high"], "act": ["read", "export"]}),
    ("r4", "B", 0, {"type": ["agent", "human"], "res": ["dataset"]}),
    ("r5", "C", 0, {"type": ["agent"], "res": ["model"], "act": ["invoke"], "tool": ["t7"]}),
    ("r6", "C", 0, {"type": ["human"], "res": ["api"], "sens": ["low"]}),
]
REQS = [("q1", "A", {"type": "tool", "res": "api", "sens": "low", "act": "delete", "tool": "t1"}),
        ("q2", "A", {"type": "human", "res": "api", "sens": "low", "act": "read", "tool": "t1"})]
HASH = {8: {"r1": 0, "r2": 1, "r3": 2, "r4": 3, "r5": 4, "r6": 5},      # injective: exact
        2: {"r1": 0, "r2": 1, "r3": 0, "r4": 1, "r5": 1, "r6": 0}}      # r4 and r5 collide

slices, vidx, off = [], {}, 0
for a, vs in ATTRS.items():
    for i, v in enumerate(vs):
        vidx[(a, v)] = off + i
    slices.append((off, off + len(vs))); off += len(vs)
V, A = off, len(ATTRS)
geo = Geometry(slices)


def encode(conds):
    h, w = np.zeros(V, np.float32), np.ones(A, np.float32)
    for a, vs in conds.items():
        w[ANAMES.index(a)] = 0
        for v in ([vs] if isinstance(vs, str) else vs):
            h[vidx[(a, v)]] = 1
    return h, w


H = np.stack([encode(c)[0] for _, _, _, c in RULES]); W = np.stack([encode(c)[1] for _, _, _, c in RULES])
EFF = np.array([e for _, _, e, _ in RULES]); DOM = np.array([d for _, d, _, _ in RULES]); NAMES = [n for n, *_ in RULES]
QH = np.stack([encode(c)[0] for _, _, c in REQS])

# ---------------------------------------------------------------- exact pooled oracle
ov, sub = pair_matrices(H.astype(np.uint8), W.astype(np.uint8), slices)
dd = DOM[:, None] != DOM[None, :]; de = EFF[:, None] != EFF[None, :]
oracle = {"x-conflict": {NAMES[i]: [NAMES[j] for j in np.flatnonzero(ov[i] & dd[i] & de[i])] for i in range(len(RULES))},
          "x-redundancy": {NAMES[i]: [NAMES[j] for j in np.flatnonzero(sub[i] & dd[i] & ~de[i])] for i in range(len(RULES))}}
cov = cover_matrix(QH.astype(np.uint8), H.astype(np.uint8), W.astype(np.uint8))
oracle["cover"] = {REQS[k][0]: [NAMES[j] for j in np.flatnonzero(cov[k])] for k in range(len(REQS))}
oracle["x-irrelevancy"] = {}
for i, (n, d, e, c) in enumerate(RULES):
    if "tool" in c:
        vs = c["tool"]
        oracle["x-irrelevancy"][n] = bool(all(ALIVE[v] == 0 for v in vs) and any(OWNER[v] != d for v in vs))


# ---------------------------------------------------------------- DyFHL-PA with the package code
def run(B):
    cfg = Config(B=B, h=0)
    model = DyFHL(geo, cfg)
    sig = torch.zeros(len(RULES), B)
    for i, n in enumerate(NAMES):
        sig[i, HASH[B][n]] = 1
    Ht, Wt, Et = torch.tensor(H), torch.tensor(W), torch.tensor(EFF)
    sc = torch.zeros(len(RULES), dtype=torch.long)
    U = {}
    for d in "ABC":
        m = torch.tensor(DOM == d)
        U[d] = model.aggregate(Ht[m], Wt[m], Et[m], sc[m], sig[m], 1)      # [1, V+A, 2, B]
    out = {"U": {d: U[d][0].numpy() for d in U}, "ctx": {}, "rules": {}, "reqs": {}}
    for d in "ABC":
        C = sum(U[x] for x in "ABC" if x != d)
        out["ctx"][d] = C[0].numpy()
        m = np.flatnonzero(DOM == d)
        ev = model.evidence(Ht[m], Wt[m], torch.zeros(len(m), dtype=torch.long), C).numpy()   # [n, 3, 2, B]
        for k, i in enumerate(m):
            e = EFF[i]
            per = per_attr(H[i], W[i], C[0].numpy())                       # (ov_a, sub_a): [A, 2, B]
            assert np.allclose(per[0].min(0), ev[k, 0]) and np.allclose(per[1].min(0), ev[k, 1])
            out["rules"][NAMES[i]] = dict(ov_opp=per[0][:, 1 - e], sub_same=per[1][:, e], OV=ev[k, 0, 1 - e], SUB=ev[k, 1, e],
                                          conflict=bool(ev[k, 0, 1 - e].max() > .5), redundancy=bool(ev[k, 1, e].max() > .5))
        if d == "A":
            Qt = torch.tensor(QH)
            evq = model.evidence(Qt, torch.zeros(len(REQS), A), torch.zeros(len(REQS), dtype=torch.long), C).numpy()
            for k, (n, _, _) in enumerate(REQS):
                per = per_attr(QH[k], np.zeros(A), C[0].numpy())
                assert np.allclose(per[0].min(0).sum(0), evq[k, 0].sum(0))      # sum over effects of the per-effect conjunction
                out["reqs"][n] = dict(ov_permit=per[0][:, 0], ov_deny=per[0][:, 1], OV=evq[k, 0].sum(0),
                                      gap=bool(evq[k, 0].sum(0).max() <= .5))
    return out


def per_attr(h, w, C):
    """Eq. (ov) and (sub) written out directly: returns arrays [A, 2, B]."""
    ovs, subs = [], []
    for a, (s, e) in enumerate(slices):
        wild = C[V + a]
        if w[a]:
            ovs.append(wild + C[s:e].sum(0)); subs.append(wild.copy())
        else:
            vs = np.flatnonzero(h[s:e]) + s
            ovs.append(wild + C[vs].sum(0)); subs.append(wild + C[vs].min(0))
    return np.stack(ovs), np.stack(subs)


res = {B: run(B) for B in (8, 2)}

# checks against the oracle
r8 = res[8]
for n in NAMES:
    assert r8["rules"][n]["conflict"] == bool(oracle["x-conflict"][n]), n
    assert r8["rules"][n]["redundancy"] == bool(oracle["x-redundancy"][n]), n
assert r8["reqs"]["q1"]["gap"] and not r8["reqs"]["q2"]["gap"]
assert not oracle["cover"]["q1"] and oracle["cover"]["q2"] == ["r6"]
for n in NAMES:   # one-sided error at B = 2
    assert res[2]["rules"][n]["conflict"] >= bool(oracle["x-conflict"][n])
    assert res[2]["rules"][n]["redundancy"] >= bool(oracle["x-redundancy"][n])

# liveness as seen by domain C
L = {d: {t: (ALIVE[t] if OWNER[t] == d else 0) for t in ATTRS["tool"]} for d in "ABC"}
Lam_C = {t: sum(L[d][t] for d in "AB") for t in ATTRS["tool"]}
live_C = {t: L["C"][t] + Lam_C[t] for t in ATTRS["tool"]}
assert live_C == {"t1": 1, "t7": 0, "t9": 1} and oracle["x-irrelevancy"]["r5"]

# delta example: Compliance withdraws r3
cfg = Config(B=8, h=0); model = DyFHL(geo, cfg)
def agg(idx):
    sig = torch.zeros(len(idx), 8)
    for k, i in enumerate(idx):
        sig[k, HASH[8][NAMES[i]]] = 1
    return model.aggregate(torch.tensor(H[idx]), torch.tensor(W[idx]), torch.tensor(EFF[idx]),
                           torch.zeros(len(idx), dtype=torch.long), sig, 1)
UB, UB_new, d3 = agg([2, 3]), agg([3]), agg([2])
assert torch.equal(UB - d3, UB_new)
delta_nnz = int((d3 != 0).sum()); full_nnz = int((UB_new != 0).sum())

summary = {
    "oracle": oracle,
    "flags_B8": {n: {k: r8["rules"][n][k] for k in ("conflict", "redundancy")} for n in NAMES},
    "flags_B2": {n: {k: res[2]["rules"][n][k] for k in ("conflict", "redundancy")} for n in NAMES},
    "gap_B8": {n: r8["reqs"][n]["gap"] for n in r8["reqs"]}, "gap_B2": {n: res[2]["reqs"][n]["gap"] for n in res[2]["reqs"]},
    "liveness_seen_by_C": live_C, "delta_nnz": delta_nnz, "full_nnz_after": full_nnz,
    "incidences": {n: int(H[i].sum() + W[i].sum()) for i, n in enumerate(NAMES)},
    "r1_ov_deny_B8": r8["rules"]["r1"]["ov_opp"].tolist(), "r1_sub_permit_B8": r8["rules"]["r1"]["sub_same"].tolist(),
    "r2_ov_permit_B8": r8["rules"]["r2"]["ov_opp"].tolist(), "r2_ov_permit_B2": res[2]["rules"]["r2"]["ov_opp"].tolist(),
    "q1_ov_permit_B8": r8["reqs"]["q1"]["ov_permit"].tolist(), "q1_ov_deny_B8": r8["reqs"]["q1"]["ov_deny"].tolist(),
    "q2_ov_permit_B8": r8["reqs"]["q2"]["ov_permit"].tolist(), "q2_ov_deny_B8": r8["reqs"]["q2"]["ov_deny"].tolist(),
}
os.makedirs("results", exist_ok=True)
json.dump(summary, open("results/running_example.json", "w"), indent=1)
print(json.dumps({k: summary[k] for k in ("oracle", "flags_B8", "flags_B2", "gap_B8", "gap_B2", "liveness_seen_by_C",
                                           "delta_nnz", "full_nnz_after", "incidences")}, indent=0))

# ---------------------------------------------------------------- LaTeX table: Compliance's aggregate U_B
vnames = [f"{a}={v}" for a, vs in ATTRS.items() for v in vs] + [f"{a}=$\\ast$" for a in ANAMES]
UBn = res[8]["U"]["B"]
rows = []
for u in range(V + A):
    cells = []
    for c, lab in ((0, "permit"), (1, "deny")):
        bins = np.flatnonzero(UBn[u, c])
        cells.append(", ".join(f"bin {b}" for b in bins) if len(bins) else "--")
    if UBn[u].sum() > 0:
        rows.append(f"\\textit{{{vnames[u]}}} & {cells[0]} & {cells[1]}\\\\")
open(f"{OUT}/ex_tab_UB.tex", "w").write("\n".join(rows) + "\n")


# ---------------------------------------------------------------- figures (millimetre canvas)
from matplotlib.patches import FancyBboxPatch, Rectangle
from dyfhl import figstyle as FS
FS.apply()
GREY, LGREY = "#6E6E6E", "#F2F2F2"
DOMCOL = FS.DOMCOL
COLX = {"type": 22.0, "res": 45.0, "sens": 68.0, "act": 91.0, "tool": 114.0}
ROWDY, NODEH = 6.4, 4.4
CH = 1.48                                       # mm per character at 8 pt (Liberation Sans, average)


def canvas(h_mm, w="onehalf"):
    fig = plt.figure(figsize=FS.figsize(w, h_mm))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, FS.WIDTH_MM[w]); ax.set_ylim(0, h_mm); ax.axis("off")
    return fig, ax


def node_w(a):
    return max(6.5, max(len(v) for v in ATTRS[a]) * CH + 2.6)


def node_xy(a, v, top):
    if v == "*":
        return COLX[a], top - 9.0 - 4 * ROWDY - 1.6
    return COLX[a], top - 9.0 - ATTRS[a].index(v) * ROWDY


def draw_hypergraph(ax, top, highlight=None, reqs=(), owners=True):
    """Hyperedges drawn as paths through one vertex per column; columns are attribute blocks."""
    wy = node_xy("type", "*", top)[1]
    ax.add_patch(Rectangle((COLX["type"] - 8.5, wy - 3.4), COLX["tool"] - COLX["type"] + 17.5, 6.8,
                           facecolor=LGREY, edgecolor="none", zorder=0))
    ax.text(COLX["tool"] + 5.6, wy, "wildcard\nvertices", fontsize=7.5, va="center", ha="left", color=GREY, linespacing=1.0)
    for a in ANAMES:
        ax.text(COLX[a], top - 3.2, a, fontsize=8, ha="center", va="center", style="italic", fontweight="bold")
    starts = {}

    def path(sets, color, ls, lw, alpha, off, z=2):
        for k in range(A - 1):
            a0, a1 = ANAMES[k], ANAMES[k + 1]
            for v0 in sets[k]:
                for v1 in sets[k + 1]:
                    x0, y0 = node_xy(a0, v0, top); x1, y1 = node_xy(a1, v1, top)
                    ax.plot([x0 + node_w(a0) / 2, x1 - node_w(a1) / 2], [y0 + off, y1 + off], color=color, ls=ls,
                            lw=lw, alpha=alpha, zorder=z, solid_capstyle="butt", dash_capstyle="butt")

    def label(v0, text, color, alpha):
        k = starts.get(v0, 0); starts[v0] = k + 1
        x, y = node_xy("type", v0, top)
        ax.text(x - node_w("type") / 2 - 1.2 - 4.6 * k, y, text, fontsize=FS.SUBSCRIPT_PT, ha="right", va="center",
                color=color, alpha=alpha)

    for i, (n, d, e, c) in enumerate(RULES):
        on = highlight is None or n in highlight
        lw, alpha = (1.3 if highlight else 1.0, 1.0) if on else (0.6, 0.22)
        sets = [c[a] if a in c else ["*"] for a in ANAMES]
        path(sets, DOMCOL[d], (0, (3.2, 1.6)) if e else "-", lw, alpha, (i - 2.5) * 0.5, z=3 if on else 2)
        label(sets[0][0], "$r_%s$" % n[1], DOMCOL[d], 1.0 if on else 0.35)
    for k, (n, _, c) in enumerate(REQS):
        if n in reqs:
            sets = [[c[a]] for a in ANAMES]
            path(sets, "black", (0, (1.0, 1.3)), 1.1, 1.0, 0.25 * (k * 2 - 1), z=3)
            label(sets[0][0], "$q_%s$" % n[1], "black", 1.0)
    for a in ANAMES:
        for v in ATTRS[a] + ["*"]:
            x, y = node_xy(a, v, top); w = node_w(a)
            dead = a == "tool" and v != "*" and ALIVE[v] == 0
            ax.add_patch(FancyBboxPatch((x - w / 2, y - NODEH / 2), w, NODEH, boxstyle="round,pad=0,rounding_size=1.2",
                                        facecolor="white" if v != "*" else "#E4E4E4", edgecolor="#333333",
                                        lw=0.6, ls=(0, (2, 1.2)) if dead else "-", zorder=5))
            lab = {"t1": "$t_1$", "t7": "$t_7$", "t9": "$t_9$"}.get(v, "*" if v == "*" else v)
            ax.text(x, y - (0.25 if v != "*" else 1.25), lab, fontsize=FS.SUBSCRIPT_PT if v in OWNER else (11 if v == "*" else 8), ha="center",
                    va="center", zorder=6, color=GREY if dead else "black")
            if owners and a == "tool" and v != "*":
                ax.text(x + w / 2 + 1.3, y, f"owner {OWNER[v]}" + (", retired" if dead else ""), fontsize=7.5,
                        va="center", color=GREY)
    return wy - NODEH / 2      # lowest y used


def legend(ax, y, x0=2.0, full=True):
    items = [("line", DOMCOL[d], "-", f"domain {d}" + (f" ({DOMNAME[d]})" if full else "")) for d in "ABC"]
    items2 = [("line", "#333333", "-", "permit rule"), ("line", "#333333", (0, (3.2, 1.6)), "deny rule"),
              ("line", "black", (0, (1.0, 1.3)), "request")]
    rows = [items, items2] if full else [items + items2]
    for r, its in enumerate(rows):
        x = x0
        for _, col, ls, txt in its:
            ax.plot([x, x + 6], [y - r * 4.6] * 2, color=col, ls=ls, lw=1.1, solid_capstyle="butt")
            ax.text(x + 7.2, y - r * 4.6, txt, fontsize=7.5, va="center")
            x += 7.2 + len(txt) * 1.38 + 5.0


def draw_matrix(ax, x0, top, M, title, sub, lastlab, cell=3.9, rows=None):
    """Evidence per attribute (rows) and bin (columns); the last row is the conjunction (min)."""
    M = np.asarray(M, float); full = np.vstack([M, M.min(0, keepdims=True)])
    rows = rows or ANAMES
    nb = M.shape[1]; labw = 11.0; gx = x0 + labw
    ax.text(x0, top, title, fontsize=FS.SUBSCRIPT_PT, va="top", ha="left")
    ax.text(x0, top - 4.6, sub, fontsize=8, va="top", ha="left", color="#333333")
    gt = top - 9.4
    for i in range(full.shape[0]):
        y = gt - (i + 1) * cell - (0.5 if i == full.shape[0] - 1 else 0)
        last = i == full.shape[0] - 1
        ax.text(gx - 1.0, y + cell / 2, lastlab if last else rows[i], fontsize=7.5, ha="right", va="center",
                style="normal" if last else "italic", fontweight="bold" if last else "normal")
        for b in range(nb):
            v = full[i, b]
            fc = ("#08519C" if v > .5 else "#EDEDED") if last else ("#C6DBEF" if v > 0 else "white")
            ax.add_patch(Rectangle((gx + b * cell, y), cell, cell, facecolor=fc, edgecolor="#8C8C8C", lw=0.4))
            ax.text(gx + (b + .5) * cell, y + cell / 2 - 0.1, f"{int(v)}", fontsize=7.5, ha="center", va="center",
                    color="white" if (last and v > .5) else "black")
    ybot = gt - full.shape[0] * cell - 0.5
    ax.plot([gx, gx + nb * cell], [ybot + cell + 0.25] * 2, color="black", lw=0.9)
    for b in range(nb):
        if full[-1, b] > .5:
            ax.add_patch(Rectangle((gx + b * cell, ybot), cell, full.shape[0] * cell + 0.5, facecolor="none",
                                   edgecolor="black", lw=1.2, zorder=4))
        ax.text(gx + (b + .5) * cell, ybot - 1.9, str(b), fontsize=7.5, ha="center", va="center")
    ax.text(gx + nb * cell / 2, ybot - 5.0, "bin", fontsize=7.5, ha="center", va="center")
    return ybot - 6.5


HG_H = 44.0
# Fig.: full hypergraph
fig, ax = canvas(51.5)
low = draw_hypergraph(ax, 51.5 - 1.0, reqs=("q1", "q2"))
legend(ax, low - 4.0)
FS.save(fig, f"{OUT}/ex_hypergraph"); plt.close(fig)


def typefig(name, h, hl, reqs, title, panels):
    fig, ax = canvas(h)
    ax.text(0.5, h - 0.8, title, fontsize=FS.SUBSCRIPT_PT, va="top")
    low = draw_hypergraph(ax, h - 5.0, highlight=hl, reqs=reqs)
    legend(ax, low - 3.6, full=False)
    top = low - 9.0
    for x0, M, t, sub, last in panels:
        draw_matrix(ax, x0, top, M, t, sub, last)
    FS.save(fig, f"{OUT}/{name}"); plt.close(fig)


H_T = 93.5
typefig("ex_conflict", H_T, {"r1", "r3"}, (), "(a) hyperedges of $r_1$ (A, permit) and $r_3$ (B, deny)", [
    (0.0, r8["rules"]["r1"]["ov_opp"], "(b) $r_1$ vs. deny rules, $B=8$", "bin 2 positive on all rows: conflict", "OV"),
    (48.5, r8["rules"]["r2"]["ov_opp"], "(c) $r_2$ vs. permit rules, $B=8$", "no positive bin: no conflict", "OV"),
    (97.0, res[2]["rules"]["r2"]["ov_opp"], "(d) $r_2$, $B=2$", "collision: false alarm", "OV")])
typefig("ex_redundancy", H_T, {"r1", "r4"}, (), "(a) hyperedges of $r_1$ (A, permit) and $r_4$ (B, permit)", [
    (0.0, r8["rules"]["r1"]["sub_same"], "(b) $r_1$ vs. permit rules, $B=8$", "bin 3 positive on all rows: redundant", "SUB"),
    (48.5, r8["rules"]["r4"]["sub_same"], "(c) $r_4$ vs. permit rules, $B=8$", "no positive bin: not redundant", "SUB"),
    (97.0, res[2]["rules"]["r4"]["sub_same"], "(d) $r_4$, $B=2$", "collision: false alarm", "SUB")])
typefig("ex_gap", H_T, {"r6"}, ("q1", "q2"), "(a) requests $q_1$, $q_2$ (dotted) and rule $r_6$ (C, permit)", [
    (0.0, r8["reqs"]["q1"]["ov_permit"], "(b) $q_1$, permit channel", "no positive bin", "OV"),
    (48.5, r8["reqs"]["q1"]["ov_deny"], "(c) $q_1$, deny channel", "no positive bin: gap", "OV"),
    (97.0, r8["reqs"]["q2"]["ov_permit"], "(d) $q_2$, permit channel", "bin 5 positive: covered", "OV")])

# irrelevancy: hypergraph + liveness table
h = 80.0
fig, ax = canvas(h)
ax.text(0.5, h - 0.8, "(a) hyperedge of $r_5$ (C, permit), which names tool $t_7$", fontsize=FS.SUBSCRIPT_PT, va="top")
low = draw_hypergraph(ax, h - 5.0, highlight={"r5"})
legend(ax, low - 3.6, full=False)
top = low - 9.0
ax.text(0.0, top, "(b) liveness evidence at domain C", fontsize=FS.SUBSCRIPT_PT, va="top")
cols = [("entity", 12), ("owner", 13), ("own $L_C$", 17), ("received $\\Lambda_C$", 24), ("sum", 11)]
x = 0.0; gt = top - 6.0; ch = 4.6
for j, (hd, wd) in enumerate(cols):
    ax.text(x + wd / 2, gt - ch / 2, hd, fontsize=FS.SUBSCRIPT_PT if "$" in hd else 8, ha="center", va="center",
            fontweight="bold" if "$" not in hd else "normal")
    for i, t in enumerate(ATTRS["tool"]):
        y = gt - (i + 2) * ch
        val = ["$t_%s$" % t[1], OWNER[t], str(L["C"][t]), str(Lam_C[t]), str(live_C[t])][j]
        hit = j == 4 and live_C[t] == 0
        fc = "#08519C" if hit else ("#FCE3D1" if t == "t7" else "white")
        ax.add_patch(Rectangle((x, y), wd, ch, facecolor=fc, edgecolor="#8C8C8C", lw=0.4))
        ax.text(x + wd / 2, y + ch / 2 - 0.1, val, fontsize=FS.SUBSCRIPT_PT if "$" in val else 8, ha="center",
                va="center", color="white" if hit else "black")
    x += wd
ax.text(x + 3.0, gt - 3 * ch, "sum = 0 for every tool named by $r_5$,\nand $t_7$ is owned by domain A:\n$r_5$ is cross-domain irrelevant",
        fontsize=FS.SUBSCRIPT_PT, va="center", linespacing=1.25)
FS.save(fig, f"{OUT}/ex_irrelevancy"); plt.close(fig)
print("figures written")
