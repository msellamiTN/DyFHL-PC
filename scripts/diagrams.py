"""Architecture overview of DyFHL-PA, drawn on a millimetre canvas in the shared figure style.
Box text is 8 pt plain text; symbols with subscripts appear only on arrow labels, at 9 pt."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from dyfhl import figstyle as FS

FS.apply()
OUT = "paper/generated"
W, H = 140.0, 70.0
fig = plt.figure(figsize=FS.figsize("onehalf", H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
FILL = {"own": "#E8F1FA", "local": "#EFEFEF", "srv": "#FDEBD9", "out": "#E5F4EE", "other": "#F4F4F4"}
BH = 8.0


def box(x, y, w, text, fill, h=BH, ls="-"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=1.4",
                                facecolor=fill, edgecolor="#333333", lw=0.6, ls=ls, zorder=3))
    ax.text(x, y - 0.2, text, ha="center", va="center", zorder=4, fontsize=8, linespacing=1.25)
    return dict(l=(x - w / 2, y), r=(x + w / 2, y), t=(x, y + h / 2), b=(x, y - h / 2))


def arrow(p, q, ls="-", both=False):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="<|-|>" if both else "-|>", mutation_scale=7, lw=0.8,
                                 color="#222222", ls=ls, shrinkA=0, shrinkB=0, zorder=2))


def label(x, y, text, ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=FS.SUBSCRIPT_PT if "$" in text else 8, zorder=4,
            linespacing=1.2)


# domain d
ax.add_patch(Rectangle((1.0, 1.0), 85.0, 64.0, facecolor="none", edgecolor="#555555", lw=0.6, ls=(0, (3, 2))))
label(3.0, 67.6, "domain $d$ (each of the $K$ domains runs the same pipeline)", ha="left")
X = 22.5; WL = 39.0
b1 = box(X, 57.0, WL, "access-control rules\n(private, never sent)", FILL["own"])
b2 = box(X, 44.0, WL, "policy hypergraph\nvalue + wildcard vertices", FILL["own"])
b3 = box(X, 31.0, WL, "per-vertex sums of hashed\nrule signatures; liveness", FILL["own"])
b4 = box(X, 18.0, WL, "conjunctive decoder\n(Algorithms 1–4)", FILL["out"])
b5 = box(X, 6.0, WL, "x-conflict, x-redundancy,\nx-gap, x-irrelevancy", FILL["out"])
bl = box(67.0, 57.0, 30.0, "exact local analyser\n(own rules only)", FILL["local"])
bi = box(67.0, 46.0, 30.0, "intra-domain\nanomalies", FILL["local"], h=7.0)
for p, q in [(b1["r"], bl["l"]), (bl["b"], bi["t"]), (b1["b"], b2["t"]), (b2["b"], b3["t"]), (b3["b"], b4["t"]),
             (b4["b"], b5["t"])]:
    arrow(p, q)

# server side
XS, WS = 114.5, 43.0
bo = box(XS, 57.0, WS, "other members of the scope\n(same pipeline)", FILL["other"])
bs = box(XS, 29.5, WS, "aggregation server\n(secure aggregation)\nreturns to each member the\nsum of the other uploads",
         FILL["srv"], h=17.0)
bf = box(XS, 7.0, WS, "FedAvg of decoder weights\n(learned decoder only)", FILL["local"], ls=(0, (3, 2)))
arrow((b3["r"][0], 31.0), (bs["l"][0], 31.0))
label(64.0, 36.3, "upload $U_d$ and $L_d$\n(on churn: only $\\Delta$)")
arrow((bs["l"][0], 24.0), (b4["r"][0], 18.0))
label(66.5, 16.6, "context $C_d$, $\\Lambda_d$")
arrow(bo["b"], bs["t"], both=True)
label(XS + 2.0, 44.5, "their uploads,\ntheir contexts", ha="left")
arrow((bf["l"][0], 7.0), (b4["r"][0] + 0.2, 15.6), ls=(0, (3, 2)), both=True)
label(62.0, 8.2, "decoder weights")
FS.save(fig, f"{OUT}/fig_architecture")
print("ok")
