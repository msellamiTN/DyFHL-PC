"""One figure style for every figure of the paper, following Elsevier's artwork instructions:
final widths of 90 / 140 / 190 mm, Arial-compatible lettering (Liberation Sans) embedded as TrueType,
text >= 7 pt and subscripts >= 6 pt at printed size, line weights within 0.1-1.5 pt, RGB colour,
vector PDF/EPS plus 1000 dpi TIFF for line art."""
from __future__ import annotations
import os, subprocess
import matplotlib as mpl

MM = 1 / 25.4
WIDTH_MM = {"single": 90.0, "onehalf": 140.0, "double": 190.0}
# printed text width of the review manuscript (elsarticle preprint, 12 pt): figures of 140 mm are shown at 98 %
MANUSCRIPT_TEXTWIDTH_MM = 137.1

# Okabe-Ito colour-blind safe palette
OI = dict(blue="#0072B2", vermillion="#D55E00", green="#009E73", sky="#56B4E9", orange="#E69F00",
          purple="#CC79A7", yellow="#F0E442", black="#000000", grey="#6E6E6E", lightgrey="#D9D9D9")
DOMCOL = {"A": OI["blue"], "B": OI["vermillion"], "C": OI["green"]}
METHOD = {"rule": OI["blue"], "learned": OI["vermillion"], "base": OI["grey"], "alt": OI["orange"], "g": OI["green"],
          "sky": OI["sky"], "purple": OI["purple"]}

BASE_PT = 8.0       # body text in figures
SMALL_PT = 7.5      # tick labels, legends
SUBSCRIPT_PT = 9.0  # any label that contains a math subscript (subscript glyphs are 0.7 x size = 6.3 pt)


def apply():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
        "mathtext.bf": "Liberation Sans:bold", "mathtext.sf": "Liberation Sans", "mathtext.default": "it",
        "font.size": BASE_PT, "axes.titlesize": BASE_PT, "axes.labelsize": BASE_PT,
        "xtick.labelsize": SMALL_PT, "ytick.labelsize": SMALL_PT, "legend.fontsize": SMALL_PT,
        "legend.title_fontsize": SMALL_PT, "figure.titlesize": BASE_PT,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "lines.linewidth": 1.0, "lines.markersize": 3.5, "lines.markeredgewidth": 0.5,
        "errorbar.capsize": 1.5, "grid.linewidth": 0.3, "grid.color": "#BFBFBF", "patch.linewidth": 0.5,
        "hatch.linewidth": 0.4,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "legend.frameon": False, "legend.handlelength": 1.8, "legend.borderaxespad": 0.3,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.pad_inches": 0.0,
        "figure.constrained_layout.h_pad": 0.02, "figure.constrained_layout.w_pad": 0.02,
    })


def figsize(width="onehalf", height_mm=50.0):
    return (WIDTH_MM[width] * MM, height_mm * MM)


def save(fig, path_noext, submission_dir="paper/figures_submission"):
    """Write vector PDF (manuscript), PNG preview, and EPS + 1000 dpi TIFF for the submission package.
    The canvas size is kept exactly (no bbox cropping), so the printed width equals the target width."""
    os.makedirs(os.path.dirname(path_noext) or ".", exist_ok=True)
    pdf = path_noext + ".pdf"
    fig.savefig(pdf)
    fig.savefig(path_noext + ".png", dpi=300)
    if submission_dir:
        os.makedirs(submission_dir, exist_ok=True)
        base = os.path.join(submission_dir, os.path.basename(path_noext))
        subprocess.run(["cp", pdf, base + ".pdf"], check=True)
        subprocess.run(["pdftops", "-eps", "-level3", pdf, base + ".eps"], check=False)
        subprocess.run(["pdftoppm", "-r", "1000", "-singlefile", "-tiff", "-tiffcompression", "lzw", pdf, base],
                       check=False)
