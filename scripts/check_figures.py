"""Check every figure file against the Elsevier artwork rules adopted in dyfhl/figstyle.py.
Reports width, fonts (embedded, Arial/Helvetica family), smallest text at printed size, line weights.
Exit code 1 if any check fails."""
import glob, os, sys
import pymupdf

TARGETS = (90.0, 140.0, 190.0)
TEXTWIDTH = 137.1
OK_FONTS = ("Liberation", "Arial", "Helvetica", "Nimbus", "TeXGyreHeros", "Symbol", "NimbusSan", "LMSans", "uhv")
fail = False
rows = []
files = sorted(glob.glob("paper/generated/fig_*.pdf") + glob.glob("paper/generated/ex_*.pdf"))
for f in files:
    doc = pymupdf.open(f)
    p = doc[0]
    w_mm, h_mm = p.rect.width / 72 * 25.4, p.rect.height / 72 * 25.4
    target = min(TARGETS, key=lambda t: abs(t - w_mm))
    scale = min(1.0, TEXTWIDTH / w_mm)                     # how the manuscript includes it
    fonts = p.get_fonts(full=True)
    not_emb = [x[3] for x in fonts if x[1] in ("n/a", "")]
    bad_family = sorted({x[3] for x in fonts if not any(k.lower() in x[3].lower() for k in OK_FONTS)})
    sizes = []
    for b in p.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            for s in l["spans"]:
                if s["text"].strip():
                    sizes.append((round(s["size"] * scale, 2), s["text"].strip()[:18]))
    body = [x for x in sizes if x[0] >= 6.95]
    small = [x for x in sizes if x[0] < 6.95]
    widths = [d["width"] for d in p.get_drawings() if d.get("width")]
    wmin, wmax = (min(widths), max(widths)) if widths else (None, None)
    problems = []
    if abs(w_mm - target) > 0.6:
        problems.append(f"width {w_mm:.1f} mm")
    if not_emb:
        problems.append(f"fonts not embedded {not_emb}")
    if bad_family:
        problems.append(f"fonts outside Arial/Helvetica family {bad_family}")
    if small and min(x[0] for x in small) < 6.0:
        problems.append(f"text below 6 pt: {sorted(small)[:3]}")
    if widths and (wmin < 0.1 * scale or wmax > 1.5):
        problems.append(f"line weights {wmin:.2f}-{wmax:.2f} pt")
    fail |= bool(problems)
    rows.append((os.path.basename(f), f"{w_mm:.1f}x{h_mm:.1f}", f"{min(x[0] for x in sizes) if sizes else '-'}",
                 len(small), f"{wmin:.2f}-{wmax:.2f}" if widths else "-", "; ".join(problems) or "ok"))
print(f"{'file':28s} {'mm':>11s} {'min pt':>7s} {'<7pt':>5s} {'lines pt':>10s}  status")
for r in rows:
    print(f"{r[0]:28s} {r[1]:>11s} {r[2]:>7} {r[3]:>5} {r[4]:>10s}  {r[5]}")
sys.exit(1 if fail else 0)
