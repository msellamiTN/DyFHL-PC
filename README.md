# DyFHL-PA: Governing Change in Cross-Organisation Agent and Tool Ecosystems

Code, experiments, results and LaTeX source of the paper *DyFHL-PA: Governing Change in Cross-Organisation Agent and Tool Ecosystems with Federated Dynamic Policy Hypergraphs* (target: Computers & Security).

## Layout
- `dyfhl/` library: `federation.py` (federated dynamic policy hypergraph, events, regions, incremental admission), `mpc.py` (two-server secret sharing, exact checks), `secure_fed.py` (flags-only cross-owner checks), `iam.py` (AWS IAM documents as rules), `composed.py` (exact analysis), plus the earlier sketch design.
- `scripts/` experiments: `extract_history.py` + `aws_history.py` (RQ2, AWS managed-policy history), `census_tools.py` (RQ3), `federation_eval.py` (RQ1, RQ4–RQ6), `secure_exact.py` (static comparison), `kill_test_demand.py` (RQ7), `make_figures_v2.py` (figures, tables, numbers).
- `tests/` property tests (`test_composed.py`, `test_v4_model.py`, `test_federation.py`).
- `results/` raw result files; `data/history/` replayed AWS change events (gzip); `paper/` LaTeX (`main.tex`) and compiled `main.pdf`.

## Reproduce
```
pip install -r requirements.txt
python3 tests/test_federation.py
gunzip -k data/history/*.gz
python3 -I scripts/aws_history.py data/history results/aws_history.json
python3 scripts/federation_eval.py results/federation.json
python3 scripts/make_figures_v2.py 
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```
External data: MAMIP (https://github.com/z0ph/MAMIP), MCP Atlas Census (https://github.com/DSHCorrectover/mcp-atlas-census), Amazon Employee Access (Kaggle).
