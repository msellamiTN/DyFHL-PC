"""Hypergraph projections of policies and their anomalies (Plotly)."""
from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

LABELS = ["none", "x-conflict", "x-redundancy", "x-irrelevancy", "several"]
COLORS = {"none": "#b8bec7", "x-conflict": "#d1495b", "x-redundancy": "#2e86ab", "x-irrelevancy": "#e0a100",
          "several": "#6a4c93", "x-gap": "#3a7d44", "covered request": "#d9dde2"}


def anomaly_class(Y):
    k = Y.sum(1)
    return np.where(k == 0, 0, np.where(k > 1, 4, 1 + Y.argmax(1)))


def embedding_projection(R, Y, pred=None, method="svd", max_points=3000, seed=0, title="Policy hypergraph projection"):
    """2-D projection of hyperedges (rules) from their incidence rows, coloured by anomaly type."""
    X = np.concatenate([R["H"], R["Wm"]], 1).astype(np.float32)
    idx = np.arange(len(X))
    if len(idx) > max_points:
        idx = np.random.default_rng(seed).choice(idx, max_points, replace=False)
    Z = TruncatedSVD(n_components=min(20, X.shape[1] - 1), random_state=seed).fit_transform(X[idx])
    Z = TSNE(n_components=2, init="pca", random_state=seed, perplexity=30).fit_transform(Z) if method == "tsne" else Z[:, :2]
    Z = Z + np.random.default_rng(seed).normal(0, 0.01 * Z.std(), Z.shape)
    cls = anomaly_class(Y[idx])
    fig = go.Figure()
    for c, name in enumerate(LABELS):
        m = cls == c
        if not m.any():
            continue
        text = [f"rule {R['uid'][i]} | domain {R['dom'][i]} | scope {R['scope'][i]} | {'deny' if R['eff'][i] else 'permit'}"
                for i in idx[m]]
        sym = "circle"
        if pred is not None:
            wrong = (pred[idx[m]] != Y[idx[m]]).any(1)
            sym = np.where(wrong, "x", "circle")
        fig.add_trace(go.Scattergl(x=Z[m, 0], y=Z[m, 1], mode="markers", name=name, text=text, hoverinfo="text",
                                   marker=dict(size=6 if c else 4, color=COLORS[name], symbol=sym,
                                               opacity=0.85 if c else 0.45)))
    fig.update_layout(title=title, template="plotly_white", height=560, legend_title="oracle label",
                      xaxis_title="component 1", yaxis_title="component 2")
    return fig


def bipartite_view(world, R, Y, scope=0, max_rules=60, seed=0):
    """Star expansion of one collaboration scope: rules (hyperedges) on the left, attribute values
    (vertices) on the right; anomalous rules are coloured, edges are incidences."""
    idx = np.flatnonzero(R["scope"] == scope)
    cls = anomaly_class(Y[idx])
    order = np.argsort(-cls, kind="stable")[:max_rules]
    idx, cls = idx[order], cls[order]
    used = np.flatnonzero(R["H"][idx].sum(0) > 0)
    names = {v: f"{a}={val}" for (a, val), v in world.vidx.items()}
    yv = {v: i for i, v in enumerate(used)}
    ex, ey = [], []
    for r, i in enumerate(idx):
        for v in np.flatnonzero(R["H"][i]):
            ex += [0, 1, None]; ey += [r * len(used) / max(1, len(idx)), yv[v], None]
    fig = go.Figure(go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=0.4, color="#c5cad1"), hoverinfo="skip",
                               showlegend=False))
    for c, name in enumerate(LABELS):
        m = cls == c
        if m.any():
            fig.add_trace(go.Scatter(x=np.zeros(m.sum()), y=np.flatnonzero(m) * len(used) / max(1, len(idx)),
                                     mode="markers", name=name, marker=dict(size=9, color=COLORS[name]),
                                     text=[f"rule {R['uid'][i]} (domain {R['dom'][i]})" for i in idx[m]], hoverinfo="text"))
    fig.add_trace(go.Scatter(x=np.ones(len(used)), y=list(range(len(used))), mode="markers", name="attribute value",
                             marker=dict(size=7, color="#30343b", symbol="square"),
                             text=[names[v] for v in used], hoverinfo="text"))
    fig.update_layout(title=f"Scope {scope}: rules (left) and attribute-value vertices (right)", template="plotly_white",
                      height=620, xaxis=dict(visible=False, range=[-0.2, 1.2]), yaxis=dict(visible=False))
    return fig
