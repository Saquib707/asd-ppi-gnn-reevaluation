"""
Step 3: full comparative evaluation.

Every model sees the identical feature matrix, identical labels and identical
folds. GNN layers are implemented directly in PyTorch so the pipeline has no
compiled graph-library dependency and is fully auditable. GAT uses the
edge-indexed formulation in gat_sparse.py (verified identical to the dense one).

Models
  majority          predict the majority class (the floor accuracy must clear)
  DC/BC/CC/EC/Clust rank by a single centrality (the source paper's only baselines)
  RWR               random walk with restart (domain-standard propagation)
  LogReg/RF/GBM/MLP non-graph learners on the SAME 5 features  <-- decisive control
  GCN / GraphSAGE   standard GNN comparators
  GAT-LR            the proposed model

Writes: results/metrics_raw.csv, results/metrics_summary.csv,
        results/significance.csv, results/rankings.csv, results/degree_overlap.json
"""
import json
import os

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import wilcoxon, spearmanr
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from gat_sparse import SparseGATLayer, build_edge_index

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

SEED = 42
N_FOLDS = 5
N_REPEATS = 5          # repeated CV -> honest variance estimates
LABEL_PCTL = 75        # dissertation's rule; sensitivity reported separately
IC_HORIZON = None      # chosen in main() from label_sensitivity.json
EPOCHS = 300
FEATURES = ["degree_centrality", "betweenness_centrality", "closeness_centrality",
            "eigenvector_centrality", "clustering_coefficient"]


# ---------------------------------------------------------------- graph utils
def norm_adj(A):
    A_hat = A + torch.eye(A.shape[0])
    d = A_hat.sum(1)
    dinv = torch.pow(d, -0.5)
    dinv[torch.isinf(dinv)] = 0
    D = torch.diag(dinv)
    return D @ A_hat @ D


def rwr_scores(A, restart=0.3, iters=100):
    """Random walk with restart, averaged over all seeds."""
    n = A.shape[0]
    P = A / np.clip(A.sum(0, keepdims=True), 1e-12, None)
    E = np.eye(n)
    R = E.copy()
    for _ in range(iters):
        R = (1 - restart) * P @ R + restart * E
    return R.mean(1)


# ---------------------------------------------------------------- GNN modules
class GCNLayer(nn.Module):
    def __init__(self, fi, fo):
        super().__init__()
        self.lin = nn.Linear(fi, fo)

    def forward(self, x, adj_n, adj=None):
        return adj_n @ self.lin(x)


class SAGELayer(nn.Module):
    def __init__(self, fi, fo):
        super().__init__()
        self.self_lin = nn.Linear(fi, fo)
        self.nbr_lin = nn.Linear(fi, fo)

    def forward(self, x, adj_n, adj=None):
        deg = adj.sum(1, keepdim=True).clamp(min=1)
        agg = (adj @ x) / deg
        return self.self_lin(x) + self.nbr_lin(agg)


class GATLayer(nn.Module):
    """Multi-head graph attention (Velickovic et al., 2018), dense formulation."""

    def __init__(self, fi, fo, heads=4, concat=True, dropout=0.2, alpha=0.01):
        super().__init__()
        self.heads, self.fo, self.concat = heads, fo, concat
        self.W = nn.Parameter(torch.empty(heads, fi, fo))
        self.a_src = nn.Parameter(torch.empty(heads, fo))
        self.a_dst = nn.Parameter(torch.empty(heads, fo))
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))
        self.leaky = nn.LeakyReLU(alpha)
        self.dropout = dropout

    def forward(self, x, adj_n, adj):
        h = torch.einsum("nf,hfo->hno", x, self.W)              # H x N x F'
        es = (h * self.a_src[:, None, :]).sum(-1)               # H x N
        ed = (h * self.a_dst[:, None, :]).sum(-1)               # H x N
        e = self.leaky(es[:, :, None] + ed[:, None, :])         # H x N x N
        mask = (adj + torch.eye(adj.shape[0], device=adj.device)) > 0
        e = e.masked_fill(~mask.unsqueeze(0), float("-inf"))
        att = torch.softmax(e, dim=-1)
        att = F.dropout(att, self.dropout, self.training)
        out = torch.einsum("hij,hjo->hio", att, h)              # H x N x F'
        self.last_attention = att.detach()
        if self.concat:
            return out.permute(1, 0, 2).reshape(x.shape[0], -1)
        return out.mean(0)


class GNN(nn.Module):
    """Two message-passing layers + a linear (logistic-regression) output head."""

    def __init__(self, kind, fin, hidden=8, heads=4, dropout=0.2):
        super().__init__()
        self.kind, self.dropout = kind, dropout
        self.edge_index = None          # set by train_gnn for the GAT path
        if kind == "gat":
            # Edge-indexed attention: mathematically identical to the dense
            # formulation in GATLayer, ~4x faster on this graph (density 0.02).
            self.l1 = SparseGATLayer(fin, hidden, heads=heads, concat=True,
                                     dropout=dropout)
            self.l2 = SparseGATLayer(hidden * heads, 2, heads=1, concat=False,
                                     dropout=dropout)
        elif kind == "gcn":
            self.l1, self.l2 = GCNLayer(fin, hidden * heads), GCNLayer(hidden * heads, 2)
        else:
            self.l1, self.l2 = SAGELayer(fin, hidden * heads), SAGELayer(hidden * heads, 2)
        self.head = nn.Linear(2, 1)          # the "LR layer"

    def forward(self, x, adj_n, adj, return_emb=False):
        if self.kind == "gat":
            h = F.leaky_relu(self.l1(x, self.edge_index), 0.01)
            h = F.dropout(h, self.dropout, self.training)
            emb = F.leaky_relu(self.l2(h, self.edge_index), 0.01)
        else:
            h = F.leaky_relu(self.l1(x, adj_n, adj), 0.01)
            h = F.dropout(h, self.dropout, self.training)
            emb = F.leaky_relu(self.l2(h, adj_n, adj), 0.01)
        logit = self.head(emb).squeeze(-1)
        return (logit, emb) if return_emb else logit


def train_gnn(kind, X, A, y, tr_idx, te_idx, seed, epochs=EPOCHS,
              lr=0.01, wd=5e-4, hidden=8, heads=4, dropout=0.2):
    torch.manual_seed(seed)
    adj = torch.tensor(A, dtype=torch.float32)
    adj_n = norm_adj(adj)
    x = torch.tensor(X, dtype=torch.float32)
    t = torch.tensor(y, dtype=torch.float32)
    tr = torch.tensor(tr_idx, dtype=torch.long)
    model = GNN(kind, X.shape[1], hidden, heads, dropout)
    if kind == "gat":
        model.edge_index = build_edge_index(A)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    pos_w = torch.tensor([(t[tr] == 0).sum() / max((t[tr] == 1).sum(), 1)])
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(x, adj_n, adj)
        loss = F.binary_cross_entropy_with_logits(out[tr], t[tr], pos_weight=pos_w)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(x, adj_n, adj)).numpy()
    return prob, model


# ---------------------------------------------------------------- evaluation
def metrics(y_true, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred) * 100,
        "precision": precision_score(y_true, pred, zero_division=0) * 100,
        "recall": recall_score(y_true, pred, zero_division=0) * 100,
        "f1": f1_score(y_true, pred, zero_division=0) * 100,
        "roc_auc": roc_auc_score(y_true, prob) if len(set(y_true)) > 1 else np.nan,
        "pr_auc": average_precision_score(y_true, prob),
    }


def main():
    feats = pd.read_csv(os.path.join(DATA, "features.csv"))
    ic = pd.read_csv(os.path.join(DATA, "ic_scores.csv"))
    sens = json.load(open(os.path.join(DATA, "label_sensitivity.json")))

    # pick the SI horizon with the most discriminative (highest-variance) IC
    horizon = max(sens["horizons"], key=lambda k: sens["horizons"][k]["cv"])
    print(f"Selected SI horizon: {horizon} (cv={sens['horizons'][horizon]['cv']:.4f})")
    ic_col = "ic_" + horizon

    df = feats.merge(ic[["gene", ic_col]], on="gene")
    genes = df["gene"].tolist()
    X = StandardScaler().fit_transform(df[FEATURES].values)
    ic_vals = df[ic_col].values
    thr = np.percentile(ic_vals, LABEL_PCTL)
    y = (ic_vals > thr).astype(int)
    n_pos = int(y.sum())
    print(f"N={len(y)}  positives={n_pos} ({100*n_pos/len(y):.2f}%)  "
          f"majority baseline acc={100*max(n_pos, len(y)-n_pos)/len(y):.2f}%")

    edges = pd.read_csv(os.path.join(DATA, "edges.csv"))
    gi = {g: i for i, g in enumerate(genes)}
    A = np.zeros((len(genes), len(genes)), dtype=np.float32)
    for a, b in zip(edges["source"], edges["target"]):
        if a in gi and b in gi:
            A[gi[a], gi[b]] = A[gi[b], gi[a]] = 1.0

    rwr = rwr_scores(A)
    cent = {c: df[c].values for c in FEATURES}

    rows = []
    prob_store = {}
    for rep in range(N_REPEATS):
        skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED + rep)
        for fold, (tr, te) in enumerate(skf.split(X, y)):
            yte = y[te]

            # --- trivial floor
            maj = np.full(len(te), y[tr].mean())
            rows.append(dict(model="Majority", rep=rep, fold=fold,
                             **metrics(yte, maj, thr=0.5)))

            # --- single-centrality rankers (the source paper's only baselines)
            for c in FEATURES:
                s = cent[c][te]
                s = (s - s.min()) / (np.ptp(s) + 1e-12)
                rows.append(dict(model=f"Centrality:{c.split('_')[0].upper()}",
                                 rep=rep, fold=fold, **metrics(yte, s)))

            # --- network propagation
            s = rwr[te]
            s = (s - s.min()) / (np.ptp(s) + 1e-12)
            rows.append(dict(model="RWR", rep=rep, fold=fold, **metrics(yte, s)))

            # --- non-graph learners on the SAME features (decisive control)
            for name, clf in [
                ("LogReg", LogisticRegression(max_iter=2000, class_weight="balanced")),
                ("RandomForest", RandomForestClassifier(
                    n_estimators=400, random_state=SEED + rep,
                    class_weight="balanced", n_jobs=-1)),
                ("GradBoost", GradientBoostingClassifier(random_state=SEED + rep)),
                ("MLP", MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000,
                                      random_state=SEED + rep)),
            ]:
                clf.fit(X[tr], y[tr])
                p = clf.predict_proba(X[te])[:, 1]
                rows.append(dict(model=name, rep=rep, fold=fold, **metrics(yte, p)))
                prob_store.setdefault(name, np.zeros(len(y)))[te] += p / N_REPEATS

            # --- graph neural networks
            for kind, label in [("gcn", "GCN"), ("sage", "GraphSAGE"), ("gat", "GAT-LR")]:
                p, model = train_gnn(kind, X, A, y, tr, te, seed=SEED + rep * 10 + fold)
                rows.append(dict(model=label, rep=rep, fold=fold,
                                 **metrics(yte, p[te])))
                prob_store.setdefault(label, np.zeros(len(y)))[te] += p[te] / N_REPEATS
        print(f"repeat {rep+1}/{N_REPEATS} done", flush=True)

    raw = pd.DataFrame(rows)
    raw.to_csv(os.path.join(RES, "metrics_raw.csv"), index=False)

    summ = (raw.groupby("model")
            .agg(["mean", "std"])[["accuracy", "precision", "recall", "f1",
                                   "roc_auc", "pr_auc"]]
            .round(4))
    summ.to_csv(os.path.join(RES, "metrics_summary.csv"))
    print("\n=== SUMMARY (mean over %d x %d folds) ===" % (N_REPEATS, N_FOLDS))
    print(summ.to_string())

    # --- paired significance: GAT-LR vs every other model, over folds
    sig = []
    base = raw[raw.model == "GAT-LR"].sort_values(["rep", "fold"])
    for m in raw.model.unique():
        if m == "GAT-LR":
            continue
        other = raw[raw.model == m].sort_values(["rep", "fold"])
        for metric in ["accuracy", "f1", "roc_auc", "pr_auc"]:
            a, b = base[metric].values, other[metric].values
            ok = ~(np.isnan(a) | np.isnan(b))
            try:
                stat, p = wilcoxon(a[ok], b[ok])
            except ValueError:
                stat, p = np.nan, np.nan
            sig.append(dict(comparison=f"GAT-LR vs {m}", metric=metric,
                            gat_mean=a[ok].mean(), other_mean=b[ok].mean(),
                            delta=a[ok].mean() - b[ok].mean(),
                            wilcoxon_stat=stat, p_value=p))
    sigdf = pd.DataFrame(sig)
    sigdf.to_csv(os.path.join(RES, "significance.csv"), index=False)

    # --- rankings + degree-overlap test
    nodes = pd.read_csv(os.path.join(DATA, "nodes.csv"))
    rank = pd.DataFrame({"gene": genes, "ic": ic_vals, "label": y,
                         "degree": df["degree"].values})
    for k, v in prob_store.items():
        rank[f"score_{k}"] = v
    rank = rank.merge(nodes[["gene", "sfari_score", "eagle", "syndromic"]],
                      on="gene", how="left")
    rank = rank.sort_values("score_GAT-LR", ascending=False)
    rank.to_csv(os.path.join(RES, "rankings.csv"), index=False)

    deg_rank = (-df["degree"].values).argsort().argsort()
    gat_rank = (-prob_store["GAT-LR"]).argsort().argsort()
    rho, pv = spearmanr(deg_rank, gat_rank)
    top = lambda s, k: set(np.argsort(-s)[:k])
    overlap = {f"top{k}_jaccard_with_degree":
               len(top(prob_store["GAT-LR"], k) & top(df["degree"].values, k)) /
               len(top(prob_store["GAT-LR"], k) | top(df["degree"].values, k))
               for k in (10, 25, 50, 100)}
    overlap.update({"spearman_rho_vs_degree": float(rho), "spearman_p": float(pv),
                    "n_positives": n_pos, "n_total": int(len(y)),
                    "positive_rate": float(n_pos / len(y)),
                    "majority_baseline_acc": float(
                        100 * max(n_pos, len(y) - n_pos) / len(y)),
                    "si_horizon": horizon, "label_percentile": LABEL_PCTL})
    with open(os.path.join(RES, "degree_overlap.json"), "w") as f:
        json.dump(overlap, f, indent=2)
    print("\n=== DEGREE-OVERLAP TEST ===")
    print(json.dumps(overlap, indent=2))
    print("\n=== SIGNIFICANCE (accuracy) ===")
    print(sigdf[sigdf.metric == "accuracy"].to_string(index=False))


if __name__ == "__main__":
    main()
