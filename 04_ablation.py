"""
Step 4: robustness, mechanism and leakage experiments (parallel over folds).

Motivation (from step 3): degree centrality alone ranks the SI influence label
with ROC-AUC 0.996, and non-graph learners on the five centralities beat
GAT-LR. The experiments below test whether that conclusion is an artefact of
one horizon or threshold, whether any model captures influence beyond degree,
why GAT underperforms, and how much transductive evaluation inflates the GNNs.

  horizon     model comparison at SI horizons T = 1,3,4,5 (T = 2 is step 3)
  threshold   model comparison at IC percentiles 50 and 90 (75 is step 3)
  residual    influence NOT explained by degree: a gene is positive if its IC
              is in the top quartile among genes of comparable degree (degree
              ventiles); bins and thresholds fit on training nodes only
  skip        GAT-LR with a root (skip) term per layer; same folds and seeds
              as step 3, so it pairs fold-by-fold with the step-3 GAT-LR
  inductive   GNNs trained with test nodes removed from the graph, scored on
              the full graph; paired with transductive training per fold
  importance  permutation importance for GAT-LR and random forest

Every GNN uses the step-3 settings (300 epochs, Adam lr 0.01, wd 5e-4,
dropout 0.2, hidden 8, 4 heads, class-weighted BCE).

The workload is ~160 independent fits on a small graph. Measurement showed the
GAT is overhead-bound (136 ms/epoch on 1 thread vs 111 ms on 4), so throughput
comes from many single-threaded worker processes, not threads or a GPU.

Usage:  python 04_ablation.py                     full run
        python 04_ablation.py --smoke             one fold per experiment, 5 epochs
        python 04_ablation.py --only residual     run a subset (comma-separated)
        python 04_ablation.py --exclude residual  run everything but a subset
Writes: results/exp_<experiment>.csv  (one row per model per fold)
"""
import importlib.util
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, RES = os.path.join(HERE, "data"), os.path.join(HERE, "results")

_spec = importlib.util.spec_from_file_location("m3", os.path.join(HERE, "03_models.py"))
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)

from gat_sparse import SparseGATLayer, build_edge_index  # noqa: E402

def _arg_list(flag):
    if flag in sys.argv and sys.argv.index(flag) + 1 < len(sys.argv):
        return set(sys.argv[sys.argv.index(flag) + 1].split(","))
    return set()


SMOKE = "--smoke" in sys.argv
ONLY, EXCLUDE = _arg_list("--only"), _arg_list("--exclude")
SEED, N_FOLDS = 42, 5
EPOCHS = 5 if SMOKE else M.EPOCHS
FEATURES = M.FEATURES
GNNS = [("gcn", "GCN"), ("sage", "GraphSAGE"), ("gat", "GAT-LR")]
_G = {}


def init_worker(payload):
    torch.set_num_threads(1)
    _G.update(payload)


# ---------------------------------------------------------------- models
class GATSkip(nn.Module):
    """GAT-LR plus a root (skip) linear term in each layer, so a node's own
    features reach the output without being averaged with its neighbours'."""

    def __init__(self, fin, hidden=8, heads=4, dropout=0.2):
        super().__init__()
        self.kind, self.dropout, self.edge_index = "gat", dropout, None
        self.l1 = SparseGATLayer(fin, hidden, heads=heads, concat=True, dropout=dropout)
        self.r1 = nn.Linear(fin, hidden * heads)
        self.l2 = SparseGATLayer(hidden * heads, 2, heads=1, concat=False,
                                 dropout=dropout)
        self.r2 = nn.Linear(hidden * heads, 2)
        self.head = nn.Linear(2, 1)

    def forward(self, x, adj_n, adj):
        h = F.leaky_relu(self.l1(x, self.edge_index) + self.r1(x), 0.01)
        h = F.dropout(h, self.dropout, self.training)
        emb = F.leaky_relu(self.l2(h, self.edge_index) + self.r2(h), 0.01)
        return self.head(emb).squeeze(-1)


def graph(kind, A):
    adj = torch.tensor(A, dtype=torch.float32)
    adj_n = M.norm_adj(adj) if kind == "gcn" else None
    ei = build_edge_index(A) if kind in ("gat", "gat_skip") else None
    return adj, adj_n, ei


def fit_gnn(kind, X, A_train, A_infer, y, tr, seed):
    """Same procedure and RNG order as 03_models.train_gnn, except that the
    graph used for training may differ from the one used for inference."""
    torch.manual_seed(seed)
    x = torch.tensor(X, dtype=torch.float32)
    t = torch.tensor(y, dtype=torch.float32)
    tri = torch.tensor(tr, dtype=torch.long)
    g_train = graph(kind, A_train)
    g_infer = g_train if A_infer is A_train else graph(kind, A_infer)
    model = GATSkip(X.shape[1]) if kind == "gat_skip" else M.GNN(kind, X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    pos_w = torch.tensor([(t[tri] == 0).sum() / max((t[tri] == 1).sum(), 1)])

    adj, adj_n, model.edge_index = g_train
    for _ in range(EPOCHS):
        model.train()
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(model(x, adj_n, adj)[tri], t[tri],
                                           pos_weight=pos_w).backward()
        opt.step()
    model.eval()
    adj, adj_n, model.edge_index = g_infer
    with torch.no_grad():
        return torch.sigmoid(model(x, adj_n, adj)).numpy(), model


# ---------------------------------------------------------------- labels
N_DEGREE_BINS = 20


def degree_bins(deg, fit_idx):
    """Degree ventiles estimated on the fitting nodes, applied to every node."""
    edges = np.unique(np.quantile(deg[fit_idx], np.linspace(0, 1, N_DEGREE_BINS + 1)))
    b = pd.cut(deg, edges, labels=False, include_lowest=True, right=True)
    b = np.where(np.isnan(b), np.where(deg < edges[0], 0, len(edges) - 2), b)
    return b.astype(int)


def labels(T, pctl, tr=None, residual=False):
    icv = _G["ic"][T]
    if not residual:
        return (icv > np.percentile(icv, pctl)).astype(int)
    # Influence beyond degree: a gene is positive if its IC exceeds the pctl-th
    # percentile of IC among genes of comparable degree (same degree ventile).
    # A plain residual from an isotonic fit on degree was rejected: degree still
    # ranked it at AUC 0.62, because IC spread grows with degree.
    # Bins and thresholds come from the fitting (training) nodes only.
    fit = np.zeros(len(icv), dtype=bool)
    fit[np.arange(len(icv)) if tr is None else tr] = True
    b = degree_bins(_G["degree"], np.flatnonzero(fit))
    y = np.zeros(len(icv), dtype=int)
    for k in np.unique(b):
        ref = icv[(b == k) & fit]
        if ref.size:
            y[b == k] = icv[b == k] > np.percentile(ref, pctl)
    return y


# ---------------------------------------------------------------- per-fold rows
def minmax(s):
    return (s - s.min()) / (np.ptp(s) + 1e-12)


def baseline_rows(y, tr, te, base):
    rows = [dict(base, model="Majority",
                 **M.metrics(y[te], np.full(len(te), y[tr].mean())))]
    for name, s in (("Degree", _G["degree"]), ("Eigenvector", _G["eig"])):
        rows.append(dict(base, model=name, **M.metrics(y[te], minmax(s[te]))))
    return rows


def sklearn_rows(X, y, tr, te, base, rep):
    rows = []
    for name, clf in (
        ("LogReg", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ("RandomForest", RandomForestClassifier(n_estimators=400, n_jobs=1,
                                                class_weight="balanced",
                                                random_state=SEED + rep)),
        ("MLP", MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000,
                              random_state=SEED + rep)),
    ):
        clf.fit(X[tr], y[tr])
        rows.append(dict(base, model=name,
                         **M.metrics(y[te], clf.predict_proba(X[te])[:, 1])))
    return rows


def run_task(task):
    exp, cfg, rep, fold, tr, te = task
    X, A = _G["X"], _G["A"]
    seed = SEED + rep * 10 + fold
    base = dict(experiment=exp, **cfg, rep=rep, fold=fold)
    t0 = time.time()
    rows = []

    if exp in ("horizon", "threshold", "residual"):
        y = labels(cfg["T"], cfg["pctl"], tr, residual=(exp == "residual"))
        base["test_positive_rate"] = float(y[te].mean())
        rows += baseline_rows(y, tr, te, base)
        rows += sklearn_rows(X, y, tr, te, base, rep)
        kinds = GNNS + ([("gat_skip", "GAT-LR+skip")] if exp == "residual" else [])
        for kind, name in kinds:
            p, _ = fit_gnn(kind, X, A, A, y, tr, seed)
            rows.append(dict(base, model=name, **M.metrics(y[te], p[te])))

    elif exp == "skip":
        y = labels(cfg["T"], cfg["pctl"])
        p, _ = fit_gnn("gat_skip", X, A, A, y, tr, seed)
        rows.append(dict(base, model="GAT-LR+skip", **M.metrics(y[te], p[te])))

    elif exp == "inductive":
        y = labels(cfg["T"], cfg["pctl"])
        A_tr = A.copy()
        A_tr[te, :] = 0.0          # test nodes send and receive no messages
        A_tr[:, te] = 0.0          # while the model is being trained
        for kind, name in GNNS:
            p_t, _ = fit_gnn(kind, X, A, A, y, tr, seed)
            p_i, _ = fit_gnn(kind, X, A_tr, A, y, tr, seed)
            rows.append(dict(base, model=name, setting="transductive",
                             **M.metrics(y[te], p_t[te])))
            rows.append(dict(base, model=name, setting="inductive",
                             **M.metrics(y[te], p_i[te])))

    elif exp == "importance":
        y = labels(cfg["T"], cfg["pctl"])
        p, gat = fit_gnn("gat", X, A, A, y, tr, seed)
        rf = RandomForestClassifier(n_estimators=400, n_jobs=1,
                                    class_weight="balanced",
                                    random_state=SEED + rep).fit(X[tr], y[tr])

        def score(name, Xq):
            if name == "RandomForest":
                return rf.predict_proba(Xq[te])[:, 1]
            with torch.no_grad():   # gat.edge_index is the full graph here
                return torch.sigmoid(
                    gat(torch.tensor(Xq, dtype=torch.float32), None, None)).numpy()[te]

        rng = np.random.default_rng(seed)
        for name in ("GAT-LR", "RandomForest"):
            ref = roc_auc_score(y[te], score(name, X))
            for j, feat in enumerate(FEATURES):
                drops = []
                for _ in range(10):
                    Xp = X.copy()
                    Xp[:, j] = rng.permutation(Xp[:, j])
                    drops.append(ref - roc_auc_score(y[te], score(name, Xp)))
                rows.append(dict(base, model=name, feature=feat, base_auc=ref,
                                 auc_drop=float(np.mean(drops))))

    for r in rows:
        r["task_seconds"] = round(time.time() - t0, 1)
    return exp, rows


# ---------------------------------------------------------------- task list
def splits(y_strat, repeats):
    out = []
    for rep in range(repeats):
        skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED + rep)
        for fold, (tr, te) in enumerate(skf.split(np.zeros(len(y_strat)), y_strat)):
            out.append((rep, fold, tr, te))
    return out


def build_tasks():
    """Heaviest experiments first, for better load balancing."""
    plan = [
        ("residual", dict(T=2, pctl=75), 5),
        ("inductive", dict(T=2, pctl=75), 3),
        *[("horizon", dict(T=T, pctl=75), 3) for T in (1, 3, 4, 5)],
        *[("threshold", dict(T=2, pctl=p), 3) for p in (50, 90)],
        ("importance", dict(T=2, pctl=75), 1),
        ("skip", dict(T=2, pctl=75), 5),
    ]
    tasks = []
    for exp, cfg, repeats in plan:
        if (ONLY and exp not in ONLY) or exp in EXCLUDE:
            continue
        # Stratify on the global label. For "residual" the global label is
        # used ONLY to build balanced splits; training labels are refit per fold.
        y_strat = labels(cfg["T"], cfg["pctl"], residual=(exp == "residual"))
        sp = splits(y_strat, repeats)
        if SMOKE:
            sp = sp[:1]
        tasks += [(exp, cfg, rep, fold, tr, te) for rep, fold, tr, te in sp]
    return tasks


def load_payload():
    feats = pd.read_csv(os.path.join(DATA, "features.csv"))
    ic = pd.read_csv(os.path.join(DATA, "ic_scores.csv"))
    df = feats.merge(ic, on="gene")
    genes = df["gene"].tolist()
    gi = {g: i for i, g in enumerate(genes)}
    A = np.zeros((len(genes), len(genes)), dtype=np.float32)
    edges = pd.read_csv(os.path.join(DATA, "edges.csv"))
    for a, b in zip(edges["source"], edges["target"]):
        if a in gi and b in gi:
            A[gi[a], gi[b]] = A[gi[b], gi[a]] = 1.0
    return {
        "X": StandardScaler().fit_transform(df[FEATURES].values),
        "A": A,
        "ic": {T: df[f"ic_T{T}"].values for T in range(1, 6)},
        "degree": df["degree"].values.astype(float),
        "eig": df["eigenvector_centrality"].values,
    }


def summarise(df, keys):
    cols = [c for c in ("roc_auc", "pr_auc", "accuracy", "f1") if c in df]
    return df.groupby(keys)[cols].agg(["mean", "std"]).round(4)


def main():
    payload = load_payload()
    _G.update(payload)
    tasks = build_tasks()
    nw = 2 if SMOKE else max(1, (os.cpu_count() or 4) - 1)
    print(f"{len(tasks)} tasks, {nw} workers, {EPOCHS} epochs"
          f"{' [SMOKE]' if SMOKE else ''}", flush=True)

    out = {}
    t0 = time.time()
    tag = "smoke_" if SMOKE else ""
    with Pool(nw, initializer=init_worker, initargs=(payload,)) as pool:
        for i, (exp, rows) in enumerate(pool.imap_unordered(run_task, tasks), 1):
            out.setdefault(exp, []).extend(rows)
            print(f"  [{i:3d}/{len(tasks)}] {exp:10s} {rows[0]['task_seconds']:6.1f}s"
                  f"  elapsed {(time.time()-t0)/60:5.1f} min", flush=True)
            if i % 10 == 0 or i == len(tasks):
                for e, r in out.items():
                    pd.DataFrame(r).to_csv(os.path.join(RES, f"{tag}exp_{e}.csv"),
                                           index=False)

    pd.set_option("display.width", 200)
    for exp, rows in out.items():
        df = pd.DataFrame(rows)
        print(f"\n=== {exp} ===")
        if exp == "importance":
            print(df.groupby(["model", "feature"])["auc_drop"]
                  .agg(["mean", "std"]).round(4).to_string())
        elif exp == "inductive":
            print(summarise(df, ["model", "setting"]).to_string())
        else:
            keys = [k for k in ("T", "pctl") if df[k].nunique() > 1] + ["model"]
            print(summarise(df, keys).to_string())
    print(f"\nTotal wall time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
