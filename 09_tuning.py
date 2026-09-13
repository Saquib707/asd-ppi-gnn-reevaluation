"""
Step 9: fair per-fold hyperparameter tuning.

Shchur et al. showed that comparisons between GNNs can change once every model
is tuned fairly, so the fixed-hyperparameter comparison of step 3 is repeated
with a small grid searched inside each training fold. For every model and
fold, each configuration is fitted on an inner training split (80% of the
training genes, stratified), scored on the inner validation split by PR-AUC,
and the best configuration is refitted on the whole training fold and scored
on the test fold. Degree is carried along untuned as the zero-parameter
reference, on the same folds, so the comparison stays paired.

Usage:  python 09_tuning.py [--smoke]
Writes: results/exp_tuning.csv (one row per model per fold, with the choice)
"""
import importlib.util
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, RES = os.path.join(HERE, "data"), os.path.join(HERE, "results")

_spec = importlib.util.spec_from_file_location("m3", os.path.join(HERE, "03_models.py"))
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)

SMOKE = "--smoke" in sys.argv
SEED, N_FOLDS = 42, 5
N_REPEATS = 1 if SMOKE else 3
EP = 5 if SMOKE else None          # None: use each configuration's own epoch count

# (lr, hidden, heads, dropout, epochs)
GAT_GRID = [(0.01, 8, 4, 0.2, 300), (0.01, 16, 4, 0.2, 300), (0.01, 8, 8, 0.2, 300),
            (0.01, 32, 4, 0.2, 300), (0.005, 16, 4, 0.2, 300), (0.05, 8, 4, 0.2, 300),
            (0.01, 8, 4, 0.5, 300), (0.01, 16, 4, 0.2, 150)]
SAGE_GRID = GAT_GRID
MLP_GRID = [dict(hidden_layer_sizes=h, alpha=a)
            for h in ((32, 16), (64, 32), (16,)) for a in (1e-4, 1e-2)]
RF_GRID = [dict(n_estimators=n, max_depth=d, min_samples_leaf=l)
           for n in (400,) for d in (None, 8) for l in (1, 5)]
LR_GRID = [dict(C=c) for c in (0.01, 0.1, 1.0, 10.0)]
_G = {}


def init_worker(payload):
    torch.set_num_threads(1)
    _G.update(payload)


def gnn_fit(kind, X, A, y, tr, seed, cfg):
    lr, hidden, heads, dropout, epochs = cfg
    return M.train_gnn(kind, X, A, y, tr, np.array([], dtype=int), seed,
                       epochs=EP or epochs, lr=lr, hidden=hidden, heads=heads,
                       dropout=dropout)[0]


def sk_fit(name, cfg, X, y, tr, rep):
    if name == "MLP":
        clf = MLPClassifier(max_iter=2000, random_state=SEED + rep, **cfg)
    elif name == "RandomForest":
        clf = RandomForestClassifier(n_jobs=1, class_weight="balanced",
                                     random_state=SEED + rep, **cfg)
    else:
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", **cfg)
    return clf.fit(X[tr], y[tr])


def run_task(task):
    rep, fold, tr, te = task
    X, A, y, deg = _G["X"], _G["A"], _G["y"], _G["degree"]
    seed = SEED + rep * 10 + fold
    base = dict(experiment="tuning", rep=rep, fold=fold)
    t0 = time.time()
    inner_tr, inner_va = train_test_split(tr, test_size=0.2, stratify=y[tr],
                                          random_state=seed)
    rows = []

    for kind, name, grid in (("gat", "GAT-LR", GAT_GRID),
                             ("sage", "GraphSAGE", SAGE_GRID)):
        scored = []
        for cfg in (grid[:2] if SMOKE else grid):
            p = gnn_fit(kind, X, A, y, inner_tr, seed, cfg)
            scored.append((M.metrics(y[inner_va], p[inner_va])["pr_auc"], cfg))
        best = max(scored, key=lambda z: z[0])[1]
        p = gnn_fit(kind, X, A, y, tr, seed, best)
        rows.append(dict(base, model=name, choice=str(best),
                         **M.metrics(y[te], p[te])))

    for name, grid in (("MLP", MLP_GRID), ("RandomForest", RF_GRID),
                       ("LogReg", LR_GRID)):
        scored = []
        for cfg in (grid[:2] if SMOKE else grid):
            clf = sk_fit(name, cfg, X, y, inner_tr, rep)
            pv = clf.predict_proba(X[inner_va])[:, 1]
            scored.append((M.metrics(y[inner_va], pv)["pr_auc"], cfg))
        best = max(scored, key=lambda z: z[0])[1]
        clf = sk_fit(name, best, X, y, tr, rep)
        rows.append(dict(base, model=name, choice=str(best),
                         **M.metrics(y[te], clf.predict_proba(X[te])[:, 1])))

    s = deg[te]
    rows.append(dict(base, model="Degree", choice="none",
                     **M.metrics(y[te], (s - s.min()) / (np.ptp(s) + 1e-12))))
    for r in rows:
        r["task_seconds"] = round(time.time() - t0, 1)
    return rows


def main():
    feats = pd.read_csv(os.path.join(DATA, "features.csv"))
    ic = pd.read_csv(os.path.join(DATA, "ic_scores.csv"))
    df = feats.merge(ic[["gene", "ic_T2"]], on="gene")
    genes = df["gene"].tolist()
    gi = {g: i for i, g in enumerate(genes)}
    A = np.zeros((len(genes), len(genes)), dtype=np.float32)
    edges = pd.read_csv(os.path.join(DATA, "edges.csv"))
    for a, b in zip(edges["source"], edges["target"]):
        if a in gi and b in gi:
            A[gi[a], gi[b]] = A[gi[b], gi[a]] = 1.0
    icv = df["ic_T2"].to_numpy()
    y = (icv > np.percentile(icv, 75)).astype(int)
    payload = {"X": StandardScaler().fit_transform(df[M.FEATURES].values), "A": A,
               "y": y, "degree": df["degree"].to_numpy(float)}

    tasks = []
    for rep in range(N_REPEATS):
        skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED + rep)
        for fold, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
            tasks.append((rep, fold, tr, te))
    if SMOKE:
        tasks = tasks[:1]
    nw = 2 if SMOKE else max(1, (os.cpu_count() or 4) - 1)
    print(f"{len(tasks)} folds, {nw} workers{' [SMOKE]' if SMOKE else ''}", flush=True)

    rows, t0 = [], time.time()
    with Pool(nw, initializer=init_worker, initargs=(payload,)) as pool:
        for i, r in enumerate(pool.imap_unordered(run_task, tasks), 1):
            rows.extend(r)
            print(f"  [{i:2d}/{len(tasks)}] {r[0]['task_seconds']:6.1f}s"
                  f"  elapsed {(time.time()-t0)/60:5.1f} min", flush=True)
            pd.DataFrame(rows).to_csv(
                os.path.join(RES, ("smoke_" if SMOKE else "") + "exp_tuning.csv"),
                index=False)

    raw = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n=== tuned models (mean, std) ===")
    print(raw.groupby("model")[["roc_auc", "pr_auc", "accuracy", "f1"]]
          .agg(["mean", "std"]).round(4).to_string())
    print("\n=== configurations chosen ===")
    for name, sub in raw.groupby("model"):
        print(f"  {name:14s} {sub['choice'].value_counts().to_dict()}")
    print(f"\nTotal wall time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
