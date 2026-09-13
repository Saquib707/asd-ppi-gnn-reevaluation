"""
Step 6: biological sanity check with an independent, curated label.

Question: do the same five network features predict which SFARI genes carry
the highest evidence score (score 1, "high confidence") rather than scores
2-3, and is any such signal simply degree -- i.e. how well studied a gene is?

  label       SFARI gene score 1 vs 2/3, among SFARI-scored genes in the network
  models      majority; degree, eigenvector and report-count rankers; LogReg,
              random forest, MLP; GCN, GraphSAGE, GAT-LR (step-3 settings)
  graph       unscored genes stay in the graph for message passing but are
              never trained on or scored
  diagnostic  Spearman correlation of degree with SFARI number of reports

The report-count ranker diagnoses study bias; it is not a competitor. SFARI
scores are assigned from published evidence, so report counts are partly the
label's own definition.

Usage:  python 06_sfari_check.py [--smoke]
Writes: results/exp_sfari.csv, results/sfari_diagnostics.json
"""
import importlib.util
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, wilcoxon
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

SMOKE = "--smoke" in sys.argv
SEED, N_FOLDS = 42, 5
N_REPEATS = 1 if SMOKE else 5
EPOCHS = 5 if SMOKE else M.EPOCHS
RANKERS = [("degree", "Degree"), ("eig", "Eigenvector"),
           ("n_reports", "SFARI reports (diagnostic)")]
_G = {}


def init_worker(payload):
    torch.set_num_threads(1)
    _G.update(payload)


def minmax(s):
    return (s - s.min()) / (np.ptp(s) + 1e-12)


def run_task(task):
    rep, fold, tr, te = task
    X, A, y = _G["X"], _G["A"], _G["y"]
    base = dict(experiment="sfari", rep=rep, fold=fold)
    t0 = time.time()
    rows = [dict(base, model="Majority",
                 **M.metrics(y[te], np.full(len(te), y[tr].mean())))]
    for key, name in RANKERS:
        rows.append(dict(base, model=name, **M.metrics(y[te], minmax(_G[key][te]))))
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
    for kind, name in (("gcn", "GCN"), ("sage", "GraphSAGE"), ("gat", "GAT-LR")):
        p, _ = M.train_gnn(kind, X, A, y, tr, te, seed=SEED + rep * 10 + fold,
                           epochs=EPOCHS)
        rows.append(dict(base, model=name, **M.metrics(y[te], p[te])))
    for r in rows:
        r["task_seconds"] = round(time.time() - t0, 1)
    return rows


def main():
    feats = pd.read_csv(os.path.join(DATA, "features.csv"))
    nodes = pd.read_csv(os.path.join(DATA, "nodes.csv"))
    df = feats.merge(nodes[["gene", "sfari_score", "n_reports"]], on="gene", how="left")
    genes = df["gene"].tolist()
    gi = {g: i for i, g in enumerate(genes)}
    A = np.zeros((len(genes), len(genes)), dtype=np.float32)
    edges = pd.read_csv(os.path.join(DATA, "edges.csv"))
    for a, b in zip(edges["source"], edges["target"]):
        if a in gi and b in gi:
            A[gi[a], gi[b]] = A[gi[b], gi[a]] = 1.0

    scored = df["sfari_score"].notna().values
    idx = np.flatnonzero(scored)
    y = np.zeros(len(genes), dtype=int)
    y[idx] = (df["sfari_score"].values[idx] == 1).astype(int)
    degree = df["degree"].values.astype(float)
    reports = df["n_reports"].fillna(0).values.astype(float)

    rho, p_rho = spearmanr(degree[idx], reports[idx])
    diag = {
        "genes_in_network": len(genes),
        "sfari_scored_in_network": int(scored.sum()),
        "score1_positives": int(y[idx].sum()),
        "positive_rate": float(y[idx].mean()),
        "spearman_degree_vs_reports": float(rho),
        "spearman_p": float(p_rho),
        "degree_auc_global": float(roc_auc_score(y[idx], degree[idx])),
        "reports_auc_global": float(roc_auc_score(y[idx], reports[idx])),
    }
    print(json.dumps(diag, indent=2), flush=True)

    payload = {"X": StandardScaler().fit_transform(df[M.FEATURES].values), "A": A,
               "y": y, "degree": degree,
               "eig": df["eigenvector_centrality"].values, "n_reports": reports}
    tasks = []
    for rep in range(N_REPEATS):
        skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED + rep)
        for fold, (tr, te) in enumerate(skf.split(idx, y[idx])):
            tasks.append((rep, fold, idx[tr], idx[te]))
    if SMOKE:
        tasks = tasks[:1]

    nw = 2 if SMOKE else max(1, (os.cpu_count() or 4) - 1)
    print(f"{len(tasks)} tasks, {nw} workers, {EPOCHS} epochs"
          f"{' [SMOKE]' if SMOKE else ''}", flush=True)
    rows, t0 = [], time.time()
    with Pool(nw, initializer=init_worker, initargs=(payload,)) as pool:
        for i, r in enumerate(pool.imap_unordered(run_task, tasks), 1):
            rows.extend(r)
            print(f"  [{i:2d}/{len(tasks)}] {r[0]['task_seconds']:6.1f}s"
                  f"  elapsed {(time.time()-t0)/60:4.1f} min", flush=True)

    raw = pd.DataFrame(rows)
    tag = "smoke_" if SMOKE else ""
    raw.to_csv(os.path.join(RES, f"{tag}exp_sfari.csv"), index=False)

    cols = ["roc_auc", "pr_auc", "accuracy", "f1"]
    pd.set_option("display.width", 200)
    print("\n=== SFARI score-1 prediction (mean, std) ===")
    print(raw.groupby("model")[cols].agg(["mean", "std"]).round(4).to_string())

    sig = []
    ref = raw[raw.model == "Degree"].sort_values(["rep", "fold"])
    for m in raw.model.unique():
        if m == "Degree":
            continue
        other = raw[raw.model == m].sort_values(["rep", "fold"])
        for metric in ("roc_auc", "pr_auc"):
            a, b = other[metric].values, ref[metric].values
            try:
                p = wilcoxon(a, b).pvalue if len(a) > 5 else np.nan
            except ValueError:
                p = np.nan
            sig.append(dict(comparison=f"{m} vs Degree", metric=metric,
                            model_mean=a.mean(), degree_mean=b.mean(),
                            delta=a.mean() - b.mean(), p_value=p))
    diag["significance_vs_degree"] = sig
    with open(os.path.join(RES, f"{tag}sfari_diagnostics.json"), "w") as f:
        json.dump(diag, f, indent=2)
    print("\n=== vs Degree ranker (paired Wilcoxon) ===")
    print(pd.DataFrame(sig).round(5).to_string(index=False))
    print(f"\nTotal wall time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
