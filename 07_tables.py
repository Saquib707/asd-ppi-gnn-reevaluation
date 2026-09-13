"""
Step 7: every number and table the paper quotes, generated from the results.

The paper never types a result by hand. This script writes LaTeX macros
(latex/generated/numbers.tex) and table bodies (latex/generated/tab_*.tex)
from the outputs of steps 1-6, so text, tables and results cannot drift
apart. An experiment that has not produced output yet appears as a red
\\TODO marker, which the pre-submission check rejects. Directional claims made
in the text are re-checked here: if a rerun contradicts one, the paper shows
a red \\ClaimCheck marker instead of silently going stale.

Usage: python 07_tables.py
"""
import json
import math
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as tdist
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, RES = os.path.join(HERE, "data"), os.path.join(HERE, "results")
OUT = os.path.join(HERE, "..", "latex", "generated")

METRICS = [("roc_auc", "Auc", 3), ("pr_auc", "Pr", 3),
           ("accuracy", "Acc", 1), ("f1", "Fone", 1)]
MAIN = [
    ("Rankers (no training)", [
        ("Majority", "Maj", "Majority class"),
        ("Centrality:DEGREE", "Deg", "Degree"),
        ("Centrality:EIGENVECTOR", "Eig", "Eigenvector"),
        ("Centrality:CLOSENESS", "Clo", "Closeness"),
        ("Centrality:BETWEENNESS", "Btw", "Betweenness"),
        ("Centrality:CLUSTERING", "Clu", "Clustering coeff."),
        ("RWR", "Rwr", "Random walk w/ restart")]),
    ("Non-graph learners, same five features", [
        ("LogReg", "Lr", "Logistic regression"),
        ("RandomForest", "Rf", "Random forest"),
        ("GradBoost", "Gb", "Gradient boosting"),
        ("MLP", "Mlp", "MLP")]),
    ("Graph neural networks", [
        ("GCN", "Gcn", "GCN"),
        ("GraphSAGE", "Sage", "GraphSAGE"),
        ("GAT-LR", "Gat", "GAT-LR")]),
]
NO_ACC = {"Deg", "Eig", "Clo", "Btw", "Clu", "Rwr"}     # scores, not probabilities
STRONG = {"Lr": "LogReg", "Rf": "RandomForest", "Gb": "GradBoost",
          "Mlp": "MLP", "Sage": "GraphSAGE"}
CODE = {"Majority": "Maj", "Degree": "Deg", "Eigenvector": "Eig", "LogReg": "Lr",
        "RandomForest": "Rf", "MLP": "Mlp", "GCN": "Gcn", "GraphSAGE": "Sage",
        "GAT-LR": "Gat", "GAT-LR+skip": "Skip", "SFARI reports (diagnostic)": "Rep"}
LABEL = {"Majority": "Majority class", "Degree": "Degree",
         "Eigenvector": "Eigenvector", "LogReg": "Logistic regression",
         "RandomForest": "Random forest", "MLP": "MLP", "GCN": "GCN",
         "GraphSAGE": "GraphSAGE", "GAT-LR": "GAT-LR",
         "GAT-LR+skip": "GAT-LR + skip",
         "SFARI reports (diagnostic)": r"SFARI reports$^\dagger$"}
FEAT = {"degree_centrality": ("Deg", "degree"),
        "betweenness_centrality": ("Btw", "betweenness"),
        "closeness_centrality": ("Clo", "closeness"),
        "eigenvector_centrality": ("Eig", "eigenvector centrality"),
        "clustering_coefficient": ("Clu", "the clustering coefficient")}
NONGRAPH = ["LogReg", "RandomForest", "MLP"]

macros, claim_failures = {}, []


# ---------------------------------------------------------------- helpers
def mac(name, value):
    assert name.isalpha(), name
    macros[name] = value


def todo(what):
    return rf"\TODO{{{what}}}"


def put(name, fn, ok, why):
    mac(name, fn() if ok else todo(why))


def fmt(x, d):
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def floor3(x):
    """Lower bound rounded down, for 'at least' statements."""
    return f"{math.floor(x * 1000) / 1000:.3f}"


def fp(p, up=False, down=False):
    """p-value for the text; up/down round outward for '<=' / '>=' statements."""
    if p is None or math.isnan(p):
        return r"\text{n/a}"
    rnd = math.ceil if up else (math.floor if down else round)
    if p < 1e-3:
        e = int(math.floor(math.log10(p)))
        c = rnd(p / 10 ** e * 10) / 10
        if c >= 10:
            c, e = c / 10, e + 1
        return rf"{c:.1f}\times10^{{{e}}}"
    return f"{rnd(p * 1000) / 1000:.3f}"


def fint(n):
    return f"{int(n):,}".replace(",", "{,}")


def load(name):
    p = os.path.join(RES, name)
    return pd.read_csv(p) if os.path.exists(p) else None


TEST_TRAIN = 1 / 4          # 5-fold CV: test/train size ratio


def corrected_t(d):
    """Nadeau-Bengio corrected resampled t-test on per-fold differences d.
    Folds share training data, so a test that treats them as independent
    (paired t, Wilcoxon) overstates significance."""
    d = np.asarray(d, float)
    if len(d) < 2 or d.var(ddof=1) == 0:
        return float("nan")
    t = d.mean() / math.sqrt((1 / len(d) + TEST_TRAIN) * d.var(ddof=1))
    return float(2 * tdist.sf(abs(t), len(d) - 1))


def holm(ps):
    """Holm step-down adjusted p-values, returned in the input order."""
    ps = np.asarray(ps, float)
    adj, running = np.empty(len(ps)), 0.0
    for rank, i in enumerate(np.argsort(ps)):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def paired(df, a, b, metric, counts=False):
    """Mean difference a - b over matched folds and its corrected p-value;
    with counts=True also the number of folds in which a < b, and J."""
    keys = [k for k in ("rep", "fold") if k in df]
    A = df[df.model == a].set_index(keys)[metric]
    B = df[df.model == b].set_index(keys)[metric]
    idx = A.index.intersection(B.index)
    d = A.loc[idx].to_numpy(float) - B.loc[idx].to_numpy(float)
    out = (d.mean(), corrected_t(d))
    return out + (int((d < 0).sum()), len(d)) if counts else out


def write(name, text):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        f.write(text)


# ---------------------------------------------------------------- data
def data_macros():
    st = json.load(open(os.path.join(DATA, "network_stats.json")))
    nodes = pd.read_csv(os.path.join(DATA, "nodes.csv"))
    n, e = st["final_nodes"], st["final_edges"]
    mac("NSfari", fint(st["sfari_genes"]))
    mac("NGenes", fint(n))
    mac("NEdges", fint(e))
    mac("MeanDeg", f"{2 * e / n:.1f}")
    mac("Density", f"{2 * e / (n * (n - 1)):.4f}")
    mac("NScored", fint(nodes["sfari_score"].notna().sum()))
    mac("NDropped", fint(st["sfari_genes_not_in_network"]))
    sens = json.load(open(os.path.join(DATA, "label_sensitivity.json")))
    cvs = [f"{sens['horizons'][f'T{t}']['cv']:.2f}" for t in range(1, 6)]
    mac("CvList", ", ".join(cvs[:-1]) + " and " + cvs[-1])
    ov = json.load(open(os.path.join(RES, "degree_overlap.json")))
    mac("NPos", fint(ov["n_positives"]))
    mac("PosRate", f"{100 * ov['positive_rate']:.2f}")
    mac("MajFloor", f"{ov['majority_baseline_acc']:.2f}")
    mac("SpearGatDeg", f"{ov['spearman_rho_vs_degree']:.3f}")
    for k, word in ((10, "Ten"), (50, "Fifty"), (100, "Hundred")):
        mac(f"Jac{word}", f"{ov[f'top{k}_jaccard_with_degree']:.2f}")


def degree_label_table():
    d = pd.read_csv(os.path.join(DATA, "features.csv")).merge(
        pd.read_csv(os.path.join(DATA, "ic_scores.csv")), on="gene")
    deg = d["degree"].to_numpy(float)
    rows, aucs, rhos = [], [], []
    for t in range(1, 6):
        icv = d[f"ic_T{t}"].to_numpy()
        rho = spearmanr(icv, deg)[0]
        rhos.append(rho)
        cells = []
        for p in (50, 75, 90):
            a = roc_auc_score((icv > np.percentile(icv, p)).astype(int), deg)
            aucs.append(a)
            cells.append(f"{a:.3f}")
        mark = r"$^\ast$" if t == 2 else ""
        rows.append(f"{t}{mark} & {rho:.3f} & " + " & ".join(cells) + r" \\")
    write("tab_degree_label.tex", "\n".join(rows) + "\n")
    mac("DegLabelAucMin", floor3(min(aucs)))
    mac("DegLabelRhoMin", floor3(min(rhos)))


# ---------------------------------------------------------------- step 3
def main_table():
    raw = load("metrics_raw.csv")
    cols = [m for m, _, _ in METRICS] + ["precision", "recall"]
    s = raw.groupby("model")[cols].agg(["mean", "std"])
    mac("NFolds", str(raw.groupby(["rep", "fold"]).ngroups))
    lines = []
    for gi, (group, models) in enumerate(MAIN):
        if gi:
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{5}}{{@{{}}l}}{{\textit{{{group}}}}}\\")
        for name, code, label in models:
            cells = []
            for met, key, d in METRICS:
                mu, sd = s.loc[name, (met, "mean")], s.loc[name, (met, "std")]
                mac(code + key, fmt(mu, d))
                mac(code + key + "Sd", fmt(sd, d))
                if key in ("Acc", "Fone") and code in NO_ACC:
                    cells.append("--")
                else:
                    cells.append(rf"{fmt(mu, d)}$\pm${fmt(sd, d)}")
            lines.append(rf"\quad {label} & " + " & ".join(cells) + r" \\")
    write("tab_main.tex", "\n".join(lines) + "\n")
    mac("DegPrecision", fmt(s.loc["Centrality:DEGREE", ("precision", "mean")], 1))
    mac("DegRecall", fmt(s.loc["Centrality:DEGREE", ("recall", "mean")], 1))

    pmin, kmin = 1.0, None
    for _, models in MAIN:
        for name, code, _ in models:
            if code == "Gat":
                continue
            for met, key, d in METRICS:
                delta, p, below, _ = paired(raw, "GAT-LR", name, met, counts=True)
                mac(f"dGatVs{code}{key}", fmt(delta, d))
                mac(f"pGatVs{code}{key}", fp(p))
                mac(f"kGatBelow{code}{key}", str(below))
                if code in STRONG or (code == "Deg" and key in ("Auc", "Pr")):
                    # Text: GAT-LR is lower on average, and the gap is not
                    # significant under the corrected test.
                    if not delta < 0:
                        claim_failures.append(f"GAT-LR not lower than {name} on {met}")
                    if not p >= 0.05:
                        claim_failures.append(f"GAT-LR vs {name} on {met} is significant "
                                              f"(p={p:.3g}) but the text says it is not")
                    pmin = min(pmin, p)
                if code in STRONG and key == "Auc":
                    kmin = below if kmin is None else min(kmin, below)
    mac("pGatMinVsStrong", fp(pmin, down=True))
    mac("kGatBelowStrongMin", str(kmin))
    sds = [s.loc[n, ("accuracy", "std")] for n in ("LogReg", "RandomForest",
                                                   "GradBoost", "MLP")]
    mac("NongraphAccSdMin", fmt(min(sds), 2))
    mac("NongraphAccSdMax", fmt(max(sds), 2))
    return raw


# ---------------------------------------------------------------- step 4
ROBUST_MODELS = ["Degree", "LogReg", "RandomForest", "MLP", "GraphSAGE", "GCN", "GAT-LR"]


def robust_table(raw3):
    h, t = load("exp_horizon.csv"), load("exp_threshold.csv")
    ok = h is not None and t is not None
    put("RobustSettings", lambda: "7", ok, "robustness pending")
    if not ok:
        write("tab_robust.tex",
              rf"\multicolumn{{8}}{{c}}{{{todo('horizon/threshold runs pending')}}}\\" + "\n")
        mac("RobustGatLower", todo("robustness pending"))
        mac("RobustGatSigLower", todo("robustness pending"))
        return
    base = raw3.assign(model=raw3["model"].replace({"Centrality:DEGREE": "Degree"}))
    settings = ([(T, 75, h[h["T"] == T]) for T in (1, 2, 3, 4, 5)]
                + [(2, p, t[t["pctl"] == p]) for p in (50, 90)])
    settings = [(T, p, base if (T, p) == (2, 75) else df) for T, p, df in settings]
    lines, lower, sig = [], 0, 0
    for T, p, df in settings:
        means = {m: df[df.model == m]["pr_auc"].mean() for m in ROBUST_MODELS}
        top = max(round(v, 3) for v in means.values())
        cells = [(rf"\textbf{{{v:.3f}}}" if round(v, 3) == top else f"{v:.3f}")
                 for v in means.values()]
        lines.append(rf"{T} & {p} & " + " & ".join(cells) + r" \\")
        best = max(NONGRAPH, key=lambda m: means[m])
        delta, pv = paired(df, "GAT-LR", best, "pr_auc")
        lower += int(delta < 0)
        sig += int(delta < 0 and pv < 0.05)
    write("tab_robust.tex", "\n".join(lines) + "\n")
    mac("RobustGatLower", str(lower))
    mac("RobustGatSigLower", str(sig))


def skip_macros(raw3):
    sk = load("exp_skip.csv")
    ok = sk is not None
    both = None
    if ok:
        keep = ["model", "rep", "fold"] + [m for m, _, _ in METRICS]
        both = pd.concat([raw3[keep], sk[keep]])
    for met, key, d in METRICS:
        put(f"Skip{key}", lambda m=met, d=d: fmt(sk[m].mean(), d), ok, "skip pending")
        put(f"Skip{key}Sd", lambda m=met, d=d: fmt(sk[m].std(), d), ok, "skip pending")
        for other, code in (("GAT-LR", "Gat"), ("MLP", "Mlp"), ("GraphSAGE", "Sage")):
            r = paired(both, "GAT-LR+skip", other, met) if ok else (None, None)
            put(f"dSkipVs{code}{key}", lambda r=r, d=d: fmt(r[0], d), ok, "skip pending")
            put(f"pSkipVs{code}{key}", lambda r=r: fp(r[1]), ok, "skip pending")


def inductive_macros():
    df = load("exp_inductive.csv")
    ok = df is not None
    for name, code in (("GCN", "Gcn"), ("GraphSAGE", "Sage"), ("GAT-LR", "Gat")):
        for met, key, d in METRICS[:2]:
            if ok:
                a = df[(df.model == name) & (df.setting == "transductive")].set_index(
                    ["rep", "fold"])[met]
                b = df[(df.model == name) & (df.setting == "inductive")].set_index(
                    ["rep", "fold"])[met]
                j = a.index.intersection(b.index)
                pv = corrected_t(b.loc[j].to_numpy() - a.loc[j].to_numpy())
                vals = (a.loc[j].mean(), b.loc[j].mean(),
                        (b.loc[j] - a.loc[j]).mean(), pv)
            else:
                vals = (None,) * 4
            put(f"Ind{code}Tr{key}", lambda v=vals, d=d: fmt(v[0], d), ok, "inductive pending")
            put(f"Ind{code}In{key}", lambda v=vals, d=d: fmt(v[1], d), ok, "inductive pending")
            put(f"Ind{code}Delta{key}", lambda v=vals, d=d: fmt(v[2], d), ok,
                "inductive pending")
            put(f"pInd{code}{key}", lambda v=vals: fp(v[3]), ok, "inductive pending")


def importance_macros():
    df = load("exp_importance.csv")
    ok = df is not None
    g = df.groupby(["model", "feature"])["auc_drop"].mean() if ok else None
    for model, mcode in (("GAT-LR", "Gat"), ("RandomForest", "Rf")):
        for feat, (fcode, _) in FEAT.items():
            put(f"Imp{mcode}{fcode}", lambda m=model, f=feat: fmt(g.loc[(m, f)], 3), ok,
                "importance pending")
        top = g.loc[model].sort_values(ascending=False) if ok else None
        put(f"Imp{mcode}Top", lambda t=top: FEAT[t.index[0]][1], ok, "importance pending")
        put(f"Imp{mcode}TopDrop", lambda t=top: fmt(t.iloc[0], 3), ok, "importance pending")


# ---------------------------------------------------------------- beyond degree
BEYOND_ROWS = ["Majority", "Degree", "Eigenvector", "LogReg", "RandomForest", "MLP",
               "GCN", "GraphSAGE", "GAT-LR", "GAT-LR+skip", "SFARI reports (diagnostic)"]


def beyond_table():
    tasks = (("Res", load("exp_residual.csv"), "residual pending"),
             ("Sf", load("exp_sfari.csv"), "SFARI check pending"))
    lines = []
    for name in BEYOND_ROWS:
        cells = []
        for prefix, df, why in tasks:
            present = df is not None and (df.model == name).any()
            for met, key, d in METRICS[:2]:
                if df is None:
                    cells.append(todo("pending"))
                elif not present:
                    cells.append("--")
                else:
                    v = df[df.model == name][met]
                    mac(f"{prefix}{CODE[name]}{key}", fmt(v.mean(), 3))
                    mac(f"{prefix}{CODE[name]}{key}Sd", fmt(v.std(), 3))
                    cells.append(fmt(v.mean(), 3))
        lines.append(rf"{LABEL[name]} & " + " & ".join(cells) + r" \\")
    write("tab_beyond.tex", "\n".join(lines) + "\n")

    for prefix, df, why in tasks:
        ok = df is not None
        put(f"{prefix}MaxSd",
            lambda df=df: fmt(df.groupby("model")[["roc_auc", "pr_auc"]].std().max().max(), 3),
            ok, why)
        for name in ("Degree", "Eigenvector", "GAT-LR", "GraphSAGE", "MLP"):
            for met, key, d in METRICS[:2]:
                if not ok:
                    mac(f"{prefix}{CODE[name]}{key}", todo(why))

    res = tasks[0][1]
    ok = res is not None
    learners = ["LogReg", "RandomForest", "MLP", "GCN", "GraphSAGE", "GAT-LR", "GAT-LR+skip"]
    best = max(learners, key=lambda m: res[res.model == m]["pr_auc"].mean()) if ok else None
    put("ResBest", lambda: LABEL[best], ok, "residual pending")
    put("ResBestPr", lambda: fmt(res[res.model == best]["pr_auc"].mean(), 3), ok,
        "residual pending")
    r = paired(res, best, "GAT-LR", "pr_auc") if ok and best != "GAT-LR" else (None, None)
    put("dResBestVsGatPr", lambda: fmt(r[0], 3), ok and r[0] is not None, "residual pending")
    put("pResBestVsGatPr", lambda: fp(r[1]), ok and r[1] is not None, "residual pending")
    r2 = paired(res, "GraphSAGE", "MLP", "pr_auc") if ok else (None, None)
    put("pResSageVsMlpPr", lambda: fp(r2[1]), ok, "residual pending")
    put("dResSageVsMlpPr", lambda: fmt(r2[0], 3), ok, "residual pending")
    ng = max(NONGRAPH, key=lambda m: res[res.model == m]["pr_auc"].mean()) if ok else None
    put("ResBestNg", lambda: LABEL[ng], ok, "residual pending")
    put("ResBestNgPr", lambda: fmt(res[res.model == ng]["pr_auc"].mean(), 3), ok,
        "residual pending")
    for model, code in (("GraphSAGE", "Sage"), ("GAT-LR+skip", "Skip"), ("GAT-LR", "Gat")):
        r3 = paired(res, model, ng, "pr_auc") if ok else (None, None)
        put(f"dRes{code}VsBestNgPr", lambda r3=r3: fmt(r3[0], 3), ok, "residual pending")
        put(f"pRes{code}VsBestNgPr", lambda r3=r3: fp(r3[1]), ok, "residual pending")
        if ok and code in ("Sage", "Skip") and not (r3[0] > 0 and r3[1] < 0.05):
            claim_failures.append(f"residual: {model} not significantly above {ng}")
    r4 = paired(res, "GAT-LR+skip", "GAT-LR", "pr_auc") if ok else (None, None)
    put("dResSkipVsGatPr", lambda: fmt(r4[0], 3), ok, "residual pending")
    put("pResSkipVsGatPr", lambda: fp(r4[1]), ok, "residual pending")
    if ok and not (r4[0] > 0 and r4[1] < 0.05):
        claim_failures.append("residual: GAT-LR+skip not significantly above GAT-LR")
    if ok:
        # the three comparisons reported in the text form one family
        fam = [("pResSageVsBestNgPr", paired(res, "GraphSAGE", ng, "pr_auc")[1]),
               ("pResSkipVsBestNgPr", paired(res, "GAT-LR+skip", ng, "pr_auc")[1]),
               ("pResSkipVsGatPr", r4[1])]
        for (name, _), padj in zip(fam, holm([p for _, p in fam])):
            mac(name, fp(padj))
            if padj >= 0.05:
                claim_failures.append(f"residual: {name} not significant after Holm")

    sf = tasks[1][1]
    ok = sf is not None
    learners = ["LogReg", "RandomForest", "MLP", "GCN", "GraphSAGE", "GAT-LR"]
    best = max(learners, key=lambda m: sf[sf.model == m]["roc_auc"].mean()) if ok else None
    put("SfBest", lambda: LABEL[best], ok, "SFARI check pending")
    put("SfBestAuc", lambda: fmt(sf[sf.model == best]["roc_auc"].mean(), 3), ok,
        "SFARI check pending")
    r = paired(sf, best, "Degree", "roc_auc") if ok else (None, None)
    put("dSfBestVsDegAuc", lambda: fmt(r[0], 3), ok, "SFARI check pending")
    put("pSfBestVsDegAuc", lambda: fp(r[1]), ok, "SFARI check pending")
    bestpr = max(learners, key=lambda m: sf[sf.model == m]["pr_auc"].mean()) if ok else None
    put("SfPrBest", lambda: LABEL[bestpr], ok, "SFARI check pending")
    put("SfPrBestPr", lambda: fmt(sf[sf.model == bestpr]["pr_auc"].mean(), 3), ok,
        "SFARI check pending")
    r5 = paired(sf, bestpr, "Degree", "pr_auc") if ok else (None, None)
    put("dSfPrBestVsDegPr", lambda: fmt(r5[0], 3), ok, "SFARI check pending")
    put("pSfPrBestVsDegPr", lambda: fp(r5[1]), ok, "SFARI check pending")
    if ok:
        if not (r[0] > 0 and r[1] < 0.05):
            claim_failures.append("SFARI: best ROC-AUC learner not significantly above degree")
        for m in learners:
            dd, pp = paired(sf, m, "Degree", "pr_auc")
            if dd > 0 and pp < 0.05:
                claim_failures.append(f"SFARI: {m} significantly above degree on PR-AUC")
        auc_p = holm([paired(sf, m, "Degree", "roc_auc")[1] for m in learners])
        adj = dict(zip(learners, auc_p))
        mac("pSfBestVsDegAucHolm", fp(adj[best]))
        words = "zero one two three four five six seven eight nine ten".split()
        mac("SfNLearners", words[len(learners)] if len(learners) <= 10 else str(len(learners)))
        if adj[best] < 0.05:
            claim_failures.append("SFARI: best learner significant after Holm; text says not")
    if not ok:
        mac("pSfBestVsDegAucHolm", todo("SFARI check pending"))
        mac("SfNLearners", todo("SFARI check pending"))
    diag_path = os.path.join(RES, "sfari_diagnostics.json")
    ok = os.path.exists(diag_path)
    diag = json.load(open(diag_path)) if ok else {}
    put("SfN", lambda: fint(diag["sfari_scored_in_network"]), ok, "SFARI check pending")
    put("SfPos", lambda: fint(diag["score1_positives"]), ok, "SFARI check pending")
    put("SfPosRate", lambda: f"{100 * diag['positive_rate']:.1f}", ok, "SFARI check pending")
    put("SfRho", lambda: f"{diag['spearman_degree_vs_reports']:.2f}", ok,
        "SFARI check pending")
    put("pSfRho", lambda: fp(diag["spearman_p"]), ok, "SFARI check pending")


def extra_claims():
    """Re-check the interpretive statements made in the Results text."""
    ind = load("exp_inductive.csv")
    if ind is not None:
        for name in ("GCN", "GraphSAGE", "GAT-LR"):
            for met in ("roc_auc", "pr_auc"):
                sel = ind[ind.model == name]
                a = sel[sel.setting == "transductive"].set_index(["rep", "fold"])[met]
                b = sel[sel.setting == "inductive"].set_index(["rep", "fold"])[met]
                j = a.index.intersection(b.index)
                if corrected_t(b.loc[j].to_numpy() - a.loc[j].to_numpy()) < 0.05:
                    claim_failures.append(f"inductive change significant: {name} {met}")
    sk, raw3 = load("exp_skip.csv"), load("metrics_raw.csv")
    if sk is not None:
        keep = ["model", "rep", "fold", "pr_auc"]
        both = pd.concat([raw3[keep], sk[keep]])
        for other in ("MLP", "GraphSAGE"):
            if paired(both, "GAT-LR+skip", other, "pr_auc")[1] < 0.05:
                claim_failures.append(f"GAT-LR+skip differs significantly from {other}")
    lo, sg = macros.get("RobustGatLower", ""), macros.get("RobustGatSigLower", "")
    if lo.isdigit() and not (lo == macros.get("RobustSettings") and sg == "0"):
        claim_failures.append("robustness: GAT-LR not below in every setting, or a gap "
                              "is significant")
    for m in ("ImpGatTop", "ImpRfTop"):
        v = macros.get(m, "")
        if v and not v.startswith(chr(92)) and v != "degree":
            claim_failures.append(f"{m} is {v}; the text says degree")


# ---------------------------------------------------------------- main
def main():
    os.makedirs(OUT, exist_ok=True)
    data_macros()
    degree_label_table()
    raw3 = main_table()
    robust_table(raw3)
    skip_macros(raw3)
    inductive_macros()
    importance_macros()
    beyond_table()
    extra_claims()
    mac("ClaimCheck", todo("CLAIM CHECK FAILED: " + "; ".join(claim_failures))
        if claim_failures else "")
    lines = ["% Generated by experiments/07_tables.py -- do not edit by hand."]
    lines += [rf"\newcommand{{\{k}}}{{{macros[k]}}}" for k in sorted(macros)]
    write("numbers.tex", "\n".join(lines) + "\n")
    pending = sorted(k for k, v in macros.items() if v.startswith(r"\TODO"))
    print(f"{len(macros)} macros written; {len(pending)} pending")
    if claim_failures:
        print("CLAIM CHECK FAILED:", *claim_failures, sep="\n  ")


if __name__ == "__main__":
    main()
