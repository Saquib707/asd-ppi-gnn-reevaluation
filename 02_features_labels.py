"""
Step 2: topological features + SI-derived influence labels.

Features (F=5): degree, betweenness, closeness, eigenvector centrality,
                clustering coefficient.

Labels: Susceptible-Infected influence. NOTE ON A DEVIATION FROM THE SOURCE
DOCUMENTS: both the dissertation and the rejected paper state the SI simulation
runs "until no new infections occur". In an SI process (no recovery) on a
connected graph this infects every reachable node for every seed, so IC == 1.0
identically and the labels carry zero signal. We therefore run SI to a FIXED
HORIZON T and report sensitivity over T. This is documented in the paper.

Writes: data/features.csv, data/ic_scores.csv, data/label_sensitivity.json
"""
import json
import os

import networkx as nx
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

BETA = 0.2
N_RUNS = 100
HORIZONS = [1, 2, 3, 4, 5]
SEED = 42


def load_graph():
    edges = pd.read_csv(os.path.join(DATA, "edges.csv"))
    G = nx.Graph()
    nodes = pd.read_csv(os.path.join(DATA, "nodes.csv"))["gene"].tolist()
    G.add_nodes_from(nodes)
    for a, b, w in zip(edges["source"], edges["target"], edges["string_score"]):
        G.add_edge(a, b, weight=float(w))
    return G


def centralities(G):
    print("  degree...", flush=True)
    dc = nx.degree_centrality(G)
    print("  betweenness...", flush=True)
    bc = nx.betweenness_centrality(G, normalized=True)
    print("  closeness...", flush=True)
    cc = nx.closeness_centrality(G)
    print("  eigenvector...", flush=True)
    ec = nx.eigenvector_centrality_numpy(G)
    print("  clustering...", flush=True)
    cl = nx.clustering(G)
    genes = sorted(G.nodes())
    return pd.DataFrame({
        "gene": genes,
        "degree_centrality": [dc[g] for g in genes],
        "betweenness_centrality": [bc[g] for g in genes],
        "closeness_centrality": [cc[g] for g in genes],
        "eigenvector_centrality": [ec[g] for g in genes],
        "clustering_coefficient": [cl[g] for g in genes],
        "degree": [G.degree(g) for g in genes],
    })


def si_influence(G, genes, horizon, beta=BETA, n_runs=N_RUNS, rng=None):
    """Mean fraction of nodes infected after `horizon` SI steps, per seed."""
    idx = {g: i for i, g in enumerate(genes)}
    n = len(genes)
    nbrs = [np.array([idx[u] for u in G.neighbors(g)], dtype=np.int64) for g in genes]
    out = np.zeros(n)
    for s in range(n):
        total = 0
        for _ in range(n_runs):
            infected = np.zeros(n, dtype=bool)
            infected[s] = True
            frontier = np.array([s], dtype=np.int64)
            for _ in range(horizon):
                if frontier.size == 0:
                    break
                cand = np.concatenate([nbrs[i] for i in frontier]) if frontier.size else \
                    np.array([], dtype=np.int64)
                if cand.size == 0:
                    break
                cand = cand[~infected[cand]]
                if cand.size == 0:
                    break
                hit = cand[rng.random(cand.size) < beta]
                if hit.size == 0:
                    frontier = np.array([], dtype=np.int64)
                    break
                new = np.unique(hit)
                new = new[~infected[new]]
                infected[new] = True
                frontier = new
            total += infected.sum()
        out[s] = total / (n_runs * n)
        if (s + 1) % 100 == 0:
            print(f"    seed {s+1}/{n}", flush=True)
    return out


def main():
    G = load_graph()
    genes = sorted(G.nodes())
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("Computing centralities...")
    feats = centralities(G)
    feats.to_csv(os.path.join(DATA, "features.csv"), index=False)

    rng = np.random.default_rng(SEED)
    sens = {}
    ic_table = {"gene": genes}
    for T in HORIZONS:
        print(f"SI simulation, horizon T={T} ...", flush=True)
        ic = si_influence(G, genes, T, rng=rng)
        ic_table[f"ic_T{T}"] = ic
        sens[f"T{T}"] = {
            "mean": float(ic.mean()), "std": float(ic.std()),
            "min": float(ic.min()), "max": float(ic.max()),
            "cv": float(ic.std() / ic.mean()) if ic.mean() else 0.0,
            "frac_at_ceiling": float((ic > 0.99).mean()),
        }
        print(f"    mean={ic.mean():.4f} std={ic.std():.4f} "
              f"max={ic.max():.4f} ceiling={sens[f'T{T}']['frac_at_ceiling']:.3f}")

    pd.DataFrame(ic_table).to_csv(os.path.join(DATA, "ic_scores.csv"), index=False)
    with open(os.path.join(DATA, "label_sensitivity.json"), "w") as f:
        json.dump({"beta": BETA, "n_runs": N_RUNS, "seed": SEED, "horizons": sens},
                  f, indent=2)
    print(json.dumps(sens, indent=2))


if __name__ == "__main__":
    main()
