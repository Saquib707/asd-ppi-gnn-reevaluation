"""
Step 1: build the ASD PPI network.

SFARI Gene export -> STRING v12.0 network among the SFARI genes -> undirected,
deduplicated graph -> genes with at least one interaction.

The network is retrieved in ONE API call. Splitting the identifier list into
chunks silently drops every interaction between chunks (an earlier chunked
version of this script returned 6,417 instead of 15,756 edges).

SFARI annotations are attached through STRING's own identifier mapping, so a
gene that STRING lists under a different preferred symbol keeps its score.

Usage:  python 01_build_network.py [--out DIR] [--sfari PATH] [--refresh]
  --out DIR     output directory (default: data/)
  --sfari PATH  SFARI export to use (default: DIR/sfari_genes_raw.csv;
                downloaded when missing). The paper used the 2026-09-10 export.
  --refresh     re-download / re-query even when cached files exist
Writes: nodes.csv, edges.csv, network_stats.json, and the raw API caches
        string_network_raw.tsv and string_id_map.tsv
"""
import argparse
import hashlib
import io
import json
import os
import time

import networkx as nx
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
SFARI_URL = ("https://gene.sfari.org/wp-content/themes/sfari-gene/utilities/"
             "download-csv.php?api-endpoint=genes")
STRING_API = "https://version-12-0.string-db.org/api"
SPECIES, REQUIRED_SCORE = 9606, 400      # >= 0.400, STRING "medium confidence"
CALLER = "asd_gat_lr_reproduction"


def cached(path, fetch, refresh):
    if refresh or not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(fetch())
    return path


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def fetch_sfari():
    r = requests.get(SFARI_URL, timeout=120, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.content


def fetch_network(symbols):
    r = requests.post(f"{STRING_API}/tsv/network", timeout=600, data={
        "identifiers": "\r".join(symbols), "species": SPECIES,
        "required_score": REQUIRED_SCORE, "caller_identity": CALLER})
    r.raise_for_status()
    return r.content


def fetch_id_map(symbols):
    """Per-identifier mapping, so chunking is safe here."""
    parts = []
    for i in range(0, len(symbols), 400):
        r = requests.post(f"{STRING_API}/tsv/get_string_ids", timeout=300, data={
            "identifiers": "\r".join(symbols[i:i + 400]), "species": SPECIES,
            "limit": 1, "echo_query": 1, "caller_identity": CALLER})
        r.raise_for_status()
        parts.append(pd.read_csv(io.StringIO(r.text), sep="\t"))
        time.sleep(1)
    return pd.concat(parts).to_csv(sep="\t", index=False).encode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "data"))
    ap.add_argument("--sfari", default=None)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    sfari_path = cached(args.sfari or os.path.join(args.out, "sfari_genes_raw.csv"),
                        fetch_sfari, args.refresh and not args.sfari)
    sf = pd.read_csv(sfari_path)
    sf["symbol"] = sf["gene-symbol"].str.strip().str.upper()
    sf = sf.dropna(subset=["symbol"]).drop_duplicates("symbol")
    symbols = sorted(sf["symbol"])
    print(f"SFARI genes: {len(symbols)}")

    net_path = cached(os.path.join(args.out, "string_network_raw.tsv"),
                      lambda: fetch_network(symbols), args.refresh)
    map_path = cached(os.path.join(args.out, "string_id_map.tsv"),
                      lambda: fetch_id_map(symbols), args.refresh)

    net = pd.read_csv(net_path, sep="\t")
    net = net[(net["score"] >= REQUIRED_SCORE / 1000)
              & (net["preferredName_A"] != net["preferredName_B"])]
    key = [tuple(sorted(p)) for p in zip(net["preferredName_A"], net["preferredName_B"])]
    net = net.assign(_k=key).drop_duplicates("_k")

    G = nx.Graph()
    for a, b, s in zip(net["preferredName_A"], net["preferredName_B"], net["score"]):
        G.add_edge(a, b, weight=float(s))
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    lcc = G.subgraph(comps[0]).copy()
    print(f"graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
          f"{len(comps)} component(s); LCC {lcc.number_of_nodes()} / "
          f"{lcc.number_of_edges()}")

    # SFARI annotation via STRING's mapping (query symbol -> preferred name).
    idmap = pd.read_csv(map_path, sep="\t")
    q2p = dict(zip(idmap["queryItem"].astype(str).str.strip().str.upper(),
                   idmap["preferredName"]))
    sf["preferred"] = sf["symbol"].map(q2p)
    unmapped = sf["preferred"].isna()               # fall back to the SFARI symbol
    sf.loc[unmapped, "preferred"] = sf.loc[unmapped, "symbol"]
    upper2node = {g.upper(): g for g in lcc.nodes()}
    sf["node"] = sf["preferred"].str.upper().map(upper2node)
    ann = (sf.dropna(subset=["node"]).groupby("node")
           .agg(sfari_symbols=("symbol", lambda s: ";".join(sorted(s))),
                sfari_score=("gene-score", "min"),      # 1 = strongest evidence
                eagle=("eagle", "max"),
                syndromic=("syndromic", "max"),
                n_reports=("number-of-reports", "max")))

    nodes = pd.DataFrame({"gene": sorted(lcc.nodes())}).merge(
        ann, left_on="gene", right_index=True, how="left")
    nodes.to_csv(os.path.join(args.out, "nodes.csv"), index=False)
    pd.DataFrame([(u, v, d["weight"]) for u, v, d in lcc.edges(data=True)],
                 columns=["source", "target", "string_score"]).to_csv(
        os.path.join(args.out, "edges.csv"), index=False)

    in_network = nodes["sfari_symbols"].dropna().str.split(";").str.len().sum()
    stats = {
        "sfari_genes": len(symbols),
        "sfari_export_sha256": sha256(sfari_path),
        "string_version": "12.0", "required_score": REQUIRED_SCORE,
        "species": SPECIES, "string_network_sha256": sha256(net_path),
        "raw_nodes": G.number_of_nodes(), "raw_edges": G.number_of_edges(),
        "n_components": len(comps),
        "final_nodes": lcc.number_of_nodes(), "final_edges": lcc.number_of_edges(),
        "mean_degree": round(2 * lcc.number_of_edges() / lcc.number_of_nodes(), 2),
        "density": round(nx.density(lcc), 5),
        "sfari_annotated": int(nodes["sfari_symbols"].notna().sum()),
        "sfari_scored": int(nodes["sfari_score"].notna().sum()),
        "eagle_annotated": int(nodes["eagle"].notna().sum()),
        "sfari_genes_not_in_network": int(len(symbols) - in_network),
        "built": time.strftime("%Y-%m-%d"),
    }
    with open(os.path.join(args.out, "network_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
