"""
Reproduces the case-grouping structural test: if a new dispatch's top-10
nearest neighbours (by embedding similarity over Stage-B root-cause text)
already carry a case_id from the existing case pipeline, how often does a
single dominant case actually emerge?

Source data, all already in this repo:
  - output/Dispatch_CaseMapped.xlsx : RootCauseText, CaseId per dispatch
    (11,189 dispatches / 955 cases, current matching logic).
  - state/embeddings.npy + state/embeddings_index.json : cached nomic-embed
    vectors (sha256(text) -> row). Reused as-is, nothing is re-embedded.

Caveat stated once, deliberately: this uses nomic-embed, the model this
repo has cached, not bge-m3, the model specified for production. It tests
the mechanism (does grouping-by-existing-case-id concentrate signal), not
bge-m3's specific recall.

Run from the repo root:
    venv/bin/python analysis/02_case_grouping_test.py

Expected output (matches the architecture doc):
    ALL sampled queries: well-supported(>=5/10) ~57%
    EXCLUDING 'No Technical Fault' placeholder text: ~55%
    'No Technical Fault' placeholder text ONLY: 100% (a known trivial case)
    case size: singleton -> ~20% well-supported
    case size: very large (100+) -> ~69% well-supported
"""
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
NO_FAULT_PLACEHOLDER = "No technical fault identified in dispatch notes."


def text_sha(t):
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def main():
    df = pd.read_excel(HERE / "output" / "Dispatch_CaseMapped.xlsx")
    df = df.dropna(subset=["RootCauseText", "CaseId"])
    df = df[df["RootCauseText"].str.len() > 0]
    print(f"Dispatches with RootCauseText + CaseId: {len(df)}")

    idx_meta = json.load(open(HERE / "state" / "embeddings_index.json"))
    sha_to_row = idx_meta["index"]
    emb = np.load(HERE / "state" / "embeddings.npy")
    print(f"Cached embeddings: {emb.shape[0]} rows, {emb.shape[1]} dims, model={idx_meta['model']}")

    rows, case_ids, is_placeholder = [], [], []
    misses = 0
    for text, cid in zip(df["RootCauseText"], df["CaseId"]):
        r = sha_to_row.get(text_sha(text))
        if r is None:
            misses += 1
            continue
        rows.append(r)
        case_ids.append(cid)
        is_placeholder.append(text.strip() == NO_FAULT_PLACEHOLDER)

    print(f"Matched to a cached embedding: {len(rows)}  (misses: {misses})")

    case_ids = np.array(case_ids)
    is_placeholder = np.array(is_placeholder)
    X = emb[rows].astype(np.float32)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    case_size = pd.Series(case_ids).value_counts()

    rng = random.Random(42)
    n = len(rows)
    sample_idx = rng.sample(range(n), min(1000, n))

    sims_all = X[sample_idx] @ X.T  # (sample, n)

    results = []
    for i, qi in enumerate(sample_idx):
        sims = sims_all[i].copy()
        sims[qi] = -1.0  # exclude self
        # stable sort so ties (duplicate-text embeddings) break the same way every run
        top10 = np.argsort(-sims, kind="stable")[:10]
        neighbour_cases = case_ids[top10]
        vals, counts = np.unique(neighbour_cases, return_counts=True)
        dom_i = np.argmax(counts)
        dom_case, dom_count = vals[dom_i], int(counts[dom_i])
        results.append({
            "query_is_placeholder": bool(is_placeholder[qi]),
            "query_case": case_ids[qi],
            "query_case_size": int(case_size.get(case_ids[qi], 1)),
            "dominant_case": dom_case,
            "dominant_count": dom_count,
            "matches_own_case": bool(dom_case == case_ids[qi]),
        })

    res = pd.DataFrame(results)

    def report(sub, label):
        n_ = len(sub)
        if n_ == 0:
            print(f"{label}: n=0")
            return
        well5 = (sub["dominant_count"] >= 5).mean() * 100
        well3 = (sub["dominant_count"] >= 3).mean() * 100
        own = sub["matches_own_case"].mean() * 100
        print(f"{label}: n={n_}  well-supported(>=5/10)={well5:.1f}%  "
              f"(>=3/10)={well3:.1f}%  dominant==own_case={own:.1f}%  "
              f"mean_dominant_count={sub['dominant_count'].mean():.2f}")

    print()
    print("=== Headline split ===")
    report(res, "ALL sampled queries")
    report(res[~res["query_is_placeholder"]], "EXCLUDING 'No Technical Fault' placeholder text")
    report(res[res["query_is_placeholder"]], "'No Technical Fault' placeholder text ONLY")

    print()
    print("=== Non-placeholder queries, by the query's OWN case size ===")
    non_ph = res[~res["query_is_placeholder"]].copy()
    bins = [0, 1, 5, 20, 100, 100000]
    labels = ["singleton (1)", "small (2-5)", "medium (6-20)", "large (21-100)", "very large (100+)"]
    non_ph["bucket"] = pd.cut(non_ph["query_case_size"], bins=bins, labels=labels)
    for b in labels:
        report(non_ph[non_ph["bucket"] == b], f"case size: {b}")


if __name__ == "__main__":
    main()
