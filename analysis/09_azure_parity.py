"""09 — Azure AI Search parity check for the flat retrieval + blended scoring.

Confirms the backfilled index reproduces the design's measured behavior, two ways:

A) Leave-one-out hit@6 over every indexed dispatch that has real recorded parts:
   query Azure with the dispatch's own cached vector (excluding itself), score
   parts with the validated formula
       score(part) = sum_{neighbors, cos>0.55} cos^3 + lambda * freq/max_global
   and count a hit when any of the dispatch's own real parts lands in the top 6.
   Expect the low-40s% ballpark at k=100 (direction-confirming, not exact — the
   corpus here is ~12.7k dispatches vs the original 2,077/265 test file).

B) HNSW-vs-exact recall: for a sample of dispatches, Azure's top-100 vs the
   exact cosine top-100 computed locally over the pushed corpus. HNSW is
   approximate; expect >90% overlap.

Run AFTER backfill_search.py has completed:
    venv/bin/python analysis/09_azure_parity.py                # sample of 800
    venv/bin/python analysis/09_azure_parity.py --all          # every labeled dispatch
    venv/bin/python analysis/09_azure_parity.py --sample 200 --recall-sample 50
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import json
import argparse
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipelib import config, search_index  # noqa: E402
from pipelib.embcache import EmbCache  # noqa: E402
from azure.search.documents.models import VectorizedQuery  # noqa: E402

K = 100
SIM_FLOOR = 0.55
LAMBDAS = [0.0, 0.05, 0.1, 0.2, 0.4]
TOP_N = 6


def norm_part(p):
    return p.strip().upper()


def load_corpus():
    state = json.load(open(config.SEARCH_STATE_FILE))
    casemap = json.load(open(config.CASEMAP_FILE))
    parts_state = json.load(open(config.PARTS_FILE))
    texts, own_parts = {}, {}
    for did in sorted(state["pushed"]):
        texts[did] = casemap["dispatches"][did]["text"]
        entry = parts_state["dispatches"].get(did, {"items": []})
        own_parts[did] = sorted({norm_part(p["part_no"])
                                 for p in search_index.real_parts(entry["items"])})
    return texts, own_parts


def cos_from_score(score):
    return max(-1.0, min(1.0, 2.0 - 1.0 / score))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=800,
                    help="labeled dispatches to evaluate (deterministic stride)")
    ap.add_argument("--all", action="store_true", help="evaluate every labeled dispatch")
    ap.add_argument("--recall-sample", type=int, default=100,
                    help="dispatches for the HNSW-vs-exact overlap check")
    args = ap.parse_args()

    config.require_search_config()
    texts, own_parts = load_corpus()
    labeled = [d for d, parts in own_parts.items() if parts]
    print(f"Corpus: {len(texts)} indexed dispatches, {len(labeled)} with real parts.")

    pop = json.load(open(config.OUT_PART_POPULARITY))
    global_freq, max_global = pop["freq"], max(pop["max_global"], 1)

    if not args.all and len(labeled) > args.sample:
        stride = len(labeled) // args.sample
        labeled = labeled[::stride][:args.sample]
    print(f"Evaluating hit@{TOP_N} on {len(labeled)} dispatches, k={K}, "
          f"floor {SIM_FLOOR}, lambdas {LAMBDAS}...")

    cache = EmbCache()
    client = search_index.search_client()
    hits = {lam: 0 for lam in LAMBDAS}
    naive_pred = [p for p, _ in sorted(global_freq.items(),
                                       key=lambda kv: (-kv[1], kv[0]))[:TOP_N]]
    naive_hits = 0

    for i, did in enumerate(labeled, 1):
        vec = cache.get(texts[did])
        results = client.search(
            search_text=None,
            vector_queries=[VectorizedQuery(vector=[float(x) for x in vec],
                                            k_nearest_neighbors=K,
                                            fields="rootCauseVector")],
            filter=f"dispatchId ne '{did}'",
            select=["dispatchId", "parts"], top=K)
        raw = Counter()
        for r in results:
            sim = cos_from_score(r["@search.score"])
            if sim <= SIM_FLOOR:
                continue
            for part in sorted({norm_part(p["partNo"]) for p in (r["parts"] or [])}):
                raw[part] += sim ** 3
        truth = set(own_parts[did])
        for lam in LAMBDAS:
            blended = Counter(raw)
            if lam:
                for part, cnt in global_freq.items():
                    blended[part] += lam * (cnt / max_global)
            pred = [p for p, _ in blended.most_common(TOP_N)]
            if truth & set(pred):
                hits[lam] += 1
        if truth & set(naive_pred):
            naive_hits += 1
        if i % 100 == 0:
            print(f"  {i}/{len(labeled)}  "
                  + "  ".join(f"lam={lam}: {hits[lam]/i:.1%}" for lam in LAMBDAS))

    n = len(labeled)
    print(f"\n=== A) hit@{TOP_N}, n={n} ===")
    print(f"  naive popularity-only baseline: {naive_hits/n:.1%}")
    for lam in LAMBDAS:
        label = "closeness only" if lam == 0 else f"blend lambda={lam}"
        print(f"  flat top-{K}, {label}: {hits[lam]/n:.1%}")

    # B) HNSW vs exact recall over the pushed corpus
    m = min(args.recall_sample, len(texts))
    sample = list(texts)[::max(1, len(texts) // m)][:m]
    all_dids = list(texts)
    mat = np.stack([cache.get(texts[d]) for d in all_dids])
    mat = mat / np.linalg.norm(mat, axis=1, keepdims=True)
    overlaps = []
    for did in sample:
        q = cache.get(texts[did]).astype(np.float64)
        q = q / np.linalg.norm(q)
        sims = mat @ q
        order = np.argsort(-sims)
        exact = {all_dids[j] for j in order[:K + 1] if all_dids[j] != did}
        exact = set(list(exact)[:K])
        results = client.search(
            search_text=None,
            vector_queries=[VectorizedQuery(vector=[float(x) for x in cache.get(texts[did])],
                                            k_nearest_neighbors=K,
                                            fields="rootCauseVector")],
            filter=f"dispatchId ne '{did}'",
            select=["dispatchId"], top=K)
        azure = {r["dispatchId"] for r in results}
        overlaps.append(len(exact & azure) / max(len(exact), 1))
    print(f"\n=== B) HNSW vs exact top-{K} overlap, n={len(sample)} ===")
    print(f"  mean {np.mean(overlaps):.1%}  min {np.min(overlaps):.1%}  "
          f"p05 {np.percentile(overlaps, 5):.1%}")


if __name__ == "__main__":
    main()
