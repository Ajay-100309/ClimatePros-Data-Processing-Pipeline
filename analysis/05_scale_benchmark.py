"""
Measures real query/retrieval latency for a vector index at the true
target production scale: 1,551,773 vectors, FieldJetXStg's own measured
count of eligible dispatches (see CLAUDE.md / output/pipeline_report.html).

This is a TIMING benchmark only. It answers "how fast," not "how
accurate." There is no real dispatch text or parts-outcome data at
1.5-million scale to test accuracy against, only 265 real labeled
examples exist (see 03_parts_prediction_accuracy.py and
04_flat_vs_case_retrieval.py). Which top-N gives the best ACCURACY still
needs real labeled data at scale before it can be re-tuned; this script
only checks whether retrieval stays fast enough, at real corpus size, to
make that a live question rather than a bottleneck.

Vectors are synthetic (random, unit-normalized) at 768 dimensions, matching
nomic-embed, the model this repo already has cached and has used for every
other test in analysis/. Production is specified to use bge-m3 (1024-dim);
a dimension-scaling note at the end extrapolates from the measured 768-dim
numbers rather than risking an out-of-memory run at 1024-dim on this
machine (see the printed hardware/memory notes).

Uses hnswlib, a standalone HNSW implementation, the same algorithm family
Azure AI Search's vector search runs on. The *shape* of the timing curve
(how query time changes with N and corpus size) is representative; the
absolute milliseconds are from this local machine, not Azure's managed
infrastructure.

Run from the repo root (needs analysis/requirements.txt installed):
    venv/bin/python analysis/05_scale_benchmark.py
Override the element count for a quick smoke test:
    N_ELEMENTS=50000 venv/bin/python analysis/05_scale_benchmark.py
"""
import os
import resource
import time

import hnswlib
import numpy as np

N_ELEMENTS = int(os.environ.get("N_ELEMENTS", 1_551_773))
DIM = 768  # nomic-embed, matches this repo's actual cached embeddings
M = 8  # HNSW links per node; lower than the library default (16) to
       # keep memory in bounds on an 8-core / 8GB machine at this scale
EF_CONSTRUCTION = 100
BATCH_SIZE = 100_000
N_QUERIES = 300
K_VALUES = [10, 50, 100, 200, 500, 1000]


def peak_rss_gb():
    # macOS reports ru_maxrss in bytes; Linux reports it in KB.
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024 ** 3) if os.uname().sysname == "Darwin" else raw / (1024 ** 2)


def main():
    rng = np.random.default_rng(42)
    print(f"Target scale: {N_ELEMENTS:,} vectors, {DIM} dims (nomic-embed), M={M}")
    est_gb = N_ELEMENTS * (DIM * 4 + M * 2 * 4 + 64) / (1024 ** 3)
    print(f"Rough memory estimate before starting: ~{est_gb:.2f} GB")
    print()

    index = hnswlib.Index(space="cosine", dim=DIM)
    index.init_index(max_elements=N_ELEMENTS, ef_construction=EF_CONSTRUCTION, M=M)
    index.set_num_threads(os.cpu_count() or 4)

    t0 = time.perf_counter()
    inserted = 0
    while inserted < N_ELEMENTS:
        n = min(BATCH_SIZE, N_ELEMENTS - inserted)
        batch = rng.standard_normal((n, DIM)).astype(np.float32)
        batch /= np.linalg.norm(batch, axis=1, keepdims=True)
        ids = np.arange(inserted, inserted + n)
        index.add_items(batch, ids)
        inserted += n
        elapsed = time.perf_counter() - t0
        print(f"  inserted {inserted:>9,} / {N_ELEMENTS:,}  "
              f"({elapsed:6.1f}s elapsed, {inserted / elapsed:,.0f} vec/s, "
              f"peak RSS so far: {peak_rss_gb():.2f} GB)")

    build_time = time.perf_counter() - t0
    print()
    print(f"Index build complete: {build_time:.1f}s for {N_ELEMENTS:,} vectors "
          f"({N_ELEMENTS / build_time:,.0f} vec/s). Peak RSS: {peak_rss_gb():.2f} GB")
    print()

    query_vecs = rng.standard_normal((N_QUERIES, DIM)).astype(np.float32)
    query_vecs /= np.linalg.norm(query_vecs, axis=1, keepdims=True)

    print(f"=== Query latency by top-N, {N_QUERIES} queries each, {N_ELEMENTS:,}-vector index ===")
    print(f"{'top-N':<8}{'ef_search':<11}{'mean ms':<10}{'median ms':<12}{'p95 ms':<10}{'max ms':<10}")
    for k in K_VALUES:
        ef = max(k * 2, 64)
        index.set_ef(ef)
        latencies_ms = []
        for i in range(N_QUERIES):
            t0 = time.perf_counter()
            index.knn_query(query_vecs[i:i + 1], k=k)
            latencies_ms.append((time.perf_counter() - t0) * 1000)
        arr = np.array(latencies_ms)
        print(f"{k:<8}{ef:<11}{arr.mean():<10.2f}{np.median(arr):<12.2f}"
              f"{np.percentile(arr, 95):<10.2f}{arr.max():<10.2f}")

    print()
    print("Extrapolation to bge-m3 (1024-dim, production spec, not directly "
          "tested here to avoid an out-of-memory run on this machine):")
    print(f"  Raw vector memory scales with dimension: {N_ELEMENTS:,} x 1024 x 4 bytes "
          f"= {N_ELEMENTS * 1024 * 4 / 1024**3:.2f} GB before graph overhead, "
          f"vs {N_ELEMENTS * DIM * 4 / 1024**3:.2f} GB measured here at 768-dim.")
    print("  Per-query distance computation scales roughly linearly with dimension "
          "(1024/768 = 1.33x), so expect query latency in the same ballpark, "
          "scaled up by roughly that factor, not a different order of magnitude. "
          "This is an estimate, not a measurement -- confirming it needs the "
          "actual production embedding model and hardware.")


if __name__ == "__main__":
    main()
