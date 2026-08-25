"""
Reproduces the real parts-prediction accuracy test: the one that changed
the scoring design.

For every dispatch with a real recorded part, retrieve its top-10 nearest
neighbours by embedding similarity (excluding itself), find the dominant
case among those neighbours, and score four candidate designs against what
was actually used:

  A. Case tier only       - rank parts by how many OTHER dispatches in the
                             dominant case (leave-one-out) used them.
  B. Fallback tier only   - sim^3-weighted accumulation over the raw top-10
                             neighbours' own recorded parts, sim > 0.55
                             (the originally-approved scoring formula).
  C. Naive baseline       - ignore retrieval and case entirely; always
                             suggest the globally most-frequent real parts.
  D. Final design         - case tier if the dominant case has >=5/10
                             agreement among neighbours AND has other
                             real-parts evidence to draw from, otherwise
                             fall back to the naive baseline (NOT the
                             sim^3 formula, which tested worse).

Source data:
  - dispatch_2k_rootcauses.numbers (repo root): DispatchNumber, RootCauseText,
    CaseId, and real recorded parts per dispatch, pulled from FieldJetXStg
    DispatchParts + the case registry.
  - state/embeddings.npy + state/embeddings_index.json : cached nomic-embed
    vectors, reused as-is.

Requires numbers-parser (not in the main pipeline's requirements.txt --
this is analysis-only, see analysis/requirements.txt):
    venv/bin/pip install -r analysis/requirements.txt

Run from the repo root:
    venv/bin/python analysis/03_parts_prediction_accuracy.py

Expected output, bit-exact on every run (the hit@6 figures are what the
architecture doc and presentation script quote; hit@3 for A and B is
reported here too but wasn't carried into either document):
    Dispatches with a real recorded part: 265 (12.8%)
    A. case tier only    hit@3=24.2%  hit@6=28.3%
    B. fallback only     hit@3=25.3%  hit@6=27.2%
    C. naive baseline    hit@3=24.9%  hit@6=36.2%
    D. final design      hit@3=28.3%  hit@6=39.6%
"""
import os

# Multi-threaded BLAS can round the matmul below differently run to run,
# occasionally flipping a near-tied neighbour at the rank-10 boundary.
# Pin single-threaded before numpy loads its BLAS backend, for a genuinely
# reproducible result rather than one that's merely close each time.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from numbers_parser import Document

HERE = Path(__file__).resolve().parent.parent
NUMBERS_FILE = HERE / "dispatch_2k_rootcauses.numbers"
SHEET_NAME = "Dispatches_RootCauses - Parts"
TEXT_COL = "RootCauseText (trimmed Notes)"
PARTS_COL = "PartsUsed (part ×qty)"


def text_sha(t):
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def load_numbers_sheet(path, sheet_name):
    doc = Document(str(path))
    for sheet in doc.sheets:
        if sheet.name == sheet_name:
            t = sheet.tables[0]
            header = [t.cell(0, c).value for c in range(t.num_cols)]
            rows = [[t.cell(r, c).value for c in range(t.num_cols)] for r in range(1, t.num_rows)]
            return pd.DataFrame(rows, columns=header)
    raise KeyError(f"sheet {sheet_name!r} not found in {path}")


def parse_parts(cell):
    if pd.isna(cell) or not str(cell).strip():
        return []
    return [seg.strip().split(" ")[0].strip() for seg in str(cell).split(";") if seg.strip()]


def main():
    df = load_numbers_sheet(NUMBERS_FILE, SHEET_NAME)
    df = df.dropna(subset=[TEXT_COL, "CaseId"])
    df["real_parts"] = df[PARTS_COL].apply(parse_parts)
    df = df[df[TEXT_COL].str.len() > 0].reset_index(drop=True)

    idx_meta = json.load(open(HERE / "state" / "embeddings_index.json"))
    sha_to_row = idx_meta["index"]
    emb = np.load(HERE / "state" / "embeddings.npy")

    df["emb_row"] = df[TEXT_COL].apply(lambda t: sha_to_row.get(text_sha(str(t))))
    n_before = len(df)
    df = df.dropna(subset=["emb_row"]).reset_index(drop=True)
    df["emb_row"] = df["emb_row"].astype(int)
    print(f"Dispatches with embedding: {len(df)}/{n_before}")

    X = emb[df["emb_row"].values].astype(np.float32)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    case_ids = df["CaseId"].values
    real_parts = df["real_parts"].tolist()
    has_real = np.array([len(p) > 0 for p in real_parts])
    print(f"Dispatches with a real recorded part: {has_real.sum()} ({has_real.mean() * 100:.1f}%)")

    global_freq = Counter()
    for p in real_parts:
        # sorted(), not set() iteration order: Counter.most_common() breaks
        # ties by insertion order, and Python's set iteration order is
        # randomized per process (PYTHONHASHSEED) unless it's fed a
        # deterministic sequence.
        for part in sorted(set(p)):
            global_freq[part] += 1
    naive_top6 = [p for p, _ in global_freq.most_common(6)]
    print(f"Naive baseline top-6 (by global dispatch frequency): {naive_top6}")

    query_idx = np.where(has_real)[0]
    print(f"\nQuery set (dispatches with a real part, embedding, case): {len(query_idx)}")

    sims_all = X[query_idx] @ X.T

    rows = []
    for qpos, qi in enumerate(query_idx):
        sims = sims_all[qpos].copy()
        sims[qi] = -1.0
        # stable sort so ties (duplicate-text embeddings) break the same way every run
        top10 = np.argsort(-sims, kind="stable")[:10]

        neighbour_cases = case_ids[top10]
        vals, counts = np.unique(neighbour_cases, return_counts=True)
        dom_i = np.argmax(counts)
        dom_case, dom_count = vals[dom_i], int(counts[dom_i])

        case_mask = (case_ids == dom_case)
        case_mask[qi] = False
        case_members = [real_parts[j] for j in np.where(case_mask & has_real)[0]]
        case_freq = Counter()
        for parts in case_members:
            for part in sorted(set(parts)):
                case_freq[part] += 1
        case_pred = [p for p, _ in case_freq.most_common(6)]

        fb_score = Counter()
        for j in top10:
            sim = sims[j]
            if sim <= 0.55:
                continue
            for part in sorted(set(real_parts[j])):
                fb_score[part] += sim ** 3
        fb_pred = [p for p, _ in fb_score.most_common(6)]

        final_pred = case_pred if (dom_count >= 5 and case_pred) else naive_top6

        actual = set(real_parts[qi])

        def hit(pred, k):
            return bool(actual & set(pred[:k]))

        rows.append({
            "dom_count": dom_count,
            "A_case_hit3": hit(case_pred, 3), "A_case_hit6": hit(case_pred, 6),
            "B_fallback_hit3": hit(fb_pred, 3), "B_fallback_hit6": hit(fb_pred, 6),
            "C_naive_hit3": hit(naive_top6, 3), "C_naive_hit6": hit(naive_top6, 6),
            "D_final_hit3": hit(final_pred, 3), "D_final_hit6": hit(final_pred, 6),
        })

    res = pd.DataFrame(rows)

    def report(sub, label, prefix):
        n_ = len(sub)
        if n_ == 0:
            print(f"{label}: n=0")
            return
        h3 = sub[f"{prefix}_hit3"].mean() * 100
        h6 = sub[f"{prefix}_hit6"].mean() * 100
        print(f"{label:<22} n={n_:<5} hit@3={h3:5.1f}%  hit@6={h6:5.1f}%")

    print()
    print("=== Design comparison, all query dispatches ===")
    report(res, "A. case tier only", "A_case")
    report(res, "B. fallback only", "B_fallback")
    report(res, "C. naive baseline", "C_naive")
    report(res, "D. final design", "D_final")

    print()
    print("=== Final design (D), segmented by case-agreement strength ===")
    report(res[res["dom_count"] >= 5], "well-supported (>=5/10)", "D_final")
    report(res[(res["dom_count"] >= 3) & (res["dom_count"] < 5)], "moderate (3-4/10)", "D_final")
    report(res[res["dom_count"] < 3], "scattered (<3/10)", "D_final")


if __name__ == "__main__":
    main()
