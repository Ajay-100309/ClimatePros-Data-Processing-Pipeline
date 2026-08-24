"""
Does the case-clustering pipeline (Stage C) actually earn its keep for parts
prediction, or does a bigger flat top-N retrieval, with no case grouping at
all, match or beat it?

For a range of N and two aggregation schemes, scores flat retrieval against
the same 265-dispatch real-parts ground truth used in
03_parts_prediction_accuracy.py, then compares against the naive popularity
baseline (C) and the current case-aware final design (D).

Source data: dispatch_2k_rootcauses.numbers (repo root) + state/embeddings.npy.
Same caveats as script 03: nomic-embed, not the production bge-m3; a
2,146-dispatch sample, not the full 1,551,773-dispatch production corpus, so
very large N here is a bigger fraction of the test pool than it would be in
production. The comparison is about which mechanism wins, not a production
accuracy number.

Run from the repo root (needs analysis/requirements.txt installed):
    venv/bin/python analysis/04_flat_vs_case_retrieval.py
"""
import os

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

N_VALUES = [5, 10, 20, 30, 50, 100, 200]


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
    df = df.dropna(subset=["emb_row"]).reset_index(drop=True)
    df["emb_row"] = df["emb_row"].astype(int)

    X = emb[df["emb_row"].values].astype(np.float32)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    case_ids = df["CaseId"].values
    real_parts = df["real_parts"].tolist()
    has_real = np.array([len(p) > 0 for p in real_parts])
    print(f"Pool size for this test: {len(df)} dispatches. Real-parts queries: {has_real.sum()}")

    global_freq = Counter()
    for p in real_parts:
        for part in sorted(set(p)):
            global_freq[part] += 1
    naive_top6 = [p for p, _ in global_freq.most_common(6)]

    query_idx = np.where(has_real)[0]
    sims_all = X[query_idx] @ X.T
    max_n = max(N_VALUES)

    rows = []
    for qpos, qi in enumerate(query_idx):
        sims = sims_all[qpos].copy()
        sims[qi] = -1.0
        ranked = np.argsort(-sims, kind="stable")[:max_n]

        record = {}
        actual = set(real_parts[qi])

        # case-aware final design (D), for reference, using the fixed top-10 it's built on
        top10 = ranked[:10]
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
        d_pred = case_pred if (dom_count >= 5 and case_pred) else naive_top6
        record["D_final_hit3"] = bool(actual & set(d_pred[:3]))
        record["D_final_hit6"] = bool(actual & set(d_pred[:6]))
        record["C_naive_hit3"] = bool(actual & set(naive_top6[:3]))
        record["C_naive_hit6"] = bool(actual & set(naive_top6[:6]))

        # flat retrieval at each N, two aggregation schemes, no case grouping at all
        for n in N_VALUES:
            topn = ranked[:n]
            sim3_score = Counter()
            freq_score = Counter()
            for j in topn:
                sim = sims[j]
                if sim <= 0.55:
                    continue
                for part in sorted(set(real_parts[j])):
                    sim3_score[part] += sim ** 3
                    freq_score[part] += 1
            sim3_pred = [p for p, _ in sim3_score.most_common(6)]
            freq_pred = [p for p, _ in freq_score.most_common(6)]
            record[f"flat_sim3_{n}_hit3"] = bool(actual & set(sim3_pred[:3]))
            record[f"flat_sim3_{n}_hit6"] = bool(actual & set(sim3_pred[:6]))
            record[f"flat_freq_{n}_hit3"] = bool(actual & set(freq_pred[:3]))
            record[f"flat_freq_{n}_hit6"] = bool(actual & set(freq_pred[:6]))

        rows.append(record)

    res = pd.DataFrame(rows)

    def pct(col):
        return res[col].mean() * 100

    print()
    print("=== Reference points (no case grouping vs. the case-aware design) ===")
    print(f"{'C. naive baseline':<28} hit@3={pct('C_naive_hit3'):5.1f}%  hit@6={pct('C_naive_hit6'):5.1f}%")
    print(f"{'D. case-aware final design':<28} hit@3={pct('D_final_hit3'):5.1f}%  hit@6={pct('D_final_hit6'):5.1f}%")

    print()
    print("=== Flat retrieval, no case pipeline, similarity-cubed weighting ===")
    for n in N_VALUES:
        print(f"top-{n:<4} hit@3={pct(f'flat_sim3_{n}_hit3'):5.1f}%  hit@6={pct(f'flat_sim3_{n}_hit6'):5.1f}%")

    print()
    print("=== Flat retrieval, no case pipeline, plain frequency weighting ===")
    for n in N_VALUES:
        print(f"top-{n:<4} hit@3={pct(f'flat_freq_{n}_hit3'):5.1f}%  hit@6={pct(f'flat_freq_{n}_hit6'):5.1f}%")

    print()
    best_col = max(
        [f"flat_sim3_{n}_hit6" for n in N_VALUES] + [f"flat_freq_{n}_hit6" for n in N_VALUES],
        key=pct,
    )
    print(f"Best flat (no-case) result: {best_col} at hit@6={pct(best_col):.1f}%, "
          f"vs. case-aware design D at {pct('D_final_hit6'):.1f}% and naive baseline at {pct('C_naive_hit6'):.1f}%.")


if __name__ == "__main__":
    main()
