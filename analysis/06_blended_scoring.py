"""
Follow-up to 04_flat_vs_case_retrieval.py: does blending a global
part-frequency term into flat retrieval recover parts that are common
across *different* root causes, the case narrow similarity search alone
would under-weight?

This directly tests a real concern: root cause and parts-needed are not
the same axis. Two dispatches can share a root cause but need different
parts (unit-model-specific), or have different root causes but need the
same common part. The second case is what this script checks: is there
signal outside the top-N textually-similar neighbours that a cheap,
precomputed "how common is this part overall" term can recover?

Source data: dispatch_2k_rootcauses.numbers + state/embeddings.npy, same
265-dispatch real-parts ground truth as script 04.

Run from the repo root:
    venv/bin/python analysis/06_blended_scoring.py
"""
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
    df = df.dropna(subset=["emb_row"]).reset_index(drop=True)
    df["emb_row"] = df["emb_row"].astype(int)

    X = emb[df["emb_row"].values].astype(np.float32)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    real_parts = df["real_parts"].tolist()
    has_real = np.array([len(p) > 0 for p in real_parts])

    global_freq = Counter()
    for p in real_parts:
        for part in sorted(set(p)):
            global_freq[part] += 1
    naive_top6 = [p for p, _ in global_freq.most_common(6)]
    max_global = max(global_freq.values())

    query_idx = np.where(has_real)[0]
    sims_all = X[query_idx] @ X.T

    def evaluate(pred_fn, label):
        hit3 = hit6 = 0
        for qpos, qi in enumerate(query_idx):
            sims = sims_all[qpos].copy()
            sims[qi] = -1.0
            ranked = np.argsort(-sims, kind="stable")
            pred = pred_fn(qi, sims, ranked)
            actual = set(real_parts[qi])
            if actual & set(pred[:3]):
                hit3 += 1
            if actual & set(pred[:6]):
                hit6 += 1
        n = len(query_idx)
        print(f"{label:<45} n={n}  hit@3={hit3 / n * 100:5.1f}%  hit@6={hit6 / n * 100:5.1f}%")

    def flat_only(n):
        def fn(qi, sims, ranked):
            score = Counter()
            for j in ranked[:n]:
                if sims[j] <= 0.55:
                    continue
                for part in sorted(set(real_parts[j])):
                    score[part] += sims[j] ** 3
            return [p for p, _ in score.most_common(6)]
        return fn

    def flat_plus_blend(n, lam):
        def fn(qi, sims, ranked):
            score = Counter()
            for j in ranked[:n]:
                if sims[j] <= 0.55:
                    continue
                for part in sorted(set(real_parts[j])):
                    score[part] += sims[j] ** 3
            for part, cnt in global_freq.items():
                score[part] += lam * (cnt / max_global)
            return [p for p, _ in score.most_common(6)]
        return fn

    print("=== Reference ===")
    evaluate(lambda qi, s, r: naive_top6, "Naive baseline alone")
    evaluate(flat_only(100), "Flat top-100, no blend")
    evaluate(flat_only(200), "Flat top-200, no blend")

    print()
    print("=== Flat top-100 + global-frequency blend, by blend weight ===")
    for lam in [0.05, 0.1, 0.2, 0.4]:
        evaluate(flat_plus_blend(100, lam), f"Flat top-100 + blend (lambda={lam})")

    print()
    print("=== Flat top-200 + global-frequency blend, by blend weight ===")
    for lam in [0.05, 0.1, 0.2, 0.4]:
        evaluate(flat_plus_blend(200, lam), f"Flat top-200 + blend (lambda={lam})")


if __name__ == "__main__":
    main()
