"""
Does embedding the raw DispatchReason field, instead of the LLM-cleaned
root-cause extraction the case pipeline already produces, change parts-
prediction accuracy?

This is a direct test of a question that turns out to already be half-
answered by how scripts 03/04/06 were built: every accuracy number they
report (36.2%, 39.6%, 42.3%, 44.2%, 44.2-44.5%) was already computed from
embeddings of the CLEANED text (state/embeddings.npy, keyed by the Stage B
extracted root-cause sentence), not raw dispatch text. Nothing in this repo
had actually embedded the raw field and measured it until this script.

dispatch_2k_rootcauses.numbers happens to carry both fields for the same
2,146 dispatches: DispatchReason (raw, unfiltered -- includes work-order
boilerplate, tracking numbers, OnCallRegion tags) and "RootCauseText
(trimmed Notes)" (the cleaned extraction, already cached). That makes a
same-population, same-methodology, same-model (nomic-embed) comparison
possible: embed the raw field fresh, re-run the identical flat top-N
sim-cubed retrieval from 04_flat_vs_case_retrieval.py on both embedding
spaces, and compare hit@6 head to head.

This calls the live embedding gateway (pipelib.llm.embed_texts) to embed
the ~2,075 raw-text values that have never been embedded before. Results
are cached locally in analysis/raw_embed_cache.{npy,json} -- a new,
analysis-only cache, separate from and never written into state/. Re-runs
after the first one are gateway-free. No new pip dependency: pipelib.llm
already needs `openai`, which is in the main requirements.txt.

Caveat on the "raw" side specifically: the DispatchReason column in this
file is a long, ticketing-system-format text block (job type, assigned-to,
tracking number, NTE dollar figure, OnCallRegion tag, the actual complaint,
all concatenated) -- not the same field 01_dispatch_reason_quality.py
checked. That script's 83%-usable figure is about state/dispatch_meta.json's
much shorter `reason` field, the one a technician's live query would
resemble. This script's "raw" result is evidence about what to embed for
the HISTORICAL corpus, not a reversal of the decision to accept a live
query's short description as-is.

Run from the repo root (needs analysis/requirements.txt installed, plus a
working .env pointing at the live LLM gateway for the first run only):
    venv/bin/python analysis/08_raw_vs_cleaned_embedding.py

Verified output (bit-exact on the cleaned side, which exactly reproduces
04_flat_vs_case_retrieval.py's numbers on this same population; the raw
side depends only on the embedding cache, not on any run-to-run randomness):
    Pool: 2,077 dispatches. Real-parts queries: 265.
    hit@6      top-10   top-100  top-200
    cleaned     27.2%    42.3%    44.2%
    raw         12.5%    26.4%    28.7%
    Mean top-100 neighbour overlap between the two embeddings: 6.8%.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from numbers_parser import Document

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))  # so `from pipelib import llm` resolves when run as a script
NUMBERS_FILE = HERE / "dispatch_2k_rootcauses.numbers"
SHEET_NAME = "Dispatches_RootCauses - Parts"
CLEAN_COL = "RootCauseText (trimmed Notes)"
RAW_COL = "DispatchReason"
PARTS_COL = "PartsUsed (part ×qty)"

RAW_CACHE_NPY = Path(__file__).resolve().parent / "raw_embed_cache.npy"
RAW_CACHE_JSON = Path(__file__).resolve().parent / "raw_embed_cache.json"

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


def embed_raw_texts(texts):
    """Analysis-only embedding cache for the raw DispatchReason field.
    Lives in analysis/, never touches state/embeddings.npy."""
    index = {}
    array = None
    if RAW_CACHE_JSON.exists() and RAW_CACHE_NPY.exists():
        index = json.loads(RAW_CACHE_JSON.read_text())
        array = np.load(RAW_CACHE_NPY)

    missing, seen = [], set()
    for t in texts:
        sha = text_sha(t)
        if sha not in index and sha not in seen:
            missing.append(t)
            seen.add(sha)

    if missing:
        print(f"Embedding {len(missing)} raw-text value(s) not yet cached "
              f"(live gateway call, nomic-embed)...")
        from pipelib import llm
        vecs = llm.embed_texts(missing)
        array = vecs if array is None else np.concatenate([array, vecs], axis=0)
        base = array.shape[0] - len(missing)
        for i, t in enumerate(missing):
            index[text_sha(t)] = base + i
        np.save(RAW_CACHE_NPY, array)
        RAW_CACHE_JSON.write_text(json.dumps({"model": "nomic-embed", "index": index}))
    else:
        print("All raw-text values already cached locally -- no gateway call needed.")

    return array, index


def flat_sim3_scores(sims_all, ranked_all, real_parts, has_real, query_idx):
    """Same methodology as 04_flat_vs_case_retrieval.py: sim^3-weighted
    accumulation over the top-N neighbours' recorded parts, sim > 0.55."""
    rows = []
    for qpos, qi in enumerate(query_idx):
        sims = sims_all[qpos]
        ranked = ranked_all[qpos]
        actual = set(real_parts[qi])
        record = {}
        for n in N_VALUES:
            topn = ranked[:n]
            score = Counter()
            for j in topn:
                sim = sims[j]
                if sim <= 0.55:
                    continue
                for part in sorted(set(real_parts[j])):
                    score[part] += sim ** 3
            pred = [p for p, _ in score.most_common(6)]
            record[f"hit3_{n}"] = bool(actual & set(pred[:3]))
            record[f"hit6_{n}"] = bool(actual & set(pred[:6]))
        rows.append(record)
    return pd.DataFrame(rows)


def rank_all(X, query_idx, max_n):
    sims_all = X[query_idx] @ X.T
    ranked_all = []
    for qpos, qi in enumerate(query_idx):
        sims = sims_all[qpos].copy()
        sims[qi] = -1.0
        sims_all[qpos] = sims
        ranked_all.append(np.argsort(-sims, kind="stable")[:max_n])
    return sims_all, ranked_all


def main():
    df = load_numbers_sheet(NUMBERS_FILE, SHEET_NAME)
    df = df.dropna(subset=[CLEAN_COL, RAW_COL, "CaseId"])
    df["real_parts"] = df[PARTS_COL].apply(parse_parts)
    df = df[df[CLEAN_COL].str.len() > 0].reset_index(drop=True)

    idx_meta = json.load(open(HERE / "state" / "embeddings_index.json"))
    sha_to_row = idx_meta["index"]
    clean_emb = np.load(HERE / "state" / "embeddings.npy")

    df["clean_row"] = df[CLEAN_COL].apply(lambda t: sha_to_row.get(text_sha(str(t))))
    n_before = len(df)
    df = df.dropna(subset=["clean_row"]).reset_index(drop=True)
    df["clean_row"] = df["clean_row"].astype(int)
    print(f"Pool with a cached cleaned-text embedding: {len(df)}/{n_before} "
          f"(matches 04_flat_vs_case_retrieval.py's population)")

    raw_array, raw_index = embed_raw_texts(df[RAW_COL].astype(str).tolist())
    df["raw_row"] = df[RAW_COL].astype(str).apply(lambda t: raw_index[text_sha(t)])

    X_clean = clean_emb[df["clean_row"].values].astype(np.float32)
    X_clean = X_clean / np.linalg.norm(X_clean, axis=1, keepdims=True)
    X_raw = raw_array[df["raw_row"].values].astype(np.float32)
    X_raw = X_raw / np.linalg.norm(X_raw, axis=1, keepdims=True)

    real_parts = df["real_parts"].tolist()
    has_real = np.array([len(p) > 0 for p in real_parts])
    query_idx = np.where(has_real)[0]
    print(f"Real-parts query dispatches: {len(query_idx)} "
          f"(same 265-dispatch ground truth as scripts 03/04/06 use, restricted "
          f"to this pool)")

    max_n = max(N_VALUES)
    sims_clean, ranked_clean = rank_all(X_clean, query_idx, max_n)
    sims_raw, ranked_raw = rank_all(X_raw, query_idx, max_n)

    res_clean = flat_sim3_scores(sims_clean, ranked_clean, real_parts, has_real, query_idx)
    res_raw = flat_sim3_scores(sims_raw, ranked_raw, real_parts, has_real, query_idx)

    print()
    print("=== hit@6, flat sim^3-weighted retrieval: cleaned text vs. raw text ===")
    print(f"{'N':<6}{'cleaned hit@6':<16}{'raw hit@6':<16}{'delta':<10}")
    for n in N_VALUES:
        c = res_clean[f"hit6_{n}"].mean() * 100
        r = res_raw[f"hit6_{n}"].mean() * 100
        print(f"{n:<6}{c:<16.1f}{r:<16.1f}{r - c:+.1f}")

    print()
    print("=== hit@3, same comparison ===")
    print(f"{'N':<6}{'cleaned hit@3':<16}{'raw hit@3':<16}{'delta':<10}")
    for n in N_VALUES:
        c = res_clean[f"hit3_{n}"].mean() * 100
        r = res_raw[f"hit3_{n}"].mean() * 100
        print(f"{n:<6}{c:<16.1f}{r:<16.1f}{r - c:+.1f}")

    # Diagnostic: how different are the neighbourhoods raw vs. cleaned text
    # retrieve for the same query, at the N that matters for production (100)?
    print()
    print("=== Neighbourhood overlap at top-100: does raw text find the same "
          "matches as cleaned text? ===")
    overlaps = []
    for qpos in range(len(query_idx)):
        top_clean = set(ranked_clean[qpos][:100].tolist())
        top_raw = set(ranked_raw[qpos][:100].tolist())
        overlaps.append(len(top_clean & top_raw) / 100.0)
    overlaps = np.array(overlaps)
    print(f"Mean fraction of the same 100 neighbours found by both: {overlaps.mean() * 100:.1f}%")
    print(f"Median: {np.median(overlaps) * 100:.1f}%  |  min: {overlaps.min() * 100:.1f}%  "
          f"|  max: {overlaps.max() * 100:.1f}%")


if __name__ == "__main__":
    main()
