"""10 — RAG evaluation harness for the Parts Finder (supersedes 09_azure_parity.py).

Measures the live retrieval system (Azure index dispatches-nomic768-v1 + the
production scoring formula from TechJays app/features/parts_finder/scoring.py)
against recorded-parts ground truth, over ONE seeded query set shared by three
query arms so every comparison is paired:

  A  own cached document vector, leave-one-out   (comparable to the 44% design baseline)
  B  raw dispatch_meta `reason` embedded fresh   (documented lower bound; ticketing blob)
  C  LLM terse technician paraphrase, embedded fresh  (headline realism arm)

Retrieval variants per arm: vector (production shape), vector_hasparts
(filter="hasParts eq true"), bm25 (keyword on rootCause). One Azure call per
query x variant at K_RETRIEVE; every k-slice, lambda, blend mode, metric, and
baseline is computed offline from the cached results.

Protocol fixes over 09: corpus text via search_index.json text_sha (09's casemap
subscript crashes on default-mode batches); popularity prior corrected per query
(leave-one-out); same-text twins excluded, not just `dispatchId ne self`;
hard-fail on missing cached vectors; seeded-shuffle sampling with ids recorded.

Production-vs-eval divergences (also recorded in eval_scores.json): the live LLM
relevance gate is not replicated; live popularity is a 6h-TTL facet query that
can degrade to zero; twin-exclusion is an eval artifact. Note the production
scorer blends popularity only over parts WITH retrieval support; analysis/06/09
injected the lambda term for every known part — both blend modes are measured
("supported" is the headline, "inject" reported at k=100 for continuity).

Usage:
    venv/bin/python analysis/10_rag_eval.py --limit 25     # smoke (~3 min)
    venv/bin/python analysis/10_rag_eval.py                # full eval, n=1000 (~1-1.5 h)
    venv/bin/python analysis/10_rag_eval.py --report-only  # rescore + report from caches

Every phase checkpoints (statefiles.save_json, atomic) and resumes; smoke-run
retrievals are reused by the full run because the seeded shuffle is shared.
Writes only analysis/eval_cache/ and output/ — never state/.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import json
import math
import random
import argparse
import hashlib
import subprocess
from collections import Counter
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
from pipelib import config, search_index, llm  # noqa: E402
from pipelib.embcache import EmbCache  # noqa: E402
from pipelib.statefiles import load_json, save_json  # noqa: E402
from azure.search.documents.models import VectorizedQuery  # noqa: E402

# ---------------------------------------------------------------- constants
K_RETRIEVE = 220            # 200 + buffer for offline self/twin exclusion
K_SLICES = [10, 50, 100, 200]
K_HEAD = 100                # headline configuration
SIM_FLOOR = 0.55            # strict >, matching production scoring.py
LAMBDAS = [0.0, 0.05, 0.1, 0.2, 0.4]
LAM_HEAD = 0.1
TOP_N = 6
SEEDS = {"queries": 4242, "judge": 4243, "bootstrap": 4244, "random_baseline": 4245}
ARMS = ["A", "B", "C"]
VARIANTS = ["vector", "vector_hasparts", "bm25"]
JUDGE_CONTEXTS = 10
COS_BANDS = [(0.55, 0.65), (0.65, 0.75), (0.75, 0.85), (0.85, 1.01)]

CACHE_DIR = os.path.join(HERE, "eval_cache")
QUERIES_FILE = os.path.join(CACHE_DIR, "queries.json")
DOCIDS_FILE = os.path.join(CACHE_DIR, "docids.json")
PARA_FILE = os.path.join(CACHE_DIR, "paraphrases.json")
JUDGE_FILE = os.path.join(CACHE_DIR, "judge_verdicts.json")
PERQ_FILE = os.path.join(CACHE_DIR, "per_query_results.json")
EVAL_EMB_JSON = os.path.join(CACHE_DIR, "eval_embed_cache.json")
EVAL_EMB_NPY = os.path.join(CACHE_DIR, "eval_embed_cache.npy")
SCORES_FILE = os.path.join(config.OUTPUT_DIR, "eval_scores.json")
REPORT_FILE = os.path.join(config.OUTPUT_DIR, "rag_eval_report.html")

DIVERGENCES = [
    "live LLM relevance gate (fail-open) is not replicated - this eval measures retrieval + scoring only",
    "live popularity is a 6h-TTL facet query that can degrade to ({}, 0); eval uses the state snapshot with per-query leave-one-out correction",
    "self- and same-text-twin exclusion is an eval artifact with no live counterpart",
    "production blends popularity over supported parts only; blend_mode=inject (analysis/06/09 style) is reported for continuity",
]


def norm_part(p):
    return p.strip().upper()


def cos_from_score(score):
    if score <= 0:
        return -1.0
    return max(-1.0, min(1.0, 2.0 - 1.0 / score))


def utcnow():
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------- corpus
class Corpus:
    """Everything the eval needs, loaded once, read-only against state/."""

    def __init__(self):
        state = load_json(config.SEARCH_STATE_FILE)
        if not state:
            sys.exit("state/search_index.json missing - nothing to evaluate.")
        self.pushed = state["pushed"]                       # did -> {text_sha,...}
        self.docids = sorted(self.pushed)                   # canonical int order
        self.did_ix = {d: i for i, d in enumerate(self.docids)}

        parts_state = load_json(config.PARTS_FILE)["dispatches"]
        self.parts_items = {}                               # did -> real part dicts
        self.truth = {}                                     # did -> set(partNo)
        for did in self.docids:
            items = search_index.real_parts(
                parts_state.get(did, {"items": []})["items"])
            self.parts_items[did] = items
            self.truth[did] = {norm_part(p["part_no"]) for p in items}
        self.labeled = [d for d in self.docids if self.truth[d]]

        extract = load_json(config.EXTRACT_FILE, {})
        self.root_cause = {d: (extract.get(d) or {}).get("root_cause", "")
                           for d in self.docids}
        self.category = {d: (extract.get(d) or {}).get("category", "")
                         for d in self.docids}
        meta = load_json(config.DISPATCH_META_FILE, {})
        self.reason = {d: (meta.get(d) or {}).get("reason", "") for d in self.docids}

        pop = load_json(config.OUT_PART_POPULARITY)
        self.freq = {norm_part(k): v for k, v in pop["freq"].items()}
        self.max_global = max(pop["max_global"], 1)

        # vectors: every pushed doc's text_sha must resolve to a cached row
        cache = EmbCache()
        rows, missing = [], []
        for did in self.docids:
            row = cache.index.get(self.pushed[did]["text_sha"])
            (missing if row is None else rows).append(row if row is not None else did)
        if missing:
            sys.exit(f"{len(missing)} pushed docs have no cached vector "
                     f"(first: {missing[0]}) - cache and push record disagree.")
        mat = cache.array[np.asarray(rows)]
        self.doc_vecs = (mat / np.linalg.norm(mat, axis=1, keepdims=True)
                         ).astype(np.float32)
        self._shared_cache = cache

        # same-text twins: sha -> dids (exclude the whole cluster per query)
        by_sha = {}
        for did in self.docids:
            by_sha.setdefault(self.pushed[did]["text_sha"], []).append(did)
        self.twin_ix = {d: frozenset(self.did_ix[t] for t in by_sha[self.pushed[d]["text_sha"]])
                        for d in self.docids}

    def doc_vec(self, did):
        return self.doc_vecs[self.did_ix[did]]

    def indexed_text(self, did, client=None):
        """The exact text whose vector is in the index: extract -> casemap ->
        Azure get_document, verified against the pushed text_sha when local."""
        sha = self.pushed[did]["text_sha"]
        rc = self.root_cause.get(did, "")
        if rc and config.text_sha(rc) == sha:
            return rc
        casemap = getattr(self, "_casemap", None)
        if casemap is None:
            casemap = self._casemap = (load_json(config.CASEMAP_FILE) or {}).get("dispatches", {})
        cm = (casemap.get(did) or {}).get("text", "")
        if cm and config.text_sha(cm) == sha:
            return cm
        if client is not None:
            return client.get_document(key=did)["rootCause"]
        return rc or cm  # best local effort


# ---------------------------------------------------------------- eval-local embedding cache
class EvalEmbCache:
    """Content-addressed cache for FRESH eval query vectors (arm B/C), stored in
    eval_cache/ so the shared state/ cache is never written (08 precedent)."""

    def __init__(self):
        meta = load_json(EVAL_EMB_JSON)
        self.index, self.array = {}, None
        if meta is not None and os.path.exists(EVAL_EMB_NPY):
            self.index = meta["index"]
            self.array = np.load(EVAL_EMB_NPY)

    def get(self, text):
        row = self.index.get(config.text_sha(text))
        return None if row is None else self.array[row]

    def ensure(self, texts):
        missing, seen = [], set()
        for t in texts:
            sha = config.text_sha(t)
            if sha not in self.index and sha not in seen:
                missing.append(t)
                seen.add(sha)
        if not missing:
            return 0
        vecs = llm.embed_texts(missing)
        self.array = vecs if self.array is None else np.concatenate([self.array, vecs])
        base = self.array.shape[0] - len(missing)
        for i, t in enumerate(missing):
            self.index[config.text_sha(t)] = base + i
        tmp = EVAL_EMB_NPY + ".tmp.npy"
        np.save(tmp, self.array)
        os.replace(tmp, EVAL_EMB_NPY)
        save_json(EVAL_EMB_JSON, {"model": config.EMBED_MODEL, "dim": 768,
                                  "index": self.index})
        return len(missing)


# ---------------------------------------------------------------- stats
def wilson_ci(hits, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = hits / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def paired_bootstrap_delta(a_bits, b_bits, n_boot=2000, seed=SEEDS["bootstrap"]):
    a = np.asarray(a_bits, dtype=np.float64)
    b = np.asarray(b_bits, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, size=(n_boot, n))
    deltas = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return (float(a.mean() - b.mean()),
            float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)))


def mcnemar_exact(a_bits, b_bits):
    b01 = sum(1 for x, y in zip(a_bits, b_bits) if x and not y)
    b10 = sum(1 for x, y in zip(a_bits, b_bits) if y and not x)
    n = b01 + b10
    if n == 0:
        return 1.0
    k = min(b01, b10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def ndcg_at(ranked, truth, n=TOP_N):
    dcg = sum(1 / math.log2(i + 2) for i, p in enumerate(ranked[:n]) if p in truth)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(truth), n)))
    return dcg / ideal if ideal else 0.0


# ---------------------------------------------------------------- scoring (mirrors TechJays scoring.py)
def aggregate(neighbors, sim_floor=SIM_FLOOR):
    """neighbors: [(cos, part_items)] -> {partNo: [raw, support]}. Matches
    production aggregate_parts: strict floor, per-neighbor part dedup, rows
    without a part number or inventory id skipped."""
    aggs = {}
    for sim, items in neighbors:
        if sim <= sim_floor:
            continue
        w = sim ** 3
        seen = set()
        for p in items:
            pn = norm_part(p["part_no"])
            if not pn or not (p.get("inventory_id") or "").strip() or pn in seen:
                continue
            seen.add(pn)
            a = aggs.setdefault(pn, [0.0, 0])
            a[0] += w
            a[1] += 1
    return aggs


def rank_parts(aggs, freq, max_global, lam, blend_mode="supported"):
    """Ranked partNo list. supported = production blend_popularity (lambda over
    retrieved parts only); inject = analysis/06/09 style (lambda term for every
    known part, so unsupported parts can enter). Tie-break mirrors production:
    (-total, -support, partNo)."""
    if blend_mode == "supported":
        totals = {pn: raw + (lam * freq.get(pn, 0) / max_global if max_global > 0 else 0.0)
                  for pn, (raw, _s) in aggs.items()}
        pool = aggs
    else:
        totals = {pn: raw for pn, (raw, _s) in aggs.items()}
        if lam and max_global > 0:
            for pn, cnt in freq.items():
                totals[pn] = totals.get(pn, 0.0) + lam * cnt / max_global
        pool = totals
    return sorted(pool, key=lambda pn: (-totals[pn], -aggs.get(pn, (0.0, 0))[1], pn))


def loo_popularity(corpus, did):
    """Memoized: called once per query per config otherwise (~200k dict copies)."""
    memo = getattr(corpus, "_loo", None)
    if memo is None:
        memo = corpus._loo = {}
    if did in memo:
        return memo[did]
    freq = dict(corpus.freq)
    for pn in corpus.truth[did]:
        if pn in freq:
            freq[pn] = freq[pn] - 1
            if freq[pn] <= 0:
                del freq[pn]
    memo[did] = (freq, max(max(freq.values(), default=0), 1))
    return memo[did]


# ---------------------------------------------------------------- phase: queries
def phase_queries(corpus, args):
    existing = load_json(QUERIES_FILE)
    population = sorted(d for d in corpus.labeled
                        if corpus.root_cause[d].strip() and corpus.reason[d].strip())
    pop_sha = hashlib.sha256(json.dumps(population).encode()).hexdigest()
    if existing and not args.reseed:
        if existing["population_sha256"] != pop_sha:
            print("WARNING: eval population changed since queries.json was frozen "
                  "(state advanced?). Keeping the frozen order; rerun with --reseed "
                  "to resample against the current corpus.")
        return existing
    order = list(population)
    random.Random(SEEDS["queries"]).shuffle(order)
    queries = {"seed": SEEDS["queries"], "population_n": len(population),
               "population_sha256": pop_sha, "order": order, "frozen_at": utcnow()}
    save_json(QUERIES_FILE, queries)
    save_json(DOCIDS_FILE, {"docids": corpus.docids})
    print(f"Frozen query order over {len(population)} labeled dispatches "
          f"(seed {SEEDS['queries']}).")
    return queries


# ---------------------------------------------------------------- phase: paraphrase (arm C)
def phase_paraphrase(corpus, dids):
    state = load_json(PARA_FILE, {})
    todo = [d for d in dids if d not in state]
    if not todo:
        return state
    prompt = config.read_prompt(os.path.join(HERE, "prompt_eval_paraphrase.txt"))
    print(f"Paraphrasing {len(todo)} root causes "
          f"(~{len(todo)} chat calls at the 60 RPM gateway limit)...")
    for i, did in enumerate(todo, 1):
        msg = prompt.replace("{root_cause}", corpus.root_cause[did])
        text = None
        for attempt in (1, 2):
            parsed, _raw = llm.chat_json([{"role": "user", "content": msg}])
            if parsed and isinstance(parsed.get("paraphrase"), str) \
                    and parsed["paraphrase"].strip():
                text = parsed["paraphrase"].strip()
                break
        state[did] = ({"text": text, "at": utcnow()} if text
                      else {"failed": True, "at": utcnow()})
        if i % 25 == 0 or i == len(todo):
            save_json(PARA_FILE, state)
            print(f"  {i}/{len(todo)} paraphrased")
    return state


# ---------------------------------------------------------------- phase: embed (arms B/C)
def phase_embed(corpus, dids, arms, paraphrases):
    ec = EvalEmbCache()
    texts = []
    if "B" in arms:
        texts += [corpus.reason[d] for d in dids]
    if "C" in arms:
        texts += [paraphrases[d]["text"] for d in dids
                  if d in paraphrases and not paraphrases[d].get("failed")]
    fresh = ec.ensure(texts)
    if fresh:
        print(f"Embedded {fresh} fresh eval query texts "
              f"({math.ceil(fresh / config.EMBED_BATCH)} gateway batches).")
    return ec


def query_vector(arm, did, corpus, ec, paraphrases):
    """(vector_for_azure, unit_vector_for_local_cos, query_text) or None."""
    if arm == "A":
        v = corpus.doc_vec(did)
        return v, v, None            # arm A text resolved lazily where needed
    text = corpus.reason[did] if arm == "B" else \
        (paraphrases.get(did) or {}).get("text")
    if not text:
        return None
    # cache is keyed by the raw text passed to ensure(); embed_texts applies
    # prefix/truncation internally, so lookup must use the same raw text
    v = ec.get(text)
    if v is None:
        sys.exit(f"eval embedding missing for arm {arm} did {did} - "
                 f"rerun the embed phase")
    unit = (v / np.linalg.norm(v)).astype(np.float32)
    return v, unit, text


# ---------------------------------------------------------------- phase: retrieve
def retrieval_path(arm, variant):
    return os.path.join(CACHE_DIR, f"retrieval_{arm}_{variant}.json")


def phase_retrieve(corpus, dids, arms, variants, ec, paraphrases):
    client = search_index.search_client()
    n_calls = 0
    for arm in arms:
        for variant in variants:
            path = retrieval_path(arm, variant)
            state = load_json(path, {"meta": {"arm": arm, "variant": variant,
                                              "k": K_RETRIEVE,
                                              "index": config.AZURE_SEARCH_INDEX,
                                              "started": utcnow()},
                                     "results": {}})
            todo = [d for d in dids if d not in state["results"]]
            if not todo:
                continue
            print(f"Retrieving arm {arm} / {variant}: {len(todo)} queries...")
            for i, did in enumerate(todo, 1):
                qv = query_vector(arm, did, corpus, ec, paraphrases)
                if qv is None:                    # failed paraphrase
                    state["results"][did] = None
                    continue
                vec, _unit, text = qv
                kwargs = dict(select=["dispatchId"], top=K_RETRIEVE)
                if variant == "vector_hasparts":
                    kwargs["filter"] = "hasParts eq true"
                if variant == "bm25":
                    qtext = text if arm != "A" else corpus.indexed_text(did, client)
                    results = client.search(search_text=qtext,
                                            search_fields=["rootCause"], **kwargs)
                else:
                    results = client.search(
                        search_text=None,
                        vector_queries=[VectorizedQuery(
                            vector=[float(x) for x in vec],
                            k_nearest_neighbors=K_RETRIEVE,
                            fields="rootCauseVector")],
                        **kwargs)
                state["results"][did] = [
                    [corpus.did_ix[r["dispatchId"]], round(float(r["@search.score"]), 6)]
                    for r in results if r["dispatchId"] in corpus.did_ix]
                n_calls += 1
                if i % 100 == 0 or i == len(todo):
                    save_json(path, state)
                    print(f"  {i}/{len(todo)}")
            save_json(path, state)
    if n_calls:
        print(f"Retrieval done: {n_calls} Azure calls this run.")


# ---------------------------------------------------------------- candidates (offline)
def candidates_for(corpus, did, entry, qunit, variant):
    """Cached Azure rows -> [(docid_int, local_cos, azure_score)] with self and
    same-text twins removed. Vector variants re-sorted by local exact cosine;
    bm25 keeps Azure's keyword order (cosine still attached for the scorer)."""
    if not entry:
        return []
    excl = corpus.twin_ix[did]
    out = []
    for docid, ascore in entry:
        if docid in excl:
            continue
        cos = float(np.dot(qunit, corpus.doc_vecs[docid]))
        out.append((docid, cos, ascore))
    if variant != "bm25":
        out.sort(key=lambda t: -t[1])
    return out


# ---------------------------------------------------------------- phase: judge
def phase_judge(corpus, dids, judge_n, paraphrases, ec):
    order = list(dids)
    random.Random(SEEDS["judge"]).shuffle(order)
    sample = order[:judge_n]
    state = load_json(JUDGE_FILE, {})
    todo = [d for d in sample if d not in state]
    if not todo:
        return state, sample
    ret = load_json(retrieval_path("A", "vector"))
    if ret is None:
        print("Judge skipped: no arm A / vector retrieval cache.")
        return state, sample
    prompt = config.read_prompt(os.path.join(HERE, "prompt_eval_judge.txt"))
    client = search_index.search_client()
    print(f"Judging {len(todo)} queries x top-{JUDGE_CONTEXTS} contexts "
          f"({len(todo)} chat calls)...")
    for i, did in enumerate(todo, 1):
        cands = candidates_for(corpus, did, ret["results"].get(did),
                               corpus.doc_vec(did), "vector")[:JUDGE_CONTEXTS]
        ctx_lines, ctx_meta = [], []
        for j, (docid, cos, _s) in enumerate(cands, 1):
            ndid = corpus.docids[docid]
            ctx_lines.append(f"{j}. {corpus.indexed_text(ndid, client)}")
            ctx_meta.append({"did": ndid, "cos": round(cos, 4)})
        msg = (prompt.replace("{query}", corpus.indexed_text(did, client))
                     .replace("{contexts}", "\n".join(ctx_lines)))
        verdicts = None
        for attempt in (1, 2):
            parsed, _raw = llm.chat_json([{"role": "user", "content": msg}])
            got = (parsed or {}).get("verdicts")
            if (isinstance(got, list) and len(got) == len(cands)
                    and all(isinstance(v, dict)
                            and v.get("verdict") in ("same", "related", "unrelated")
                            for v in got)):
                verdicts = [v["verdict"] for v in
                            sorted(got, key=lambda v: int(v.get("id", 0)))]
                break
        state[did] = {"contexts": ctx_meta,
                      "verdicts": verdicts or ["error"] * len(cands),
                      "at": utcnow()}
        if i % 10 == 0 or i == len(todo):
            save_json(JUDGE_FILE, state)
            print(f"  {i}/{len(todo)} judged")
    return state, sample


# ---------------------------------------------------------------- phase: score
def evaluate_config(corpus, per_query_cands, dids, k, lam, blend_mode):
    """Metrics for one (variant-arm query set, k, lambda, blend) config."""
    hits = {n: 0 for n in (1, 3, 6, 10)}
    mrr = ndcg = p6_num = r6_macro = inter6 = truth_total = 0.0
    n_eval = 0
    bits, detail = [], {}
    ret_any = ret_floor = 0
    for did in dids:
        cands = per_query_cands.get(did)
        if cands is None:
            continue
        n_eval += 1
        truth = corpus.truth[did]
        top = cands[:k]
        has_any = any(corpus.truth[corpus.docids[d]] & truth for d, _c, _s in top)
        has_floor = any(corpus.truth[corpus.docids[d]] & truth
                        for d, c, _s in top if c > SIM_FLOOR)
        ret_any += has_any
        ret_floor += has_floor
        freq, mg = loo_popularity(corpus, did)
        aggs = aggregate([(c, corpus.parts_items[corpus.docids[d]])
                          for d, c, _s in top])
        ranked = rank_parts(aggs, freq, mg, lam, blend_mode)
        for n in hits:
            if truth & set(ranked[:n]):
                hits[n] += 1
        first = next((i + 1 for i, pn in enumerate(ranked) if pn in truth), 0)
        mrr += 1 / first if first else 0.0
        ndcg += ndcg_at(ranked, truth)
        inter = len(truth & set(ranked[:TOP_N]))
        inter6 += inter
        p6_num += inter / TOP_N
        r6_macro += inter / len(truth)
        truth_total += len(truth)
        hit6 = 1 if truth & set(ranked[:TOP_N]) else 0
        bits.append(hit6)
        detail[did] = {
            "hit6": hit6, "first_rank": first,
            "fail": ("hit" if hit6 else
                     "retrieval_miss" if not has_any else
                     "below_floor" if not has_floor else "scorer_misrank")}
    if n_eval == 0:
        return None
    lo, hi = wilson_ci(hits[6], n_eval)
    return {
        "n": n_eval,
        "hit": {f"@{n}": round(hits[n] / n_eval, 4) for n in (1, 3, 6, 10)},
        "hit6_ci95": [round(lo, 4), round(hi, 4)],
        "mrr": round(mrr / n_eval, 4),
        "ndcg6": round(ndcg / n_eval, 4),
        "precision6": {"macro": round(p6_num / n_eval, 4),
                       "micro": round(inter6 / (TOP_N * n_eval), 4)},
        "recall6": {"macro": round(r6_macro / n_eval, 4),
                    "micro": round(inter6 / truth_total, 4)},
        "retrieval_recall": {"topk": round(ret_any / n_eval, 4),
                             "topk_above_floor": round(ret_floor / n_eval, 4)},
        "_bits": bits, "_detail": detail,
    }


def slice_breakdowns(corpus, detail, dids):
    by_cat, by_size = {}, {"1": [], "2": [], "3+": []}
    head_parts = {pn for pn, _ in sorted(corpus.freq.items(),
                                         key=lambda kv: (-kv[1], kv[0]))[:10]}
    tiers = {"head": [], "mid": [], "tail": []}
    mid_parts = {pn for pn, _ in sorted(corpus.freq.items(),
                                        key=lambda kv: (-kv[1], kv[0]))[10:100]}
    excl_head = []
    for did in dids:
        rec = detail.get(did)
        if rec is None:
            continue
        cat = corpus.category.get(did) or "Unknown"
        by_cat.setdefault(cat, []).append(rec["hit6"])
        size = len(corpus.truth[did])
        by_size["1" if size == 1 else "2" if size == 2 else "3+"].append(rec["hit6"])
        truth = corpus.truth[did]
        tier = ("head" if truth & head_parts else
                "mid" if truth & mid_parts else "tail")
        tiers[tier].append(rec["hit6"])
        if not truth & head_parts:
            excl_head.append(rec["hit6"])

    def pack(groups, min_n=1):
        return {g: {"n": len(v), "hit6": round(sum(v) / len(v), 4)}
                for g, v in sorted(groups.items(), key=lambda kv: -len(kv[1]))
                if len(v) >= min_n}
    return {"by_category": pack(by_cat, min_n=10),
            "by_truth_size": pack(by_size),
            "by_part_tier": pack(tiers),
            "hit6_excl_head_truth": (round(sum(excl_head) / len(excl_head), 4)
                                     if excl_head else None),
            "excl_head_n": len(excl_head)}


def baselines_for(corpus, dids):
    rng = random.Random(SEEDS["random_baseline"])
    all_parts = sorted(corpus.freq)
    rnd = pop = 0
    for did in dids:
        truth = corpus.truth[did]
        if truth & set(rng.sample(all_parts, TOP_N)):
            rnd += 1
        freq, _mg = loo_popularity(corpus, did)
        pop_pred = [p for p, _ in sorted(freq.items(),
                                         key=lambda kv: (-kv[1], kv[0]))[:TOP_N]]
        if truth & set(pop_pred):
            pop += 1
    n = len(dids)
    out = {}
    for name, h in (("random@6", rnd), ("popularity_only@6", pop)):
        lo, hi = wilson_ci(h, n)
        out[name] = {"n": n, "hit6": round(h / n, 4),
                     "ci95": [round(lo, 4), round(hi, 4)]}
    return out


def plain_freq_baseline(corpus, per_query_cands, dids, k=K_HEAD):
    """analysis/04's second scheme: count-of-neighbors weighting, no sim^3."""
    hits = n_eval = 0
    for did in dids:
        cands = per_query_cands.get(did)
        if cands is None:
            continue
        n_eval += 1
        cnt = Counter()
        for d, c, _s in cands[:k]:
            if c <= SIM_FLOOR:
                continue
            for pn in {norm_part(p["part_no"])
                       for p in corpus.parts_items[corpus.docids[d]]
                       if norm_part(p["part_no"])}:
                cnt[pn] += 1
        pred = [p for p, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_N]]
        if corpus.truth[did] & set(pred):
            hits += 1
    if not n_eval:
        return None
    lo, hi = wilson_ci(hits, n_eval)
    return {"n": n_eval, "hit6": round(hits / n_eval, 4),
            "ci95": [round(lo, 4), round(hi, 4)]}


def parity_check(corpus, per_query_cands, dids, n_sample):
    try:
        tj_root = os.path.join(os.path.dirname(REPO), "TechJays")
        sys.path.insert(0, tj_root)
        from app.features.parts_finder.scoring import build_suggestions  # noqa
    except Exception as e:  # pragma: no cover - environment-dependent
        return {"import_ok": False, "error": str(e)[:200], "n": 0}
    from dataclasses import dataclass, field as dfield

    @dataclass(frozen=True)
    class _Part:
        inventory_id: str
        part_no: str
        name: str
        qty: float
        united_part_no: str = ""

    @dataclass(frozen=True)
    class _Neighbor:
        dispatch_id: str
        similarity: float
        parts: tuple = dfield(default_factory=tuple)

    order = list(dids)
    random.Random(SEEDS["queries"]).shuffle(order)
    sample = order[:n_sample]
    mismatches = []
    for did in sample:
        cands = per_query_cands.get(did)
        if cands is None:
            continue
        top = cands[:K_HEAD]
        neighbors = []
        for d, _c, ascore in top:
            ndid = corpus.docids[d]
            sim = cos_from_score(ascore)
            neighbors.append(_Neighbor(
                dispatch_id=ndid, similarity=sim,
                parts=tuple(_Part(inventory_id=p["inventory_id"],
                                  part_no=p["part_no"], name=p.get("name", ""),
                                  qty=float(p.get("qty", 1)),
                                  united_part_no=p.get("united_part_no", "") or "")
                            for p in corpus.parts_items[ndid])))
        prod = [s.partNumber for s in build_suggestions(
            neighbors, corpus.freq, corpus.max_global,
            sim_floor=SIM_FLOOR, lam=LAM_HEAD, limit=TOP_N)]
        aggs = aggregate([(nb.similarity, corpus.parts_items[nb.dispatch_id])
                          for nb in neighbors])
        mine = rank_parts(aggs, corpus.freq, corpus.max_global,
                          LAM_HEAD, "supported")[:TOP_N]
        if prod != mine:
            mismatches.append({"did": did, "prod": prod, "mine": mine})
    return {"import_ok": True, "n": len(sample),
            "identical_top6": len(sample) - len(mismatches),
            "mismatches": mismatches[:5]}


def judge_metrics(corpus, judge_state, sample):
    per_q_strict, per_q_lenient = [], []
    bands = {f"{lo:.2f}-{hi if hi <= 1 else 1.0:.2f}":
             Counter() for lo, hi in COS_BANDS}
    failed = 0
    for did in sample:
        rec = judge_state.get(did)
        if rec is None:
            continue
        vs = rec["verdicts"]
        if "error" in vs:
            failed += 1
            continue
        per_q_strict.append(sum(v == "same" for v in vs) / len(vs))
        per_q_lenient.append(sum(v in ("same", "related") for v in vs) / len(vs))
        for meta, v in zip(rec["contexts"], vs):
            for lo, hi in COS_BANDS:
                if lo <= meta["cos"] < hi:
                    bands[f"{lo:.2f}-{hi if hi <= 1 else 1.0:.2f}"][v] += 1
                    break
    def mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else None
    band_out = {}
    for band, cnt in bands.items():
        tot = sum(cnt.values())
        if tot:
            band_out[band] = {"n": tot,
                              **{v: round(cnt[v] / tot, 4)
                                 for v in ("same", "related", "unrelated")}}
    return {"n": len(per_q_strict), "contexts": JUDGE_CONTEXTS,
            "context_precision10_same": mean(per_q_strict),
            "context_precision10_same_or_related": mean(per_q_lenient),
            "by_cosine_band": band_out, "failed_queries": failed}


def git_sha(path):
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=path,
                              capture_output=True, text=True, timeout=10
                              ).stdout.strip() or None
    except Exception:
        return None


def phase_score(corpus, dids, arms, variants, ec, paraphrases, judge_state,
                judge_sample, parity_n):
    cand_cache = {}
    results, comparisons = [], []
    headline_bits = {}          # (arm, variant) -> {did: bit} at K_HEAD/LAM_HEAD
    perq_out = {}

    for arm in arms:
        for variant in variants:
            ret = load_json(retrieval_path(arm, variant))
            if ret is None:
                continue
            cands = {}
            for did in dids:
                entry = ret["results"].get(did)
                if entry is None:
                    cands[did] = None
                    continue
                qv = query_vector(arm, did, corpus, ec, paraphrases)
                if qv is None:
                    cands[did] = None
                    continue
                _vec, unit, _t = qv
                cands[did] = candidates_for(corpus, did, entry, unit, variant)
            cand_cache[(arm, variant)] = cands

            for k in K_SLICES:
                for lam in LAMBDAS:
                    r = evaluate_config(corpus, cands, dids, k, lam, "supported")
                    if r is None:
                        continue
                    bits, detail = r.pop("_bits"), r.pop("_detail")
                    block = {"arm": arm, "variant": variant, "k": k,
                             "lambda": lam, "blend": "supported", **r}
                    if k == K_HEAD and lam == LAM_HEAD:
                        block.update(slice_breakdowns(corpus, detail, dids))
                        headline_bits[(arm, variant)] = {
                            d: detail[d]["hit6"] for d in detail}
                        fails = Counter(v["fail"] for v in detail.values())
                        block["failure_split"] = {
                            f: round(fails[f] / r["n"], 4)
                            for f in ("hit", "retrieval_miss", "below_floor",
                                      "scorer_misrank")}
                        perq_out[f"{arm}_{variant}"] = detail
                    results.append(block)
            # inject blend for continuity with 06/09, k=100 only
            for lam in LAMBDAS:
                r = evaluate_config(corpus, cands, dids, K_HEAD, lam, "inject")
                if r is None:
                    continue
                r.pop("_bits"), r.pop("_detail")
                results.append({"arm": arm, "variant": variant, "k": K_HEAD,
                                "lambda": lam, "blend": "inject", **r})

    # paired comparisons at the headline config
    def bits_pair(a_key, b_key):
        a, b = headline_bits.get(a_key), headline_bits.get(b_key)
        if not a or not b:
            return None
        shared = sorted(set(a) & set(b))
        return [a[d] for d in shared], [b[d] for d in shared], len(shared)

    PAIRS = [(("A", "vector"), ("A", "vector_hasparts")),
             (("A", "vector"), ("A", "bm25")),
             (("A", "vector"), ("C", "vector")),
             (("B", "vector"), ("C", "vector"))]
    for a_key, b_key in PAIRS:
        pair = bits_pair(a_key, b_key)
        if pair is None:
            continue
        a_bits, b_bits, n_pairs = pair
        delta, lo, hi = paired_bootstrap_delta(a_bits, b_bits)
        comparisons.append({
            "a": {"arm": a_key[0], "variant": a_key[1]},
            "b": {"arm": b_key[0], "variant": b_key[1]},
            "metric": f"hit@{TOP_N} (k={K_HEAD}, lambda={LAM_HEAD})",
            "n_pairs": n_pairs, "delta": round(delta, 4),
            "bootstrap_ci95": [round(lo, 4), round(hi, 4)],
            "mcnemar_p": round(mcnemar_exact(a_bits, b_bits), 5)})

    base = baselines_for(corpus, dids)
    pf = plain_freq_baseline(corpus, cand_cache.get(("A", "vector"), {}), dids)
    if pf:
        base["plain_freq_agg@6 (A/vector, k=100)"] = pf

    parity = parity_check(corpus, cand_cache.get(("A", "vector"), {}),
                          dids, parity_n)

    judge = (judge_metrics(corpus, judge_state, judge_sample)
             if judge_state else None)

    ids_sha = hashlib.sha256(json.dumps(sorted(dids)).encode()).hexdigest()[:16]
    scores = {
        "schema": 1,
        "run": {
            "generated_at": utcnow(),
            "index": config.AZURE_SEARCH_INDEX,
            "index_doc_count": len(corpus.docids),
            "labeled_n": len(corpus.labeled),
            "distinct_truth_parts": len({p for d in corpus.labeled
                                         for p in corpus.truth[d]}),
            "constants": {"k_retrieve": K_RETRIEVE, "k_slices": K_SLICES,
                          "sim_floor": SIM_FLOOR, "lambdas": LAMBDAS,
                          "top_n": TOP_N, "embed_model": config.EMBED_MODEL,
                          "dim": 768, "loo_popularity": True},
            "seeds": SEEDS,
            "sample": {"n": len(dids), "ids_sha256_16": ids_sha,
                       "arm_c_failed_paraphrases":
                           sum(1 for d in dids
                               if (paraphrases.get(d) or {}).get("failed"))},
            "git": {"pipeline": git_sha(REPO),
                    "techjays": git_sha(os.path.join(os.path.dirname(REPO), "TechJays"))},
            "divergences_from_live": DIVERGENCES,
        },
        "baselines": base,
        "results": results,
        "comparisons": comparisons,
        "judge": judge,
        "parity_check": parity,
    }
    save_json(PERQ_FILE, perq_out)
    with open(SCORES_FILE + ".tmp", "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=1, sort_keys=True, ensure_ascii=False)
    os.replace(SCORES_FILE + ".tmp", SCORES_FILE)
    print(f"Scores written: {SCORES_FILE}")
    return scores


# ---------------------------------------------------------------- phase: report
def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}%"


def tbl(headers, rows):
    h = "".join(f"<th>{esc(x)}</th>" for x in headers)
    b = "".join("<tr>" + "".join(f"<td>{x}</td>" for x in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"


def ci_bar(p, lo, hi):
    return (f'<div class="bar"><div class="ci" style="left:{100*lo:.1f}%;'
            f'width:{max(0.5, 100*(hi-lo)):.1f}%"></div>'
            f'<div class="pt" style="left:{100*p:.1f}%"></div></div>'
            f'<span class="v">{pct(p)} <small>[{pct(lo)}-{pct(hi)}]</small></span>')


ARM_LABELS = {"A": "A - own doc vector (LOO)", "B": "B - raw reason field",
              "C": "C - technician paraphrase"}


def phase_report(scores):
    res = scores["results"]

    def find(arm, variant, k=K_HEAD, lam=LAM_HEAD, blend="supported"):
        for r in res:
            if (r["arm"], r["variant"], r["k"], r["lambda"], r["blend"]) == \
                    (arm, variant, k, lam, blend):
                return r
        return None

    head_rows = []
    for arm in ARMS:
        for variant in VARIANTS:
            r = find(arm, variant)
            if r:
                head_rows.append([
                    esc(ARM_LABELS.get(arm, arm)), esc(variant), r["n"],
                    ci_bar(r["hit"]["@6"], *r["hit6_ci95"]),
                    pct(r["hit"]["@1"]), pct(r["hit"]["@3"]), pct(r["hit"]["@10"]),
                    f'{r["mrr"]:.3f}', pct(r["recall6"]["macro"]),
                    pct(r["retrieval_recall"]["topk"])])

    base_rows = [[esc(k), v["n"], ci_bar(v["hit6"], *v["ci95"])]
                 for k, v in scores["baselines"].items()]

    sens_rows = []
    for variant in VARIANTS:
        for k in K_SLICES:
            row = [esc(variant), k]
            for lam in LAMBDAS:
                r = find("A", variant, k, lam)
                row.append(pct(r["hit"]["@6"]) if r else "-")
            sens_rows.append(row)

    r_head = find("A", "vector")
    fail_rows, cat_rows, tier_rows = [], [], []
    if r_head:
        for arm in ARMS:
            for variant in VARIANTS:
                r = find(arm, variant)
                if r and "failure_split" in r:
                    fs = r["failure_split"]
                    fail_rows.append([esc(f"{arm}/{variant}"), pct(fs["hit"]),
                                      pct(fs["retrieval_miss"]),
                                      pct(fs["below_floor"]),
                                      pct(fs["scorer_misrank"])])
        cat_rows = [[esc(c), v["n"], pct(v["hit6"])]
                    for c, v in r_head.get("by_category", {}).items()]
        tier_rows = [[esc(t), v["n"], pct(v["hit6"])]
                     for t, v in r_head.get("by_part_tier", {}).items()]
        tier_rows.append(["(hit@6 excluding head-tier truths)",
                          r_head.get("excl_head_n", "-"),
                          pct(r_head.get("hit6_excl_head_truth"))])

    comp_rows = [[esc(f'{c["a"]["arm"]}/{c["a"]["variant"]}'),
                  esc(f'{c["b"]["arm"]}/{c["b"]["variant"]}'), c["n_pairs"],
                  f'{c["delta"]:+.1%}',
                  f'[{c["bootstrap_ci95"][0]:+.1%}, {c["bootstrap_ci95"][1]:+.1%}]',
                  f'{c["mcnemar_p"]:.4f}'] for c in scores["comparisons"]]

    judge = scores.get("judge")
    judge_html = "<p>Judge phase not run.</p>"
    if judge and judge.get("n"):
        band_rows = [[esc(b), v["n"], pct(v["same"]), pct(v["related"]),
                      pct(v["unrelated"])]
                     for b, v in judge["by_cosine_band"].items()]
        judge_html = (
            f'<p>n={judge["n"]} queries x top-{judge["contexts"]} contexts '
            f'(arm A / vector). Context precision@10: '
            f'<b>{pct(judge["context_precision10_same"])}</b> strict (same failure), '
            f'<b>{pct(judge["context_precision10_same_or_related"])}</b> lenient '
            f'(same or related). Failed judge calls: {judge["failed_queries"]}.</p>'
            + tbl(["cosine band", "n contexts", "same", "related", "unrelated"],
                  band_rows))

    par = scores["parity_check"]
    par_html = (f'<p class="ok">{par["identical_top6"]}/{par["n"]} sampled queries '
                f'produce an identical top-6 to the production scorer '
                f'(TechJays build_suggestions).</p>'
                if par.get("import_ok") and not par.get("mismatches") else
                f'<p class="warn">Parity: {esc(json.dumps(par)[:400])}</p>')

    run = scores["run"]
    div_html = "".join(f"<li>{esc(d)}</li>" for d in run["divergences_from_live"])

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Parts Finder RAG Evaluation</title><style>
body{{font:14px/1.5 -apple-system,Segoe UI,sans-serif;margin:2rem auto;max-width:1100px;
padding:0 1rem;color:#1a1a2e;background:#fff}}
h1{{font-size:1.5rem}} h2{{font-size:1.15rem;margin-top:2.2rem;border-bottom:2px solid #114D9A;
padding-bottom:.3rem}} table{{border-collapse:collapse;width:100%;margin:.8rem 0;font-size:13px}}
th,td{{border:1px solid #d8dce6;padding:.35rem .6rem;text-align:left;vertical-align:middle}}
th{{background:#f0f3f9}} tr:nth-child(even) td{{background:#fafbfd}}
.bar{{position:relative;display:inline-block;width:180px;height:12px;background:#eef1f7;
border-radius:6px;vertical-align:middle;margin-right:.5rem}}
.ci{{position:absolute;top:2px;height:8px;background:#b9cdec;border-radius:4px}}
.pt{{position:absolute;top:0;width:3px;height:12px;background:#114D9A;border-radius:2px}}
.v{{white-space:nowrap}} small{{color:#667}} .ok{{color:#1a7f37}} .warn{{color:#b35900}}
.caveat{{background:#fff8e6;border:1px solid #e6cf8b;border-radius:6px;padding:.8rem 1.2rem}}
footer{{margin-top:2.5rem;color:#667;font-size:12px;border-top:1px solid #d8dce6;padding-top:.8rem}}
</style></head><body>
<h1>Parts Finder RAG Evaluation</h1>
<p>Index <b>{esc(run["index"])}</b> - {run["index_doc_count"]:,} documents,
{run["labeled_n"]:,} with real recorded parts ({run["distinct_truth_parts"]} distinct part numbers).
Evaluated on <b>n={run["sample"]["n"]}</b> seeded leave-one-out queries shared by all arms.
Scoring formula and constants mirror the production service
(k={K_HEAD}, floor {SIM_FLOOR}, lambda {LAM_HEAD}, top-{TOP_N}).</p>
<div class="caveat"><b>What this eval is not:</b><ul>{div_html}</ul></div>

<h2>Headline - hit@6 by query arm x retrieval variant (k={K_HEAD}, lambda={LAM_HEAD})</h2>
<p>Arm C (technician-style paraphrase, embedded fresh) is the realism headline;
arm A is comparable to the original 44% design baseline; arm B is a documented lower bound.</p>
{tbl(["query arm", "retrieval", "n", "hit@6 [95% CI]", "hit@1", "hit@3", "hit@10",
      "MRR", "recall@6", "retr.recall@100"], head_rows)}

<h2>Baselines</h2>
{tbl(["baseline", "n", "hit@6 [95% CI]"], base_rows)}

<h2>Sensitivity - arm A hit@6 across k and lambda (blend=supported)</h2>
{tbl(["variant", "k"] + [f"lambda={l}" for l in LAMBDAS], sens_rows)}

<h2>Failure decomposition (k={K_HEAD}, lambda={LAM_HEAD})</h2>
<p>retrieval_miss: no retrieved neighbor held a true part - below_floor: evidence
retrieved but under the {SIM_FLOOR} cosine floor - scorer_misrank: evidence above
the floor but the true part missed the top-{TOP_N}.</p>
{tbl(["arm/variant", "hit", "retrieval miss", "below floor", "scorer misrank"], fail_rows)}

<h2>Per-category (arm A / vector, categories with n>=10)</h2>
{tbl(["category", "n", "hit@6"], cat_rows)}

<h2>Part-popularity tiers (arm A / vector)</h2>
<p>head = a truth part in the 10 most-used parts; tail = none in the top 100.
The exclusion row answers "is it just predicting the most common parts?"</p>
{tbl(["tier", "n", "hit@6"], tier_rows)}

<h2>LLM-judged context relevance</h2>
{judge_html}

<h2>Paired comparisons (hit@6, k={K_HEAD}, lambda={LAM_HEAD})</h2>
{tbl(["A", "B", "n pairs", "delta (A-B)", "bootstrap 95% CI", "McNemar p"], comp_rows)}

<h2>Production-scorer parity</h2>
{par_html}

<footer>Generated {esc(run["generated_at"])} - pipeline {esc(run["git"]["pipeline"])}
/ techjays {esc(run["git"]["techjays"])} - seeds {esc(json.dumps(run["seeds"]))}
- sample ids sha {esc(run["sample"]["ids_sha256_16"])}
- machine-readable scores: output/eval_scores.json</footer>
</body></html>"""
    with open(REPORT_FILE + ".tmp", "w", encoding="utf-8") as f:
        f.write(html)
    os.replace(REPORT_FILE + ".tmp", REPORT_FILE)
    print(f"Report written: {REPORT_FILE}")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=int, default=1000)
    ap.add_argument("--judge-sample", type=int, default=200)
    ap.add_argument("--parity-sample", type=int, default=50)
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke run: cap queries at N (judge capped at 3)")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--arms", default="A,B,C")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--reseed", action="store_true",
                    help="refreeze the query order against the current corpus")
    ap.add_argument("--report-only", action="store_true",
                    help="rescore + regenerate report from caches, no network")
    args = ap.parse_args()

    arms = [a for a in args.arms.split(",") if a in ARMS]
    variants = [v for v in args.variants.split(",") if v in VARIANTS]
    config.require_search_config()
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("Loading corpus (state + vectors)...")
    corpus = Corpus()
    print(f"Corpus: {len(corpus.docids)} indexed docs, "
          f"{len(corpus.labeled)} with real parts.")

    queries = phase_queries(corpus, args)
    n = min(args.sample, queries["population_n"])
    if args.limit:
        n = min(args.limit, n)
    dids = queries["order"][:n]
    judge_n = 3 if args.limit else min(args.judge_sample, n)
    parity_n = min(args.parity_sample, n)
    print(f"Query set: n={n} (arms {','.join(arms)}; variants {','.join(variants)})")

    paraphrases = load_json(PARA_FILE, {})
    if not args.report_only:
        if "C" in arms:
            paraphrases = phase_paraphrase(corpus, dids)
        ec = phase_embed(corpus, dids, arms, paraphrases)
        phase_retrieve(corpus, dids, arms, variants, ec, paraphrases)
    else:
        ec = EvalEmbCache()

    judge_state, judge_sample = ({}, [])
    if not args.no_judge:
        if args.report_only:
            judge_state = load_json(JUDGE_FILE, {})
            order = list(dids)
            random.Random(SEEDS["judge"]).shuffle(order)
            judge_sample = order[:judge_n]
        else:
            judge_state, judge_sample = phase_judge(corpus, dids, judge_n,
                                                    paraphrases, ec)

    print("Scoring (offline)...")
    scores = phase_score(corpus, dids, arms, variants, ec, paraphrases,
                         judge_state, judge_sample, parity_n)
    phase_report(scores)

    r = next((r for r in scores["results"]
              if (r["arm"], r["variant"], r["k"], r["lambda"], r["blend"]) ==
              ("A", "vector", K_HEAD, LAM_HEAD, "supported")), None)
    if r:
        print(f"\nHeadline (arm A / vector, k={K_HEAD}, lambda={LAM_HEAD}): "
              f"hit@6 {r['hit']['@6']:.1%} "
              f"[{r['hit6_ci95'][0]:.1%}-{r['hit6_ci95'][1]:.1%}], "
              f"MRR {r['mrr']:.3f}, n={r['n']}")


if __name__ == "__main__":
    main()
