"""Periodic case consolidation: a retroactive safety net for Stage C.

Stage C is strictly sequential and forward-only — a case's exemplars are
compared only against dispatches that arrive *after* it exists, never against
each other, and a case created early from a poorly-worded first member is
frozen the moment a later, better-matching case is created instead. Nothing in
the online path ever revisits that decision. This module is the offline
counterpart: it looks at the *finished* case catalog, finds pairs of cases
whose exemplars are suspiciously close, and asks the exact same conservative
LLM judge used online whether they are in fact the same root cause. Only a
judge-confirmed pair is ever merged — a high cosine similarity by itself never
merges anything, for the same reason Stage C never trusts it alone.

Merges are resolved with union-find so transitive chains (A~B, B~C) collapse
onto one survivor. The survivor is the case with more mapped dispatches (ties
broken by the lower case number, i.e. the older case) — the more-established
case's canonical text is the one kept as the display name in reports.
"""
from collections import Counter

from . import config, ledger
from .statefiles import load_json, save_json
from .stage_casemap import (
    load_state, init_chroma, reconcile, ensure_exemplars, _exemplar_id, _judge, _cosine,
)


class _UnionFind:
    def __init__(self, ids):
        self.parent = {i: i for i in ids}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra
        return self.find(a)


def _candidate_pairs(col, state):
    """One Chroma query per exemplar, deduped to distinct case pairs, best
    exemplar-pair similarity first. Reuses the already-cached exemplar
    embeddings and the collection's own ANN index — no new embedding calls."""
    best = {}  # frozenset({case_a, case_b}) -> best_sim
    case_ids = list(state["cases"])
    for case_id in case_ids:
        exemplars = ensure_exemplars(state["cases"][case_id])
        for i, ex in enumerate(exemplars):
            eid = _exemplar_id(case_id, i)
            got = col.get(ids=[eid], include=["embeddings"])
            if not got["ids"] or got["embeddings"] is None or len(got["embeddings"]) == 0:
                continue
            res = col.query(query_embeddings=[got["embeddings"][0]],
                             n_results=min(10, col.count()),
                             include=["metadatas", "distances"])
            for dist, meta in zip(res["distances"][0], res["metadatas"][0]):
                other = (meta or {}).get("case_id")
                if other is None or other == case_id:
                    continue
                sim = 1 - dist
                if sim < config.CONSOLIDATE_SIM_FLOOR:
                    continue
                key = frozenset((case_id, other))
                if key not in best or sim > best[key]:
                    best[key] = sim
    return sorted(best.items(), key=lambda kv: -kv[1])


def run(dry_run=False):
    state = load_state()
    cache_texts = [ex["text"] for c in state["cases"].values() for ex in ensure_exemplars(c)]
    from .embcache import EmbCache
    cache = EmbCache()
    cache.ensure(cache_texts)

    col = init_chroma()
    reconcile(col, state, cache)

    print(f"Consolidate: {len(state['cases'])} cases, scanning for merge candidates "
          f"(similarity floor {config.CONSOLIDATE_SIM_FLOOR})...")
    pairs = _candidate_pairs(col, state)
    print(f"Consolidate: {len(pairs)} candidate case pair(s) above the floor.")

    member_counts = Counter(r["case_id"] for r in state["dispatches"].values() if r["case_id"])
    uf = _UnionFind(state["cases"])

    def survivor_of(a, b):
        ra, rb = uf.find(a), uf.find(b)
        na, nb = member_counts.get(ra, 0), member_counts.get(rb, 0)
        if na != nb:
            return (ra, rb) if na > nb else (rb, ra)
        return (ra, rb) if ra < rb else (rb, ra)

    merges = []
    checked = 0
    for key, sim in pairs:
        a, b = tuple(key)
        ra, rb = uf.find(a), uf.find(b)
        if ra == rb:
            continue  # already unified by an earlier, higher-similarity pair
        survivor, absorbed = survivor_of(ra, rb)
        checked += 1
        cand_text = state["cases"][absorbed]["canonical"]
        surv_text = state["cases"][survivor]["canonical"]
        surv_cat = state["cases"][survivor]["category"]
        verdict = _judge(cand_text, [(survivor, 1 - sim, surv_text, surv_cat)])
        if verdict == survivor:
            uf.union(survivor, absorbed)
            merges.append({"survivor": survivor, "absorbed": absorbed,
                           "similarity": round(sim, 4),
                           "absorbed_canonical": cand_text,
                           "survivor_canonical": surv_text})
            print(f"  MERGE  {absorbed} -> {survivor}  (sim={sim:.3f})  "
                  f"{cand_text[:50]!r} == {surv_text[:50]!r}")
        else:
            outcome = "judge said different" if verdict is None else f"judge said {verdict}"
            print(f"  keep separate  {a} / {b}  (sim={sim:.3f})  {outcome}")

    print(f"Consolidate: {checked} pair(s) judged, {len(merges)} merge(s) confirmed.")
    if dry_run or not merges:
        print("Consolidate: dry-run or no merges — nothing written." if dry_run
              else "Consolidate: no merges — nothing to write.")
        return state, merges

    # apply merges: reassign dispatches, merge exemplars (capped, deduped),
    # delete absorbed vectors, drop absorbed case records
    absorbed_to_survivor = {}
    for case_id in state["cases"]:
        root = uf.find(case_id)
        if root != case_id:
            absorbed_to_survivor[case_id] = root

    for did, rec in state["dispatches"].items():
        if rec["case_id"] in absorbed_to_survivor:
            rec["case_id"] = absorbed_to_survivor[rec["case_id"]]

    for absorbed, survivor in absorbed_to_survivor.items():
        surv_case = state["cases"][survivor]
        surv_exemplars = ensure_exemplars(surv_case)
        for ex in ensure_exemplars(state["cases"][absorbed]):
            if len(surv_exemplars) >= config.MAX_EXEMPLARS_PER_CASE:
                break
            vec = cache.get(ex["text"])
            if vec is None:
                continue
            too_close = False
            for existing in surv_exemplars:
                ev = cache.get(existing["text"])
                if ev is not None and _cosine(vec, ev) >= config.EXEMPLAR_DEDUP_SIM:
                    too_close = True
                    break
            if not too_close:
                idx = len(surv_exemplars)
                surv_exemplars.append(ex)
                col.add(ids=[_exemplar_id(survivor, idx)], embeddings=[vec.tolist()],
                        documents=[ex["text"]],
                        metadatas=[{"case_id": survivor, "category": surv_case["category"]}])
        drop_ids = [_exemplar_id(absorbed, i)
                    for i in range(len(ensure_exemplars(state["cases"][absorbed])))]
        col.delete(ids=drop_ids)
        del state["cases"][absorbed]

    state.setdefault("consolidation_log", []).extend(merges)
    save_json(config.CASEMAP_FILE, state)

    led = ledger.load()
    for rec in led["dispatches"].values():
        if rec.get("case_id") in absorbed_to_survivor:
            rec["case_id"] = absorbed_to_survivor[rec["case_id"]]
    ledger.save(led)

    from . import reports
    reports.write_all(state)

    print(f"Consolidate: {len(merges)} merge(s) applied. "
          f"Cases now: {len(state['cases'])}.")
    return state, merges
