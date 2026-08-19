"""Stage C: map extracted root causes onto canonical cases. Strictly sequential —
a new case must be visible to the very next query. Ported from main_casemap.py,
re-keyed by DispatchId and backed by the per-text embedding cache.
"""
import sys
import json

import numpy as np

from . import config, llm
from .statefiles import load_json, save_json

_PROMPT = config.read_prompt(config.PROMPT_CASEMAP)


def load_state():
    state = load_json(config.CASEMAP_FILE)
    if state is None:
        raise SystemExit(
            "state/casemap.json missing — run migrate_casemap_state.py first.")
    if state.get("version") != 2:
        raise SystemExit(
            f"state/casemap.json has version {state.get('version')}, expected 2 — "
            "run migrate_casemap_state.py.")
    if state.get("fingerprint") != config.casemap_fingerprint():
        raise SystemExit(
            "Casemap fingerprint mismatch: prompt_casemap.txt, thresholds, or "
            "models changed since the case DB was built. Restore the previous "
            "config, or start a fresh case DB (delete state/casemap.json and "
            f"{config.CHROMA_DIR}).")
    return state


def _cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _exemplar_id(case_id, i):
    return f"{case_id}#{i}"


def ensure_exemplars(case):
    """Backward-compat migration: seed exemplars from the frozen canonical text
    for case records written before multi-exemplar retrieval existed."""
    if "exemplars" not in case:
        case["exemplars"] = [{"text": case["canonical"], "text_sha": case["text_sha"]}]
    return case["exemplars"]


def init_chroma():
    import chromadb
    from chromadb.config import Settings

    cclient = chromadb.PersistentClient(
        path=config.CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False))
    col = cclient.get_or_create_collection(
        name=config.CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"})
    space = (col.metadata or {}).get("hnsw:space")
    if space != "cosine":
        sys.exit(f"Collection uses space '{space}', not cosine. "
                 f"Delete {config.CHROMA_DIR} and rebuild.")
    return col


def reconcile(col, state, cache):
    """Self-heals drift between the case registry and Chroma. One vector per
    exemplar (id "CASE-XXXX#i"), not one per case — a case registry written
    before multi-exemplar retrieval existed has no "exemplars" list yet, so
    this also doubles as the one-time migration: every old single-vector case
    becomes exemplar #0 under the new id scheme, rebuilt entirely from the
    embedding cache with no new embedding calls."""
    expected = {}  # exemplar_id -> (case_id, exemplar)
    for case_id, case in state["cases"].items():
        for i, ex in enumerate(ensure_exemplars(case)):
            expected[_exemplar_id(case_id, i)] = (case_id, ex)

    chroma_ids = set(col.get(include=[])["ids"])
    expected_ids = set(expected)

    orphans = chroma_ids - expected_ids
    if orphans:
        print(f"Reconcile: deleting {len(orphans)} orphan vector(s).")
        col.delete(ids=list(orphans))

    missing = expected_ids - chroma_ids
    if missing:
        print(f"Reconcile: restoring {len(missing)} exemplar vector(s).")
        for eid in missing:
            case_id, ex = expected[eid]
            vec = cache.get(ex["text"])
            assert vec is not None, \
                f"{eid} exemplar text missing from embedding cache"
            col.upsert(ids=[eid],
                       embeddings=[vec.tolist()],
                       documents=[ex["text"]],
                       metadatas=[{"case_id": case_id,
                                   "category": state["cases"][case_id]["category"]}])

    assert col.count() == len(expected_ids), \
        f"Chroma count {col.count()} != expected exemplar count {len(expected_ids)}"


def _judge(text, candidates):
    cand_lines = "\n".join(
        f"{i}. {cid} — \"{doc}\" (Category: {cat})"
        for i, (cid, _dist, doc, cat) in enumerate(candidates, start=1))
    prompt = _PROMPT.replace("{summary}", text).replace("{candidates}", cand_lines)
    valid_ids = {cid for cid, _, _, _ in candidates}

    messages = [
        {"role": "system", "content": "You are an expert HVAC service analyst."},
        {"role": "user", "content": prompt},
    ]
    for attempt in range(2):
        parsed, raw = llm.chat_json(messages)
        problem = None
        if parsed is None or "match" not in parsed:
            problem = "invalid JSON shape"
        else:
            match = parsed["match"]
            if match is None:
                return None
            if match in valid_ids:
                return match
            problem = f"'{match}' is not one of the listed case ids"
        if attempt == 0:
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content":
                             f"Your previous answer was invalid: {problem}. "
                             'Answer again with JSON only: {"match": "CASE-XXXX"} or {"match": null}.'})
    return "UNRESOLVED"


def _create_case(col, state, dup_cache, dispatch_id, text, category, cache):
    case_id = f"CASE-{state['next_case_num']:04d}"
    state["next_case_num"] += 1
    vec = cache.get(text)
    exemplar = {"text": text, "text_sha": config.text_sha(text)}
    col.add(ids=[_exemplar_id(case_id, 0)], embeddings=[vec.tolist()],
            documents=[text], metadatas=[{"case_id": case_id, "category": category}])
    state["cases"][case_id] = {"canonical": text, "category": category,
                               "dispatch_id": dispatch_id,
                               "text_sha": config.text_sha(text),
                               "exemplars": [exemplar]}
    dup_cache[text.strip().casefold()] = case_id
    return case_id


def _add_exemplar_if_useful(col, state, cache, case_id, text):
    """Grow a case's exemplar set with a genuinely new phrasing (capped, deduped)
    so future retrieval isn't stuck matching only the one dispatch that happened
    to create the case — the drift this fixes is directly visible in production
    data: same-case dispatches matched by exact text can be as low as 0.75
    cosine similarity from the original frozen canonical exemplar."""
    case = state["cases"][case_id]
    exemplars = ensure_exemplars(case)
    if len(exemplars) >= config.MAX_EXEMPLARS_PER_CASE:
        return
    vec = cache.get(text)
    if vec is None:
        return
    for ex in exemplars:
        ex_vec = cache.get(ex["text"])
        if ex_vec is not None and _cosine(vec, ex_vec) >= config.EXEMPLAR_DEDUP_SIM:
            return
    idx = len(exemplars)
    exemplars.append({"text": text, "text_sha": config.text_sha(text)})
    col.add(ids=[_exemplar_id(case_id, idx)], embeddings=[vec.tolist()],
            documents=[text], metadatas=[{"case_id": case_id, "category": case["category"]}])


def run(batch, extract_state, useful_by_dispatch):
    state = load_state()
    cache_texts = []
    todo = []
    for d in batch["dispatches"]:
        did = d["dispatch_id"]
        if did in state["dispatches"]:
            continue
        if did not in useful_by_dispatch:
            continue  # stage A incomplete for this dispatch
        if not useful_by_dispatch[did]:
            continue  # zero useful notes — terminal without mapping
        if did not in extract_state:
            continue  # stage B incomplete
        text = extract_state[did]["root_cause"].strip() or config.NO_FAULT_PLACEHOLDER
        todo.append((did, text, extract_state[did]["category"]))
        cache_texts.append(text)

    if not todo:
        print("Stage C: nothing to map (all done).")
        return state

    from .embcache import EmbCache
    cache = EmbCache()
    cache.ensure(cache_texts + [c["canonical"] for c in state["cases"].values()])

    col = init_chroma()
    reconcile(col, state, cache)

    dup_cache = {}
    for did, rec in state["dispatches"].items():
        if rec["case_id"]:
            dup_cache[rec["text"].strip().casefold()] = rec["case_id"]

    print(f"Stage C: mapping {len(todo)} dispatches (cases now: {len(state['cases'])})...")

    def process(items, phase):
        done = 0
        for did, text, category in items:
            if did in state["dispatches"]:
                continue
            key = text.strip().casefold()
            rec = {"text": text, "text_sha": config.text_sha(text)}

            if key in dup_cache:
                rec.update(case_id=dup_cache[key], match_type="matched_exact")
            else:
                k_raw = min(config.CANDIDATE_RAW_K, col.count())
                candidates = []
                if k_raw > 0:
                    res = col.query(query_embeddings=[cache.get(text).tolist()],
                                    n_results=k_raw,
                                    include=["documents", "metadatas", "distances"])
                    # Each case can have multiple exemplar vectors; keep only the
                    # closest exemplar per case_id, then rank cases by that. The
                    # judge is still shown each candidate's stable canonical text,
                    # not the matched exemplar, so its input format is unchanged.
                    best_per_case = {}
                    for dist, meta in zip(res["distances"][0], res["metadatas"][0]):
                        if dist > 1 - config.SIM_FLOOR:
                            continue
                        cid = (meta or {}).get("case_id")
                        if cid is None or cid not in state["cases"]:
                            continue
                        if cid not in best_per_case or dist < best_per_case[cid]:
                            best_per_case[cid] = dist
                    ranked = sorted(best_per_case.items(), key=lambda kv: kv[1])[:config.N_CANDIDATES]
                    candidates = [
                        (cid, dist, state["cases"][cid]["canonical"],
                         state["cases"][cid]["category"])
                        for cid, dist in ranked
                    ]
                if not candidates:
                    case_id = _create_case(col, state, dup_cache, did, text, category, cache)
                    rec.update(case_id=case_id,
                               match_type="new_first" if k_raw == 0 else "new_no_candidates")
                else:
                    verdict = _judge(text, candidates)
                    if verdict == "UNRESOLVED":
                        rec.update(case_id="", match_type="unresolved")
                    elif verdict is None:
                        case_id = _create_case(col, state, dup_cache, did, text, category, cache)
                        rec.update(case_id=case_id, match_type="new_llm_rejected")
                    else:
                        rec.update(case_id=verdict, match_type="matched_llm")
                        dup_cache[key] = verdict
                        _add_exemplar_if_useful(col, state, cache, verdict, text)

            state["dispatches"][did] = rec
            if rec["case_id"]:
                resolved = sum(1 for r in state["dispatches"].values() if r["case_id"])
                state["growth"].append([resolved, len(state["cases"])])

            done += 1
            print(f"  [C:{phase} {done}/{len(items)}] {did} -> "
                  f"{rec['match_type']} {rec['case_id']} (cases: {len(state['cases'])})")
            if done % config.CHECKPOINT_EVERY == 0:
                save_json(config.CASEMAP_FILE, state)

    try:
        process(todo, "main")
        unresolved = [(did, text, cat) for did, text, cat in todo
                      if state["dispatches"].get(did, {}).get("match_type") == "unresolved"]
        if unresolved:
            print(f"Stage C: retry pass over {len(unresolved)} unresolved dispatch(es)...")
            for did, _t, _c in unresolved:
                del state["dispatches"][did]
            process(unresolved, "retry")
    finally:
        save_json(config.CASEMAP_FILE, state)
        print("Stage C: checkpoint flushed.")
    return state
