"""Stage D: embed extracted root causes and upsert them into Azure AI Search.

Runs in both pipeline modes — the index must gain a document for every dispatch
with a real extracted root cause regardless of whether Stage C also ran. Never
opens a database connection: parts come from state/parts.json (written by the
fetch half), display metadata from state/dispatch_meta.json, vectors from the
shared embedding cache. state/search_index.json records what was pushed (keyed
by dispatch id with the text sha), saved incrementally after every upload batch,
so a killed run resumes exactly where it stopped and re-pushes are skipped
unless the root-cause text changed. Empty root causes are never indexed — the
runner records those as terminal "no_fault".
"""
from datetime import datetime, timezone

from . import config, search_index
from .embcache import EmbCache
from .statefiles import load_json, save_json


def load_state():
    state = load_json(config.SEARCH_STATE_FILE)
    if state is None:
        state = {"schema": 1, "index": config.AZURE_SEARCH_INDEX, "pushed": {}}
    return state


def save_state(state):
    save_json(config.SEARCH_STATE_FILE, state)


def index_items(items, cache, state, client=None, force=False):
    """items: [{did, text, category}], text non-empty. Embeds anything missing
    from the cache, builds documents, upserts in batches, records progress.
    force=True re-pushes even when the root-cause text is unchanged — needed
    after an additive schema update, where only non-text fields (e.g. parts
    enrichment) changed and the text_sha skip would otherwise skip every doc."""
    meta = load_json(config.DISPATCH_META_FILE, {})
    parts_state = load_json(config.PARTS_FILE, {"schema": 1, "dispatches": {}})

    if force:
        todo = list(items)
    else:
        todo = [it for it in items
                if state["pushed"].get(it["did"], {}).get("text_sha")
                != config.text_sha(it["text"])]
    print(f"\n=== Stage D: search index — {len(items)} candidates, "
          f"{len(items) - len(todo)} already pushed, {len(todo)} to push ===")
    if not todo:
        return state

    cache.ensure([it["text"] for it in todo])
    if client is None:
        client = search_index.search_client()

    no_parts_entry = 0
    for i in range(0, len(todo), config.SEARCH_UPLOAD_BATCH):
        chunk = todo[i:i + config.SEARCH_UPLOAD_BATCH]
        docs = []
        for it in chunk:
            entry = parts_state["dispatches"].get(it["did"])
            if entry is None:
                no_parts_entry += 1
                entry = {"items": []}
            docs.append(search_index.build_document(
                it["did"], it["text"], it["category"],
                meta.get(it["did"]), entry["items"], cache.get(it["text"])))
        succeeded, failed = search_index.upload_documents(client, docs)
        now = datetime.now(timezone.utc).isoformat()
        for it, doc in zip(chunk, docs):
            if it["did"] in succeeded:
                state["pushed"][it["did"]] = {
                    "text_sha": config.text_sha(it["text"]),
                    "at": now,
                    "has_parts": doc["hasParts"],
                }
        save_state(state)
        done = min(i + config.SEARCH_UPLOAD_BATCH, len(todo))
        print(f"  Pushed {done}/{len(todo)} (failures this run: {len(failed)})")
    if no_parts_entry:
        print(f"  Warning: {no_parts_entry} dispatches had no parts.json entry "
              "(pre-upgrade batch?) — indexed with empty parts; "
              "backfill_search.py --refresh-parts reconciles them.")
    return state


def run(batch, extract_state, useful):
    """Pipeline wrapper: index every batch dispatch with useful notes and a
    non-empty extracted root cause."""
    config.require_search_config()
    items = []
    for d in batch["dispatches"]:
        did = d["dispatch_id"]
        if not useful.get(did):
            continue
        rec = extract_state.get(did)
        if rec is None or not rec["root_cause"].strip():
            continue
        items.append({"did": did, "text": rec["root_cause"],
                      "category": rec.get("category", "")})
    state = load_state()
    return index_items(items, EmbCache(), state)


def write_popularity(state):
    """output/part_popularity.json: distinct-dispatch count per real partNo
    across everything pushed — the same popularity definition the blended
    scoring experiments used, for the parity harness and offline analysis.
    (The query service computes its own equivalent via a facet query.)"""
    parts_state = load_json(config.PARTS_FILE, {"schema": 1, "dispatches": {}})
    freq = {}
    for did in state["pushed"]:
        entry = parts_state["dispatches"].get(did)
        if not entry:
            continue
        seen = set()
        for p in search_index.real_parts(entry["items"]):
            key = p["part_no"].strip().upper()
            if key not in seen:
                seen.add(key)
                freq[key] = freq.get(key, 0) + 1
    save_json(config.OUT_PART_POPULARITY, {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "indexed_dispatches": len(state["pushed"]),
        "max_global": max(freq.values(), default=0),
        "freq": freq,
    })
    print(f"Popularity prior: {len(freq)} distinct parts across "
          f"{len(state['pushed'])} indexed dispatches -> {config.OUT_PART_POPULARITY}")
