"""Backfill (and reconcile) the Azure AI Search index from processed history.

Source of truth is state/casemap.json's dispatches map — its text field carries
the extracted root cause for every dispatch ever mapped, including the legacy
seed that predates state/extract.json. Dispatches whose text is the no-fault
placeholder are skipped (a constant string would form a degenerate vector
cluster and carries no parts signal). Categories come from extract.json when
present, else from the dispatch's case record.

This is a fetch-half command: it opens one DB connection to pull recorded parts
for dispatches missing from state/parts.json (or all of them with
--refresh-parts). Embeddings come from the shared cache — for the historical
corpus that is a 100% hit, so no gateway calls are made. Pushes are recorded in
state/search_index.json after every batch, so reruns skip what's already
pushed; rerunning after failures, after cases-mode runs, or against a freshly
recreated empty index is the supported reconciliation path. (Note: after
--recreate, also delete state/search_index.json so the push record matches the
empty index.)

Usage:
    venv/bin/python backfill_search.py --dry-run       # counts only, no writes
    venv/bin/python backfill_search.py --limit 50      # small slice first
    venv/bin/python backfill_search.py                 # full backfill / reconcile
    venv/bin/python backfill_search.py --refresh-parts # force re-pull of parts
"""
import sys
import argparse
from datetime import datetime, timezone

from pipelib import config, db, stage_index
from pipelib.embcache import EmbCache
from pipelib.statefiles import ensure_dirs, load_json, save_json


def collect_items():
    casemap = load_json(config.CASEMAP_FILE)
    if casemap is None:
        sys.exit("state/casemap.json missing — nothing to backfill from.")
    if casemap.get("version") != 2:
        sys.exit(f"casemap version {casemap.get('version')} != 2 — refusing.")
    extract = load_json(config.EXTRACT_FILE, {})
    items, skipped = [], 0
    for did in sorted(casemap["dispatches"]):
        rec = casemap["dispatches"][did]
        text = rec["text"]
        if text == config.NO_FAULT_PLACEHOLDER:
            skipped += 1
            continue
        category = extract.get(did, {}).get("category") \
            or casemap["cases"].get(rec["case_id"], {}).get("category", "")
        items.append({"did": did, "text": text, "category": category})
    return items, skipped


def ensure_parts(items, refresh):
    parts_state = load_json(config.PARTS_FILE, {"schema": 1, "dispatches": {}})
    if refresh:
        need = [it["did"] for it in items]
    else:
        need = [it["did"] for it in items
                if it["did"] not in parts_state["dispatches"]]
    if not need:
        print("Parts: all dispatches already in state/parts.json.")
        return
    print(f"Parts: pulling {len(need)} dispatches from FieldJetXStg "
          f"({(len(need) + 499) // 500} chunked queries)...")
    conn = db.connect()
    try:
        fetched = db.fetch_parts_for(conn, need)
    finally:
        conn.close()
    now = datetime.now(timezone.utc).isoformat()
    for did in need:
        parts_state["dispatches"][did] = {
            "fetched_at": now, "items": fetched.get(did, [])}
    save_json(config.PARTS_FILE, parts_state)
    with_parts = sum(1 for did in need if fetched.get(did))
    print(f"Parts: {with_parts}/{len(need)} pulled dispatches have "
          "recorded parts.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="report source counts and exit; writes nothing")
    ap.add_argument("--limit", type=int,
                    help="only process the first N dispatches (by id)")
    ap.add_argument("--refresh-parts", action="store_true",
                    help="re-pull parts from the DB even if already cached")
    args = ap.parse_args()

    ensure_dirs()
    items, skipped = collect_items()
    state = stage_index.load_state()
    already = sum(1 for it in items
                  if state["pushed"].get(it["did"], {}).get("text_sha")
                  == config.text_sha(it["text"]))
    print(f"Backfill source: {len(items)} indexable dispatches "
          f"({skipped} no-fault placeholders skipped, "
          f"{already} already pushed).")
    if args.limit:
        items = items[:args.limit]
        print(f"--limit {args.limit}: processing {len(items)} dispatches.")
    if args.dry_run:
        print("--dry-run: nothing written.")
        return

    config.require_search_config()
    ensure_parts(items, args.refresh_parts)

    cache = EmbCache()
    state = stage_index.index_items(items, cache, state)
    stage_index.write_popularity(state)
    print(f"\nDone. {len(state['pushed'])} dispatches recorded as pushed to "
          f"index '{state['index']}'.")


if __name__ == "__main__":
    main()
