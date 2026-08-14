"""Stage 0: fetch candidates from the DB, exclude processed, stage the batch.

The staged batch (state/batch_current.json) is the work order for the whole
run — the DB is never touched again for this batch, so resume is DB-free.
"""
from datetime import datetime, timezone

from . import config, db, ledger
from .statefiles import load_json, save_json


def stage_batch(count, dry_run=False):
    existing = load_json(config.BATCH_FILE)
    if existing is not None:
        print(f"Staged batch {existing['batch_id']} already exists "
              f"({len(existing['dispatches'])} dispatches) — resuming it.")
        if existing["count_requested"] != count:
            print(f"  (note: it was staged with --count {existing['count_requested']}, "
                  f"current --count {count} ignored)")
        return existing

    led = ledger.load()
    casemap = load_json(config.CASEMAP_FILE)
    exclude = ledger.processed_ids(led) | set(casemap["dispatches"].keys())

    conn = db.connect()
    try:
        selected = _select(conn, count, exclude)
        if dry_run:
            print(f"\n--dry-run: would stage {len(selected)} dispatches "
                  f"(excluded pool: {len(exclude)}):")
            for h in selected[:20]:
                print(f"  {h['dispatch_id']}  {h['received_dt']}  {h['reason'][:60]}")
            if len(selected) > 20:
                print(f"  ... and {len(selected) - 20} more")
            return None

        ids = [h["dispatch_id"] for h in selected]
        notes = db.fetch_notes_for(conn, ids)
    finally:
        conn.close()

    dispatches = []
    for h in selected:
        n = notes.get(h["dispatch_id"], [])
        if not n:
            print(f"  Skipping {h['dispatch_id']}: no non-empty notes returned.")
            continue
        dispatches.append({**h, "notes": n})

    batch = {
        "batch_id": "batch_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "count_requested": count,
        "dispatches": dispatches,
    }
    save_json(config.BATCH_FILE, batch)

    # display metadata for cumulative reports
    meta = load_json(config.DISPATCH_META_FILE, {})
    for d in dispatches:
        combined = "\n---\n".join(x["text"] for x in d["notes"])
        if len(combined) > config.XLSX_CELL_LIMIT:
            combined = combined[:config.XLSX_CELL_LIMIT] + " [TRUNCATED]"
        meta[d["dispatch_id"]] = {
            "dispatch_number": d["dispatch_number"],
            "reason": d["reason"],
            "received_dt": d["received_dt"],
            "note_count": len(d["notes"]),
            "combined_notes": combined,
        }
    save_json(config.DISPATCH_META_FILE, meta)

    print(f"Staged batch {batch['batch_id']}: {len(dispatches)} dispatches.")
    return batch


def _select(conn, count, exclude):
    for factor in (3, 10):
        cands = db.fetch_candidates(conn, count * factor)
        fresh = [h for h in cands if h["dispatch_id"] not in exclude]
        if len(fresh) >= count:
            return fresh[:count]
        print(f"Only {len(fresh)} unprocessed among top {count * factor} candidates"
              + ("; escalating fetch..." if factor == 3 else "."))
    assert all(h["dispatch_id"] not in exclude for h in fresh)
    return fresh
