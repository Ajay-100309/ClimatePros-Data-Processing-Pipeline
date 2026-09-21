"""Stage 0: fetch candidates from the DB, exclude processed, stage the batch.

The staged batch (state/batch_current.json.gz) is the work order for the whole
run — the DB is never touched again for this batch, so resume is DB-free. It is
gzipped and git-tracked so the fetch machine (which needs VPN to the database)
can hand it to a processing machine that has none.

Two selection strategies. The default is newest-first, which is what built the
existing corpus. Passing `month="YYYY-MM"` selects that month instead and
spreads the picks evenly across it (see _select_month), so the corpus can cover
all twelve calendar months rather than only the most recent few.
"""
import os
from datetime import datetime, timezone

from . import config, db, ledger, monthplan
from .statefiles import load_json, save_json


def load_batch():
    """The staged work order, or None. Falls back to the pre-gzip filename so
    a batch staged by an older checkout is still picked up."""
    for path in (config.BATCH_FILE, config.BATCH_FILE_LEGACY):
        batch = load_json(path)
        if batch is not None:
            return batch
    return None


def remove_batch():
    for path in (config.BATCH_FILE, config.BATCH_FILE_LEGACY):
        if os.path.exists(path):
            os.remove(path)


def meta_record(d, light=False):
    """The dispatch_meta entry for a staged dispatch. Everything here comes
    from the work order itself, which is what lets the process half rebuild
    its own metadata on a machine that never ran the fetch.

    `light` drops combined_notes — the whole note text, and ~90% of the file's
    bulk. Only reports.py reads it (for the legacy --with-cases spreadsheet,
    via .get with a default); Stage D needs just the three display fields.
    """
    rec = {
        "dispatch_number": d["dispatch_number"],
        "reason": d["reason"],
        "received_dt": d["received_dt"],
        "note_count": len(d["notes"]),
    }
    if light:
        return rec
    combined = "\n---\n".join(x["text"] for x in d["notes"])
    if len(combined) > config.XLSX_CELL_LIMIT:
        combined = combined[:config.XLSX_CELL_LIMIT] + " [TRUNCATED]"
    rec["combined_notes"] = combined
    return rec


def merge_dispatch_meta(dispatches, light=True):
    """Add display metadata for any staged dispatch missing from
    state/dispatch_meta.json. Returns the number added.

    dispatch_meta.json is 100MB+ on the fetch machine and untracked, so it
    never crosses machines; the process half calls this so Stage D still has
    reason / dispatchNumber / receivedDt for the documents it pushes. Records
    are written light by default, which keeps a process-only machine's copy at
    a few MB instead of mirroring the fetch machine's — the notes it would
    otherwise duplicate are already in the work order.
    """
    meta = load_json(config.DISPATCH_META_FILE, {})
    added = 0
    for d in dispatches:
        if d["dispatch_id"] not in meta:
            meta[d["dispatch_id"]] = meta_record(d, light=light)
            added += 1
    if added:
        save_json(config.DISPATCH_META_FILE, meta)
    return added


def _clear_if_already_processed(existing):
    """True when the staged batch was already processed on another machine.

    When the work order travels by git the processing machine deletes it on
    finalize and that deletion comes back on the next pull, so this mostly
    matters for rsync handoffs, where a stale copy would block every future
    fetch. The batch archive is the proof it finished, so drop the stale work
    order instead of refusing.
    """
    archive = os.path.join(config.BATCH_ARCHIVE_DIR, existing["batch_id"] + ".json")
    if not os.path.exists(archive):
        return False
    print(f"Staged batch {existing['batch_id']} was already processed elsewhere "
          f"(archive present) — clearing the stale work order.")
    remove_batch()
    return True


def stage_batch(count, dry_run=False, month=None):
    # a dry run writes nothing, so an in-flight batch must not block a preview
    existing = None if dry_run else load_batch()
    if existing is not None and not _clear_if_already_processed(existing):
        print(f"Staged batch {existing['batch_id']} already exists "
              f"({len(existing['dispatches'])} dispatches) — using it; "
              "nothing new fetched.")
        if existing["count_requested"] != count:
            print(f"  (note: it was staged with --count {existing['count_requested']}, "
                  f"current --count {count} ignored)")
        return existing

    led = ledger.load()
    casemap = load_json(config.CASEMAP_FILE)
    exclude = ledger.processed_ids(led) | set(casemap["dispatches"].keys())

    conn = db.connect()
    try:
        selected = (_select_month(conn, count, exclude, month) if month
                    else _select(conn, count, exclude))
        if dry_run:
            print(f"\n--dry-run: would stage {len(selected)} dispatches "
                  f"(excluded pool: {len(exclude)}):")
            if month:
                _print_spread(selected, month)
            for h in selected[:20]:
                print(f"  {h['dispatch_id']}  {h['received_dt']}  {h['reason'][:60]}")
            if len(selected) > 20:
                print(f"  ... and {len(selected) - 20} more")
            return None

        ids = [h["dispatch_id"] for h in selected]
        notes = db.fetch_notes_for(conn, ids)
        parts = db.fetch_parts_for(conn, ids)
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
        "strategy": "month_spread" if month else "newest_first",
        "month": month,
        "dispatches": dispatches,
    }
    save_json(config.BATCH_FILE, batch)

    # display metadata for cumulative reports
    meta = load_json(config.DISPATCH_META_FILE, {})
    for d in dispatches:
        meta[d["dispatch_id"]] = meta_record(d)
    save_json(config.DISPATCH_META_FILE, meta)

    # recorded parts per staged dispatch — an explicit [] means "checked, none
    # recorded", so the process half never needs the DB to tell the difference
    parts_state = load_json(config.PARTS_FILE, {"schema": 1, "dispatches": {}})
    for d in dispatches:
        parts_state["dispatches"][d["dispatch_id"]] = {
            "fetched_at": batch["fetched_at"],
            "items": parts.get(d["dispatch_id"], []),
        }
    save_json(config.PARTS_FILE, parts_state)

    if month:
        plan = monthplan.load_plan()
        if plan:
            monthplan.save_plan(monthplan.record_fetched(plan, month, len(dispatches)))

    with_parts = sum(1 for d in dispatches if parts.get(d["dispatch_id"]))
    print(f"Staged batch {batch['batch_id']}"
          + (f" [{month}]" if month else "")
          + f": {len(dispatches)} dispatches ({with_parts} with recorded parts).")
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


# A month holds 14-23k eligible dispatches, so one query returns the whole
# month's headers and the selection happens here. That ordering matters:
# excluding first and spreading second keeps the spacing even no matter how
# much of the month has already been processed.
MONTH_FETCH_CAP = 100_000


def _select_month(conn, count, exclude, month):
    dt_min, dt_max = monthplan.month_bounds(month)
    cands = db.fetch_candidates(conn, MONTH_FETCH_CAP, dt_min=dt_min, dt_max=dt_max)
    fresh = [h for h in cands if h["dispatch_id"] not in exclude]
    fresh.sort(key=lambda h: (h["received_dt"] or "", h["dispatch_id"]))
    print(f"{month}: {len(cands)} eligible, {len(fresh)} unprocessed.")

    if len(fresh) <= count:
        if len(fresh) < count:
            print(f"  Month has only {len(fresh)} unprocessed — taking all of them.")
        return fresh

    # even stride over the date-sorted month: every week is represented in
    # proportion to its dispatch volume, rather than the month's tail only
    stride = len(fresh) / count
    return [fresh[min(int(i * stride), len(fresh) - 1)] for i in range(count)]


def _print_spread(selected, month):
    """Week-by-week histogram, so a dry run shows the spread actually worked."""
    weeks = {}
    for h in selected:
        day = int((h["received_dt"] or "0000-00-00")[8:10] or 0)
        weeks[min(4, max(1, (day - 1) // 7 + 1))] = \
            weeks.get(min(4, max(1, (day - 1) // 7 + 1)), 0) + 1
    total = sum(weeks.values()) or 1
    print(f"  spread across {month}: "
          + "  ".join(f"wk{w}: {weeks.get(w, 0)} ({weeks.get(w, 0)/total:.0%})"
                      for w in sorted(weeks)))
