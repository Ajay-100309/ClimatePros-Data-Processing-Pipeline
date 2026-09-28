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
import sys
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


def archive_exists(batch_id):
    """True if this batch was already finalized (any archive extension —
    archives moved from .json to .json.gz partway through the project)."""
    base = os.path.join(config.BATCH_ARCHIVE_DIR, batch_id)
    return os.path.exists(base + ".json") or os.path.exists(base + ".json.gz")


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
    if not archive_exists(existing["batch_id"]):
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

        return _finish_and_save(conn, selected, count_requested=count,
                                months=[month] if month else [],
                                strategy="month_spread" if month else "newest_first")
    finally:
        conn.close()


def stage_multi_month_batch(month_counts, dry_run=False):
    """Stage several months into ONE combined work order.

    month_counts: [(month, count)], oldest first — typically straight from
    monthplan.next_months(). Each month is selected independently (same
    exclude-then-spread rule as a single month; date ranges never overlap
    between months so there is no double-selection risk), then notes/parts
    are fetched once over the combined id list — this is what lets one
    process.py run churn through several months without a git round trip
    after each one.
    """
    existing = None if dry_run else load_batch()
    if existing is not None and not _clear_if_already_processed(existing):
        print(f"Staged batch {existing['batch_id']} already exists "
              f"({len(existing['dispatches'])} dispatches) — using it; "
              "nothing new fetched.")
        return existing

    led = ledger.load()
    casemap = load_json(config.CASEMAP_FILE)
    exclude = ledger.processed_ids(led) | set(casemap["dispatches"].keys())

    conn = db.connect()
    try:
        selected, by_month = [], {}
        for month, count in month_counts:
            picked = _select_month(conn, count, exclude, month)
            by_month[month] = picked
            selected.extend(picked)

        if dry_run:
            print(f"\n--dry-run: would stage {len(selected)} dispatches across "
                  f"{len(month_counts)} months (excluded pool: {len(exclude)}):")
            _gzip_heads_up(len(selected))
            for month, _ in month_counts:
                _print_spread(by_month[month], month)
            for h in selected[:20]:
                print(f"  {h['dispatch_id']}  {h['received_dt']}  {h['reason'][:60]}")
            if len(selected) > 20:
                print(f"  ... and {len(selected) - 20} more")
            return None

        return _finish_and_save(conn, selected,
                                count_requested=sum(c for _, c in month_counts),
                                months=[m for m, _ in month_counts],
                                strategy="month_spread_multi")
    finally:
        conn.close()


def _batch_months(batch):
    """Schema-compatible months list — the pre-refactor schema stored a single
    'month' string; the current one always stores a 'months' list."""
    if batch.get("months"):
        return list(batch["months"])
    if batch.get("month"):
        return [batch["month"]]
    return []


def append_months_to_batch(month_counts, dry_run=False):
    """Add more months onto the currently staged (NOT YET PROCESSED) batch,
    instead of starting a fresh one. If nothing is staged (or the staged
    batch turns out to be stale/already-processed elsewhere), this falls back
    to stage_multi_month_batch. Safe because months are disjoint date ranges
    — a dispatch selected for a new month can never collide with one already
    in the batch.

    Only append BEFORE the batch has been handed to the processing machine
    (i.e. before commit+push, or before process.py has started on it there).
    process.py loads the work order once at the start of a run and deletes it
    on finish based on that in-memory copy — an append made and pushed after
    processing has already begun there would be silently lost when that run
    finalizes, not corrupted, just wasted (the appended dispatches were never
    marked processed anywhere, so they're simply re-selected next time).
    """
    existing = load_batch()
    if existing is not None and _clear_if_already_processed(existing):
        existing = None
    if existing is None:
        print("No batch currently staged — staging fresh instead of appending.")
        return stage_multi_month_batch(month_counts, dry_run=dry_run)

    already = _batch_months(existing)
    overlap = sorted(set(already) & {m for m, _ in month_counts})
    if overlap:
        sys.exit(f"{overlap} already in the staged batch {existing['batch_id']} — "
                 f"nothing to append (check state/fetch_plan.json / --show-plan).")

    led = ledger.load()
    casemap = load_json(config.CASEMAP_FILE)
    exclude = ledger.processed_ids(led) | set(casemap["dispatches"].keys())

    conn = db.connect()
    try:
        selected, by_month = [], {}
        for month, count in month_counts:
            picked = _select_month(conn, count, exclude, month)
            by_month[month] = picked
            selected.extend(picked)

        if dry_run:
            print(f"\n--dry-run: would ADD {len(selected)} dispatches across "
                  f"{len(month_counts)} months to batch {existing['batch_id']} "
                  f"(currently {len(existing['dispatches'])} staged, months {already}):")
            _gzip_heads_up(len(existing['dispatches']) + len(selected))
            for month, _ in month_counts:
                _print_spread(by_month[month], month)
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

    new_dispatches = []
    for h in selected:
        n = notes.get(h["dispatch_id"], [])
        if not n:
            print(f"  Skipping {h['dispatch_id']}: no non-empty notes returned.")
            continue
        new_dispatches.append({**h, "notes": n})

    existing["dispatches"].extend(new_dispatches)
    existing["count_requested"] = (existing.get("count_requested", 0)
                                   + sum(c for _, c in month_counts))
    existing["months"] = sorted(set(already) | {m for m, _ in month_counts})
    existing.pop("month", None)  # superseded by the months list
    existing["strategy"] = "month_spread_multi"
    save_json(config.BATCH_FILE, existing)

    meta = load_json(config.DISPATCH_META_FILE, {})
    for d in new_dispatches:
        meta[d["dispatch_id"]] = meta_record(d)
    save_json(config.DISPATCH_META_FILE, meta)

    parts_state = load_json(config.PARTS_FILE, {"schema": 1, "dispatches": {}})
    fetched_at = datetime.now(timezone.utc).isoformat()
    for d in new_dispatches:
        parts_state["dispatches"][d["dispatch_id"]] = {
            "fetched_at": fetched_at,
            "items": parts.get(d["dispatch_id"], []),
        }
    save_json(config.PARTS_FILE, parts_state)

    by_month_new = {}
    for d in new_dispatches:
        ym = (d.get("received_dt") or "")[:7]
        by_month_new[ym] = by_month_new.get(ym, 0) + 1
    plan = monthplan.load_plan()
    if plan:
        for m, _ in month_counts:
            monthplan.record_fetched(plan, m, by_month_new.get(m, 0))
        monthplan.save_plan(plan)

    with_parts_new = sum(1 for d in new_dispatches if parts.get(d["dispatch_id"]))
    print(f"Added {len(new_dispatches)} dispatches ({with_parts_new} with parts) to "
          f"batch {existing['batch_id']} — now {len(existing['dispatches'])} total "
          f"across months {existing['months']}.")
    _gzip_heads_up(len(existing["dispatches"]))
    return existing


# rough measured ratio for the git-blob-size heads up below: a 10k-dispatch
# work order gzips to ~14.3MB (build_document-shaped JSON, mostly note text)
GZIP_MB_PER_1K = 1.43


def _gzip_heads_up(n_dispatches):
    est = n_dispatches * GZIP_MB_PER_1K / 1000
    if est > 30:
        print(f"  Work order is ~{est:.0f}MB gzipped (est.) — GitHub warns above "
              f"50MB and hard-blocks above 100MB per file.")


def _finish_and_save(conn, selected, count_requested, months, strategy):
    ids = [h["dispatch_id"] for h in selected]
    notes = db.fetch_notes_for(conn, ids)
    parts = db.fetch_parts_for(conn, ids)

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
        "count_requested": count_requested,
        "strategy": strategy,
        "months": months,
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

    if months:
        # group the FINAL (post notes-skip) dispatches back to their source
        # month via received_dt — months are disjoint date ranges so this is
        # exact, and it works uniformly for one month or several
        by_month = {}
        for d in dispatches:
            ym = (d.get("received_dt") or "")[:7]
            by_month[ym] = by_month.get(ym, 0) + 1
        plan = monthplan.load_plan()
        if plan:
            for m in months:
                monthplan.record_fetched(plan, m, by_month.get(m, 0))
            monthplan.save_plan(plan)

    with_parts = sum(1 for d in dispatches if parts.get(d["dispatch_id"]))
    print(f"Staged batch {batch['batch_id']}"
          + (f" [{'+'.join(months)}]" if months else "")
          + f": {len(dispatches)} dispatches ({with_parts} with recorded parts).")
    _gzip_heads_up(len(dispatches))
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
