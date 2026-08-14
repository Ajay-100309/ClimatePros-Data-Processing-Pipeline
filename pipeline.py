"""New-dispatch automation pipeline.

Fetch N never-processed dispatches from FieldJetXStg, classify note usefulness,
extract verbatim root causes, map them into the persistent case vector DB, and
extend the dispatches-vs-cases growth series. Fully resumable; no dispatch is
ever processed twice.

Usage:
    venv/bin/python pipeline.py --count 200       # run a batch
    venv/bin/python pipeline.py --count 5 --dry-run
    venv/bin/python pipeline.py --skip-fetch      # resume the staged batch
    venv/bin/python pipeline.py --stats
"""
import os
import sys
import argparse

from pipelib import config, ledger, reports, statefiles
from pipelib import stage_fetch, stage_notes, stage_extract, stage_casemap
from pipelib.statefiles import load_json, save_json


def cmd_stats():
    led = ledger.load()
    casemap = load_json(config.CASEMAP_FILE) or {}
    statuses = {}
    for rec in led["dispatches"].values():
        statuses[rec["status"]] = statuses.get(rec["status"], 0) + 1
    growth = casemap.get("growth", [])
    print(f"Ledger: {len(led['dispatches'])} processed dispatches {statuses}")
    print(f"Cases: {len(casemap.get('cases', {}))} "
          f"(next: CASE-{casemap.get('next_case_num', 0):04d})")
    print(f"Mapped dispatches: {len(casemap.get('dispatches', {}))}")
    if growth:
        print(f"Growth series: {len(growth)} points, last {growth[-1]}")
    staged = load_json(config.BATCH_FILE)
    if staged:
        print(f"In-flight batch: {staged['batch_id']} "
              f"({len(staged['dispatches'])} dispatches)")


def finalize(batch, notes_state, extract_state, casemap_state):
    led = ledger.load()
    useful = stage_notes.useful_notes(batch, notes_state)
    outcomes = {}
    for d in batch["dispatches"]:
        did = d["dispatch_id"]
        if did not in notes_state:
            outcomes[did] = "incomplete_stage_a"
            continue
        if not useful.get(did):
            ledger.mark(led, did, "no_useful_notes", "", batch["batch_id"])
            outcomes[did] = "no_useful_notes"
            continue
        rec = casemap_state["dispatches"].get(did)
        if rec is None:
            outcomes[did] = "incomplete_stage_b" if did not in extract_state \
                else "incomplete_stage_c"
            continue
        if not rec["case_id"]:
            outcomes[did] = "unresolved"
            continue
        ledger.mark(led, did, "mapped", rec["case_id"], batch["batch_id"])
        outcomes[did] = f"mapped:{rec['case_id']}"

    incomplete = [d for d, o in outcomes.items()
                  if o.startswith("incomplete") or o == "unresolved"]

    ledger.save(led)
    reports.write_all(casemap_state)

    archive = dict(batch)
    archive["outcomes"] = outcomes
    archive_path = os.path.join(config.BATCH_ARCHIVE_DIR, batch["batch_id"] + ".json")
    save_json(archive_path, archive)
    os.remove(config.BATCH_FILE)

    terminal = len(outcomes) - len(incomplete)
    print(f"\nBatch {batch['batch_id']} finalized: "
          f"{terminal} dispatches terminal, {len(incomplete)} incomplete "
          f"(eligible for a future batch).")
    if incomplete:
        print(f"  Incomplete: {incomplete}")
    print(f"Ledger now {len(led['dispatches'])} dispatches; "
          f"cases {len(casemap_state['cases'])}; "
          f"growth last point {casemap_state['growth'][-1]}.")
    print(f"Batch archive: {archive_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, help="dispatches to process this batch")
    ap.add_argument("--dry-run", action="store_true",
                    help="fetch + exclusion report only; nothing written")
    ap.add_argument("--skip-fetch", action="store_true",
                    help="resume the staged batch (error if none)")
    ap.add_argument("--stats", action="store_true",
                    help="print ledger/case/growth summary and exit")
    args = ap.parse_args()

    statefiles.ensure_dirs()

    if args.stats:
        cmd_stats()
        return

    if args.skip_fetch:
        batch = load_json(config.BATCH_FILE)
        if batch is None:
            sys.exit("--skip-fetch: no staged batch found.")
        print(f"Resuming staged batch {batch['batch_id']} "
              f"({len(batch['dispatches'])} dispatches).")
    else:
        if not args.count:
            sys.exit("--count N is required (or --skip-fetch / --stats).")
        # fail fast on state problems before touching the DB
        ledger.load()
        stage_casemap.load_state()
        batch = stage_fetch.stage_batch(args.count, dry_run=args.dry_run)
        if args.dry_run:
            return

    notes_state = stage_notes.run(batch)
    useful = stage_notes.useful_notes(batch, notes_state)
    n_zero = sum(1 for v in useful.values() if not v)
    print(f"Useful-note dispatches: {sum(1 for v in useful.values() if v)}; "
          f"zero-useful: {n_zero}")

    extract_state = stage_extract.run(batch, useful)
    casemap_state = stage_casemap.run(batch, extract_state, useful)

    finalize(batch, notes_state, extract_state, casemap_state)


if __name__ == "__main__":
    main()
