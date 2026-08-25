"""Orchestration shared by the fetch-only, process-only, and combined CLIs.

The two halves meet at state/batch_current.json: fetch() is the only step that
opens a DB connection and it just writes the work order; process() consumes
that file and never touches the DB. Nothing else crosses the boundary, so the
halves can run as separate commands, on separate schedules, or from hosts with
different network access — as long as they share the same state/ directory.
"""
import os

from . import config, ledger, reports, statefiles
from . import (stage_fetch, stage_notes, stage_extract, stage_diagnostic,
              stage_casemap, stage_consolidate)
from .statefiles import load_json, save_json


def check_state():
    """Fail fast on unusable ledger/casemap state before any expensive work —
    a fingerprint mismatch must not surface only after a batch of LLM calls."""
    ledger.load()
    stage_casemap.load_state()


def staged_batch():
    """The staged work order, or None if no batch is in flight."""
    return load_json(config.BATCH_FILE)


def stats():
    statefiles.ensure_dirs()
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
    staged = staged_batch()
    if staged:
        print(f"In-flight batch: {staged['batch_id']} "
              f"({len(staged['dispatches'])} dispatches)")


def fetch(count, dry_run=False):
    """Stage the next `count` never-processed dispatches as the work order.

    Returns the staged batch, or None for a dry run. Nothing is marked
    processed here — a staged batch that is never processed only leaves its
    dispatches staged, still eligible.
    """
    statefiles.ensure_dirs()
    check_state()
    return stage_fetch.stage_batch(count, dry_run=dry_run)


def consolidate(dry_run=False):
    """Offline maintenance pass over the finished case catalog — see
    stage_consolidate for why this exists. No DB access, no staged batch."""
    statefiles.ensure_dirs()
    check_state()
    return stage_consolidate.run(dry_run=dry_run)


def process(batch):
    """Run stages A/B/C over a staged batch and finalize it. No DB access."""
    statefiles.ensure_dirs()
    check_state()

    notes_state = stage_notes.run(batch)
    useful = stage_notes.useful_notes(batch, notes_state)
    print(f"Useful-note dispatches: {sum(1 for v in useful.values() if v)}; "
          f"zero-useful: {sum(1 for v in useful.values() if not v)}")

    extract_state = stage_extract.run(batch, useful)
    diag_state = stage_diagnostic.run(batch, extract_state)
    diagnostic_extract = {did: rec for did, rec in extract_state.items()
                          if diag_state.get(did, {}).get("diagnostic", True)}
    casemap_state = stage_casemap.run(batch, diagnostic_extract, useful)

    finalize(batch, notes_state, extract_state, diag_state, casemap_state)


def finalize(batch, notes_state, extract_state, diag_state, casemap_state):
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
        if did not in extract_state:
            outcomes[did] = "incomplete_stage_b"
            continue
        if did not in diag_state:
            outcomes[did] = "incomplete_stage_b2"
            continue
        if not diag_state[did]["diagnostic"]:
            ledger.mark(led, did, "non_diagnostic", "", batch["batch_id"])
            outcomes[did] = "non_diagnostic"
            continue
        rec = casemap_state["dispatches"].get(did)
        if rec is None:
            outcomes[did] = "incomplete_stage_c"
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
