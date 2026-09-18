"""Orchestration shared by the fetch-only, process-only, and combined CLIs.

The two halves meet at state/batch_current.json: fetch() is the only step that
opens a DB connection and it just writes the work order (dispatch headers,
notes, and recorded parts); process() consumes that file and never touches the
DB. Nothing else crosses the boundary, so the halves can run as separate
commands, on separate schedules, or from hosts with different network access —
as long as they share the same state/ directory.

process() works through the batch in slices (config.PROCESS_CHUNK), running
A -> B -> D per slice, so documents reach the search index throughout a long
run rather than only after every dispatch has been extracted.

Two process modes share the A/B stages:
- default (parts-finder indexing): A -> B -> Stage D (embed + push to Azure AI
  Search). Terminal ledger statuses: indexed, no_fault, no_useful_notes. The
  casemap fingerprint guard is NOT consulted — Stage C never runs.
- --with-cases (legacy case mapping): A -> B -> C -> D. Stage C behavior and
  its "mapped" ledger status are unchanged; documents are still pushed so a
  cases-mode run can never leave a dispatch terminal-but-unindexed.
"""
import os

from . import config, ledger, reports, statefiles
from . import stage_fetch, stage_notes, stage_extract, stage_casemap, stage_index
from .statefiles import load_json, save_json


def check_state(with_cases=False):
    """Fail fast on unusable state before any expensive work — a casemap
    fingerprint mismatch must not surface only after a batch of LLM calls,
    but it only matters when Stage C is actually going to run."""
    ledger.load()
    if with_cases:
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
    search_state = load_json(config.SEARCH_STATE_FILE)
    if search_state:
        print(f"Search index '{search_state.get('index')}': "
              f"{len(search_state.get('pushed', {}))} documents pushed")
    staged = staged_batch()
    if staged:
        print(f"In-flight batch: {staged['batch_id']} "
              f"({len(staged['dispatches'])} dispatches)")


def fetch(count, dry_run=False, month=None):
    """Stage the next `count` never-processed dispatches as the work order.

    With `month="YYYY-MM"`, selects from that month and spreads the picks
    evenly across it instead of taking the newest.

    Returns the staged batch, or None for a dry run. Nothing is marked
    processed here — a staged batch that is never processed only leaves its
    dispatches staged, still eligible.
    """
    statefiles.ensure_dirs()
    check_state()
    return stage_fetch.stage_batch(count, dry_run=dry_run, month=month)


def _slices(dispatches, size):
    """The batch split into A->B->D passes. An empty batch still yields one
    (empty) slice so the stages run once and hand back their global state."""
    if not size or size >= len(dispatches):
        return [dispatches] if dispatches else [[]]
    return [dispatches[i:i + size] for i in range(0, len(dispatches), size)]


def process(batch, with_cases=False, chunk_size=None):
    """Run stages A/B (+C with --with-cases) then Stage D, and finalize.

    The batch is processed in slices of `chunk_size` (config.PROCESS_CHUNK by
    default): each slice goes A -> B -> (C) -> D before the next one starts, so
    a long run keeps feeding the search index instead of pushing everything
    only after the last dispatch is extracted. Results are identical either
    way — every stage is keyed globally by dispatch id and skips what it has
    already done — so the slice size is purely about when documents land.
    Pass chunk_size=0 to run each stage over the whole batch, as before.
    """
    statefiles.ensure_dirs()
    check_state(with_cases)
    if not with_cases:
        # Stage D is the only reason this mode runs — refuse before LLM spend
        config.require_search_config()

    size = config.PROCESS_CHUNK if chunk_size is None else chunk_size
    chunks = _slices(batch["dispatches"], size)
    total = len(batch["dispatches"])
    if len(chunks) > 1:
        print(f"Processing {total} dispatches in {len(chunks)} slices of "
              f"up to {size} — the search index is updated after each.")

    notes_state = extract_state = index_state = casemap_state = None
    done = 0
    for i, chunk in enumerate(chunks, 1):
        sub = dict(batch, dispatches=chunk)
        if len(chunks) > 1:
            print(f"\n{'=' * 62}\n=== Slice {i}/{len(chunks)}: dispatches "
                  f"{done + 1}-{done + len(chunk)} of {total}\n{'=' * 62}")

        notes_state = stage_notes.run(sub)
        useful = stage_notes.useful_notes(sub, notes_state)
        print(f"Useful-note dispatches: {sum(1 for v in useful.values() if v)}; "
              f"zero-useful: {sum(1 for v in useful.values() if not v)}")

        extract_state = stage_extract.run(sub, useful)
        # Stage C stays strictly sequential: slices run in order, and within a
        # slice a case minted for dispatch n is visible to dispatch n+1.
        casemap_state = (stage_casemap.run(sub, extract_state, useful)
                         if with_cases else None)
        index_state = stage_index.run(sub, extract_state, useful)

        done += len(chunk)
        if len(chunks) > 1:
            # keep the popularity prior coherent with the index mid-run, so a
            # killed run leaves the two consistent rather than skewed
            stage_index.write_popularity(index_state)
            print(f"--- Slice {i}/{len(chunks)} complete: {done}/{total} dispatches "
                  f"through the pipeline; index holds "
                  f"{len(index_state['pushed'])} documents. ---")

    finalize(batch, notes_state, extract_state, index_state, casemap_state)


def finalize(batch, notes_state, extract_state, index_state, casemap_state=None):
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
        if casemap_state is not None:
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
        else:
            rec = extract_state.get(did)
            if rec is None:
                outcomes[did] = "incomplete_stage_b"
                continue
            if not rec["root_cause"].strip():
                # nothing to index; terminal so it is never refetched
                ledger.mark(led, did, "no_fault", "", batch["batch_id"])
                outcomes[did] = "no_fault"
                continue
            if did in index_state["pushed"]:
                ledger.mark(led, did, "indexed", "", batch["batch_id"])
                outcomes[did] = "indexed"
            else:
                # push failed — stays out of the ledger, eligible next run,
                # where the committed A/B results are reused for free
                outcomes[did] = "incomplete_stage_d"

    incomplete = [d for d, o in outcomes.items()
                  if o.startswith("incomplete") or o == "unresolved"]

    ledger.save(led)
    if casemap_state is not None:
        reports.write_all(casemap_state)
    stage_index.write_popularity(index_state)

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
          f"search index has {len(index_state['pushed'])} documents pushed.")
    if casemap_state is not None:
        print(f"Cases {len(casemap_state['cases'])}; "
              f"growth last point {casemap_state['growth'][-1]}.")
    print(f"Batch archive: {archive_path}")
