"""Stage 0 on its own: collect dispatches from FieldJetXStg and stage them.

Selects dispatches not already in the ledger or the case map, pulls all of
their notes, and writes the work order to state/batch_current.json — then
stops. This is the only command that opens a database connection: no LLM calls
are made, no dispatch is marked processed, and nothing in the case catalog
changes. Run process.py afterwards to consume the staged batch.

If a batch is already staged, it is left untouched and reported — finish it
with process.py before staging another. (A batch already processed on another
machine is detected by its archive and cleared automatically.)

Two selection strategies:

  newest-first (default)  the most recent unprocessed dispatches. This is what
                          built the existing corpus, and it concentrates it in
                          the last few months.
  month-stratified        --month YYYY-MM takes one month and spreads the picks
                          evenly across it, so the corpus can cover all twelve
                          calendar months. --plan-months builds a multi-month
                          budget once; --next-month then walks it.

Usage:
    venv/bin/python fetch.py --count 200
    venv/bin/python fetch.py --count 5 --dry-run

    venv/bin/python fetch.py --plan-months --per-month 5000 --months 24
    venv/bin/python fetch.py --show-plan
    venv/bin/python fetch.py --next-month
    venv/bin/python fetch.py --month 2025-02 --count 5000
"""
import re
import sys
import argparse

from pipelib import config, db, monthplan, runner
from pipelib.statefiles import load_json, ensure_dirs
from pipelib import ledger

MONTH_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


def build_plan(per_month, months, end_month):
    """Query eligible counts for the window and write state/fetch_plan.json."""
    ensure_dirs()
    if not MONTH_RE.match(end_month):
        sys.exit(f"--end-month must be YYYY-MM, got {end_month!r}")
    wanted = monthplan.month_list(end_month, months)
    dt_min, _ = monthplan.month_bounds(wanted[0])
    _, dt_max = monthplan.month_bounds(wanted[-1])
    print(f"Counting eligible dispatches {wanted[0]} .. {wanted[-1]} "
          f"({months} months; this takes ~{max(5, months * 5)}s)...")

    conn = db.connect()
    try:
        eligible = db.count_candidates_by_month(conn, dt_min, dt_max)
    finally:
        conn.close()

    led = ledger.load()
    casemap = load_json(config.CASEMAP_FILE) or {"dispatches": {}}
    exclude = ledger.processed_ids(led) | set(casemap["dispatches"].keys())
    meta = load_json(config.DISPATCH_META_FILE, {})
    processed = monthplan.processed_by_month(meta, exclude)

    plan = monthplan.build_plan(eligible, processed, per_month, wanted)
    monthplan.save_plan(monthplan.merge_progress(plan, monthplan.load_plan()))
    print(monthplan.format_plan(plan))
    print(f"\nPlan written: {config.FETCH_PLAN_FILE}")
    print("Next: venv/bin/python fetch.py --next-month")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, help="dispatches to stage")
    ap.add_argument("--dry-run", action="store_true",
                    help="fetch + exclusion report only; nothing written")
    ap.add_argument("--month", metavar="YYYY-MM",
                    help="stage from this month, spread evenly across it")
    ap.add_argument("--next-month", action="store_true",
                    help="stage the oldest month still short of its plan quota")
    ap.add_argument("--plan-months", action="store_true",
                    help="(re)build state/fetch_plan.json for a month window")
    ap.add_argument("--show-plan", action="store_true",
                    help="print the current fetch plan and exit")
    ap.add_argument("--per-month", type=int, default=5000,
                    help="planning: dispatches to target per month (default 5000)")
    ap.add_argument("--months", type=int, default=24,
                    help="planning: months in the window (default 24 — every "
                         "calendar month covered twice)")
    ap.add_argument("--end-month", metavar="YYYY-MM", default="2026-06",
                    help="planning: newest month in the window (default 2026-06)")
    args = ap.parse_args()

    if args.show_plan:
        plan = monthplan.load_plan()
        if not plan:
            sys.exit("No fetch plan yet — run: venv/bin/python fetch.py --plan-months")
        print(monthplan.format_plan(plan))
        return

    if args.plan_months:
        build_plan(args.per_month, args.months, args.end_month)
        return

    month, count = args.month, args.count

    if args.next_month:
        if month:
            sys.exit("--next-month and --month are mutually exclusive.")
        plan = monthplan.load_plan()
        if not plan:
            sys.exit("No fetch plan yet — run: venv/bin/python fetch.py --plan-months")
        month = monthplan.next_month(plan)
        if month is None:
            print("Fetch plan complete — every month has met its quota.")
            print(monthplan.format_plan(plan))
            return
        rec = plan["months"][month]
        count = count or (rec["quota"] - rec["fetched"])
        print(f"Next month from plan: {month} "
              f"(quota {rec['quota']:,}, fetched {rec['fetched']:,}, "
              f"staging {count:,})")

    if month and not MONTH_RE.match(month):
        sys.exit(f"--month must be YYYY-MM, got {month!r}")
    if not count:
        sys.exit("--count N is required (or --next-month, which reads the plan).")

    batch = runner.fetch(count, dry_run=args.dry_run, month=month)
    if batch is None:
        return  # dry run; stage_fetch already printed the report

    print(f"Work order: {config.BATCH_FILE}")
    print("Next: venv/bin/python process.py")


if __name__ == "__main__":
    main()
