"""New-dispatch automation pipeline: fetch and process in one run.

Fetch N never-processed dispatches from FieldJetXStg, classify note usefulness,
extract verbatim root causes, map them into the persistent case vector DB, and
extend the dispatches-vs-cases growth series. Fully resumable; no dispatch is
ever processed twice.

The same two halves are also available as separate commands — fetch.py stages
the batch, process.py consumes it — split at state/batch_current.json. All
three entry points share pipelib/runner.py, so the behaviour is identical
either way.

Usage:
    venv/bin/python pipeline.py --count 200       # fetch + process a batch
    venv/bin/python pipeline.py --count 5 --dry-run
    venv/bin/python pipeline.py --skip-fetch      # process the staged batch
    venv/bin/python pipeline.py --stats
"""
import sys
import argparse

from pipelib import runner


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

    if args.stats:
        runner.stats()
        return

    if args.skip_fetch:
        batch = runner.staged_batch()
        if batch is None:
            sys.exit("--skip-fetch: no staged batch found.")
        print(f"Resuming staged batch {batch['batch_id']} "
              f"({len(batch['dispatches'])} dispatches).")
    else:
        if not args.count:
            sys.exit("--count N is required (or --skip-fetch / --stats).")
        batch = runner.fetch(args.count, dry_run=args.dry_run)
        if args.dry_run:
            return

    runner.process(batch)


if __name__ == "__main__":
    main()
