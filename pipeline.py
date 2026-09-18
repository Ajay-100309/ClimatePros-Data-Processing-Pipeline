"""New-dispatch automation pipeline: fetch and process in one run.

Fetch N never-processed dispatches from FieldJetXStg, classify note usefulness,
extract verbatim root causes, then embed each root cause and push it (with the
dispatch's recorded parts) into the Azure AI Search index that serves the Parts
Finder. Fully resumable; no dispatch is ever processed twice.

Case mapping (the unique-case catalog and its reports) no longer runs by
default — pass --with-cases to also run Stage C exactly as before.

The same two halves are also available as separate commands — fetch.py stages
the batch, process.py consumes it — split at state/batch_current.json. All
three entry points share pipelib/runner.py, so the behaviour is identical
either way.

Usage:
    venv/bin/python pipeline.py --count 200       # fetch + A/B + index a batch
    venv/bin/python pipeline.py --count 200 --with-cases   # legacy: also Stage C
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
    ap.add_argument("--month", metavar="YYYY-MM",
                    help="fetch from this month, spread evenly across it "
                         "(default: newest-first). Multi-month planning lives "
                         "in fetch.py --plan-months / --next-month")
    ap.add_argument("--dry-run", action="store_true",
                    help="fetch + exclusion report only; nothing written")
    ap.add_argument("--skip-fetch", action="store_true",
                    help="resume the staged batch (error if none)")
    ap.add_argument("--stats", action="store_true",
                    help="print ledger/case/growth summary and exit")
    ap.add_argument("--with-cases", action="store_true",
                    help="also run Stage C case-mapping and regenerate the "
                         "case reports (legacy default)")
    ap.add_argument("--chunk-size", type=int, default=None, metavar="N",
                    help="dispatches per A->B->D slice (default 500 from config.PROCESS_CHUNK); the search index is updated after each slice. 0 runs each stage over the whole batch")
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
        batch = runner.fetch(args.count, dry_run=args.dry_run, month=args.month)
        if args.dry_run:
            return

    runner.process(batch, with_cases=args.with_cases,
                   chunk_size=args.chunk_size)


if __name__ == "__main__":
    main()
