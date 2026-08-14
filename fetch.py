"""Stage 0 on its own: collect dispatches from FieldJetXStg and stage them.

Selects the newest dispatches not already in the ledger or the case map, pulls
all of their notes, and writes the work order to state/batch_current.json —
then stops. This is the only command that opens a database connection: no LLM
calls are made, no dispatch is marked processed, and nothing in the case
catalog changes. Run process.py afterwards to consume the staged batch.

If a batch is already staged, it is left untouched and reported — finish it
with process.py before staging another.

Usage:
    venv/bin/python fetch.py --count 200
    venv/bin/python fetch.py --count 5 --dry-run
"""
import sys
import argparse

from pipelib import config, runner


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, help="dispatches to stage")
    ap.add_argument("--dry-run", action="store_true",
                    help="fetch + exclusion report only; nothing written")
    args = ap.parse_args()

    if not args.count:
        sys.exit("--count N is required.")

    batch = runner.fetch(args.count, dry_run=args.dry_run)
    if batch is None:
        return  # dry run; stage_fetch already printed the report

    print(f"Work order: {config.BATCH_FILE}")
    print("Next: venv/bin/python process.py")


if __name__ == "__main__":
    main()
