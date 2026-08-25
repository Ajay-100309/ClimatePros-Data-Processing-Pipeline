"""Offline maintenance pass: retroactively merge near-duplicate cases.

Stage C (process.py) is strictly sequential and never revisits a case once
created, so near-duplicate cases can and do accumulate as the catalog grows.
This scans the finished case catalog for pairs whose exemplar embeddings are
suspiciously close, then asks the same conservative LLM judge used online
whether they're really the same root cause — nothing is ever merged on
similarity alone. Confirmed merges collapse transitively via union-find, the
case with more mapped dispatches survives, and all reports are regenerated.

No DB access, no staged batch required. Safe to re-run: a clean catalog with
no merge candidates left just reports zero merges.

Usage:
    venv/bin/python consolidate.py --dry-run   # report candidate pairs and judge verdicts only
    venv/bin/python consolidate.py             # apply confirmed merges
"""
import argparse

from pipelib import runner


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="judge candidate pairs and report; write nothing")
    args = ap.parse_args()

    runner.consolidate(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
