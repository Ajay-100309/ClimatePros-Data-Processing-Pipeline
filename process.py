"""The process half on its own: consume the batch staged by fetch.py.

Classifies note usefulness, extracts verbatim root causes, embeds them, and
pushes each dispatch (with its recorded parts) into the Azure AI Search index,
then finalizes the batch — appending terminal outcomes to the ledger and
archiving the work order. Pass --with-cases to also run the legacy Stage C
case mapping and regenerate the case reports.

Reads state/batch_current.json and never opens a database connection, so this
can run wherever the LLM gateway and Azure AI Search are reachable. Fully
resumable: re-run the same command after any interruption and it continues
from the last checkpoint.

Usage:
    venv/bin/python process.py
    venv/bin/python process.py --with-cases
    venv/bin/python process.py --stats
"""
import sys
import argparse

from pipelib import runner


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stats", action="store_true",
                    help="print ledger/case/growth summary and exit")
    ap.add_argument("--with-cases", action="store_true",
                    help="also run Stage C case-mapping and regenerate the "
                         "case reports (legacy default)")
    args = ap.parse_args()

    if args.stats:
        runner.stats()
        return

    batch = runner.staged_batch()
    if batch is None:
        sys.exit("No staged batch found — run: venv/bin/python fetch.py --count N")

    print(f"Processing staged batch {batch['batch_id']} "
          f"({len(batch['dispatches'])} dispatches).")
    runner.process(batch, with_cases=args.with_cases)


if __name__ == "__main__":
    main()
