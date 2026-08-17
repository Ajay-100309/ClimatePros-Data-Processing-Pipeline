"""Reset the pipeline to a cold start — empty ledger, empty case catalog.

Clears the processed-dispatch ledger, the case registry, every per-stage
checkpoint, the batch archive, and the Chroma vector DB, then writes the two
seed files the loaders require (ledger.load() and stage_casemap.load_state()
both hard-exit if theirs is missing). Afterwards the next fetch selects
dispatches as though none had ever been processed, and case numbering restarts
at CASE-0001.

The embedding cache is KEPT by default: it is content-addressed by sha256 of
the text and the model is unchanged, so it stays valid and saves re-embedding
everything. Pass --cold to drop it too.

This is destructive and requires --yes. In this repo state/ and output/ are
committed to git, so a reset is recoverable with:
    git checkout <commit> -- state output

Usage:
    venv/bin/python reset_state.py --dry-run
    venv/bin/python reset_state.py --yes
    venv/bin/python reset_state.py --yes --cold
"""
import os
import sys
import glob
import json
import shutil
import argparse
from datetime import datetime, timezone

from pipelib import config, statefiles
from pipelib.statefiles import load_json


def current_totals():
    """What the reset would discard, for the confirmation summary."""
    led = load_json(config.LEDGER_FILE) or {}
    cm = load_json(config.CASEMAP_FILE) or {}
    return (len(led.get("dispatches", {})), len(cm.get("cases", {})),
            len(cm.get("dispatches", {})))


def targets(cold):
    """(label, path) pairs this reset removes, in report order."""
    out = [
        ("ledger", config.LEDGER_FILE),
        ("case registry", config.CASEMAP_FILE),
        ("stage A checkpoints", config.NOTES_CLASS_FILE),
        ("stage B checkpoints", config.EXTRACT_FILE),
        ("dispatch display metadata", config.DISPATCH_META_FILE),
        ("staged work order", config.BATCH_FILE),
        ("batch archive", config.BATCH_ARCHIVE_DIR),
        ("case vector DB", config.CHROMA_DIR),
    ]
    if cold:
        out += [("embedding cache", config.EMB_NPY),
                ("embedding index", config.EMB_INDEX)]
    return out


def seed():
    """Write the minimal state both loaders accept.

    The fingerprint is derived from config rather than copied, so it can never
    drift from the live prompt/threshold/model values and trip the guard.
    """
    statefiles.ensure_dirs()
    statefiles.save_json(config.LEDGER_FILE, {
        "version": 1,
        "seeded_from": "reset_state.py",
        "seeded_at": datetime.now(timezone.utc).isoformat(),
        "dispatches": {},
    })
    statefiles.save_json(config.CASEMAP_FILE, {
        "version": 2,
        "fingerprint": config.casemap_fingerprint(),
        "next_case_num": 1,
        "cases": {},
        "dispatches": {},
        "growth": [],
    })


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yes", action="store_true",
                    help="actually do it (required; destructive)")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be removed and exit")
    ap.add_argument("--cold", action="store_true",
                    help="also drop the embedding cache (slower next run)")
    args = ap.parse_args()

    n_proc, n_cases, n_mapped = current_totals()
    print(f"Current state: {n_proc:,} processed dispatches, {n_cases:,} cases, "
          f"{n_mapped:,} mapped.")
    print(f"Reset would remove ({'cold' if args.cold else 'keeping embedding cache'}):")
    for label, path in targets(args.cold):
        if os.path.isdir(path):
            n = len(glob.glob(os.path.join(path, "*")))
            print(f"  {label:<28} {path}  ({n} entr{'y' if n == 1 else 'ies'})")
        elif os.path.exists(path):
            mb = os.path.getsize(path) / 1e6
            print(f"  {label:<28} {path}  ({mb:.1f} MB)")
        else:
            print(f"  {label:<28} {path}  (absent)")

    if args.dry_run:
        print("\n--dry-run: nothing removed.")
        return
    if not args.yes:
        sys.exit("\nRefusing to reset without --yes. "
                 "Re-run with --dry-run to preview, or --yes to proceed.")

    for _label, path in targets(args.cold):
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.exists(path):
            os.remove(path)

    seed()
    print(f"\nReset complete. Ledger and case catalog are empty; "
          f"next case will be CASE-0001.")
    print("Next: venv/bin/python fetch.py --count 3000")
    print("      venv/bin/python process.py")


if __name__ == "__main__":
    main()
