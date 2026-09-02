"""
Reproduces the "83% of DispatchReason values are usable as-is" finding.

Question: is the raw DispatchReason field (what a Parts Finder query would
use if nothing were cleaned up first) mostly administrative noise, or mostly
real problem descriptions?

Source data: state/dispatch_meta.json, the 2,000 real staged dispatches
already sitting in this repo. No DB access, no LLM calls.

Run from the repo root:
    venv/bin/python analysis/01_dispatch_reason_quality.py

Expected output (matches the architecture doc and presentation script):
    total dispatches: 2000
    OnCallRegion: tag ............. ~4.0%
    Work-order assignment notice ... ~7.1%
    Billing only / no action ....... ~1.1%
    Under 40 chars .................. ~4.0%
    Under 80 chars ................... ~8.6%
    Any known non-descriptive pattern: ~16.7%
    Remaining, unclassified: ~83.3%
"""
import json
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
META_FILE = HERE / "state" / "dispatch_meta.json"

PATTERNS = {
    "OnCallRegion: tag": lambda r: r.strip().startswith("OnCallRegion:"),
    "Work-order assignment notice": lambda r: bool(re.search(r"has assigned Work Order", r)),
    "Billing only / no action needed": lambda r: bool(re.search(r"billing only|No action needed", r, re.I)),
    "Under 40 chars (MIN_NOTE_LEN)": lambda r: len(r.strip()) < 40,
    "Under 80 chars": lambda r: len(r.strip()) < 80,
}


def main():
    meta = json.load(open(META_FILE))
    reasons = [v["reason"] for v in meta.values()]
    n = len(reasons)
    print(f"total dispatches: {n}")

    lens = [len(r) for r in reasons]
    print(f"length min/median/max: {min(lens)} / {statistics.median(lens)} / {max(lens)}")
    print()

    covered = [False] * n
    for name, fn in PATTERNS.items():
        hits = [fn(r) for r in reasons]
        c = sum(hits)
        print(f"{name}: {c} ({c / n * 100:.1f}%)")
        covered = [a or b for a, b in zip(covered, hits)]

    print()
    print(f"Any known non-descriptive pattern: {sum(covered)} ({sum(covered) / n * 100:.1f}%)")
    print(f"Remaining, unclassified (assumed usable): {n - sum(covered)} ({(n - sum(covered)) / n * 100:.1f}%)")


if __name__ == "__main__":
    main()
