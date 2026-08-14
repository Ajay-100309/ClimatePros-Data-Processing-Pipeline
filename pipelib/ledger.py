"""Processed-dispatch ledger: terminal states only.

A dispatch enters the ledger exactly once, at batch finalize, with status
"mapped" (has a CaseId) or "no_useful_notes". Dispatches that failed a stage
never enter — they stay eligible for a future batch, where their committed
stage results are reused.
"""
from . import config
from .statefiles import load_json, save_json


def load():
    data = load_json(config.LEDGER_FILE)
    if data is None:
        raise SystemExit(
            "state/ledger.json missing — run migrate_casemap_state.py first.")
    return data


def processed_ids(data):
    return set(data["dispatches"].keys())


def mark(data, dispatch_id, status, case_id, batch_id):
    data["dispatches"][dispatch_id] = {
        "status": status, "case_id": case_id, "batch": batch_id}


def save(data):
    save_json(config.LEDGER_FILE, data)
