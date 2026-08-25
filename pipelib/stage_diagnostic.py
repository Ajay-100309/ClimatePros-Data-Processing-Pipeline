"""Stage B2: classify whether an already-extracted root-cause summary reflects
a real diagnostic finding, or non-diagnostic content that slipped through
Stage A/B (installation work, parts logistics, scheduling, warranty/billing
disputes) — see Root Cause 2 in the diagnostic report.

Runs on the short extracted summary only (1-4 sentences), not the raw note
history, so it is cheap and — critically — can be applied retroactively to the
whole existing extract.json backlog without re-running Stage B.
"""
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import config, llm
from .statefiles import load_json, save_json

_PROMPT = config.read_prompt(config.PROMPT_DIAGNOSTIC)


def _classify_one(dispatch_id, summary):
    prompt = _PROMPT.replace("{summary}", summary)
    for attempt in range(3):
        parsed, _raw = llm.chat_json([
            {"role": "system", "content": "You are an expert HVAC service analyst."},
            {"role": "user", "content": prompt},
        ])
        if parsed is None or "diagnostic" not in parsed:
            continue
        return {"diagnostic": bool(parsed["diagnostic"]),
                "reason": str(parsed.get("reason", "")).strip()}
    raise RuntimeError(f"diagnostic classification failed after retries for {dispatch_id}")


def run(batch, extract_state):
    state = load_json(config.DIAGNOSTIC_FILE, {})
    pending = [(d["dispatch_id"], extract_state[d["dispatch_id"]])
               for d in batch["dispatches"]
               if d["dispatch_id"] in extract_state and d["dispatch_id"] not in state]
    if not pending:
        print("Stage B2: nothing to classify (all done).")
        return state
    print(f"Stage B2: classifying {len(pending)} extracted summaries for diagnostic content...")

    lock = threading.Lock()
    done = 0
    failed = []
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as ex:
        futures = {ex.submit(_classify_one, did, rec["root_cause"] or config.NO_FAULT_PLACEHOLDER): did
                   for did, rec in pending}
        for fut in as_completed(futures):
            did = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                print(f"  Stage B2 failed for {did}: {type(e).__name__}: {e}")
                traceback.print_exc()
                failed.append(did)
                continue
            with lock:
                state[did] = res
                done += 1
                tag = "diagnostic" if res["diagnostic"] else "NON-diagnostic"
                print(f"  [B2 {done}/{len(pending)}] {did} -> {tag} ({res['reason']})")
                if done % config.CHECKPOINT_EVERY == 0:
                    save_json(config.DIAGNOSTIC_FILE, state)

    save_json(config.DIAGNOSTIC_FILE, state)
    if failed:
        print(f"Stage B2: {len(failed)} dispatch(es) failed and stay pending: {failed}")
    return state
