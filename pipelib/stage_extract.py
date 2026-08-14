"""Stage B: verbatim root-cause extraction per dispatch, from useful notes only."""
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import config, llm
from .statefiles import load_json, save_json

_PROMPT = config.read_prompt(config.PROMPT_EXTRACT)


def _extract_from_history(dispatch_id, history):
    prompt = _PROMPT.replace("{dispatch_notes}", history)

    last_err = None
    for attempt in range(3):
        try:
            parsed, _raw = llm.chat_json([
                {"role": "system", "content": "You are an expert HVAC service analyst."},
                {"role": "user", "content": prompt},
            ])
        except Exception as e:
            last_err = e
            print(f"  {dispatch_id}: extraction call error (attempt {attempt + 1}/3): "
                  f"{type(e).__name__}: {e}")
            continue
        if parsed is None or "root_cause" not in parsed:
            continue
        return {"root_cause": str(parsed["root_cause"]).strip(),
                "category": str(parsed.get("category", "")).strip()}
    raise RuntimeError("extraction failed after retries"
                       + (f" (last error: {type(last_err).__name__}: {last_err})"
                          if last_err else ""))


def _extract_dispatch(dispatch_id, notes):
    if len(notes) <= config.NOTE_CHUNK:
        history = "\n---\n".join(n["text"] for n in notes)
        return _extract_from_history(dispatch_id, history)

    # Large note counts risk overflowing the model's context window in one
    # shot, and that failure never recovers on retry (same oversized prompt
    # every time). Extract per chunk instead, then consolidate the verbatim
    # chunk extracts with a second pass over the same prompt.
    chunk_results = []
    for i in range(0, len(notes), config.NOTE_CHUNK):
        chunk = notes[i:i + config.NOTE_CHUNK]
        chunk_history = "\n---\n".join(n["text"] for n in chunk)
        chunk_results.append(_extract_from_history(dispatch_id, chunk_history))

    combined = "\n---\n".join(r["root_cause"] for r in chunk_results if r["root_cause"])
    if not combined:
        fallback_cat = next((r["category"] for r in chunk_results if r["category"]), "")
        return {"root_cause": "", "category": fallback_cat}
    return _extract_from_history(dispatch_id, combined)


def run(batch, useful_by_dispatch):
    state = load_json(config.EXTRACT_FILE, {})
    pending = [(did, notes) for did, notes in useful_by_dispatch.items()
               if notes and did not in state]
    if not pending:
        print("Stage B: nothing to extract (all done).")
        return state
    print(f"Stage B: extracting root causes for {len(pending)} dispatches...")

    lock = threading.Lock()
    done = 0
    failed = []
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as ex:
        futures = {ex.submit(_extract_dispatch, did, notes): did
                   for did, notes in pending}
        for fut in as_completed(futures):
            did = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                print(f"  Stage B failed for {did}: {type(e).__name__}: {e}")
                traceback.print_exc()
                failed.append(did)
                continue
            with lock:
                state[did] = res
                done += 1
                print(f"  [B {done}/{len(pending)}] {did} ({res['category']}): "
                      f"{res['root_cause'][:70]}")
                if done % config.CHECKPOINT_EVERY == 0:
                    save_json(config.EXTRACT_FILE, state)

    save_json(config.EXTRACT_FILE, state)
    if failed:
        print(f"Stage B: {len(failed)} dispatch(es) failed and stay pending: {failed}")
    return state
