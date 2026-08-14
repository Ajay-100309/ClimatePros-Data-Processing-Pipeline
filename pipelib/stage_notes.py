"""Stage A: note-usefulness classification, chunked 20 notes per request.

A dispatch is committed to state/notes_class.json only when ALL of its chunks
succeeded — partial dispatches stay pending and are retried on the next run.
"""
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import config, llm
from .statefiles import load_json, save_json

_PROMPT = config.read_prompt(config.PROMPT_NOTES)


def _norm_note_id(value):
    return str(value).replace(" ", "").strip().upper()


def _classify_chunk(dispatch_id, notes):
    note_text = ""
    for n in notes:
        note_text += (f"DispatchNotesId: {n['note_id']}\n"
                      f"DispatchNote:\n{n['text']}\n"
                      "-------------------------\n")
    prompt = _PROMPT.replace("{dispatch_notes}", note_text)
    valid = {_norm_note_id(n["note_id"]): n["note_id"] for n in notes}

    for attempt in range(3):
        parsed, _raw = llm.chat_json([
            {"role": "system", "content": "You are an HVAC dispatch analyst."},
            {"role": "user", "content": prompt},
        ])
        if parsed is None or "notes" not in parsed or not isinstance(parsed["notes"], list):
            continue
        out = []
        for item in parsed["notes"]:
            try:
                nid = _norm_note_id(item["dispatch_notes_id"])
            except (KeyError, TypeError):
                continue
            if nid not in valid:
                print(f"  {dispatch_id}: skipping unknown note id from model: "
                      f"{item.get('dispatch_notes_id')}")
                continue
            out.append({
                "dispatch_notes_id": valid[nid],
                "useful": bool(item.get("useful")),
                "category": str(item.get("category", "")),
                "reason": str(item.get("reason", "")),
            })
        if out:
            return out
    raise RuntimeError(f"chunk classification failed after retries "
                       f"({len(notes)} notes)")


def _classify_dispatch(dispatch):
    results = []
    notes = dispatch["notes"]
    for i in range(0, len(notes), config.NOTE_CHUNK):
        results.extend(_classify_chunk(dispatch["dispatch_id"],
                                       notes[i:i + config.NOTE_CHUNK]))
    return results


def run(batch):
    state = load_json(config.NOTES_CLASS_FILE, {})
    pending = [d for d in batch["dispatches"]
               if d["dispatch_id"] not in state]
    if not pending:
        print("Stage A: nothing to classify (all done).")
        return state
    print(f"Stage A: classifying notes for {len(pending)} dispatches...")

    lock = threading.Lock()
    done = 0
    failed = []
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as ex:
        futures = {ex.submit(_classify_dispatch, d): d["dispatch_id"]
                   for d in pending}
        for fut in as_completed(futures):
            did = futures[fut]
            try:
                results = fut.result()
            except Exception as e:
                print(f"  Stage A failed for {did}: {type(e).__name__}: {e}")
                traceback.print_exc()
                failed.append(did)
                continue
            with lock:
                state[did] = results
                done += 1
                useful = sum(1 for r in results if r["useful"])
                print(f"  [A {done}/{len(pending)}] {did}: "
                      f"{useful}/{len(results)} notes useful")
                if done % config.CHECKPOINT_EVERY == 0:
                    save_json(config.NOTES_CLASS_FILE, state)

    save_json(config.NOTES_CLASS_FILE, state)
    if failed:
        print(f"Stage A: {len(failed)} dispatch(es) failed and stay pending: {failed}")
    return state


def useful_notes(batch, state):
    """{DISPATCH_ID: [note dicts (chronological, useful only)]} for classified
    dispatches of this batch."""
    out = {}
    for d in batch["dispatches"]:
        did = d["dispatch_id"]
        if did not in state:
            continue
        useful_ids = {config.norm_guid(r["dispatch_notes_id"])
                      for r in state[did] if r["useful"]}
        out[did] = [n for n in d["notes"]
                    if config.norm_guid(n["note_id"]) in useful_ids]
    return out
