# Dispatch Case-Mapping Pipeline — Deployment & Run Guide

## What this does

Processes HVAC/refrigeration service dispatches from the `FieldJetXStg` database and groups
them into a catalog of unique root-cause "cases." Each dispatch's notes are filtered down to
the technically useful ones, the verbatim root cause is extracted, and that root cause is
compared against every existing case (via embedding similarity + an AI judge) to decide
whether it's a repeat of a known problem or a genuinely new one. The output is a growing,
deduplicated case catalog plus a chart showing how many unique cases exist relative to how
many dispatches have been processed.

**This pipeline is resumable and never reprocesses a dispatch twice** — it tracks everything
it has already done and always picks up where it left off.

## Current status (as of this handoff)

- **6,009 dispatches processed**, **821 unique cases** identified
- The included `state/` folder and `output/chroma_cases_extracted/` vector database contain
  this full history — **do not delete or replace them**, they're what makes "process the next
  dispatches" possible instead of starting over from zero

## Files included / required

```
pipeline.py               entry point — fetch and process in one run
fetch.py                  entry point — database collection only
process.py                entry point — processing only (no database access)
pipelib/                  all pipeline logic (config, db, stages, llm client, reports)
prompt_notes.txt          Stage A prompt (note-usefulness classification)
prompt_extract.txt        Stage B prompt (verbatim root-cause extraction)
prompt_casemap.txt        Stage C prompt (case-matching judge)
requirements.txt          Python dependencies
state/                    existing progress — ledger, case registry, checkpoints (KEEP)
output/chroma_cases_extracted/   existing vector database of all 821 cases (KEEP)
```

Everything else in the original project folder (`main_*.py`, `rag_parts_recommender.py`,
duplicate-audit scripts, etc.) belongs to separate/older workflows and is **not needed** to
run this pipeline or produce the case-growth report.

## Setup

1. **Python 3.12**, then install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. **Create a `.env` file** in the project root (this is never included in any file transfer —
   it must be created fresh with your own values). Required keys:
   ```
   API_KEY=
   BASE_URL=
   MODEL=
   DB_HOST=
   DB_PORT=
   DB_NAME=
   DB_USER=
   DB_PASSWORD=
   ```
   `API_KEY`/`BASE_URL`/`MODEL` point at the LLM gateway (chat model + `nomic-embed` for
   embeddings); the `DB_*` values point at the `FieldJetXStg` SQL Server database.

3. **Copy `state/` and `output/chroma_cases_extracted/` into place** exactly as provided —
   these two must stay in sync with each other. If they ever get out of sync (e.g. one was
   restored from an older backup than the other), the pipeline will detect a mismatch on
   startup and refuse to run rather than silently corrupt the case catalog.

## Running it

```
python pipeline.py --count 3000          # fetch and process the next 3000 never-seen dispatches
python pipeline.py --count 5 --dry-run   # preview what would be fetched, writes nothing
python pipeline.py --skip-fetch          # resume an interrupted batch without re-fetching
python pipeline.py --stats               # print current ledger/case/growth totals, no processing
```

Each run only ever touches dispatches it hasn't seen before — dispatch #1 through #6,009 will
never be re-selected or reprocessed.

### Running the two halves as separate commands

The same work can be split into a collection step and a processing step:

```
python fetch.py --count 3000             # step 1: database only — stage the batch and stop
python process.py                        # step 2: process the staged batch (no database access)

python fetch.py --count 5 --dry-run      # preview what would be fetched, writes nothing
python process.py --stats                # ledger/case/growth totals, no processing
```

`fetch.py` is the **only** command that opens a database connection. It selects the dispatches,
pulls their notes, writes the work order to `state/batch_current.json`, and exits — no AI calls
are made and nothing is marked processed, so a staged batch that never gets processed simply
leaves those dispatches staged and still eligible.

`process.py` picks that file up and needs only the LLM gateway, so the two steps can run on
different schedules, or on different machines, as long as they share the same `state/` folder.
Both are safe to re-run: `fetch.py` will not stage a second batch while one is already in
flight (it reports the existing one instead), and re-running `process.py` after an interruption
resumes from the last checkpoint.

`python pipeline.py --count 3000` remains exactly equivalent to running the two in sequence —
all three commands share the same code (`pipelib/runner.py`), so there is no behavioural
difference between them.

### A known operational note: keep concurrency at 1

`pipelib/config.py` currently has `MAX_WORKERS = 1`. This was deliberately lowered from a
higher value after diagnosing recurring `ReadTimeout` errors — the LLM gateway couldn't
reliably keep up when multiple requests arrived at once, causing dropped/timed-out requests
under concurrency. **Recommend leaving this at 1 unless you've re-tested concurrency
specifically against the server's own gateway setup** — a different server/network path may
behave differently, but this hasn't been re-verified.

### Stopping and resuming is always safe

State is checkpointed every 10 dispatches per stage, and every write is atomic (write-then-
rename), so the process can be killed at any point (Ctrl+C, terminal close, server restart)
without corrupting anything. Simply re-running the same command resumes from the last
checkpoint — no manual recovery steps needed.

## How cases are formed

Three sequential stages, run once per batch:

1. **Stage A — Note classification.** Each dispatch's raw notes (technician entries, customer
   complaints, scheduling chatter, etc.) are classified note-by-note as technically useful or
   not (admin/scheduling/billing text gets filtered out).

2. **Stage B — Root-cause extraction.** From only the useful notes, the verbatim sentence(s)
   stating the actual technical root cause are extracted — not summarized or reworded, just
   the exact original wording, typically 1-4 sentences.

3. **Stage C — Case matching.** This is where "is this new or a repeat?" gets decided, in two
   steps:
   - **Similarity search (mechanical):** the extracted text is embedded and compared against
     every existing case; the **5** most similar existing cases above a **60%** similarity
     threshold become candidates. (Both values configurable in `pipelib/config.py` as
     `N_CANDIDATES` and `SIM_FLOOR`.)
   - **AI judgment (semantic):** an LLM is shown the new text alongside those candidates and
     asked whether it's the *same underlying root cause* as one of them — same component and
     same failure mode required, not just similar wording or category. If it's uncertain, it
     is instructed to say no and create a new case rather than risk an incorrect merge — the
     system is deliberately tuned to favor correctness of matches over catching every possible
     match.

Every dispatch ends up either matched to an existing case or creating a new one; a running
`(dispatches processed, unique cases)` series is recorded after every resolved dispatch.

## Case-growth reporting

This is produced automatically — no separate script needed. After every completed batch,
`pipelib/reports.py` regenerates:

- `output/case_growth.xlsx` — the raw `(dispatches processed, unique cases)` series
- `output/case_growth.png` — a chart of that series, with a dashed reference line showing
  what "zero deduplication" (1 new case per dispatch) would look like for comparison
- `output/Dispatch_CaseMapped.xlsx` — one row per dispatch with its assigned case
- `output/case_summary_extracted.xlsx` — one row per case with its member dispatch count

**What to look for in the chart:** the gap between the actual curve and the reference
diagonal shows how much deduplication is happening. Based on the history to date, the new-
case rate started around 40%+ early on and has settled into a steady ~15% for the last several
thousand dispatches — meaning roughly 1 in 6-7 new dispatches introduces a genuinely new
problem, and the rest are recognized repeats of known issues.
