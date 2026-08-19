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

- **Catalog reset to zero** — `state/` and `output/chroma_cases_extracted/` were deliberately
  cleared (`reset_state.py --yes`) before this handoff so the first run exercises the current,
  fixed matching logic from a clean start. Next case will be `CASE-0001`.
- The embedding cache (`state/embeddings.npy` / `embeddings_index.json`) was **kept** — it's
  content-addressed by text hash, so it stays valid and saves re-embedding.
- Once you run the pipeline, `state/` and `output/chroma_cases_extracted/` again become live,
  irreplaceable progress — **do not delete or replace them** once real dispatches have been
  processed. Both are committed to this repo, so anyone else cloning it after that point
  inherits the full history and continues from wherever it was left.
- A prior run against ~13,000 dispatches (before this reset) had produced 1,808 cases with the
  old matching logic; a full offline replay of that same data with the fixes below produced
  **955 cases** from a clean catalog — see *How cases are formed* for what changed and why.
- **A batch of 2,000 real dispatches is already fetched and staged** (`state/batch_current.json`,
  from `fetch.py --count 2000`) so the very first run doesn't need database access at all — see
  *Running it for the first time* below.

## Files included / required

```
pipeline.py               entry point — fetch and process in one run
fetch.py                  entry point — database collection only
process.py                entry point — processing only (no database access)
consolidate.py            entry point — offline maintenance pass, merges near-duplicate cases
reset_state.py            wipe progress and start the catalog over from zero
make_report.py            regenerate output/pipeline_report.html from state
pipelib/                  all pipeline logic (config, db, stages, llm client, reports)
prompt_notes.txt          Stage A prompt (note-usefulness classification)
prompt_extract.txt        Stage B prompt (verbatim root-cause extraction)
prompt_diagnostic.txt     Stage B2 prompt (filters out non-diagnostic content)
prompt_casemap.txt        Stage C prompt (case-matching judge)
requirements.txt          Python dependencies
state/                    progress — ledger, case registry, checkpoints (KEEP once populated)
output/chroma_cases_extracted/   vector database of all cases (KEEP once populated)
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

## Running it for the first time

A batch of 2,000 real dispatches is already staged (`state/batch_current.json`), so you can see
the whole pipeline work end-to-end **without setting up database access first**:

```
python process.py            # processes the already-staged 2,000 dispatches, no DB needed
python pipeline.py --stats   # check progress / final totals
```

You still need a complete `.env` — `pipelib/config.py` validates all 8 keys at startup
regardless of which command you run, even though `process.py` never actually connects to the
database. If you don't have `FieldJetXStg` access yet, any placeholder values for the `DB_*`
keys will satisfy the check; just don't run `fetch.py` until they're real. If `process.py` gets
interrupted partway through, just re-run the same command; it resumes from the last checkpoint
(every 10 dispatches per stage).

## Running it

```
python pipeline.py --count 3000          # fetch and process the next 3000 never-seen dispatches
python pipeline.py --count 5 --dry-run   # preview what would be fetched, writes nothing
python pipeline.py --skip-fetch          # resume an interrupted batch without re-fetching
python pipeline.py --stats               # print current ledger/case/growth totals, no processing
```

Each run only ever touches dispatches it hasn't seen before — anything already recorded in
`state/ledger.json` or `state/casemap.json` will never be re-selected or reprocessed.

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

## Starting over from scratch

Because the case history ships with the code, a new machine continues the existing catalog by
default. To ignore it and rebuild from zero:

```
python reset_state.py --dry-run     # show exactly what would be cleared, change nothing
python reset_state.py --yes         # clear it and seed an empty catalog
python fetch.py --count 3000        # then run as normal
python process.py
```

That one command clears the ledger, the case registry, both stage checkpoints, the batch
archive and the vector database, then writes the two seed files the pipeline requires on
startup. Case numbering restarts at `CASE-0001`. It refuses to run without `--yes`, and prints
the current totals first so you can see what you are about to discard.

Two details worth knowing:

- **The embedding cache is kept by default.** It is keyed by a hash of the text, so it stays
  valid across a reset and saves re-embedding everything — a straight time and API saving with
  no effect on results. Add `--cold` if you want it dropped too.
- **A reset is recoverable.** `state/` and `output/` are tracked in git, so the previous
  catalog can be restored with `git checkout <commit> -- state output` as long as it was
  committed. Check `git status` is clean before resetting.

Expect the rebuilt catalog to reach similar *totals* but different *case IDs*. Selection is
newest-first with nothing excluded, so the same dispatches come back in the same order, but
which text mints `CASE-0001` depends on AI judgement — so case IDs are not comparable between
runs. Compare the growth curve, not the identifiers.

### A known operational note: the LLM gateway is not fully reliable under load

`pipelib/config.py` currently has `MAX_WORKERS = 8` for Stages A, B and B2 (Stage C is a plain
sequential loop by design — see above). In practice, expect two failure modes from the gateway,
both retried automatically with backoff by `pipelib/llm.py`, up to 8 attempts:

- **Transient connection blips** (`ReadTimeout`, DNS resolution failures).
- **Gateway-side `500 InternalServerError`** (e.g. `"Model Group=qwen3-vl... Connection
  error"`, reported when the backend model itself is temporarily unavailable) — observed
  repeatedly during live testing and now retried the same way as a connection blip.

Both usually self-heal within a few attempts. If retries are ever exhausted: Stages A and B
handle it per-dispatch and just leave that dispatch pending for the next run; **Stage C has no
equivalent per-dispatch handler**, so exhausting retries there still crashes `process.py`. This
is safe to resume (checkpointed every 10 items, `finalize()` never ran so
`state/batch_current.json` is untouched) — just re-run `process.py` — but it's not automatic.
If your environment sees this often, wrapping `process.py` in a restart loop is a reasonable
stopgap; a proper fix would give Stage C the same per-dispatch resilience Stage A/B already
have.

### Stopping and resuming is always safe

State is checkpointed every 10 dispatches per stage, and every write is atomic (write-then-
rename), so the process can be killed at any point (Ctrl+C, terminal close, server restart)
without corrupting anything. Simply re-running the same command resumes from the last
checkpoint — no manual recovery steps needed.

## How cases are formed

Four sequential stages, run once per batch:

1. **Stage A — Note classification.** Each dispatch's raw notes (technician entries, customer
   complaints, scheduling chatter, etc.) are classified note-by-note as technically useful or
   not (admin/scheduling/billing text gets filtered out).

2. **Stage B — Root-cause extraction.** From only the useful notes, the verbatim sentence(s)
   stating the actual technical root cause are extracted — not summarized or reworded, just
   the exact original wording, typically 1-4 sentences.

3. **Stage B2 — Diagnostic filter.** Not every dispatch with "useful" notes is actually about a
   technical fault — some are installation jobs, parts logistics, warranty disputes, or
   scheduling calls that Stage B still extracted *something* from. This step reads the short
   extracted summary and classifies whether it's a genuine diagnostic finding (including a
   real "no fault found, confirmed by inspection" conclusion) or non-diagnostic content that
   should never have produced a case. Non-diagnostic dispatches get a terminal ledger status
   (`non_diagnostic`) instead of going to Stage C. Defaults to "diagnostic" when uncertain — a
   missed filter costs one extra case; a wrongly-filtered real diagnosis costs real data.

4. **Stage C — Case matching.** This is where "is this new or a repeat?" gets decided, in two
   steps:
   - **Similarity search (mechanical):** the extracted text is embedded and compared against
     every existing case. Each case keeps up to **5** representative example texts (not just
     the one that created it — a case's wording drifts as more dispatches join it, and a single
     frozen example turned out to be the biggest single cause of unnecessary new cases). The
     **5** most similar *cases* (best-matching example per case) above a **60%** similarity
     threshold become candidates. (Configurable in `pipelib/config.py`: `N_CANDIDATES`,
     `SIM_FLOOR`, `MAX_EXEMPLARS_PER_CASE`.)
   - **AI judgment (semantic):** an LLM is shown the new text alongside those candidates and
     asked whether it's the *same underlying root cause* as one of them — same component and
     same failure mode required, not just similar wording or category. If it's uncertain, it
     is instructed to say no and create a new case rather than risk an incorrect merge — the
     system is deliberately tuned to favor correctness of matches over catching every possible
     match.

Every dispatch ends up mapped to a case, marked non-diagnostic, or (rarely) left unresolved for
retry; a running `(dispatches processed, unique cases)` series is recorded after every resolved
dispatch.

### Cleaning up duplicate cases after the fact (`consolidate.py`)

Stage C only ever compares a new dispatch against cases that already existed *before* it — it
never revisits old decisions. That's a real constraint (case *n* must be visible to dispatch
*n+1*, not the other way around), but it means two near-duplicate cases can end up created at
different points and nothing catches it automatically. `consolidate.py` is the backstop: it
scans the existing catalog for suspiciously similar case pairs and asks the *same* cautious AI
judge used in Stage C to confirm before merging anything — never merges on similarity alone.

```
python consolidate.py --dry-run     # see candidate pairs and judge verdicts, write nothing
python consolidate.py               # apply confirmed merges, regenerate reports
```

Safe to run repeatedly (a clean catalog with nothing left to merge just reports zero merges).
**Recommend running it periodically** — e.g. after every few batches — rather than only once.

## Case-growth reporting

This is produced automatically — no separate script needed. After every completed batch,
`pipelib/reports.py` regenerates:

- `output/case_growth.xlsx` — the raw `(dispatches processed, unique cases)` series
- `output/case_growth.png` — a chart of that series, with a dashed reference line showing
  what "zero deduplication" (1 new case per dispatch) would look like for comparison
- `output/Dispatch_CaseMapped.xlsx` — one row per dispatch with its assigned case
- `output/case_summary_extracted.xlsx` — one row per case with its member dispatch count

**What to look for in the chart:** the gap between the actual curve and the reference
diagonal shows how much deduplication is happening. On a fresh catalog, expect the new-case
rate to start high (nothing to match against yet) and fall as the catalog matures. Watch this
curve early and compare it against `consolidate.py --dry-run`'s merge count over time — a
climbing new-case rate alongside a rising merge count suggests retrieval is missing matches
that consolidation later has to clean up, worth investigating rather than ignoring.
