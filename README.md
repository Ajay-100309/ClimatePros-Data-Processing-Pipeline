# Dispatch Pipeline — Deployment & Run Guide

## What this does

Processes HVAC/refrigeration service dispatches from the `FieldJetXStg` database. Each
dispatch's notes are filtered down to the technically useful ones and the verbatim root
cause is extracted; by default that root cause is then **embedded and pushed — together with
the dispatch's recorded parts — into the Azure AI Search index that serves the Parts Finder**
(`AZURE_SEARCH_INDEX`, default `dispatches-nomic768-v1` on `cp-search-dev`).

The original unique-case catalog (Stage C: embedding similarity + an AI judge deciding
repeat-vs-new, plus the case reports and growth chart) still exists in full, but no longer
runs by default — pass `--with-cases` to run it exactly as before.

**This pipeline is resumable and never reprocesses a dispatch twice** — it tracks everything
it has already done and always picks up where it left off.

## Current status (as of this handoff)

- **16,000 dispatches processed**, **1,808 unique cases** identified
- The included `state/` folder and `output/chroma_cases_extracted/` vector database contain
  this full history — **do not delete or replace them**, they're what makes "process the next
  dispatches" possible instead of starting over from zero
- Both are committed to this repo, so **a fresh clone arrives with all 16,000 dispatches
  already processed** and continues from #16,001. Starting over is a deliberate step — see
  *Starting over from scratch* below.

## Files included / required

```
pipeline.py               entry point — fetch and process in one run
fetch.py                  entry point — database collection only
process.py                entry point — processing only (no database access)
create_search_index.py    create/inspect/rebuild the Azure AI Search index
backfill_search.py        push processed history into the index; also the reconciler
reset_state.py            wipe progress and start the catalog over from zero
make_report.py            regenerate output/pipeline_report.html from state
pipelib/                  all pipeline logic (config, db, stages, llm client, reports)
prompt_notes.txt          Stage A prompt (note-usefulness classification)
prompt_extract.txt        Stage B prompt (verbatim root-cause extraction)
prompt_casemap.txt        Stage C prompt (case-matching judge)
requirements.txt          Python dependencies
state/                    existing progress — ledger, case registry, checkpoints,
                          recorded parts (parts.json), search push record
                          (search_index.json) (KEEP)
output/chroma_cases_extracted/   existing vector database of the cases (KEEP)
output/part_popularity.json      part-usage prior over the indexed corpus (regenerated)
analysis/                 the evaluation scripts behind the Parts Finder design numbers
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
   AZURE_SEARCH_ENDPOINT=
   AZURE_SEARCH_API_KEY=
   AZURE_SEARCH_INDEX=
   ```
   `API_KEY`/`BASE_URL`/`MODEL` point at the LLM gateway (chat model + `nomic-embed` for
   embeddings); the `DB_*` values point at the `FieldJetXStg` SQL Server database; the
   `AZURE_SEARCH_*` values point at the Azure AI Search service (`AZURE_SEARCH_INDEX`
   defaults to `dispatches-nomic768-v1` if left unset). The three search keys are only
   enforced by the commands that push documents — `--stats`, `make_report.py`, and
   `reset_state.py` work without them.

3. **Copy `state/` and `output/chroma_cases_extracted/` into place** exactly as provided —
   these two must stay in sync with each other. If they ever get out of sync (e.g. one was
   restored from an older backup than the other), the pipeline will detect a mismatch on
   startup and refuse to run rather than silently corrupt the case catalog.

## Running it

```
python pipeline.py --count 3000          # fetch, extract, embed + push the next 3000 never-seen dispatches
python pipeline.py --count 3000 --with-cases   # same, plus Stage C case-mapping and the case reports
python pipeline.py --count 5 --dry-run   # preview what would be fetched, writes nothing
python pipeline.py --skip-fetch          # resume an interrupted batch without re-fetching
python pipeline.py --stats               # ledger / case / search-index totals, no processing
```

Each run only ever touches dispatches it hasn't seen before — everything already in the
ledger or the case map will never be re-selected or reprocessed.

### The Azure AI Search index

One-time setup (already done for `cp-search-dev`) and the backfill/reconcile command:

```
python create_search_index.py            # create the index (refuses if it exists)
python create_search_index.py --show     # print the live schema
python create_search_index.py --update   # additive in-place field update
python backfill_search.py --dry-run      # what would be pushed, writes nothing
python backfill_search.py                # push all processed history not yet indexed
python backfill_search.py --refresh-parts --force-push   # after --update: repopulate all docs
```

`backfill_search.py` is rerunnable and skips whatever is already pushed, which makes it the
reconciler for any gap (a failed push mid-batch, history processed by `--with-cases` runs
before this feature, a freshly re-created index — after `--recreate`, also delete
`state/search_index.json` so the push record matches the empty index). The embedding model
is pinned (`nomic-embed`, 768-dim, no prefix); switching models later means a **new**
`AZURE_SEARCH_INDEX` name plus a backfill rerun — vector dimensions cannot change on a live
index. Additive non-vector fields are the exception: `--update` pushes them onto the live
index, and `--refresh-parts --force-push` re-pulls parts from the DB and re-pushes every
document so the new field is populated (documents keep null for it until re-pushed).

Each indexed part carries `unitedPartNo` — United Refrigeration's own catalog number,
resolved from `InventorySupplierXREF` (active United suppliers only, false self-referencing
cross-references excluded; empty when the item has no United mapping).

### Running the two halves as separate commands

The same work can be split into a collection step and a processing step:

```
python fetch.py --count 3000             # step 1: database only — stage the batch and stop
python process.py                        # step 2: process the staged batch (no database access)

python fetch.py --count 5 --dry-run      # preview what would be fetched, writes nothing
python process.py --stats                # ledger/case/growth totals, no processing
```

The **fetch half** (`fetch.py` and `backfill_search.py`) are the only commands that open a
database connection. `fetch.py` selects the dispatches, pulls their notes and recorded parts
(`state/parts.json`), writes the work order to `state/batch_current.json`, and exits — no AI
calls are made and nothing is marked processed, so a staged batch that never gets processed
simply leaves those dispatches staged and still eligible.

`process.py` picks that file up and needs only the LLM gateway and Azure AI Search, so the
two steps can run on different schedules, or on different machines, as long as they share the
same `state/` folder.
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

### A known operational note: gateway concurrency

`MAX_WORKERS` in `pipelib/config.py` was deliberately lowered to 1 at one point after
diagnosing recurring `ReadTimeout` errors — the LLM gateway couldn't reliably keep up when
multiple requests arrived at once. **The code currently has `MAX_WORKERS = 8`, which
contradicts that guidance** — someone raised it without updating the docs; whether the
gateway now tolerates concurrency has not been re-verified. If Stage A/B runs start hitting
`ReadTimeout`s, set it back to 1. (Stage D adds no new gateway concurrency either way —
embedding calls are sequential.)

### Stopping and resuming is always safe

State is checkpointed every 10 dispatches per stage, and every write is atomic (write-then-
rename), so the process can be killed at any point (Ctrl+C, terminal close, server restart)
without corrupting anything. Simply re-running the same command resumes from the last
checkpoint — no manual recovery steps needed.

## How a batch is processed

Stages A and B always run; Stage C only with `--with-cases`; Stage D always runs last:

1. **Stage A — Note classification.** Each dispatch's raw notes (technician entries, customer
   complaints, scheduling chatter, etc.) are classified note-by-note as technically useful or
   not (admin/scheduling/billing text gets filtered out).

2. **Stage B — Root-cause extraction.** From only the useful notes, the verbatim sentence(s)
   stating the actual technical root cause are extracted — not summarized or reworded, just
   the exact original wording, typically 1-4 sentences.

3. **Stage C — Case matching** (only with `--with-cases`). This is where "is this new or a
   repeat?" gets decided, in two steps:
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

4. **Stage D — Search indexing.** Every dispatch with a non-empty extracted root cause is
   embedded (`nomic-embed`, 768-dim, via the shared content-addressed cache — already-seen
   text costs nothing) and upserted into the Azure AI Search index together with its real
   recorded parts (catalog number present, not a consumable, quantity > 0, inventory id
   resolved). Dispatches whose notes contain no technical fault are recorded as terminal
   `no_fault` and never indexed. `state/search_index.json` tracks what was pushed;
   `output/part_popularity.json` (distinct-dispatch count per part) is regenerated after
   every batch. Terminal ledger statuses in the default mode are `indexed`, `no_fault`, and
   `no_useful_notes`; a push failure leaves the dispatch out of the ledger so the next run
   retries it with the Stage A/B results reused for free.

## Case-growth reporting (only regenerated by `--with-cases` runs)

After every completed `--with-cases` batch, `pipelib/reports.py` regenerates:

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
