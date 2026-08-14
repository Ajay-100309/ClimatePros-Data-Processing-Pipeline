# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A standalone, resumable LLM data-mining pipeline that reads HVAC/refrigeration service dispatches from the **FieldJetXStg** SQL Server (directly via pymssql — it bypasses both ClimatePros APIs) and folds them into a growing, deduplicated catalog of canonical root-cause **cases** (`CASE-XXXX`). Output is the case catalog plus a "unique cases vs dispatches processed" growth series.

This is its own git repo, nested in the ClimatePros workspace (the workspace `CLAUDE.md` one level up calls this project `Dispatch/` and describes the surrounding FieldJetX apps/APIs — this is the trimmed deployment cut of it; the legacy `main_*.py` one-shot scripts and audit tooling are deliberately not here).

`README.md` is the operator-facing deployment guide and is accurate; `pipeline_architecture.html` is a standalone visual of the flow, the Stage C decision, and the thresholds. Module docstrings in `pipelib/*.py` are authoritative on behavior.

## Commands

```bash
venv/bin/python pipeline.py --count 3000        # fetch + process the next N never-seen dispatches
venv/bin/python pipeline.py --count 5 --dry-run # fetch + exclusion report only; writes nothing
venv/bin/python pipeline.py --skip-fetch        # resume the staged batch without touching the DB
venv/bin/python pipeline.py --stats             # ledger / case / growth totals, no processing

venv/bin/python fetch.py --count 3000           # DB collection only — stage the batch and stop
venv/bin/python process.py                      # process the staged batch — no DB access
pip install -r requirements.txt                 # Python 3.12; venv/ is present but gitignored
```

Three entry points, one implementation: `fetch.py`, `process.py`, and `pipeline.py` are all thin
CLIs over `pipelib/runner.py`, so `pipeline.py --count N` is exactly `fetch.py --count N` followed
by `process.py`. The split point is `state/batch_current.json` — put new orchestration in
`runner.py`, not in an entry point, or the three will drift.

There is **no test suite, no linter, and no CI** in this repo — don't invent commands for them. Paths are anchored to the repo root via `config.HERE`, so the CWD doesn't matter, but `.env` is only read from the repo root.

A real run needs both live dependencies: the Tailscale LLM gateway (chat + `nomic-embed`) and direct SQL access to FieldJetXStg. `--stats` needs neither beyond a valid `.env` (`pipelib/config.py` hard-exits at import naming any missing key).

## Live state — the thing to be careful about

`state/` (48M) and `output/` (8.8M) are **committed to git and are irreplaceable live data, not build artifacts**. They hold the append-only ledger, the case registry, the embedding cache, and the Chroma vector DB of every case. Current history: **6,009 dispatches in the ledger, 4,617 mapped, 821 cases**.

- Never `git checkout` / `reset --hard` / `stash` these paths casually — that rewinds case identity. Deleting them permanently resets `CASE-XXXX` numbering.
- `state/casemap.json` and `output/chroma_cases_extracted/` must stay in sync; `stage_casemap.reconcile()` repairs drift (deletes orphan vectors, restores missing ones from the embedding cache) and then **asserts** `chroma.count() == len(registry)`.
- **Casemap fingerprint guard**: `state/casemap.json` stores a fingerprint of `MODEL`, `EMBED_MODEL`, `EMBED_PREFIX`, `SIM_FLOOR`, `N_CANDIDATES`, and `sha256(prompt_casemap.txt)`. Changing any of them makes Stage C refuse to run against the existing case DB. This is deliberate — restore the previous config, or start a fresh case DB.
- The state error messages tell you to "run `migrate_casemap_state.py` first" — **that script is not in this repo** (it belongs to the original project folder and has already run). If those files are missing here, they need restoring from a backup, not regenerating.
- `.env` is gitignored but present, with 8 required keys: `API_KEY`/`BASE_URL`/`MODEL` (LLM gateway) and `DB_HOST`/`DB_PORT`/`DB_NAME`/`DB_USER`/`DB_PASSWORD` (FieldJetXStg). Config validation reports missing key *names* only, never values — keep it that way.

## Architecture

`pipelib/runner.py` orchestrates four `pipelib` stages over a staged batch, then finalizes; the entry points only parse arguments. All state writes go through `statefiles.save_json` (tmp + `os.replace`, atomic) and every stage checkpoints every `CHECKPOINT_EVERY = 10` items, so the process can be killed at any point and resumed by re-running the same command. `runner.check_state()` validates the ledger and the casemap fingerprint up front in **both** halves — a fingerprint mismatch must never surface only after a batch of LLM calls has already been paid for.

**Stage 0 — `stage_fetch`.** Selects newest-first candidates from `dbo.Dispatch` / `DispatchStatus` / `DispatchNotes` (parameterized pymssql; `%%` escapes LIKE wildcards), excluding the union of ledger IDs and already-mapped IDs, escalating the fetch multiplier 3× → 10× if too few are fresh. It writes `state/batch_current.json` — **the work order for the entire run. The DB is never touched again for that batch, which is what makes `process.py` / `--skip-fetch` DB-free and lets collection and processing run as separate commands.** Also appends display metadata to `state/dispatch_meta.json` for the cumulative reports.

**Stage A — `stage_notes`.** Classifies each note as technically useful or not (admin/scheduling/billing filtered out), 20 notes per LLM call. A dispatch is committed to `state/notes_class.json` **only when all of its chunks succeed** — partials stay pending and are retried next run.

**Stage B — `stage_extract`.** Pulls the *verbatim* root-cause sentences (never paraphrased) from the useful notes only → `state/extract.json`. Dispatches with more notes than `NOTE_CHUNK` are extracted per chunk and then consolidated by a second pass over the same prompt — a single oversized prompt would overflow context and never recover on retry.

**Stage C — `stage_casemap`.** Must stay **strictly sequential**: a case minted for dispatch *n* has to be visible to the query for dispatch *n+1*. For each dispatch: exact-duplicate cache → embed (sha256-keyed cache in `embcache.py`, npy written before the index so the index never references a missing row) → query top `N_CANDIDATES = 5` from the cosine Chroma collection `root_cause_cases` above `SIM_FLOOR = 0.60` → LLM judge decides same-component-and-same-failure-mode, answering `null` when uncertain (a wrong merge is worse than a duplicate case). Outcomes are recorded as `match_type`: `matched_exact`, `matched_llm`, `new_no_candidates`, `new_llm_rejected`, `new_first`, or `unresolved` (judge returned an invalid shape twice). A retry pass re-runs the unresolved ones. The `growth` series in `casemap.json` gets one `[resolved_dispatches, unique_cases]` point per resolved dispatch.

**Finalize.** Appends **terminal** statuses to `state/ledger.json` (`mapped` or `no_useful_notes`), regenerates all reports, archives the batch with its per-dispatch outcomes to `state/batches/<batch_id>.json`, and deletes `batch_current.json`.

### Ledger vs. casemap — why the counts differ

The ledger is the "never process this again" set and holds only terminal outcomes (6,009 = 4,617 mapped + 1,392 with no useful notes). Dispatches that **failed** a stage are deliberately never written to the ledger, so they stay eligible for a future batch — and because `notes_class.json` / `extract.json` are keyed globally by `DispatchId` (not per batch), their already-committed stage results are reused rather than recomputed.

### LLM gateway quirks (all already handled in `pipelib/llm.py`)

`MAX_WORKERS = 1` is **deliberate** — it was lowered from a higher value after recurring `ReadTimeout`s under concurrency; leave it at 1 unless concurrency has been re-tested against the specific gateway. `RateLimitError` sleeps 60s and retries indefinitely (60 RPM chat limit). `APITimeoutError` at `temperature=0` is a known repetition-loop hang: retried once at `temperature=0.4` with a token cap. `APIConnectionError` (Tailscale DNS blips) backs off 15s × attempt, up to 8 attempts. Embedding responses are re-sorted by `.index` because order isn't guaranteed.

### Reports (`reports.py`, regenerated from state after every batch)

`output/Dispatch_CaseMapped.xlsx` (one row per dispatch), `case_summary_extracted.xlsx` (one row per case with member count), `case_growth.xlsx` + `case_growth.png` (the growth series, plotted against a dashed "1 case per dispatch" reference diagonal). Strings are stripped of Excel-illegal control chars and truncated at `XLSX_CELL_LIMIT = 32000`.

## Conventions worth keeping

- Prompts are plain `.txt` at the repo root (`prompt_notes`, `prompt_extract`, `prompt_casemap`) with `{placeholder}` substitution — not f-strings in code. Editing `prompt_casemap.txt` trips the fingerprint guard.
- GUIDs are normalized with `config.norm_guid` (upper, stripped) at every boundary; note IDs additionally strip internal spaces.
- `RECEIVED_CUTOFF = "2026-07-01"` is hardcoded because the DB snapshot goes quiet after early June 2026.
- Stage functions are idempotent by design: each skips items already present in its state file, which is what lets a re-run be a no-op rather than a duplicate.
