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

venv/bin/python reset_state.py --dry-run        # preview a cold start; changes nothing
venv/bin/python reset_state.py --yes            # wipe progress, seed an empty catalog
venv/bin/python consolidate.py --dry-run        # preview near-duplicate case merges; changes nothing
venv/bin/python consolidate.py                  # apply confirmed merges, regenerate reports
venv/bin/python make_report.py                  # regenerate output/pipeline_report.html
pip install -r requirements.txt                 # Python 3.12; venv/ is present but gitignored
```

Three entry points, one implementation: `fetch.py`, `process.py`, and `pipeline.py` are all thin
CLIs over `pipelib/runner.py`, so `pipeline.py --count N` is exactly `fetch.py --count N` followed
by `process.py`. The split point is `state/batch_current.json` — put new orchestration in
`runner.py`, not in an entry point, or the three will drift.

There is **no test suite, no linter, and no CI** in this repo — don't invent commands for them. Paths are anchored to the repo root via `config.HERE`, so the CWD doesn't matter, but `.env` is only read from the repo root.

A real run needs both live dependencies: the Tailscale LLM gateway (chat + `nomic-embed`) and direct SQL access to FieldJetXStg. `--stats` needs neither beyond a valid `.env` (`pipelib/config.py` hard-exits at import naming any missing key).

## Live state — the thing to be careful about

`state/` and `output/` are **committed to git and are irreplaceable live data, not build artifacts**. They hold the append-only ledger, the case registry, the embedding cache, and the Chroma vector DB of every case.

The catalog was deliberately reset to empty (`reset_state.py --yes`, embedding cache kept) as part of shipping the fixes described below, so a fresh clone right now starts at `CASE-0001` with zero processed dispatches — check `pipeline.py --stats` for the true current count rather than trusting any number written here, since it will drift the moment the pipeline runs again.

- Never `git checkout` / `reset --hard` / `stash` these paths casually — that rewinds case identity. Deleting them permanently resets `CASE-XXXX` numbering.
- `state/casemap.json` and `output/chroma_cases_extracted/` must stay in sync; `stage_casemap.reconcile()` repairs drift (deletes orphan vectors, restores missing ones from the embedding cache) and then **asserts** `chroma.count() == total exemplar count` (one Chroma vector per exemplar, up to `MAX_EXEMPLARS_PER_CASE` per case — not one per case).
- **Casemap fingerprint guard**: `state/casemap.json` stores a fingerprint of `MODEL`, `EMBED_MODEL`, `EMBED_PREFIX`, `SIM_FLOOR`, `N_CANDIDATES`, and `sha256(prompt_casemap.txt)`. Changing any of them makes Stage C refuse to run against the existing case DB. This is deliberate — restore the previous config, or start a fresh case DB.
- The state error messages tell you to "run `migrate_casemap_state.py` first" — **that script is not in this repo** (it belonged to the original project folder). `ledger.load()` and `stage_casemap.load_state()` both `SystemExit` when their file is missing, so deleting state files does *not* give a working cold start. `reset_state.py` is the supported path: it clears progress and writes the two seed files those loaders require (`{"dispatches": {}}` and a v2 casemap with a config-derived fingerprint, `next_case_num: 1`). To recover an existing catalog instead, restore from git or a backup.
- Because `state/` is committed, **a fresh clone inherits all processed dispatches** and continues from the next unseen one. Starting over is always explicit.
- `.env` is gitignored but present, with 8 required keys: `API_KEY`/`BASE_URL`/`MODEL` (LLM gateway) and `DB_HOST`/`DB_PORT`/`DB_NAME`/`DB_USER`/`DB_PASSWORD` (FieldJetXStg). Config validation reports missing key *names* only, never values — keep it that way.

## Architecture

`pipelib/runner.py` orchestrates five `pipelib` stages over a staged batch, then finalizes; the entry points only parse arguments. All state writes go through `statefiles.save_json` (tmp + `os.replace`, atomic) and every stage checkpoints every `CHECKPOINT_EVERY = 10` items, so the process can be killed at any point and resumed by re-running the same command. `runner.check_state()` validates the ledger and the casemap fingerprint up front in **both** halves — a fingerprint mismatch must never surface only after a batch of LLM calls has already been paid for.

**Stage 0 — `stage_fetch`.** Selects newest-first candidates from `dbo.Dispatch` / `DispatchStatus` / `DispatchNotes` (parameterized pymssql; `%%` escapes LIKE wildcards), excluding the union of ledger IDs and already-mapped IDs, escalating the fetch multiplier 3× → 10× if too few are fresh. It writes `state/batch_current.json` — **the work order for the entire run. The DB is never touched again for that batch, which is what makes `process.py` / `--skip-fetch` DB-free and lets collection and processing run as separate commands.** Also appends display metadata to `state/dispatch_meta.json` for the cumulative reports.

**Stage A — `stage_notes`.** Classifies each note as technically useful or not (admin/scheduling/billing filtered out), 20 notes per LLM call. A dispatch is committed to `state/notes_class.json` **only when all of its chunks succeed** — partials stay pending and are retried next run.

**Stage B — `stage_extract`.** Pulls the *verbatim* root-cause sentences (never paraphrased) from the useful notes only → `state/extract.json`. Dispatches with more notes than `NOTE_CHUNK` are extracted per chunk and then consolidated by a second pass over the same prompt — a single oversized prompt would overflow context and never recover on retry.

**Stage B2 — `stage_diagnostic`.** Reads the short extracted summary (not the raw notes) and classifies whether it's a genuine diagnostic finding vs. non-diagnostic content that slipped through Stage A/B (installation work, parts logistics, warranty/billing disputes, scheduling) → `state/diagnostic.json`. Cheap enough to have been run retroactively over the entire historical `extract.json` backlog without re-running Stage B. Defaults to `diagnostic: true` when uncertain. Non-diagnostic dispatches skip Stage C entirely.

**Stage C — `stage_casemap`.** Must stay **strictly sequential**: a case minted for dispatch *n* has to be visible to the query for dispatch *n+1*. For each dispatch: exact-duplicate cache → embed (sha256-keyed cache in `embcache.py`, npy written before the index so the index never references a missing row) → query the cosine Chroma collection `root_cause_cases` for the nearest exemplar vectors (each case keeps up to `MAX_EXEMPLARS_PER_CASE = 5` representative texts, one Chroma id per exemplar as `CASE-XXXX#i`; results are grouped back to the best-matching **case**, not exemplar) → top `N_CANDIDATES = 5` cases above `SIM_FLOOR = 0.60` go to an LLM judge that decides same-component-and-same-failure-mode, answering `null` when uncertain (a wrong merge is worse than a duplicate case). Outcomes are recorded as `match_type`: `matched_exact`, `matched_llm`, `new_no_candidates`, `new_llm_rejected`, `new_first`, or `unresolved` (judge returned an invalid shape twice). A retry pass re-runs the unresolved ones. On a confirmed match, `_add_exemplar_if_useful()` may grow that case's exemplar set (capped, deduplicated by `EXEMPLAR_DEDUP_SIM = 0.985`). The `growth` series in `casemap.json` gets one `[resolved_dispatches, unique_cases]` point per resolved dispatch.

**Finalize.** Appends **terminal** statuses to `state/ledger.json` (`mapped`, `no_useful_notes`, or `non_diagnostic`), regenerates all reports, archives the batch with its per-dispatch outcomes to `state/batches/<batch_id>.json`, and deletes `batch_current.json`.

**`consolidate.py` / `stage_consolidate`** (separate from the `fetch`/`process` flow, no staged batch or DB access needed). An offline maintenance pass over the *finished* case catalog — Stage C never revisits a decision once made, so near-duplicate cases can accumulate across a growing catalog. Generates candidate case pairs via Chroma's own ANN index over exemplar vectors (`CONSOLIDATE_SIM_FLOOR = 0.80` pre-filter, no new embeddings), judges each pair with the *same* `_judge()` used online, and merges confirmed pairs with union-find so transitive chains collapse onto one survivor (more mapped dispatches wins; ties go to the lower case number). Reassigns affected dispatches in both `casemap.json` and `ledger.json`; appends every merge to `casemap.json["consolidation_log"]` for audit. Safe to run repeatedly.

### Ledger vs. casemap — why the counts differ

The ledger is the "never process this again" set and holds only terminal outcomes: `mapped`, `no_useful_notes`, or `non_diagnostic`. Dispatches that **failed** a stage are deliberately never written to the ledger, so they stay eligible for a future batch — and because `notes_class.json` / `extract.json` / `diagnostic.json` are keyed globally by `DispatchId` (not per batch), their already-committed stage results are reused rather than recomputed.

### LLM gateway quirks (all already handled in `pipelib/llm.py`, except one)

`MAX_WORKERS = 8` for Stages A/B/B2 (Stage C is a plain sequential loop by design, not a concurrency setting — see above). `RateLimitError` sleeps 60s and retries indefinitely (60 RPM chat limit). `APITimeoutError` at `temperature=0` is a known repetition-loop hang: retried once at `temperature=0.4` with a token cap. `APIConnectionError` (Tailscale DNS blips / DNS resolution failures) backs off 15s × attempt, up to 8 attempts. Embedding responses are re-sorted by `.index` because order isn't guaranteed.

A gateway-side `openai.InternalServerError` (HTTP 500, e.g. "Model Group=... Connection error" when the backend model itself is temporarily unavailable) was observed repeatedly during live testing and, before this fix, was not retried — if it hit Stage C's judge call, there was no per-dispatch handler around it (unlike Stage A/B) and it crashed the whole `process.py` run. `llm.py` now retries it with the same backoff as `APIConnectionError`. If it still exhausts all 8 attempts, Stage C has no per-dispatch fallback of its own, so `process.py` exits — safe to resume (checkpointed every 10 items, `finalize()` never ran so `batch_current.json` is untouched), just not automatic; re-run the same command.

### Reports (`reports.py`, regenerated from state after every batch)

`output/Dispatch_CaseMapped.xlsx` (one row per dispatch), `case_summary_extracted.xlsx` (one row per case with member count), `case_growth.xlsx` + `case_growth.png` (the growth series, plotted against a dashed "1 case per dispatch" reference diagonal). Strings are stripped of Excel-illegal control chars and truncated at `XLSX_CELL_LIMIT = 32000`.

## Conventions worth keeping

- Prompts are plain `.txt` at the repo root (`prompt_notes`, `prompt_extract`, `prompt_diagnostic`, `prompt_casemap`) with `{placeholder}` substitution — not f-strings in code. Editing `prompt_casemap.txt` trips the fingerprint guard; the others don't (they're outside Stage C's fingerprint scope), but changing them still only affects dispatches processed after the change — `notes_class.json`/`extract.json`/`diagnostic.json` entries already committed are never recomputed.
- GUIDs are normalized with `config.norm_guid` (upper, stripped) at every boundary; note IDs additionally strip internal spaces.
- `RECEIVED_CUTOFF = "2026-07-01"` is hardcoded because the DB snapshot goes quiet after early June 2026.
- Stage functions are idempotent by design: each skips items already present in its state file, which is what lets a re-run be a no-op rather than a duplicate.
