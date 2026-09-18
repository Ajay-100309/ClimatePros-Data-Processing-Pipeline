# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A standalone, resumable LLM data-mining pipeline that reads HVAC/refrigeration service dispatches from the **FieldJetXStg** SQL Server (directly via pymssql — it bypasses both ClimatePros APIs), extracts each dispatch's verbatim root cause, and — **by default** — embeds it and pushes it, with the dispatch's recorded parts, into the Azure AI Search index serving the Parts Finder (`dispatches-nomic768-v1` on `cp-search-dev`). The original deduplicated catalog of canonical root-cause **cases** (`CASE-XXXX`) plus its growth series still runs in full behind `--with-cases`.

This is its own git repo, nested in the ClimatePros workspace (the workspace `CLAUDE.md` one level up calls this project `Dispatch/` and describes the surrounding FieldJetX apps/APIs — this is the trimmed deployment cut of it; the legacy `main_*.py` one-shot scripts and audit tooling are deliberately not here).

`README.md` is the operator-facing deployment guide and is accurate; `pipeline_architecture.html` is a standalone visual of the flow, the Stage C decision, and the thresholds; `parts-finder-system-design.html` is the Parts Finder design doc whose measured numbers the `analysis/` suite reproduces. Module docstrings in `pipelib/*.py` are authoritative on behavior; `analysis/README.md` is authoritative on the analysis suite.

## Commands

```bash
venv/bin/python pipeline.py --count 3000        # fetch + A/B + embed + push to Azure AI Search
venv/bin/python pipeline.py --count 3000 --with-cases   # also run Stage C + case reports (legacy)
venv/bin/python pipeline.py --count 5 --dry-run # fetch + exclusion report only; writes nothing
venv/bin/python pipeline.py --skip-fetch        # resume the staged batch without touching the DB
venv/bin/python pipeline.py --stats             # ledger / case / search-index totals, no processing

venv/bin/python fetch.py --count 3000           # DB collection only — stage the batch and stop
venv/bin/python fetch.py --plan-months --per-month 5000 --months 24   # build the seasonal fetch plan
venv/bin/python fetch.py --show-plan            # per-month quotas + progress, no DB
venv/bin/python fetch.py --next-month           # stage the oldest month still short of quota
venv/bin/python fetch.py --month 2025-02 --count 5000   # one specific month, spread evenly
venv/bin/python process.py                      # process the staged batch — no DB access
venv/bin/python process.py --with-cases         # same, plus Stage C

venv/bin/python create_search_index.py          # create the Azure index (--show / --update / --recreate --yes)
venv/bin/python backfill_search.py              # push processed history; rerunnable reconciler
venv/bin/python backfill_search.py --dry-run    # counts only, writes nothing
venv/bin/python backfill_search.py --refresh-parts --force-push   # after --update: re-pull parts + re-push all docs

venv/bin/python reset_state.py --dry-run        # preview a cold start; changes nothing
venv/bin/python reset_state.py --yes            # wipe progress, seed an empty catalog
venv/bin/python reset_state.py --yes --cold     # same, and drop the embedding cache too
venv/bin/python make_report.py                  # regenerate output/pipeline_report.html
pip install -r requirements.txt                 # Python 3.12; venv/ is present but gitignored
```

Three entry points, one implementation: `fetch.py`, `process.py`, and `pipeline.py` are all thin
CLIs over `pipelib/runner.py`, so `pipeline.py --count N` is exactly `fetch.py --count N` followed
by `process.py`. The split point is `state/batch_current.json` — put new orchestration in
`runner.py`, not in an entry point, or the three will drift.

There is **no test suite, no linter, and no CI** in this repo — don't invent commands for them. Paths are anchored to the repo root via `config.HERE`, so the CWD doesn't matter, but `.env` is only read from the repo root.

A real run needs three live dependencies: the Tailscale LLM gateway (chat + `nomic-embed`), direct SQL access to FieldJetXStg (fetch half only — `fetch.py` and `backfill_search.py`), and Azure AI Search (`AZURE_SEARCH_*` keys — optional at import, enforced by `config.require_search_config()` only in the commands that push). `--stats` needs none of them beyond a valid `.env` (`pipelib/config.py` hard-exits at import naming any missing required key).

## Live state — the thing to be careful about

`state/` and `output/` are **irreplaceable live data, not build artifacts** — the small, identity-critical files are committed to git, but four bulk items are deliberately untracked (GitHub's 100 MB blob limit): `state/embeddings.npy` + `embeddings_index.json` (content-addressed embedding cache — rebuildable via the gateway, `stage_index` re-embeds on demand), `state/dispatch_meta.json` (display metadata — re-queryable only from the DB), and `state/batches/` + `batch_current.json` (audit archives / transient work order). Move the untracked bulk between machines with `./sync_state.sh pull|push user@host [remote_repo_path]`. Together they hold the append-only ledger, the case registry, the embedding cache, the recorded-parts snapshot, the search push record, and the Chroma vector DB of every case. Check `pipeline.py --stats` for current totals (16,000 ledger dispatches / 1,808 cases / 12,744 indexed as of 2026-08-28).

- Never `git checkout` / `reset --hard` / `stash` these paths casually — that rewinds case identity. Deleting them permanently resets `CASE-XXXX` numbering.
- `state/casemap.json` and `output/chroma_cases_extracted/` must stay in sync; `stage_casemap.reconcile()` repairs drift (deletes orphan vectors, restores missing ones from the embedding cache) and then **asserts** `chroma.count() == len(registry)`.
- **Casemap fingerprint guard**: `state/casemap.json` stores a fingerprint of `MODEL`, `EMBED_MODEL`, `EMBED_PREFIX`, `SIM_FLOOR`, `N_CANDIDATES`, and `sha256(prompt_casemap.txt)`. Changing any of them makes Stage C refuse to run against the existing case DB. This is deliberate — restore the previous config, or start a fresh case DB. **The guard is consulted only in `--with-cases` mode** (`runner.check_state(with_cases=True)`); the default index mode never loads casemap state, so it cannot trip.
- `state/fetch_plan.json` (small, git-tracked) is the month-stratified budget: per-month `eligible`/`processed`/`available`/`quota`/`fetched`. Rebuild it with `fetch.py --plan-months` after a batch lands — `merge_progress` carries the `fetched` counters across, so re-planning against fresher ledger numbers never loses progress. Deleting it only loses `--next-month`; `--month YYYY-MM` works without a plan.
- **Stale work order**: `batch_current.json` is untracked, so a batch processed on another machine leaves a stale copy behind that would block every future fetch. `stage_fetch._clear_if_already_processed` removes it when `state/batches/<batch_id>.json` exists (sync the archives first with `./sync_state.sh pull`). `--dry-run` ignores the guard entirely, since it writes nothing.
- `state/parts.json` (recorded parts per fetched dispatch; an explicit `[]` means "checked, none recorded") and `state/search_index.json` (what was pushed to Azure, keyed by dispatch id + text sha) are the Stage D state files. Deleting `search_index.json` makes the next backfill re-push everything (harmless — upserts), but do it deliberately, e.g. right after `create_search_index.py --recreate`. The Azure index itself is NOT covered by `reset_state.py`.
- **The search index is versioned by name** (`dispatches-nomic768-v1`): vector dimensions are immutable on a live index, so an embedding-model change means a new `AZURE_SEARCH_INDEX` name + backfill rerun. The one allowed in-place edit is an **additive non-vector field**: `create_search_index.py --update` pushes the current definition onto the live index, then `backfill_search.py --refresh-parts --force-push` populates the new field on all existing documents (existing docs keep null for it until re-pushed). The embedding path for Stage D is the same `nomic-embed`/768/no-prefix as Stage C on purpose (shared cache; fingerprint untouched).
- The state error messages tell you to "run `migrate_casemap_state.py` first" — **that script is not in this repo** (it belonged to the original project folder). `ledger.load()` and `stage_casemap.load_state()` both `SystemExit` when their file is missing, so deleting state files does *not* give a working cold start. `reset_state.py` is the supported path: it clears progress and writes the two seed files those loaders require (`{"dispatches": {}}` and a v2 casemap with a config-derived fingerprint, `next_case_num: 1`). To recover an existing catalog instead, restore from git or a backup.
- Because the small state is committed, **a fresh clone inherits all processed dispatches** and continues from the next unseen one; starting over is always explicit. But run `./sync_state.sh pull user@server` before real work on a fresh clone — without the untracked bulk, Stage D re-embeds everything it touches (gateway cost) and re-pushed docs lose display fields (`dispatch_meta.json` cannot be rebuilt without DB access).
- `.env` is gitignored but present, with 8 required keys — `API_KEY`/`BASE_URL`/`MODEL` (LLM gateway) and `DB_HOST`/`DB_PORT`/`DB_NAME`/`DB_USER`/`DB_PASSWORD` (FieldJetXStg) — plus 3 optional search keys `AZURE_SEARCH_ENDPOINT`/`AZURE_SEARCH_API_KEY`/`AZURE_SEARCH_INDEX` (enforced only by pushing commands, so `--stats`/`make_report.py`/`reset_state.py` work without them). Config validation reports missing key *names* only, never values — keep it that way.

## Architecture

`pipelib/runner.py` orchestrates the `pipelib` stages over a staged batch, then finalizes; the entry points only parse arguments. All state writes go through `statefiles.save_json` (tmp + `os.replace`, atomic) and every stage checkpoints every `CHECKPOINT_EVERY = 10` items, so the process can be killed at any point and resumed by re-running the same command. `runner.check_state(with_cases)` validates the ledger up front in both halves; it additionally validates the casemap fingerprint **only when Stage C will run** — a fingerprint mismatch must never surface after a batch of LLM calls has been paid for, and `runner.process()` likewise calls `config.require_search_config()` before any LLM spend in the default mode.

**Stage 0 — `stage_fetch`.** Two selection strategies. The default, `_select`, takes **newest-first** candidates from `dbo.Dispatch` / `DispatchStatus` / `DispatchNotes` (parameterized pymssql; `%%` escapes LIKE wildcards), excluding the union of ledger IDs and already-mapped IDs, escalating the fetch multiplier 3× → 10× if too few are fresh. It writes `state/batch_current.json` — **the work order for the entire run. The DB is never touched again for that batch, which is what makes `process.py` / `--skip-fetch` DB-free and lets collection and processing run as separate commands.** `_select_month` (used when `--month YYYY-MM` / `--next-month` is given) instead pulls **one month's** candidates via the same SQL with per-month `dt_min`/`dt_max`, filters the exclusion set, and then picks at an **even stride over the date-sorted month** — excluding before spreading is what keeps the spacing even however much of the month is already processed. `state/fetch_plan.json` (built by `fetch.py --plan-months`, logic in `pipelib/monthplan.py`) holds per-month quotas: availability-capped with the shortfall redistributed across months that still have headroom, plus a `fetched` counter that survives re-planning. Batches record `strategy` and `month`. Also appends display metadata to `state/dispatch_meta.json` and each dispatch's recorded parts to `state/parts.json` (3-hop join `DispatchParts → InventoryLocationXREF → Inventory` — `DispatchParts` has no `InventoryId`; consumables kept but flagged). Each part also carries `united_part_no` — United Refrigeration's own catalog number from `InventorySupplierXREF` (empty when unmapped; ~10% of the active catalog but ~87% of recent real-part usage), the namespace the United inventory API requires — indexed as `parts/unitedPartNo`.

**Stage A — `stage_notes`.** Classifies each note as technically useful or not (admin/scheduling/billing filtered out), 20 notes per LLM call. A dispatch is committed to `state/notes_class.json` **only when all of its chunks succeed** — partials stay pending and are retried next run.

**Stage B — `stage_extract`.** Pulls the *verbatim* root-cause sentences (never paraphrased) from the useful notes only → `state/extract.json`. Dispatches with more notes than `NOTE_CHUNK` are extracted per chunk and then consolidated by a second pass over the same prompt — a single oversized prompt would overflow context and never recover on retry.

**Stage C — `stage_casemap`** (only with `--with-cases`). Must stay **strictly sequential**: a case minted for dispatch *n* has to be visible to the query for dispatch *n+1*. For each dispatch: exact-duplicate cache → embed (sha256-keyed cache in `embcache.py`, npy written before the index so the index never references a missing row) → query top `N_CANDIDATES = 5` from the cosine Chroma collection `root_cause_cases` above `SIM_FLOOR = 0.60` → LLM judge decides same-component-and-same-failure-mode, answering `null` when uncertain (a wrong merge is worse than a duplicate case). Outcomes are recorded as `match_type`: `matched_exact`, `matched_llm`, `new_no_candidates`, `new_llm_rejected`, `new_first`, or `unresolved` (judge returned an invalid shape twice). A retry pass re-runs the unresolved ones. The `growth` series in `casemap.json` gets one `[resolved_dispatches, unique_cases]` point per resolved dispatch.

**Stage D — `stage_index`** (both modes; never opens the DB). Every batch dispatch with useful notes and a non-empty extracted root cause is embedded through the shared `EmbCache` and upserted (`merge_or_upload`, `SEARCH_UPLOAD_BATCH = 500`) into the Azure index with its **real** parts only (`search_index.real_parts`: catalog number present, `NonPart = 0`, qty > 0, InventoryId resolved). Pushes are recorded in `state/search_index.json` incrementally (resume-safe; same-text re-pushes skipped); `write_popularity()` regenerates `output/part_popularity.json` (distinct-dispatch count per partNo — the same definition the query service computes via a facet query). Runs in cases mode too, so a `--with-cases` run can never leave a dispatch terminal-but-unindexed.

**Finalize.** Appends **terminal** statuses to `state/ledger.json` — default mode: `indexed`, `no_fault` (empty extracted root cause; never indexed), `no_useful_notes`; cases mode: `mapped` or `no_useful_notes`, unchanged. A failed push stays out of the ledger (`incomplete_stage_d`), eligible next run with the A/B results reused. Case reports regenerate only in cases mode. Archives the batch with per-dispatch outcomes to `state/batches/<batch_id>.json` and deletes `batch_current.json`.

### Ledger vs. casemap — why the counts differ

The ledger is the "never process this again" set and holds only terminal outcomes. Dispatches that **failed** a stage are deliberately never written to the ledger, so they stay eligible for a future batch — and because `notes_class.json` / `extract.json` are keyed globally by `DispatchId` (not per batch), their already-committed stage results are reused rather than recomputed.

### LLM gateway quirks (all already handled in `pipelib/llm.py`)

`MAX_WORKERS` was deliberately lowered to 1 after recurring `ReadTimeout`s under concurrency, but **the code currently says 8** — an undocumented change that has not been re-verified against the gateway; if Stage A/B hit `ReadTimeout`s, set it back to 1. `RateLimitError` sleeps 60s and retries indefinitely (60 RPM chat limit). `APITimeoutError` at `temperature=0` is a known repetition-loop hang: retried once at `temperature=0.4` with a token cap. `APIConnectionError` (Tailscale DNS blips) backs off 15s × attempt, up to 8 attempts. Embedding responses are re-sorted by `.index` because order isn't guaranteed.

### Reports (`reports.py`, regenerated only after `--with-cases` batches)

`output/Dispatch_CaseMapped.xlsx` (one row per dispatch), `case_summary_extracted.xlsx` (one row per case with member count), `case_growth.xlsx` + `case_growth.png` (the growth series, plotted against a dashed "1 case per dispatch" reference diagonal). Strings are stripped of Excel-illegal control chars and truncated at `XLSX_CELL_LIMIT = 32000`.

## Conventions worth keeping

- Prompts are plain `.txt` at the repo root (`prompt_notes`, `prompt_extract`, `prompt_casemap`) with `{placeholder}` substitution — not f-strings in code. Editing `prompt_casemap.txt` trips the fingerprint guard.
- GUIDs are normalized with `config.norm_guid` (upper, stripped) at every boundary; note IDs additionally strip internal spaces.
- `RECEIVED_CUTOFF = "2026-07-01"` is hardcoded because the DB snapshot goes quiet after early June 2026.
- Stage functions are idempotent by design: each skips items already present in its state file, which is what lets a re-run be a no-op rather than a duplicate.
- `make_report.py` hardcodes `TOTAL_ELIGIBLE = 1_551_773` (eligible `dbo.Dispatch` rows under the pipeline's candidate filters, measured 2026-08-14) for its coverage figures — recompute via `db.CAND_SQL` wrapped in `COUNT(*)` if the DB snapshot is ever refreshed.

## Analysis suite (`analysis/`) — the evidence behind the Parts Finder design

Nine numbered scripts that reproduce every measured number in `parts-finder-system-design.html`. `analysis/README.md` is authoritative for setup, run order, and the reasoning trail of scripts 01–08; `09_azure_parity.py` postdates that README — it validates the **live** Azure index (leave-one-out hit@6 with the blended scoring formula, plus HNSW-vs-exact recall) and must run only after `backfill_search.py` has completed.

```bash
venv/bin/pip install -r analysis/requirements.txt   # adds numbers-parser + hnswlib
venv/bin/python analysis/01_dispatch_reason_quality.py   # …through 09; run from the repo root
```

- All read-only: none touch FieldJetXStg or write into `state/`. The only gateway callers are `08_raw_vs_cleaned_embedding.py` (first run only; caches to the committed `analysis/raw_embed_cache.{json,npy}`) and 09's queries against Azure (needs `AZURE_SEARCH_*`; vectors come from the shared cache).
- Scripts 03/04/06/07/08 read `dispatch_2k_rootcauses.numbers` / `case_parts_dataset.numbers` expected at the repo root — **those files are not in this repo** (they belong to the original project folder) and the scripts cannot run without them. 01/02/05/09 run from committed state/output alone.
- `05_scale_benchmark.py`'s full run takes hours and holds its 1,551,773-vector HNSW index in memory only; `05_scale_benchmark_output_1551773.txt` is the saved log of the one real run the design's latency numbers come from.
