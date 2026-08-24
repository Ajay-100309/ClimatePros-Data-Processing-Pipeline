# Parts Finder feasibility analysis

Eight scripts. Together they reproduce every measured number in the Parts
Finder System Design architecture doc and presentation script, in the
order the actual reasoning happened, not just a flat list of outputs. Run
them in order and you get the same evidence trail that produced the final
design, including the places where a first assumption turned out wrong
and got corrected.

Seven of the eight are read-only against files already in this repo or
its root, and never call an LLM. The eighth, `08_raw_vs_cleaned_embedding.py`,
is the one exception: on its first run it calls the live embedding
gateway to embed roughly 2,000 text values that had never been embedded
before, then caches the result locally so every later run is gateway-free.
None of the eight touch the database, and none write into `state/`.

## Setup

Scripts 01 and 02 need nothing beyond this repo's existing venv
(`pandas`, `numpy`, `openpyxl` are already in the top-level
`requirements.txt`). Scripts 03, 04, 06, 07, and 08 read `.numbers` files
directly and need `numbers-parser`. Script 05 needs `hnswlib`, a real
HNSW vector-index library, for the scale benchmark. Script 08 reuses
`pipelib.llm`, which needs `openai` (already in the top-level
`requirements.txt`) and a working `.env` for its first run only.

```bash
venv/bin/pip install -r analysis/requirements.txt
```

## Run, in order

```bash
venv/bin/python analysis/01_dispatch_reason_quality.py
venv/bin/python analysis/02_case_grouping_test.py
venv/bin/python analysis/03_parts_prediction_accuracy.py
venv/bin/python analysis/04_flat_vs_case_retrieval.py
venv/bin/python analysis/05_scale_benchmark.py
venv/bin/python analysis/06_blended_scoring.py
venv/bin/python analysis/07_corpus_composition.py
venv/bin/python analysis/08_raw_vs_cleaned_embedding.py
```

Run from the repo root. All paths inside the scripts are resolved
relative to it. Script `05_scale_benchmark.py`'s full run takes several
hours on modest hardware (see its own docstring) and builds its index
in memory only, discarding it on exit; `05_scale_benchmark_output_1551773.txt`
is the saved log from the one full-scale run this doc's timing numbers
come from. Script `08_raw_vs_cleaned_embedding.py` takes a couple of
minutes on its first run (one round of live embedding calls) and seconds
after that. Everything else runs in seconds to a couple of minutes and
produces bit-identical output every time.

## The reasoning, in the order it actually happened

The scripts are numbered by when each one was written, which is not the
same order the underlying questions came up in conversation. This section
walks the real order, each point labelled with the script that answers
it, so the numbering mismatch never has to be untangled by a reader.

**A. What does the underlying data actually look like, before assuming
anything about it?** `07_corpus_composition.py`. Case sizes, the
No-Technical-Fault share, and parts-recording sparsity, each pulled
straight from source with a named example row, not asserted from memory.

**B. Is the case-clustering pipeline's own output good enough evidence to
use directly for parts prediction?** No, on its own: over half its cases
are singletons, and a meaningful share of the corpus lands in a case with
no fault found at all (`07_corpus_composition.py`, case-size distribution).

**C. Is the raw dispatch-reason field usable as-is, without an LLM
cleanup pass?** Mostly yes, once actually checked instead of assumed:
83% of 2,000 real records carry no known non-descriptive pattern
(`01_dispatch_reason_quality.py`).

**D. If a new dispatch's nearest neighbours already have a case
assigned, does grouping by that case actually concentrate signal?** Yes,
and it scales exactly as expected, strong for common problems, weak for
rare ones (`02_case_grouping_test.py`). This result is what originally
justified routing predictions through the case pipeline.

**E. Does that case-based design actually predict the right part, against
real outcomes, not just retrieval behaviour?** This is the first real
accuracy test, and it's the one that mattered most: case-matched
evidence scored 39.6% top-6 hit rate against 265 real labeled dispatches,
beating a naive "most common parts" guess (36.2%) and a pure similarity
score with no case grouping at all (27.2%) (`03_parts_prediction_accuracy.py`).

**F. Given step E made the case pipeline look necessary, is it actually
necessary, or is 10 matches just too few?** This is the question that
changed the architecture. Retrieving 100 or 200 matches directly, with
no case step at all, beat the case-matched design outright, 42.3% and
44.2% respectively (`04_flat_vs_case_retrieval.py`). The case pipeline
was dropped from the live design because of this result, not an opinion.

**G. Does removing the case pipeline lose anything real, specifically a
part that's common across different root causes rather than tied to one
description?** Tested directly rather than assumed away: blending in
how often a part is used across all of history, alongside the top-100
search, recovers that signal, reaching 44.2-44.5%, matching the cost of
retrieving 200 matches without actually retrieving 200
(`06_blended_scoring.py`).

**H. Retrieving 100-200 matches instead of 10, at the real 1,551,773-
dispatch production scale, how much does that actually cost?** Real
numbers, not an assumption: 122ms at top-10, 348ms at top-100, 601ms at
top-200, measured with a real HNSW index built at full scale
(`05_scale_benchmark.py`). This script is also what caught an earlier
wrong claim in the architecture doc, that retrieval cost was close to
flat regardless of how many matches were requested. It isn't.

Running `07_corpus_composition.py` again, after the fact, is also what
caught a stale claim that had briefly existed earlier ("about one in
eight" dispatches with no fault found, carried over from an older,
superseded pipeline run) no longer matching the 955-case file the
documents actually cite. That claim had already been removed from both
documents by the time this check confirmed it; the script stays in
`analysis/` as the check that would catch it again if it came back.

**I. The design embeds the cleaned root-cause extraction, not the raw
dispatch-reason field, for the historical corpus. Does that actually
matter, or would raw text have worked just as well?** Tested directly
rather than left as an assumption: embedding the raw DispatchReason field
instead scored 12.5% to 28.7% (top-6 hit rate, top-10 through top-200),
against 27.2% to 44.2% for the cleaned extraction on the same dispatches,
the same model, the same retrieval code. The two embeddings agree on only
6.8% of their top-100 matches for the same query, on average
(`08_raw_vs_cleaned_embedding.py`). Cleaning the text first is not a
convenience carried over from the case pipeline; it is most of where this
design's accuracy comes from.

## What each script answers, at a glance

| Script | Question | Data source | Headline result |
|---|---|---|---|
| `01_dispatch_reason_quality.py` | Is the raw `DispatchReason` field usable on its own? | `state/dispatch_meta.json` (2,000 real staged dispatches) | 83.3% carry no known non-descriptive pattern |
| `02_case_grouping_test.py` | Does grouping retrieved neighbours by their existing case concentrate signal? | `output/Dispatch_CaseMapped.xlsx` + `state/embeddings.npy` | 55.3% well-supported (>=5/10 agree), scaling from 20% for one-off cases to 69% for common ones |
| `03_parts_prediction_accuracy.py` | Does the case-matched design actually predict the right part? | `dispatch_2k_rootcauses.numbers` + `state/embeddings.npy` | Case-aware design: 39.6% top-6 hit rate, against 36.2% naive and 27.2% for similarity alone |
| `04_flat_vs_case_retrieval.py` | Is the case pipeline actually needed, or does a wider direct search beat it? | Same as script 03 | Direct search, no case step: 42.3% (top 100) to 44.2% (top 200), beating 39.6% |
| `05_scale_benchmark.py` | How fast is retrieval at the real 1,551,773-dispatch scale? | Synthetic vectors at real dimensionality + a real HNSW index (hnswlib) | 122ms / 348ms / 601ms at top-10 / top-100 / top-200; corrected an earlier "cost stays flat" claim |
| `06_blended_scoring.py` | Does a part common across *different* root causes get missed by narrow similarity search? | Same as script 03 | Blending in global part frequency recovers it: 44.2-44.5%, matching top-200 at top-100's cost |
| `07_corpus_composition.py` | What does the underlying data look like, with named examples for every summary stat? | `output/Dispatch_CaseMapped.xlsx` + `case_parts_dataset.numbers` + `dispatch_2k_rootcauses.numbers` | 531/955 cases are singletons; 9.8% of dispatches are "No Technical Fault"; 21.2% of case_parts_dataset's mapped dispatches have any part recorded; top real parts by frequency |
| `08_raw_vs_cleaned_embedding.py` | Does embedding the raw dispatch-reason field instead of the cleaned extraction change accuracy? | Same as script 03, plus a live embedding call for the raw field | Cleaned text: 27.2-44.2% (top-10 to top-200). Raw text: 12.5-28.7%. Only 6.8% neighbour overlap between the two |

## Caveats these scripts do not hide

- Scripts 02-04 and 06 embed with `nomic-embed`, because that's the
  model this repo has cached. Production is specified to use `bge-m3`.
  Re-run once that's live before trusting the exact percentages.
- Script 05's absolute timing numbers were measured on an 8-core, 8GB
  local test machine, not Azure's managed infrastructure, and show signs
  of hitting a memory ceiling above roughly 1 million vectors (see its
  own output and docstring). The *shape* of the result, retrieval cost
  scaling with how many matches you ask for, is real; the exact
  milliseconds need confirming on real Azure AI Search infrastructure.
- Script 03/04/06's sample is 265 dispatches with a genuinely recorded
  part, out of 2,146 in a snapshot pulled from FieldJetX. Directionally
  informative, not a final production number.
- Case IDs are not stable across different runs of the case-clustering
  pipeline (this repo's own README says as much). `07_corpus_composition.py`
  reads case examples directly from whichever file a claim is about,
  rather than reusing an example across files, specifically to avoid
  repeating the kind of mismatch described in step H above.
- All eight scripts are read-only against files already in this repo.
  None touch FieldJetXStg or write into `state/`. Seven never call the
  LLM gateway; `08_raw_vs_cleaned_embedding.py` does, once, to embed text
  that had no cached vector anywhere in this repo, and caches the result
  locally afterward.
- `08_raw_vs_cleaned_embedding.py`'s "raw" field is the DispatchReason
  column in dispatch_2k_rootcauses.numbers, a long ticketing-system text
  block (job type, tracking number, NTE amount, OnCallRegion tag, and the
  actual complaint, concatenated). It is not the same field as
  `state/dispatch_meta.json`'s short `reason` field that
  `01_dispatch_reason_quality.py` checked, the one a technician's live
  query is expected to resemble. This script's result says the historical
  corpus should embed cleaned text, not that a live query needs cleaning
  too, those are two different fields and two separate decisions.
- `05_scale_benchmark.py` never writes its index to disk. The
  1,551,773-vector HNSW index it builds exists only in the process's
  memory for the ~3h41m the script runs, then is discarded when it
  exits, whether or not the run succeeded. `05_scale_benchmark_output_1551773.txt`
  is the saved console log from the one full-scale run behind this doc's
  122/348/601ms numbers; it's the only record of that run, not a
  reloadable index. Re-running the script rebuilds a fresh index from the
  same seeded random vectors (bit-identical data, since the seed is
  fixed) and will land in the same ballpark on similar hardware, but not
  at exactly the same millisecond, because it's a live timing
  measurement, not a stored result.
