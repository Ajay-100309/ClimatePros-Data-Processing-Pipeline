"""
Reproduces every descriptive fact about the historical corpus cited in the
architecture doc and presentation script that scripts 01-06 don't already
cover: case-size distribution, the No Technical Fault share, parts-recording
sparsity, and the global part-frequency counts that scripts 03/04/06 use to
build their naive baseline (the "how many times is a part used across
history" question).

Two sources, read directly, not hand-copied:
  - output/Dispatch_CaseMapped.xlsx : the pipeline's own case-mapping file,
    955 cases / 11,189 dispatches under current matching logic.
  - case_parts_dataset.numbers (repo root) : cases joined to real
    FieldJetXStg parts data, built by a teammate's analysis (see the
    ReadMe sheet inside the file for exact provenance).

This script is also what caught a real error: an earlier draft of both
documents cited "about one in eight" dispatches landing in a
No-Technical-Fault cluster. That number came from a different, older,
now-superseded pipeline run (1,808 cases, pre-fix matching logic) and was
never re-checked against the 955-case file both documents actually cite.
The real figure, measured here, is 1,091/11,189 = 9.7%, closer to one in
ten. This script exists specifically so a mistake like that gets caught
by running code, not by trusting a number carried over from an earlier
conversation.

Run from the repo root (needs analysis/requirements.txt installed):
    venv/bin/python analysis/07_corpus_composition.py
"""
from pathlib import Path

import pandas as pd
from numbers_parser import Document

HERE = Path(__file__).resolve().parent


def load_numbers_sheet(path, sheet_name):
    doc = Document(str(path))
    for sheet in doc.sheets:
        if sheet.name == sheet_name:
            t = sheet.tables[0]
            header = [t.cell(0, c).value for c in range(t.num_cols)]
            rows = [[t.cell(r, c).value for c in range(t.num_cols)] for r in range(1, t.num_rows)]
            return pd.DataFrame(rows, columns=header)
    raise KeyError(f"sheet {sheet_name!r} not found in {path}")


def section(title):
    print()
    print(f"=== {title} ===")


def main():
    repo_root = HERE.parent

    section("Case-size distribution (output/Dispatch_CaseMapped.xlsx)")
    dcm = pd.read_excel(repo_root / "output" / "Dispatch_CaseMapped.xlsx")
    case_sizes = dcm["CaseId"].value_counts()
    n_cases = len(case_sizes)
    n_singleton = int((case_sizes == 1).sum())
    print(f"Total cases: {n_cases}")
    print(f"Singleton cases (exactly one dispatch): {n_singleton} "
          f"({n_singleton / n_cases * 100:.1f}% of cases)")
    print(f"Cited in the docs as: \"more than 500 of its 955 clusters hold exactly one dispatch\"")

    example_case = dcm[dcm["CaseId"] == "CASE-0417"]
    if len(example_case):
        row = example_case.iloc[0]
        print(f"\nCASE-0417 example: DispatchCount in file = "
              f"{(dcm['CaseId'] == 'CASE-0417').sum()}, "
              f"CaseName = {row.get('CaseName', row.get('RootCauseText', '?'))!r}")

    section("No Technical Fault share (same file)")
    n_no_fault = int((dcm["Category"] == "No Technical Fault").sum())
    n_total = len(dcm)
    print(f"No Technical Fault dispatches: {n_no_fault} / {n_total} "
          f"({n_no_fault / n_total * 100:.1f}%, roughly 1 in {n_total / n_no_fault:.0f})")
    print("This is the figure that corrects the earlier \"about one in eight\" claim, "
          "which came from a different, older pipeline run (1,808 cases) and did not "
          "match this file (955 cases) when re-checked here.")

    section("Parts-recording sparsity (case_parts_dataset.numbers)")
    cases = load_numbers_sheet(repo_root / "case_parts_dataset.numbers", "Cases")
    total_dispatches = cases["DispatchCount"].sum()
    total_with_parts = cases["DispatchesWithParts"].sum()
    print(f"Mapped dispatches: {int(total_dispatches)}")
    print(f"Dispatches with any part recorded (real or consumable): {int(total_with_parts)} "
          f"({total_with_parts / total_dispatches * 100:.1f}%, roughly 1 in "
          f"{total_dispatches / total_with_parts:.0f})")
    print('Cited in the docs as: "only one in five dispatches ... has any part recorded"')

    section("CASE-0012 example (case_parts_dataset.numbers, CasePartsLong sheet)")
    long_df = load_numbers_sheet(repo_root / "case_parts_dataset.numbers", "CasePartsLong")
    case_0012 = long_df[long_df["CaseId"] == "CASE-0012"]
    if len(case_0012):
        case_dispatches = case_0012["CaseDispatches"].iloc[0]
        leak = case_0012[case_0012["PartDesc"].astype(str).str.contains("LEAK DETECTOR", case=False)]
        if len(leak):
            row = leak.iloc[0]
            print(f"CASE-0012 has {int(case_dispatches)} dispatches. "
                  f"{row['PartDesc']} was used by {int(row['DispatchesUsingPart'])} of them "
                  f"({row['ShareOfCaseDispatches'] * 100:.1f}% share).")
        else:
            print(f"CASE-0012 has {int(case_dispatches)} dispatches (leak detector row not found "
                  f"in this pull; check CasePartsLong directly).")

    section("Global part-frequency count (dispatch_2k_rootcauses.numbers)")
    print("This is the exact computation scripts 03, 04, and 06 use to build the naive")
    print("popularity baseline and the popularity blend; shown here on its own, directly")
    print('answering "how many times is a part used across history."')
    dr = load_numbers_sheet(repo_root / "dispatch_2k_rootcauses.numbers", "Dispatches_RootCauses - Parts")

    def parse_parts(cell):
        if pd.isna(cell) or not str(cell).strip():
            return []
        return [seg.strip().split(" ")[0].strip() for seg in str(cell).split(";") if seg.strip()]

    dr["real_parts"] = dr["PartsUsed (part ×qty)"].apply(parse_parts)
    from collections import Counter
    freq = Counter()
    for parts in dr["real_parts"]:
        for part in sorted(set(parts)):
            freq[part] += 1
    n_with_real = sum(1 for p in dr["real_parts"] if len(p) > 0)
    print(f"Dispatches with a real recorded part in this file: {n_with_real} / {len(dr)}")
    print(f"Distinct real parts seen: {len(freq)}")
    print("Top 10 most-frequently-used real parts, by dispatch count:")
    for part, count in freq.most_common(10):
        print(f"  {part:<10} used on {count} dispatches ({count / n_with_real * 100:.1f}% "
              f"of dispatches with any real part)")


if __name__ == "__main__":
    main()
