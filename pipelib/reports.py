"""Regenerate cumulative masters (xlsx) + growth chart (png) from state only."""
import re
from collections import Counter

import pandas as pd

from . import config
from .statefiles import load_json

_ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _clean(value):
    if isinstance(value, str):
        s = _ILLEGAL.sub("", value)
        if len(s) > config.XLSX_CELL_LIMIT:
            s = s[:config.XLSX_CELL_LIMIT] + " [TRUNCATED]"
        return s
    return value


def write_all(casemap_state):
    meta = load_json(config.DISPATCH_META_FILE, {})
    cases = casemap_state["cases"]
    dispatches = casemap_state["dispatches"]

    rows = []
    for did, rec in dispatches.items():
        m = meta.get(did, {})
        rows.append({
            "DispatchId": did,
            "DispatchNumber": m.get("dispatch_number", ""),
            "DispatchReason": m.get("reason", ""),
            "ReceivedDateTime": m.get("received_dt", ""),
            "NoteCount": m.get("note_count", ""),
            "CombinedNotes": m.get("combined_notes", ""),
            "RootCauseText": rec["text"],
            "CaseId": rec["case_id"],
            "CaseName": cases[rec["case_id"]]["canonical"] if rec["case_id"] else "",
            "Category": cases[rec["case_id"]]["category"] if rec["case_id"] else "",
            "MatchType": rec["match_type"],
        })
    mapped_df = pd.DataFrame(rows).map(_clean)
    mapped_df.to_excel(config.OUT_MAPPED, index=False)

    member_counts = Counter(r["case_id"] for r in dispatches.values() if r["case_id"])
    case_rows = []
    for case_id, case in cases.items():
        case_rows.append({
            "CaseId": case_id,
            "CaseName": case["canonical"],
            "MemberCount": member_counts.get(case_id, 0),
            "Category": case["category"],
            "CreatedFromDispatchId": case["dispatch_id"],
        })
    cases_df = pd.DataFrame(case_rows).sort_values("MemberCount", ascending=False)
    cases_df = cases_df.map(_clean)
    cases_df.to_excel(config.OUT_CASES, index=False)

    growth_df = pd.DataFrame(casemap_state["growth"],
                             columns=["DispatchesProcessed", "UniqueCases"])
    growth_df.to_excel(config.OUT_GROWTH_XLSX, index=False)

    _plot_growth(growth_df)
    print(f"Reports regenerated: {len(mapped_df)} mapped dispatches, "
          f"{len(cases_df)} cases, {len(growth_df)} growth points.")


def _plot_growth(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURFACE, SERIES = "#fcfcfb", "#2a78d6"
    GRID, BASELINE = "#e1e0d9", "#c3c2b7"
    MUTED, INK, SECONDARY = "#898781", "#0b0b0b", "#52514e"

    x, y = df["DispatchesProcessed"], df["UniqueCases"]
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    lim = x.max()
    ax.plot([0, lim], [0, lim], color=BASELINE, linewidth=1, linestyle=(0, (4, 4)), zorder=1)
    diag_t = y.max() * 0.72
    ax.annotate("1 case per dispatch", xy=(diag_t, diag_t), xytext=(6, 2),
                textcoords="offset points", color=MUTED, fontsize=9,
                rotation=59, rotation_mode="anchor", ha="left", va="bottom")

    ax.plot(x, y, color=SERIES, linewidth=2, zorder=3)
    fx, fy = int(x.iloc[-1]), int(y.iloc[-1])
    ax.scatter([fx], [fy], s=28, color=SERIES, zorder=4)
    ax.annotate(f"{fy} cases", xy=(fx, fy), xytext=(-8, 10),
                textcoords="offset points", ha="right", color=INK,
                fontsize=10, fontweight="bold")

    ax.set_xlim(0, lim * 1.04)
    ax.set_ylim(0, max(fy * 1.25, 50))
    ax.set_title("Unique root-cause cases vs dispatches processed",
                 color=INK, fontsize=13, loc="left", pad=14)
    ax.set_xlabel("Dispatches processed", color=SECONDARY, fontsize=10)
    ax.set_ylabel("Unique cases", color=SECONDARY, fontsize=10)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    fig.tight_layout()
    fig.savefig(config.OUT_GROWTH_PNG, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
