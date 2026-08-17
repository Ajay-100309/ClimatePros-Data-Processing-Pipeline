"""Build the HTML data report from pipeline state.

Reads state/ only — no database, no LLM, no network — so it can be re-run after
any batch to refresh output/pipeline_report.html.

Usage:
    venv/bin/python make_report.py
"""
import os
import json
import glob
from collections import Counter
from datetime import datetime, timezone

import numpy as np

from pipelib import config
from pipelib.statefiles import load_json

# Eligible rows in dbo.Dispatch under the pipeline's own candidate filters
# (IsConstruction=0, dated, reason present, not cancelled, >=1 note over
# MIN_NOTE_LEN). Measured 2026-08-14; recompute with db.CAND_SQL wrapped in a
# COUNT(*) if the snapshot is ever refreshed.
TOTAL_ELIGIBLE = 1_551_773

OUT = os.path.join(config.OUTPUT_DIR, "pipeline_report.html")

C_MATCH = "#1a56b0"    # existing case  (repo --accent-dup)
C_NEW = "#a8480a"      # new case       (repo --accent-new)
RAMP = ["#86b6ef", "#3987e5", "#256abf", "#184f95"]  # validated ordinal ramp
GRID, AXIS, MUTED, INK = "#dedcd3", "#c3c2b7", "#78766f", "#14140f"


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def fmt(n):
    return f"{n:,}"


# ----------------------------------------------------------------- metrics


def collect():
    cm = load_json(config.CASEMAP_FILE)
    led = load_json(config.LEDGER_FILE)
    nc = load_json(config.NOTES_CLASS_FILE, {})
    m = {}

    cases, disp = cm["cases"], cm["dispatches"]
    m["cases"], m["disp"] = cases, disp
    m["growth"] = np.array(cm["growth"], dtype=float)
    m["n_processed"] = len(led["dispatches"])
    m["status"] = Counter(r["status"] for r in led["dispatches"].values())
    m["n_cases"], m["n_mapped"] = len(cases), len(disp)
    m["match_types"] = Counter(r["match_type"] for r in disp.values())

    # per-case membership and the long tail
    members = Counter(r["case_id"] for r in disp.values() if r["case_id"])
    m["members"] = members
    m["singletons"] = sum(1 for v in members.values() if v == 1)
    m["coverage"] = [(n, sum(v for _, v in members.most_common(n)))
                     for n in (10, 25, 50, 100, 200, 500)]

    # categories, weighted by dispatches rather than by case
    cats = Counter()
    for r in disp.values():
        if r["case_id"]:
            cats[cases[r["case_id"]]["category"]] += 1
    m["cats"] = cats
    m["n_cats"] = len(cats)
    m["cats_thin"] = sum(1 for v in cats.values() if v <= 10)

    # the placeholder category — dispatches that mapped but named no real fault
    m["no_fault"] = sum(1 for r in disp.values() if r["case_id"]
                        and cases[r["case_id"]]["category"] == "No Technical Fault")
    m["diagnostic"] = m["n_mapped"] - m["no_fault"]

    # note-level funnel
    m["notes_total"] = sum(len(v) for v in nc.values())
    m["notes_useful"] = sum(1 for v in nc.values() for r in v if r["useful"])
    m["notes_disp"] = len(nc)

    # marginal new-case rate per 1,000 resolved dispatches
    g, step, bins = m["growth"], 1000, []
    pd = pc = 0
    for i in range(step, len(g) + 1, step):
        d, c = g[i - 1]
        bins.append((int(pd), int(d), int(c - pc), (c - pc) / (d - pd)))
        pd, pc = d, c
    if len(g) % step:
        d, c = g[-1]
        bins.append((int(pd), int(d), int(c - pc), (c - pc) / (d - pd)))
    m["bins"] = bins

    # Headline rate over the trailing 3,000 resolved dispatches. The final bin is
    # usually a partial block of a few hundred, far too noisy to headline.
    tail = min(3000, len(g) - 1)
    d0, c0 = g[-1 - tail]
    d1, c1 = g[-1]
    m["recent_rate"] = (c1 - c0) / (d1 - d0)
    m["recent_span"] = int(d1 - d0)

    # Heaps' law: C = K * N^beta, the standard model for vocabulary growth.
    # Fit on the recent regime too — the opening 1,000 dispatches are a
    # cold-start artifact (every case is new when the catalog is empty).
    N, Cs = g[:, 0], g[:, 1]
    sel = N >= 100
    beta, a = np.polyfit(np.log(N[sel]), np.log(Cs[sel]), 1)
    K = np.exp(a)
    pred = K * N ** beta
    m["heaps"] = (K, beta, 1 - ((Cs - pred) ** 2).sum() / ((Cs - Cs.mean()) ** 2).sum())
    sel2 = N >= 6000
    b2, a2 = np.polyfit(np.log(N[sel2]), np.log(Cs[sel2]), 1)
    m["heaps_recent"] = (np.exp(a2), b2)

    m["batches"] = []
    for p in sorted(glob.glob(os.path.join(config.BATCH_ARCHIVE_DIR, "*.json"))):
        b = json.load(open(p))
        oc = Counter("mapped" if v.startswith("mapped") else v
                     for v in b.get("outcomes", {}).values())
        if b["dispatches"]:
            m["batches"].append((b["batch_id"], len(b["dispatches"]), oc))
    return m


# ------------------------------------------------------------------ charts


def nice_ticks(vmax, counts=(4, 5, 6)):
    """Round tick values covering vmax, so axes read 15,000 not 13,284.

    Tries several tick counts and keeps whichever overshoots least — a fixed
    count leaves a third of the plot empty whenever vmax lands just past a step.
    """
    best = None
    for n in counts:
        rough = vmax / n
        mag = 10 ** int(np.floor(np.log10(rough)))
        step = next(x for x in (1, 2, 2.5, 5, 10) if rough / mag <= x) * mag
        top = step * n
        if top < vmax * 1.04:  # keep a little air above the final point
            continue
        if best is None or top < best[1]:
            best = ([i * step for i in range(n + 1)], top)
    return best


def chart_growth(m):
    """Cases vs dispatches, against the no-deduplication diagonal."""
    W, H = 1100, 430
    L, R, T, B = 66, 30, 30, 46
    g = m["growth"]
    xt, xmax = nice_ticks(g[-1][0])
    yt, ymax = nice_ticks(g[-1][1])
    px = lambda v: L + v / xmax * (W - L - R)
    py = lambda v: H - B - v / ymax * (H - B - T)

    idx = np.linspace(0, len(g) - 1, min(len(g), 420)).astype(int)
    pts = " ".join(f"{px(g[i][0]):.1f},{py(g[i][1]):.1f}" for i in idx)

    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Unique cases against '
         f'dispatches processed. The curve bends well below the one-case-per-dispatch '
         f'diagonal, ending at {fmt(int(g[-1][1]))} cases for {fmt(int(g[-1][0]))} dispatches.">']
    for v in yt:
        y = py(v)
        s.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W-R}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        s.append(f'<text x="{L-10}" y="{y+4:.1f}" text-anchor="end" font-size="12" '
                 f'fill="{MUTED}" style="font-variant-numeric:tabular-nums">{int(v):,}</text>')
    for v in xt:
        s.append(f'<text x="{px(v):.1f}" y="{H-B+22}" text-anchor="middle" font-size="12" '
                 f'fill="{MUTED}" style="font-variant-numeric:tabular-nums">{int(v):,}</text>')

    # the 1:1 reference — what zero deduplication would look like. It leaves the
    # top of the plot early, so the label rides just inside that exit point.
    dx = min(xmax, ymax)
    s.append(f'<line x1="{px(0)}" y1="{py(0)}" x2="{px(dx):.1f}" y2="{py(dx):.1f}" '
             f'stroke="{AXIS}" stroke-width="1" stroke-dasharray="5 5"/>')
    lx, ly = px(dx * 0.62), py(dx * 0.62)
    ang = np.degrees(np.arctan2(py(dx) - py(0), px(dx) - px(0)))
    s.append(f'<text x="{lx:.1f}" y="{ly-8:.1f}" font-size="12" fill="{MUTED}" '
             f'transform="rotate({ang:.1f} {lx:.1f} {ly-8:.1f})">1 new case per dispatch</text>')

    s.append(f'<polyline points="{pts}" fill="none" stroke="{C_MATCH}" stroke-width="2" '
             f'stroke-linejoin="round"/>')
    fx, fy = px(g[-1][0]), py(g[-1][1])
    s.append(f'<circle cx="{fx:.1f}" cy="{fy:.1f}" r="4.5" fill="{C_MATCH}" '
             f'stroke="#ffffff" stroke-width="2"/>')
    s.append(f'<text x="{fx-12:.1f}" y="{fy-12:.1f}" text-anchor="end" font-size="13" '
             f'font-weight="600" fill="{INK}">{int(g[-1][1]):,} cases</text>')
    s.append(f'<line x1="{L}" y1="{py(0)}" x2="{W-R}" y2="{py(0)}" stroke="{AXIS}" stroke-width="1"/>')
    s.append(f'<text x="{(L+W-R)/2:.0f}" y="{H-4}" text-anchor="middle" font-size="12" '
             f'fill="{MUTED}">Dispatches with a resolved root cause</text>')
    s.append("</svg>")
    return "".join(s)


def chart_rate(m):
    """Marginal new-case rate per 1,000 dispatches."""
    W, H = 1100, 320
    L, R, T, B = 62, 28, 20, 60
    bins = m["bins"]
    yt, ytop = nice_ticks(max(b[3] for b in bins) * 100)   # ticks in percent
    ymax = ytop / 100
    bw = (W - L - R) / len(bins)
    py = lambda v: H - B - v / ymax * (H - B - T)

    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="New-case rate per 1,000 '
         f'dispatches, falling from {bins[0][3]*100:.0f} percent in the first block to '
         f'about {bins[-1][3]*100:.0f} percent, then holding roughly flat.">']
    for v in yt:
        y = py(v / 100)
        s.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W-R}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        s.append(f'<text x="{L-10}" y="{y+4:.1f}" text-anchor="end" font-size="12" '
                 f'fill="{MUTED}">{v:.0f}%</text>')
    for i, (d0, d1, n, rate) in enumerate(bins):
        x = L + i * bw + 3
        w = bw - 6
        y = py(rate)
        h = py(0) - y
        # the opening block is the cold-start artifact; keep it in a muted tone
        fill = MUTED if i == 0 else C_NEW
        s.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="4" '
                 f'fill="{fill}"><title>{d0:,}–{d1:,}: {n} new cases ({rate*100:.1f}%)</title></rect>')
        s.append(f'<text x="{x+w/2:.1f}" y="{y-7:.1f}" text-anchor="middle" font-size="11.5" '
                 f'fill="{INK}" style="font-variant-numeric:tabular-nums">{rate*100:.1f}%</text>')
        # the final block is usually partial — give it a decimal so it does not
        # print the same "13k" label as the full block before it
        lab = f"{d1/1000:.0f}k" if d1 % 1000 == 0 else f"{d1/1000:.1f}k"
        s.append(f'<text x="{x+w/2:.1f}" y="{H-B+20}" text-anchor="middle" font-size="11" '
                 f'fill="{MUTED}" style="font-variant-numeric:tabular-nums">{lab}</text>')
    s.append(f'<line x1="{L}" y1="{py(0)}" x2="{W-R}" y2="{py(0)}" stroke="{AXIS}" stroke-width="1"/>')
    s.append(f'<text x="{(L+W-R)/2:.0f}" y="{H-8}" text-anchor="middle" font-size="12" '
             f'fill="{MUTED}">Cumulative dispatches processed (blocks of 1,000)</text>')
    s.append("</svg>")
    return "".join(s)


def chart_bars(rows, total, color=C_MATCH, width=1100, label_w=250):
    """Horizontal magnitude bars — one hue, since this is one series."""
    rowh, T = 30, 8
    H = T + rowh * len(rows) + 10
    xmax = max(v for _, v in rows)
    plot = width - label_w - 145  # right margin holds the "1,673 · 12.6%" label
    s = [f'<svg viewBox="0 0 {width} {H}" role="img" aria-label="Ranked bar chart.">']
    for i, (name, v) in enumerate(rows):
        y = T + i * rowh
        w = v / xmax * plot
        s.append(f'<text x="{label_w-12}" y="{y+15}" text-anchor="end" font-size="13" '
                 f'fill="{INK}">{esc(name)}</text>')
        s.append(f'<rect x="{label_w}" y="{y+3}" width="{max(w,2):.1f}" height="16" rx="4" '
                 f'fill="{color}"><title>{esc(name)}: {v:,} dispatches ({v/total*100:.1f}%)</title></rect>')
        s.append(f'<text x="{label_w+w+10:.1f}" y="{y+16}" font-size="12.5" fill="{MUTED}" '
                 f'style="font-variant-numeric:tabular-nums">{v:,}  ·  {v/total*100:.1f}%</text>')
    s.append("</svg>")
    return "".join(s)


def chart_funnel(stages, width=1100):
    """Ordered stages — ordinal ramp, darkening as the funnel narrows."""
    rowh, T = 46, 10
    H = T + rowh * len(stages) + 8
    top = stages[0][1]
    label_w, plot = 300, width - 300 - 150
    s = [f'<svg viewBox="0 0 {width} {H}" role="img" aria-label="Processing funnel from '
         f'dispatches fetched down to dispatches naming a real technical fault.">']
    for i, (name, v, note) in enumerate(stages):
        y = T + i * rowh
        w = v / top * plot
        col = RAMP[min(i, len(RAMP) - 1)]
        s.append(f'<text x="{label_w-14}" y="{y+18}" text-anchor="end" font-size="13" '
                 f'font-weight="600" fill="{INK}">{esc(name)}</text>')
        s.append(f'<text x="{label_w-14}" y="{y+34}" text-anchor="end" font-size="11.5" '
                 f'fill="{MUTED}">{esc(note)}</text>')
        s.append(f'<rect x="{label_w}" y="{y+6}" width="{w:.1f}" height="26" rx="4" fill="{col}">'
                 f'<title>{esc(name)}: {v:,} ({v/top*100:.1f}% of fetched)</title></rect>')
        s.append(f'<text x="{label_w+w+12:.1f}" y="{y+24}" font-size="13" fill="{INK}" '
                 f'style="font-variant-numeric:tabular-nums">{v:,}</text>')
        s.append(f'<text x="{label_w+w+12:.1f}" y="{y+38}" font-size="11" fill="{MUTED}" '
                 f'style="font-variant-numeric:tabular-nums">{v/top*100:.1f}%</text>')
    s.append("</svg>")
    return "".join(s)


def chart_split(m, width=1100):
    """Matched vs newly created — two semantic colors, 2px surface gap."""
    H, bar_y, bar_h = 108, 24, 34
    mt = m["match_types"]
    tot = sum(mt.values())
    matched = mt["matched_llm"] + mt["matched_exact"]
    new = tot - matched
    segs = [("Matched to an existing case", matched, C_MATCH),
            ("Created a new case", new, C_NEW)]
    x = 0
    s = [f'<svg viewBox="0 0 {width} {H}" role="img" aria-label="Of every resolved dispatch, '
         f'{matched/tot*100:.1f} percent matched an existing case and {new/tot*100:.1f} percent '
         f'created a new one.">']
    for name, v, col in segs:
        w = v / tot * width
        s.append(f'<rect x="{x:.1f}" y="{bar_y}" width="{max(w-2,2):.1f}" height="{bar_h}" rx="4" '
                 f'fill="{col}"><title>{esc(name)}: {v:,} ({v/tot*100:.1f}%)</title></rect>')
        anchor = "start" if x < width * 0.5 else "end"
        tx = x + 2 if anchor == "start" else x + w - 4
        s.append(f'<text x="{tx:.1f}" y="{bar_y-10}" text-anchor="{anchor}" font-size="13" '
                 f'font-weight="600" fill="{INK}">{v/tot*100:.1f}%</text>')
        s.append(f'<text x="{tx:.1f}" y="{bar_y+bar_h+20}" text-anchor="{anchor}" font-size="12.5" '
                 f'fill="{MUTED}">{esc(name)} · {v:,}</text>')
        x += w
    s.append("</svg>")
    return "".join(s)


# ------------------------------------------------------------------- page


def build(m):
    K, beta, r2 = m["heaps"]
    K2, b2 = m["heaps_recent"]
    n_proc, n_cases = m["n_processed"], m["n_cases"]
    mapped, no_use = m["status"]["mapped"], m["status"]["no_useful_notes"]
    tot_mt = sum(m["match_types"].values())
    matched = m["match_types"]["matched_llm"] + m["match_types"]["matched_exact"]
    ratio = m["n_mapped"] / n_cases
    last = m["recent_rate"]
    generated = datetime.now(timezone.utc).strftime("%d %B %Y")

    proj_rows = "".join(
        f"<tr><td>{fmt(n)}</td><td>{fmt(int(K*n**beta))}</td><td>{fmt(int(K2*n**b2))}</td>"
        f"<td>{n/TOTAL_ELIGIBLE*100:.1f}%</td></tr>"
        for n in (25_000, 50_000, 100_000, 250_000, 500_000, TOTAL_ELIGIBLE))

    top_cases = "".join(
        f"<tr><td><code>{cid}</code></td><td class=num>{n}</td>"
        f"<td>{esc(m['cases'][cid]['category'])}</td>"
        f"<td class=q>{esc(m['cases'][cid]['canonical'][:150])}"
        f"{'…' if len(m['cases'][cid]['canonical'])>150 else ''}</td></tr>"
        for cid, n in m["members"].most_common(15))

    mt_rows = "".join(
        f"<tr><td><code>{k}</code></td><td class=num>{fmt(v)}</td>"
        f"<td class=num>{v/tot_mt*100:.1f}%</td><td>{esc(d)}</td></tr>"
        for k, d in [
            ("matched_llm", "Similarity search proposed candidates; the AI judge confirmed one"),
            ("new_llm_rejected", "Candidates existed, but the judge rejected all of them — new case"),
            ("matched_exact", "Byte-identical root-cause text already seen — cache hit, no AI call"),
            ("new_no_candidates", "Nothing cleared the 0.60 similarity floor — new case"),
            ("new_first", "The very first case in an empty catalog"),
        ] if (v := m["match_types"].get(k, 0)))

    cov_rows = "".join(
        f"<tr><td>Top {n} cases</td><td class=num>{fmt(c)}</td>"
        f"<td class=num>{c/m['n_mapped']*100:.1f}%</td></tr>"
        for n, c in m["coverage"])

    batch_rows = "".join(
        f"<tr><td><code>{esc(bid)}</code></td><td class=num>{fmt(n)}</td>"
        f"<td class=num>{fmt(oc.get('mapped',0))}</td><td class=num>{fmt(oc.get('no_useful_notes',0))}</td>"
        f"<td class=num>{fmt(sum(v for k,v in oc.items() if k.startswith('incomplete') or k=='unresolved'))}</td></tr>"
        for bid, n, oc in m["batches"])

    cat_rows = [(k, v) for k, v in m["cats"].most_common(14)]
    door = sum(v for k, v in m["cats"].items() if "door" in k.lower())
    n_door = sum(1 for k in m["cats"] if "door" in k.lower())

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dispatch Case-Mapping — Data Report</title>
<style>
  :root {{
    --surface: #fcfcfb; --card: #ffffff; --ink: #14140f; --secondary: #52514e;
    --muted: #78766f; --grid: #dedcd3; --match: {C_MATCH}; --new: {C_NEW};
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--surface); color: var(--ink);
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; line-height: 1.55; }}
  .wrap {{ max-width: 1180px; margin: 0 auto; padding: 40px 24px 90px; }}
  h1 {{ font-size: 1.7rem; margin: 0 0 6px; letter-spacing: -0.01em; }}
  .subtitle {{ color: var(--secondary); margin: 0 0 8px; font-size: 1.03rem; }}
  .stamp {{ color: var(--muted); font-size: 0.85rem; margin: 0 0 34px; }}
  h2 {{ font-size: 1.16rem; border-bottom: 1px solid var(--grid); padding-bottom: 8px;
       margin-top: 56px; }}
  h3 {{ font-size: 1rem; margin: 30px 0 6px; }}
  p {{ margin: 12px 0; }}
  .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
           gap: 14px; margin: 26px 0 8px; }}
  .kpi {{ background: var(--card); border: 1px solid var(--grid); border-radius: 12px;
          padding: 18px 20px; }}
  .kpi .v {{ font-size: 2.05rem; font-weight: 650; letter-spacing: -0.02em; line-height: 1.1; }}
  .kpi .k {{ color: var(--secondary); font-size: 0.84rem; text-transform: uppercase;
             letter-spacing: 0.04em; margin-bottom: 7px; }}
  .kpi .n {{ color: var(--muted); font-size: 0.85rem; margin-top: 5px; }}
  figure {{ margin: 22px 0 8px; padding: 24px; background: var(--card);
            border: 1px solid var(--grid); border-radius: 12px; overflow-x: auto; }}
  figure svg {{ width: 100%; min-width: 640px; height: auto; display: block; }}
  figcaption {{ color: var(--secondary); font-size: 0.92rem; margin-top: 16px; padding: 0 2px; }}
  table {{ width: 100%; border-collapse: collapse; margin: 16px 0 6px; font-size: 0.94rem; }}
  th, td {{ text-align: left; padding: 9px 12px; border-bottom: 1px solid var(--grid);
            vertical-align: top; }}
  th {{ color: var(--secondary); font-weight: 600; font-size: 0.78rem;
        text-transform: uppercase; letter-spacing: 0.03em; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
  td.q {{ color: var(--secondary); font-size: 0.89rem; }}
  code {{ font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 0.88em;
          background: var(--surface); border: 1px solid var(--grid); padding: 1px 5px;
          border-radius: 4px; white-space: nowrap; }}
  .callout {{ background: var(--card); border: 1px solid var(--grid);
              border-left: 3px solid var(--new); border-radius: 10px;
              padding: 16px 20px; margin: 20px 0; }}
  .callout h3 {{ margin-top: 0; }}
  .legend {{ display: flex; gap: 22px; flex-wrap: wrap; font-size: 0.9rem;
             color: var(--secondary); margin: 14px 0 0; }}
  .legend span {{ display: inline-flex; align-items: center; gap: 8px; }}
  .sw {{ width: 12px; height: 12px; border-radius: 3px; display: inline-block; }}
  .tw {{ overflow-x: auto; }}
</style>
</head>
<body>
<div class="wrap">

  <h1>Dispatch Case-Mapping — Data Report</h1>
  <p class="subtitle">What {fmt(n_proc)} processed dispatches reveal about the shape of
     ClimatePros' root-cause catalog.</p>
  <p class="stamp">Generated {generated} from <code>state/</code> · covers every batch to date</p>

  <div class="kpis">
    <div class="kpi"><div class="k">Dispatches processed</div><div class="v">{fmt(n_proc)}</div>
      <div class="n">{n_proc/TOTAL_ELIGIBLE*100:.2f}% of {fmt(TOTAL_ELIGIBLE)} eligible</div></div>
    <div class="kpi"><div class="k">Unique cases</div><div class="v">{fmt(n_cases)}</div>
      <div class="n">from {fmt(m['n_mapped'])} resolved dispatches</div></div>
    <div class="kpi"><div class="k">Deduplication</div><div class="v">{ratio:.1f}&times;</div>
      <div class="n">dispatches per case on average</div></div>
    <div class="kpi"><div class="k">New-case rate</div><div class="v">{last*100:.1f}%</div>
      <div class="n">trailing {fmt(m['recent_span'])} dispatches, from {m['bins'][0][3]*100:.1f}%</div></div>
  </div>

  <h2>1. The catalog is deduplicating well — and still growing</h2>

  <figure>
    {chart_growth(m)}
    <figcaption>Every resolved dispatch adds a point. The dashed diagonal is what the curve
      would look like if no two dispatches ever shared a root cause. The gap between the two
      is the deduplication the pipeline is buying: {fmt(m['n_mapped'])} dispatches collapsed
      into {fmt(n_cases)} distinct cases.</figcaption>
  </figure>

  <h2>2. But the new-case rate has flattened, not fallen to zero</h2>

  <figure>
    {chart_rate(m)}
    <figcaption>New cases created per 1,000 resolved dispatches. The first block (muted) is a
      cold-start artifact — with an empty catalog, nearly everything is new. After that the rate
      fell steadily to about 10%, and has held there for the last several thousand dispatches
      rather than continuing to decline.</figcaption>
  </figure>

  <div class="callout">
    <h3>This is the report's main finding</h3>
    <p>A catalog approaching saturation would show the new-case rate decaying toward zero.
      This one decays and then <strong>plateaus around {last*100:.0f}%</strong>. Roughly one in
      {1/last:.0f} dispatches still introduces a genuinely new root cause after
      {fmt(m['n_mapped'])} of them — so the catalog should be treated as <em>usefully
      representative but not complete</em>. It is a working knowledge base, not a finished
      taxonomy.</p>
  </div>

  <h2>3. What that implies at full corpus scale</h2>

  <p>Case growth follows Heaps' law — <code>cases = K &times; dispatches<sup>β</sup></code>, the
     standard model for how a vocabulary grows as a corpus expands. The fit is unusually tight:
     <strong>β = {beta:.3f}</strong>, R² = {r2:.4f} across all {fmt(len(m['growth']))} points.
     A β below 1 is precisely why deduplication works, and it predicts how the catalog scales.</p>

  <div class="tw"><table>
    <thead><tr><th>Dispatches processed</th><th>Projected cases (full fit)</th>
      <th>Projected cases (recent regime)</th><th>Share of corpus</th></tr></thead>
    <tbody>{proj_rows}</tbody>
  </table></div>

  <p class="stamp">Two fits are shown because the opening 1,000 dispatches distort the curve:
     the "recent regime" fit uses only N ≥ 6,000. They agree closely, which is reassuring —
     but the final row extrapolates roughly 100× beyond observed data, so treat it as an
     order-of-magnitude estimate, not a forecast. The mid-range rows (25k–100k) are the
     defensible ones.</p>

  <h2>4. Where dispatches drop out</h2>

  <figure>
    {chart_funnel([
      ("Fetched & processed", n_proc, "selected from the staging database"),
      ("Mapped to a case", m['n_mapped'], f"{no_use:,} had only admin or scheduling text"),
      ("Named a real technical fault", m['diagnostic'], f"{m['no_fault']:,} concluded “no fault identified”"),
    ])}
    <figcaption>Of {fmt(n_proc)} dispatches processed, {fmt(m['diagnostic'])}
      ({m['diagnostic']/n_proc*100:.1f}%) ended up describing an actual technical failure. The
      two losses are different in kind: {fmt(no_use)} dispatches had no usable notes at all,
      while {fmt(m['no_fault'])} were processed successfully but concluded that no fault was
      found.</figcaption>
  </figure>

  <p>The note filter is doing most of the heavy lifting. Across
     {fmt(m['notes_disp'])} dispatches, <strong>{fmt(m['notes_total'])} individual notes</strong>
     were classified and only <strong>{fmt(m['notes_useful'])} ({m['notes_useful']/m['notes_total']*100:.1f}%)</strong>
     were judged technically useful — about {m['notes_useful']/m['notes_disp']:.1f} useful notes
     per dispatch out of {m['notes_total']/m['notes_disp']:.1f} written. Three of every four
     notes are scheduling, dispatch, or billing chatter.</p>

  <h2>5. How each dispatch gets assigned</h2>

  <figure>
    {chart_split(m)}
    <figcaption>{matched/tot_mt*100:.1f}% of resolved dispatches matched a case that already
      existed.</figcaption>
    <div class="legend">
      <span><i class="sw" style="background:{C_MATCH}"></i> Matched an existing case</span>
      <span><i class="sw" style="background:{C_NEW}"></i> Created a new case</span>
    </div>
  </figure>

  <div class="tw"><table>
    <thead><tr><th>Match type</th><th>Dispatches</th><th>Share</th><th>What it means</th></tr></thead>
    <tbody>{mt_rows}</tbody>
  </table></div>

  <p>The exact-duplicate cache resolves {m['match_types']['matched_exact']/tot_mt*100:.1f}% of
     dispatches with no AI call at all — technicians writing byte-identical root causes. Only
     {m['match_types'].get('new_no_candidates',0)} dispatches
     ({m['match_types'].get('new_no_candidates',0)/tot_mt*100:.2f}%) failed to surface any
     candidate above the 0.60 similarity floor, which says the embedding retrieval is nearly
     always finding something plausible — the AI judge, not the vector search, is what decides
     new-versus-repeat.</p>

  <h2>6. What actually breaks</h2>

  <figure>
    {chart_bars(cat_rows, m['n_mapped'])}
    <figcaption>Top {len(cat_rows)} categories by dispatch count, out of {m['n_cats']} distinct
      categories the model produced. Drain blockages, refrigerant leaks, and fan-motor failures
      dominate the real faults.</figcaption>
  </figure>

  <h2>7. A long tail, and a concentrated head</h2>

  <p>Case sizes are extremely uneven. <strong>{fmt(m['singletons'])} cases
     ({m['singletons']/n_cases*100:.1f}%) have exactly one member dispatch</strong>, while the
     largest single case covers {fmt(m['members'].most_common(1)[0][1])}.</p>

  <div class="tw"><table>
    <thead><tr><th>Coverage</th><th>Dispatches</th><th>Share of resolved</th></tr></thead>
    <tbody>{cov_rows}</tbody>
  </table></div>

  <p>This is the practically useful shape: a technician-facing tool built on the top ~100 cases
     would already recognise about half of all incoming work, while the long tail is where the
     catalog keeps growing.</p>

  <h3>Largest cases</h3>
  <div class="tw"><table>
    <thead><tr><th>Case</th><th>Dispatches</th><th>Category</th><th>Canonical root cause</th></tr></thead>
    <tbody>{top_cases}</tbody>
  </table></div>

  <h2>8. Data-quality findings</h2>

  <div class="callout">
    <h3>The largest "case" is not a fault</h3>
    <p><code>CASE-0012</code> — <em>"No technical fault identified in dispatch notes"</em> — is the
      biggest case in the catalog with {fmt(m['members'].get('CASE-0012',0))} member dispatches.
      It is the pipeline's own placeholder for a successful run that found no fault, not a
      diagnosis. Together with similar entries, the <em>No Technical Fault</em> category accounts
      for {fmt(m['no_fault'])} dispatches ({m['no_fault']/m['n_mapped']*100:.1f}% of everything
      mapped). Any downstream consumer should filter it out before treating the catalog as a
      fault taxonomy.</p>
  </div>

  <div class="callout">
    <h3>Category labels are fragmenting</h3>
    <p>The extraction prompt asks for a standardised 2–4 word category, but
      <strong>{m['n_cats']} distinct categories</strong> have emerged, and
      {m['cats_thin']} of them ({m['cats_thin']/m['n_cats']*100:.0f}%) carry 10 or fewer
      dispatches. Door-related failures alone are split across <strong>{n_door} separate
      categories</strong> ({fmt(door)} dispatches) — <em>Door Seal</em>, <em>Door Gasket</em>,
      <em>Door Handle</em>, <em>Door Hardware</em>, <em>Door Mechanism</em> and more. Fan and
      motor failures similarly split across overlapping labels.</p>
    <p>Note this affects the <em>category</em> field only — it is metadata, not the matching key.
      Case identity comes from the root-cause text and the AI judge, so fragmented categories do
      not merge or split cases. It does mean category counts should be read as indicative, and
      a normalisation pass would make them reportable.</p>
  </div>

  <div class="callout">
    <h3>Category and root cause can disagree</h3>
    <p>Spot-checking the largest cases surfaces mislabels: <code>CASE-0308</code> is filed under
      <em>Condenser Fan Motor Failure</em>, but its canonical text describes a broken blower
      motor belt. The root-cause text — the field that drives matching — is correct; the
      category assigned alongside it is not. Same conclusion as above: trust the text, treat the
      category as a hint.</p>
  </div>

  <h2>9. Batch history</h2>

  <div class="tw"><table>
    <thead><tr><th>Batch</th><th>Dispatches</th><th>Mapped</th><th>No useful notes</th>
      <th>Incomplete</th></tr></thead>
    <tbody>{batch_rows}</tbody>
  </table></div>

  <p>Incomplete dispatches are not lost — they never enter the ledger, so they stay eligible and
     are retried in a future batch with their committed stage results reused.</p>

  <h2>10. What this supports, and what it doesn't</h2>

  <p><strong>Supported today.</strong> The catalog is a sound basis for recognising repeat work:
     {matched/tot_mt*100:.0f}% of dispatches match something already known, the top 100 cases
     cover roughly half of all resolved dispatches, and the deduplication ratio of {ratio:.1f}×
     is stable and improving. Retrieval-style uses — "has this failure been seen before, and
     what was done about it" — are well served.</p>

  <p><strong>Not yet supported.</strong> Treating the catalog as complete. At
     {n_proc/TOTAL_ELIGIBLE*100:.2f}% of the corpus with a new-case rate still near
     {last*100:.0f}%, roughly one dispatch in {1/last:.0f} is still novel, and the Heaps fit puts
     the eventual catalog in the tens of thousands of cases. Frequency statistics drawn from it
     describe the sample, not the business — and the category field needs normalisation before
     it can carry reporting.</p>

</div>
</body>
</html>
"""


def main():
    m = collect()
    html = build(m)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {OUT} ({len(html):,} bytes)")
    print(f"  {m['n_processed']:,} processed · {m['n_cases']:,} cases · "
          f"{m['n_mapped']:,} mapped · beta={m['heaps'][1]:.3f}")


if __name__ == "__main__":
    main()
