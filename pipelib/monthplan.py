"""Month-stratified fetch planning: which months to mine, and how deep.

Newest-first fetching concentrates the corpus in the two or three most recent
months, which hides seasonal failure modes (heat/defrost/freeze-up faults never
appear in a summer-only corpus). The plan spreads a fetch budget across a
window of months instead — by default 24 months, so every calendar month is
covered twice.

Quotas are capped by what each month actually still has (eligible minus
already-processed) and any shortfall is redistributed across months with
surplus, so an exhausted month costs the plan nothing. Pure functions plus one
small state file (state/fetch_plan.json); no DB and no network here — the
caller supplies the eligible counts from db.count_candidates_by_month.
"""
from datetime import datetime, timezone

from . import config
from .statefiles import load_json, save_json

PLAN_VERSION = 1


def month_bounds(ym):
    """('YYYY-MM') -> (dt_min, dt_max) as SQL-ready strings, upper end exclusive."""
    y, m = (int(x) for x in ym.split("-"))
    nxt = f"{y + 1:04d}-01-01" if m == 12 else f"{y:04d}-{m + 1:02d}-01"
    return f"{y:04d}-{m:02d}-01", nxt


def month_list(end_ym, n_months):
    """The n_months months ending at (and including) end_ym, oldest first."""
    y, m = (int(x) for x in end_ym.split("-"))
    out = []
    for _ in range(n_months):
        out.append(f"{y:04d}-{m:02d}")
        m -= 1
        if m == 0:
            y, m = y - 1, 12
    return list(reversed(out))


def processed_by_month(meta, exclude):
    """{'YYYY-MM': n} for dispatches already processed, from dispatch_meta's
    received_dt. Entries without a date (the legacy Excel seed) are ignored."""
    out = {}
    for did in exclude:
        dt = (meta.get(did, {}).get("received_dt") or "")[:7]
        if len(dt) == 7:
            out[dt] = out.get(dt, 0) + 1
    return out


def build_plan(eligible, processed, per_month, months):
    """Per-month quotas, capped by availability, shortfall redistributed.

    Target is per_month * len(months). A month that cannot fill its share
    (exhausted, or simply small) keeps only what it has; the difference is
    spread over months that still have headroom, repeatedly, until the target
    is met or no headroom is left anywhere.
    """
    avail = {m: max(0, eligible.get(m, 0) - processed.get(m, 0)) for m in months}
    quota = {m: min(per_month, avail[m]) for m in months}
    target = per_month * len(months)

    while True:
        shortfall = target - sum(quota.values())
        headroom = {m: avail[m] - quota[m] for m in months if avail[m] > quota[m]}
        if shortfall <= 0 or not headroom:
            break
        total_headroom = sum(headroom.values())
        added = 0
        for m, room in sorted(headroom.items()):
            take = min(room, max(1, shortfall * room // total_headroom))
            take = min(take, shortfall - added)
            quota[m] += take
            added += take
            if added >= shortfall:
                break
        if added == 0:
            break

    return {
        "schema": PLAN_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window": {"first": months[0], "last": months[-1], "n_months": len(months)},
        "per_month": per_month,
        "months": {m: {"eligible": eligible.get(m, 0),
                       "processed": processed.get(m, 0),
                       "available": avail[m],
                       "quota": quota[m],
                       "fetched": 0} for m in months},
    }


def load_plan():
    return load_json(config.FETCH_PLAN_FILE)


def save_plan(plan):
    save_json(config.FETCH_PLAN_FILE, plan)


def merge_progress(new_plan, old_plan):
    """Carry `fetched` counters across a re-plan so refreshing the plan never
    loses track of what has already been staged."""
    if not old_plan:
        return new_plan
    for m, rec in new_plan["months"].items():
        prev = (old_plan.get("months") or {}).get(m)
        if prev:
            rec["fetched"] = prev.get("fetched", 0)
    return new_plan


def next_month(plan):
    """The oldest month still short of its quota, or None when the plan is done."""
    for m in sorted(plan["months"]):
        rec = plan["months"][m]
        if rec["fetched"] < rec["quota"]:
            return m
    return None


def record_fetched(plan, ym, n):
    rec = plan["months"].setdefault(
        ym, {"eligible": 0, "processed": 0, "available": 0, "quota": 0, "fetched": 0})
    rec["fetched"] = rec.get("fetched", 0) + n
    return plan


def format_plan(plan):
    lines = [f"Fetch plan {plan['window']['first']} .. {plan['window']['last']} "
             f"({plan['window']['n_months']} months, target {plan['per_month']}/month, "
             f"generated {plan['generated_at'][:19]}Z)",
             f"  {'month':9s} {'eligible':>9s} {'processed':>10s} "
             f"{'available':>10s} {'quota':>7s} {'fetched':>8s}"]
    tq = tf = 0
    for m in sorted(plan["months"]):
        r = plan["months"][m]
        tq += r["quota"]
        tf += r.get("fetched", 0)
        flag = "  (done)" if r.get("fetched", 0) >= r["quota"] > 0 else ""
        lines.append(f"  {m:9s} {r['eligible']:9,d} {r['processed']:10,d} "
                     f"{r['available']:10,d} {r['quota']:7,d} "
                     f"{r.get('fetched', 0):8,d}{flag}")
    lines.append(f"  {'TOTAL':9s} {'':9s} {'':10s} {'':10s} {tq:7,d} {tf:8,d}")
    return "\n".join(lines)
