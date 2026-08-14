"""FieldJetXStg access. Candidate SQL adapted from build_1k/extract.py (proven),
minus the parts-usage requirement (root-cause pipeline needs notes, not parts).
All queries parameterized; %% escapes LIKE wildcards for pymssql.
"""
from . import config

CAND_SQL = """
SELECT TOP (%(n)s)
       d.DispatchId, d.DispatchNumber, d.DispatchReason,
       d.ReceivedDateTime, ds.DispatchStatusName
FROM dbo.Dispatch d
JOIN dbo.DispatchStatus ds ON ds.DispatchStatusId = d.DispatchStatusId
WHERE d.IsConstruction = 0
  AND d.ReceivedDateTime IS NOT NULL
  AND d.ReceivedDateTime >= %(dt_min)s
  AND d.ReceivedDateTime <  %(dt_max)s
  AND d.DispatchReason IS NOT NULL AND LEN(d.DispatchReason) > 3
  AND ds.DispatchStatusName NOT LIKE '%%Cancel%%'
  AND EXISTS (SELECT 1 FROM dbo.DispatchNotes dn
              WHERE dn.DispatchId = d.DispatchId
                AND LEN(dn.DispatchNotes) > %(min_note)s)
ORDER BY d.ReceivedDateTime DESC;
"""


def connect():
    import pymssql
    return pymssql.connect(
        server=config.DB["server"], port=config.DB["port"],
        database=config.DB["database"], user=config.DB["user"],
        password=config.DB["password"], timeout=60, login_timeout=30,
    )


def fetch_candidates(conn, n):
    """Newest-first candidate dispatch headers. Returns list of dicts with
    UPPERCASE GUID strings."""
    cur = conn.cursor(as_dict=True)
    cur.execute(CAND_SQL, {
        "n": n,
        "dt_min": config.RECEIVED_MIN,
        "dt_max": config.RECEIVED_CUTOFF,
        "min_note": config.MIN_NOTE_LEN,
    })
    out = []
    for r in cur.fetchall():
        out.append({
            "dispatch_id": config.norm_guid(r["DispatchId"]),
            "dispatch_number": (r["DispatchNumber"] or "").strip(),
            "reason": (r["DispatchReason"] or "").strip(),
            "received_dt": r["ReceivedDateTime"].isoformat() if r["ReceivedDateTime"] else None,
            "status_name": (r["DispatchStatusName"] or "").strip(),
        })
    return out


def fetch_notes_for(conn, dispatch_ids):
    """All notes for the given dispatch ids, chronological. Returns
    {DISPATCH_ID: [{note_id, text, insert_dt}, ...]} with deterministic order."""
    notes = {}
    cur = conn.cursor(as_dict=True)
    CHUNK = 500
    for i in range(0, len(dispatch_ids), CHUNK):
        chunk = dispatch_ids[i:i + CHUNK]
        placeholders = ",".join(["%s"] * len(chunk))
        cur.execute(
            f"""SELECT DispatchId, DispatchNotesId, DispatchNotes, InsertDt
                FROM dbo.DispatchNotes
                WHERE DispatchId IN ({placeholders})
                ORDER BY InsertDt, DispatchNotesId""",
            tuple(chunk))
        for r in cur.fetchall():
            did = config.norm_guid(r["DispatchId"])
            text = (r["DispatchNotes"] or "").strip()
            if not text:
                continue
            notes.setdefault(did, []).append({
                "note_id": config.norm_guid(r["DispatchNotesId"]),
                "text": text,
                "insert_dt": r["InsertDt"].isoformat() if r["InsertDt"] else None,
            })
    return notes
