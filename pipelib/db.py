"""FieldJetXStg access. Candidate SQL adapted from build_1k/extract.py (proven).
Parts SQL adapted from the Dispatch research repo's build_case_parts_dataset.py:
DispatchParts carries no InventoryId, so the InventoryLocationXREF hop is
mandatory to reach the catalog item. All queries parameterized; %% escapes
LIKE wildcards for pymssql.

united_part_no is United Refrigeration's own catalog number (the namespace the
United inventory API's `item` field requires), resolved per item from
InventorySupplierXREF. A correlated TOP-1 subquery — not a join — because the
xref averages ~3.7 rows per item and a join would fan out SUM(dp.Quantity).
Active suppliers matched by name (12 duplicate United rows exist; no canonical
SupplierId), and xref rows whose "United number" merely repeats the internal
Inventory.Number are excluded — those are known false cross-references.
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


PARTS_SQL = """
SELECT dp.DispatchId,
       xr.InventoryId,
       COALESCE(NULLIF(LTRIM(RTRIM(inv.Number)), ''), '')          AS part_no,
       COALESCE(NULLIF(LTRIM(RTRIM(inv.InventoryDesc)), ''), '')   AS desc_short,
       COALESCE(NULLIF(LTRIM(RTRIM(inv.InventoryName)), ''), '')   AS inv_name,
       inv.NonPart,
       COALESCE(c.InventoryCategoryName, '')                       AS inv_cat,
       COALESCE(sc.InventorySubCategoryName, '')                   AS inv_subcat,
       SUM(dp.Quantity) AS qty,
       (SELECT TOP 1 LTRIM(RTRIM(sx.PartNumber))
        FROM dbo.InventorySupplierXREF sx
        JOIN dbo.Supplier s ON s.SupplierId = sx.SupplierId
        WHERE sx.InventoryId = xr.InventoryId
          AND s.IsActive = 1
          AND s.SupplierName LIKE '%%united%%refrig%%'
          AND LTRIM(RTRIM(ISNULL(sx.PartNumber, ''))) <> ''
          AND LTRIM(RTRIM(sx.PartNumber)) <> LTRIM(RTRIM(ISNULL(inv.Number, '')))
        ORDER BY sx.UpdateDt DESC) AS united_part_no
FROM dbo.DispatchParts dp
LEFT JOIN dbo.InventoryLocationXREF xr ON xr.InventoryLocationXREFId = dp.InventoryLocationXREFId
LEFT JOIN dbo.Inventory inv ON inv.InventoryId = xr.InventoryId
LEFT JOIN dbo.InventoryCategory c ON c.InventoryCategoryId = inv.InventoryCategoryId
LEFT JOIN dbo.InventorySubCategory sc ON sc.InventorySubCategoryId = inv.InventorySubCategoryId
WHERE dp.DispatchId IN ({ph})
GROUP BY dp.DispatchId, xr.InventoryId, inv.Number, inv.InventoryDesc,
         inv.InventoryName, inv.NonPart,
         c.InventoryCategoryName, sc.InventorySubCategoryName
HAVING SUM(dp.Quantity) > 0
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


def fetch_parts_for(conn, dispatch_ids):
    """Recorded parts usage per dispatch, consumables included but flagged.
    Returns {DISPATCH_ID: [{inventory_id, part_no, name, qty, consumable,
    inv_cat, inv_subcat}, ...]}. Dispatches with no rows are simply absent —
    callers store an explicit empty list to record "checked, none found".
    """
    parts = {}
    cur = conn.cursor(as_dict=True)
    CHUNK = 500
    for i in range(0, len(dispatch_ids), CHUNK):
        chunk = dispatch_ids[i:i + CHUNK]
        placeholders = ",".join(["%s"] * len(chunk))
        cur.execute(PARTS_SQL.format(ph=placeholders), tuple(chunk))
        for r in cur.fetchall():
            did = config.norm_guid(r["DispatchId"])
            name = r["desc_short"] or r["inv_name"] or r["part_no"]
            parts.setdefault(did, []).append({
                "inventory_id": config.norm_guid(r["InventoryId"]) if r["InventoryId"] else "",
                "part_no": r["part_no"],
                "name": name,
                "qty": float(r["qty"]),
                "consumable": bool(r["NonPart"]),
                "inv_cat": r["inv_cat"],
                "inv_subcat": r["inv_subcat"],
                "united_part_no": (r["united_part_no"] or "").strip(),
            })
    return parts
