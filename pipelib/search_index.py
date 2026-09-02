"""Azure AI Search index: schema, clients, document building, batched upsert.

One index, one document per dispatch, versioned by name (config.AZURE_SEARCH_INDEX,
default dispatches-nomic768-v1). Vector dimensions are immutable on a live index,
so an embedding-model change means a NEW index name plus a backfill rerun — never
an in-place edit. Only real parts are indexed (catalog number present, NonPart=0,
qty>0, InventoryId resolved); consumables stay in state/parts.json only.
"""
from datetime import datetime, timezone

from . import config


def _models():
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient
    from azure.search.documents.indexes import models as m
    return AzureKeyCredential, SearchClient, SearchIndexClient, m


def index_definition():
    _, _, _, m = _models()
    return m.SearchIndex(
        name=config.AZURE_SEARCH_INDEX,
        fields=[
            m.SimpleField(name="dispatchId", type=m.SearchFieldDataType.String,
                          key=True, filterable=True),
            m.SearchableField(name="rootCause", type=m.SearchFieldDataType.String),
            m.SearchableField(name="reason", type=m.SearchFieldDataType.String),
            m.SimpleField(name="category", type=m.SearchFieldDataType.String,
                          filterable=True, facetable=True),
            m.SimpleField(name="dispatchNumber", type=m.SearchFieldDataType.String,
                          filterable=True),
            m.SimpleField(name="receivedDt", type=m.SearchFieldDataType.DateTimeOffset,
                          filterable=True, sortable=True),
            m.SimpleField(name="hasParts", type=m.SearchFieldDataType.Boolean,
                          filterable=True),
            m.ComplexField(name="parts", collection=True, fields=[
                m.SimpleField(name="inventoryId", type=m.SearchFieldDataType.String,
                              filterable=True),
                m.SimpleField(name="partNo", type=m.SearchFieldDataType.String,
                              filterable=True, facetable=True),
                m.SimpleField(name="name", type=m.SearchFieldDataType.String),
                m.SimpleField(name="qty", type=m.SearchFieldDataType.Double),
            ]),
            m.SimpleField(name="embedModel", type=m.SearchFieldDataType.String,
                          filterable=True),
            m.SimpleField(name="indexedAt", type=m.SearchFieldDataType.DateTimeOffset,
                          filterable=True),
            m.SearchField(name="rootCauseVector",
                          type=m.SearchFieldDataType.Collection(m.SearchFieldDataType.Single),
                          searchable=True, hidden=True,
                          vector_search_dimensions=768,
                          vector_search_profile_name="vec-profile"),
        ],
        vector_search=m.VectorSearch(
            algorithms=[m.HnswAlgorithmConfiguration(
                name="hnsw-cosine",
                parameters=m.HnswParameters(metric="cosine", m=4,
                                            ef_construction=400, ef_search=500))],
            profiles=[m.VectorSearchProfile(
                name="vec-profile", algorithm_configuration_name="hnsw-cosine")],
        ),
    )


def index_client():
    config.require_search_config()
    AzureKeyCredential, _, SearchIndexClient, _ = _models()
    return SearchIndexClient(config.AZURE_SEARCH_ENDPOINT,
                             AzureKeyCredential(config.AZURE_SEARCH_API_KEY))


def search_client():
    config.require_search_config()
    AzureKeyCredential, SearchClient, _, _ = _models()
    return SearchClient(config.AZURE_SEARCH_ENDPOINT, config.AZURE_SEARCH_INDEX,
                        AzureKeyCredential(config.AZURE_SEARCH_API_KEY))


def create_index(recreate=False):
    client = index_client()
    existing = [n for n in client.list_index_names()]
    if config.AZURE_SEARCH_INDEX in existing:
        if not recreate:
            raise SystemExit(
                f"Index '{config.AZURE_SEARCH_INDEX}' already exists — refusing to "
                "touch it. Use --recreate --yes to delete and rebuild (destructive), "
                "or set AZURE_SEARCH_INDEX to a new name for a schema change.")
        client.delete_index(config.AZURE_SEARCH_INDEX)
        print(f"Deleted existing index '{config.AZURE_SEARCH_INDEX}'.")
    client.create_index(index_definition())
    print(f"Created index '{config.AZURE_SEARCH_INDEX}' "
          f"(768-dim cosine HNSW, key=dispatchId).")


def _received_dt_utc(value):
    # dispatch_meta stores tz-naive SQL Server datetimes; Edm.DateTimeOffset
    # requires an offset, and staging times are treated as UTC by convention
    if not value:
        return None
    if value.endswith("Z") or "+" in value[10:]:
        return value
    return value + "Z"


def real_parts(part_items):
    return [p for p in part_items
            if p["part_no"] and not p["consumable"] and p["qty"] > 0
            and p["inventory_id"]]


def build_document(did, text, category, meta_rec, part_items, vector):
    meta_rec = meta_rec or {}
    parts = [{"inventoryId": p["inventory_id"], "partNo": p["part_no"],
              "name": p["name"], "qty": float(p["qty"])}
             for p in real_parts(part_items)]
    doc = {
        "dispatchId": did,
        "rootCause": text,
        "reason": meta_rec.get("reason", ""),
        "category": category or "",
        "dispatchNumber": meta_rec.get("dispatch_number", ""),
        "hasParts": bool(parts),
        "parts": parts,
        "embedModel": config.SEARCH_EMBED_MODEL_TAG,
        "indexedAt": datetime.now(timezone.utc).isoformat(),
        "rootCauseVector": [float(x) for x in vector],
    }
    received = _received_dt_utc(meta_rec.get("received_dt"))
    if received:
        doc["receivedDt"] = received
    return doc


def upload_documents(client, docs):
    """merge_or_upload in config.SEARCH_UPLOAD_BATCH chunks. Returns
    (succeeded_ids, failed_ids); one bounded retry for per-document failures
    (throttling surfaces as succeeded=False rows, not exceptions)."""
    succeeded, failed = set(), {}
    for i in range(0, len(docs), config.SEARCH_UPLOAD_BATCH):
        chunk = docs[i:i + config.SEARCH_UPLOAD_BATCH]
        for result in client.merge_or_upload_documents(documents=chunk):
            if result.succeeded:
                succeeded.add(result.key)
            else:
                failed[result.key] = result.error_message
    if failed:
        retry = [d for d in docs if d["dispatchId"] in failed]
        print(f"  Retrying {len(retry)} failed uploads once...")
        failed = {}
        for result in client.merge_or_upload_documents(documents=retry):
            if result.succeeded:
                succeeded.add(result.key)
            else:
                failed[result.key] = result.error_message
        for key, msg in failed.items():
            print(f"  Upload failed for {key}: {msg}")
    return succeeded, set(failed)
