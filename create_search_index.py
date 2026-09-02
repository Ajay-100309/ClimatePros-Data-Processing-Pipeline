"""Create (or inspect, or deliberately rebuild) the Azure AI Search index.

The index name is versioned by embedding model (AZURE_SEARCH_INDEX, default
dispatches-nomic768-v1) because vector dimensions are immutable on a live
index: an embedding-model change means a NEW name here plus a backfill rerun.
Additive non-vector fields are the one allowed in-place edit (--update); after
one, run backfill_search.py --refresh-parts --force-push to populate the new
field on existing documents. Creation refuses to touch an existing index —
--recreate --yes is the only (destructive) way to rebuild in place.

Usage:
    venv/bin/python create_search_index.py              # create if absent
    venv/bin/python create_search_index.py --show       # print the live schema
    venv/bin/python create_search_index.py --update     # additive in-place field update
    venv/bin/python create_search_index.py --recreate --yes   # delete + rebuild
"""
import sys
import argparse

from pipelib import config, search_index


def show():
    client = search_index.index_client()
    names = list(client.list_index_names())
    if config.AZURE_SEARCH_INDEX not in names:
        sys.exit(f"Index '{config.AZURE_SEARCH_INDEX}' does not exist. "
                 f"Indexes on the service: {names or '(none)'}")
    idx = client.get_index(config.AZURE_SEARCH_INDEX)

    def field_line(f, prefix=""):
        flags = [a for a in ("key", "searchable", "filterable", "facetable",
                             "sortable", "hidden") if getattr(f, a, False)]
        dims = getattr(f, "vector_search_dimensions", None)
        extra = f" dims={dims}" if dims else ""
        print(f"  {prefix}{f.name:<18} {str(f.type):<38} "
              f"{','.join(flags)}{extra}")

    print(f"Index '{idx.name}' ({len(idx.fields)} top-level fields):")
    for f in idx.fields:
        field_line(f)
        for sub in (getattr(f, "fields", None) or []):
            field_line(sub, prefix="  parts/")
    algo = idx.vector_search.algorithms[0]
    print(f"Vector search: {algo.name} kind={algo.kind} "
          f"metric={algo.parameters.metric} m={algo.parameters.m} "
          f"efConstruction={algo.parameters.ef_construction} "
          f"efSearch={algo.parameters.ef_search}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show", action="store_true",
                    help="print the live index schema and exit")
    ap.add_argument("--update", action="store_true",
                    help="push additive field changes onto the existing index")
    ap.add_argument("--recreate", action="store_true",
                    help="delete the existing index first (DESTRUCTIVE)")
    ap.add_argument("--yes", action="store_true",
                    help="required with --recreate")
    args = ap.parse_args()

    if args.show:
        show()
        return
    if args.update:
        search_index.update_index()
        return
    if args.recreate and not args.yes:
        sys.exit("--recreate deletes every indexed document. "
                 "Re-run with --recreate --yes if that is really the intent.")
    search_index.create_index(recreate=args.recreate)


if __name__ == "__main__":
    main()
