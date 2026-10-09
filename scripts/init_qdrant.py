"""Start Qdrant and index the collections the demo needs. Qdrant only.

Usage:
        uv run python scripts/init_qdrant.py                    # naive + sparse + hybrid
        uv run python scripts/init_qdrant.py rerank             # rerank only
        uv run python scripts/init_qdrant.py --all              # every registered architecture
        uv run python scripts/init_qdrant.py --all naive hybrid # explicit subset

Ingestion reads every PDF under ./documents. Fetch the corpus first with
`uv run rag-analysis fetch-corpus` when the directory is empty.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.registry import UnknownArchitecture, all_architectures, architecture
from vector.store import collection_points, ensure_qdrant_up, get_qdrant_client

DEFAULT_ARCHITECTURES = ("naive", "sparse", "hybrid")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Start Qdrant and index its collections."
    )
    parser.add_argument(
        "architectures",
        nargs="*",
        help="Architectures to ingest. Defaults to naive, sparse, hybrid.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Ingest every registered architecture instead of the default three.",
    )
    args = parser.parse_args()

    if args.architectures:
        names = args.architectures
    elif args.all:
        names = sorted(registered.name for registered in all_architectures())
    else:
        names = list(DEFAULT_ARCHITECTURES)

    try:
        resolved = [architecture(name) for name in names]
    except UnknownArchitecture as error:
        parser.error(str(error))

    print("[INIT] starting Qdrant...")
    ensure_qdrant_up()
    print("[INIT] Qdrant is ready")

    for registered in resolved:
        print(f"[INIT] ingesting {registered.name} into {registered.collection}...")
        registered.ingest()

    client = get_qdrant_client()
    print("[INIT] collections:")
    for registered in resolved:
        count = collection_points(client, registered.collection)
        print(f"[INIT]   {registered.collection}: {count} points")


if __name__ == "__main__":
    main()
