from config.params import Params
from core.chunk import Chunk, Provenance
from core.registry import DENSE, SPARSE
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

import subprocess
import time
import urllib.request
from pathlib import Path


def get_qdrant_client():
    return QdrantClient(url=Params.QDRANT_URL)


def chunk_from_point(point: models.ScoredPoint) -> Chunk:
    """Read one scored store point as a chunk carrying its text and its provenance.

    Indexed nodes record the pages they span as a list, so a chunk is placed on the first of
    them. A point without text is an indexing fault, not an empty chunk, so it raises rather
    than being retrieved as a blank that still counts as a hit.
    """
    metadata = (point.payload or {}).get("metadata") or {}
    pages = metadata.get("pages") or []

    return Chunk(
        text=point.payload["text"],
        score=float(point.score),
        provenance=Provenance(
            source=metadata.get("source_file"),
            section=metadata.get("section"),
            page=pages[0] if pages else None,
            node_id=str(point.id) if point.id is not None else None,
        ),
    )


def chunks_from_result(result: models.QueryResponse) -> list[Chunk]:
    """Read a scored store response as chunks, in the order the store ranked them."""
    return [chunk_from_point(point) for point in result.points]


def create_collection(
    client: QdrantClient, collection_name: str, vectors: tuple[str, ...] = (DENSE,)
):
    """
    Creates collection with a config for each vector the architecture declares.

    An architecture that retrieves on sparse vectors alone gets a collection with no dense
    config, so a comparison between it and a dense one is not a comparison against a collection
    that happens to hold both.
    """

    vectors_config, sparse_vectors_config = {}, {}

    for name in vectors:
        if name == SPARSE:
            sparse_vectors_config[name] = models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False)
            )
        else:
            vectors_config[name] = models.VectorParams(
                size=384,  # matches all-MiniLM-L6-v2
                distance=models.Distance.COSINE,
            )

    options = {"vectors_config": vectors_config}
    if sparse_vectors_config:
        options["sparse_vectors_config"] = sparse_vectors_config

    client.create_collection(collection_name=collection_name, **options)

    indexed = ", ".join(vectors)
    print(f"[INFO] created collection: {collection_name} ({indexed})")


def reset_vector_store(collection_name: str):
    """
    Resets a specific collection by name
    """
    client = get_qdrant_client()

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
        print(f"[INFO] deleted collection: {collection_name}")
    else:
        print(f"[INFO] collection '{collection_name}' does not exist")


def collection_points(client: QdrantClient, collection_name: str) -> int | None:
    """How many points a collection holds, or nothing when it does not exist."""
    try:
        return client.get_collection(collection_name).points_count
    except UnexpectedResponse as error:
        if error.status_code == 404:
            return None
        raise


def ensure_qdrant_up(timeout_s: int = 60) -> None:
    """Start Qdrant with compose and wait until it answers, or fail loudly.

    Raises RuntimeError when docker is missing or Qdrant never becomes ready,
    so callers never ingest into a store that is not there.
    """
    root = Path(__file__).resolve().parent.parent

    try:
        subprocess.run(
            ["docker", "compose", "up", "-d", "qdrant"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            "docker is not installed, so Qdrant cannot be started"
        ) from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"could not start Qdrant: {error.stderr.strip()}") from error

    ready_url = f"{Params.QDRANT_URL.rstrip('/')}/readyz"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(ready_url, timeout=3) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(2)

    raise RuntimeError(f"Qdrant never became ready at {ready_url}")
