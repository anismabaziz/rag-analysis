"""The store the project talks to, and the collections it is asked to hold."""

import importlib


class RecordingClient:
	"""A store that records the collection it was asked to create instead of creating it."""

	def __init__(self):
		self.created = {}

	def create_collection(self, **options):
		self.created = options


def test_the_client_is_built_from_the_configured_url(monkeypatch):
	store = importlib.import_module("vector.store")

	seen = {}
	monkeypatch.setattr(store, "QdrantClient", lambda **kwargs: seen.update(kwargs))
	monkeypatch.setattr(store.Params, "QDRANT_URL", "http://qdrant.internal:6333")

	store.get_qdrant_client()

	assert seen == {"url": "http://qdrant.internal:6333"}


def test_a_dense_architecture_gets_a_collection_with_one_dense_vector():
	store = importlib.import_module("vector.store")
	client = RecordingClient()

	store.create_collection(client, "rag_naive", ("dense",))

	assert list(client.created["vectors_config"]) == ["dense"]
	assert "sparse_vectors_config" not in client.created


def test_a_hybrid_architecture_gets_a_collection_holding_both_signals():
	store = importlib.import_module("vector.store")
	client = RecordingClient()

	store.create_collection(client, "rag_hybrid", ("dense", "sparse"))

	assert list(client.created["vectors_config"]) == ["dense"]
	assert list(client.created["sparse_vectors_config"]) == ["sparse"]


def test_a_sparse_only_architecture_does_not_get_a_dense_vector_too():
	"""Otherwise a sparse-only run would retrieve on a signal it never claimed to index."""
	store = importlib.import_module("vector.store")
	client = RecordingClient()

	store.create_collection(client, "rag_sparse", ("sparse",))

	assert client.created["vectors_config"] == {}
	assert list(client.created["sparse_vectors_config"]) == ["sparse"]
