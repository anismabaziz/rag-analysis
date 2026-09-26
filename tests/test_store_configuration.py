"""The store the project talks to has to be the one the reader configured."""

import importlib


def test_the_client_is_built_from_the_configured_url(monkeypatch):
	store = importlib.import_module("vector.store")

	seen = {}
	monkeypatch.setattr(store, "QdrantClient", lambda **kwargs: seen.update(kwargs))
	monkeypatch.setattr(store.Params, "QDRANT_URL", "http://qdrant.internal:6333")

	store.get_qdrant_client()

	assert seen == {"url": "http://qdrant.internal:6333"}
