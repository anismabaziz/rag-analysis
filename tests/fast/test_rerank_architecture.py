"""The rerank architecture: hybrid retrieval widened, then ordered by a cross-encoder.

Each clause below gets a test. The architecture is a declaration like the others, so
ingestion needs no dispatcher edit. Retrieval fetches wider than the committed depth
and returns the committed depth, with the width recorded in the run configuration.
Against the same store content the rerank run holds the same chunks as the hybrid run
in a different order, which is what lets a gain over hybrid be read as reranking doing
work rather than a different setup being measured. The summary reports it pooled and
per domain like the rest.
"""

import asyncio
from types import SimpleNamespace

import pytest

from architectures.hybrid import HybridRAG
from architectures.rerank import RerankRAG, rerank_chunks
from config.configuration import frozen_configuration
from config.params import Params
from core import registry
from core.chunk import Chunk, Provenance
from tests.fast.test_evaluation_run import found_everywhere, offline  # noqa: F401


class DenseEmbedModel:
    """Stands in for the encoder, which no test may download."""

    def embed(self, texts):
        return [[0.0] * 384 for _ in texts]


class SparseValues(list):
    """The sparse encoder returns arrays; the architecture only ever calls tolist on them."""

    def tolist(self):
        return list(self)


class SparseVector:
    """The shape the sparse encoder returns, without loading a model."""

    indices = SparseValues([0])
    values = SparseValues([1.0])


class SparseEmbedModel:
    """Stands in for the sparse encoder, which no test may download."""

    def embed(self, texts):
        return [SparseVector() for _ in texts]


class FakeClient:
    """Stands in for the store, answering every query with the points it was given."""

    def __init__(self, count):
        self.count = count
        self.limits = []

    def query_points(self, collection_name, prefetch, query):
        self.limits = [asked.limit for asked in prefetch]
        points = [
            SimpleNamespace(
                id=f"node-{index}",
                score=float(self.count - index),
                payload={
                    "text": f"chunk {index}",
                    "metadata": {
                        "source_file": "./documents/papers/a-paper.pdf",
                        "section": "A section",
                        "pages": [index + 1],
                    },
                },
            )
            for index in range(self.count)
        ]
        return SimpleNamespace(points=points)


class ReversingReranker:
    """Scores the last candidate highest, so a reranked ranking reads backwards."""

    def __init__(self):
        self.queries = []

    def rerank(self, query, documents):
        documents = list(documents)
        self.queries.append(query)
        return [float(index) for index in range(len(documents))]


def build_rerank(monkeypatch, count, reranker=None):
    """A rerank pipeline whose store holds `count` chunks and whose reranker is given."""
    monkeypatch.setattr(
        "architectures.rerank.get_qdrant_client", lambda: FakeClient(count)
    )
    monkeypatch.setattr(
        "architectures.rerank.get_sparse_embed_model", lambda: SparseEmbedModel()
    )
    pipeline = RerankRAG("llm", DenseEmbedModel(), "rag_rerank")
    if reranker is not None:
        pipeline._reranker = reranker
    return pipeline


def build_hybrid(monkeypatch, count):
    """A hybrid pipeline against the same store content, for the ordering comparison."""
    monkeypatch.setattr(
        "architectures.hybrid.get_qdrant_client", lambda: FakeClient(count)
    )
    monkeypatch.setattr(
        "architectures.hybrid.get_sparse_embed_model", lambda: SparseEmbedModel()
    )
    return HybridRAG("llm", DenseEmbedModel(), "rag_hybrid")


def test_rerank_is_registered_with_its_own_collection_and_hybrid_vectors():
    declared = registry.architecture("rerank")

    assert declared.collection == "rag_rerank"
    assert declared.vectors == (registry.DENSE, registry.SPARSE)
    assert declared.pipeline is RerankRAG


def test_rerank_is_ingestible_without_a_dispatcher_edit(monkeypatch):
    ingested = []
    monkeypatch.setattr(
        "build_index.build_index",
        lambda collection, vectors, **kwargs: ingested.append((collection, vectors)),
    )

    registry.architecture("rerank").ingest()

    assert ingested == [("rag_rerank", (registry.DENSE, registry.SPARSE))]


def test_retrieve_fetches_wider_than_the_committed_depth_and_returns_it(monkeypatch):
    pipeline = build_rerank(monkeypatch, Params.RERANK_CANDIDATES, ReversingReranker())

    chunks = asyncio.run(pipeline.retrieve("a question"))

    assert pipeline.client.limits == [
        Params.RERANK_CANDIDATES,
        Params.RERANK_CANDIDATES,
    ]
    assert len(chunks) == Params.TOP_K
    assert all(isinstance(chunk, Chunk) for chunk in chunks)


def test_retrieve_orders_by_the_reranker_rather_than_the_store(monkeypatch):
    reranker = ReversingReranker()
    pipeline = build_rerank(monkeypatch, Params.RERANK_CANDIDATES, reranker)

    chunks = asyncio.run(pipeline.retrieve("a question"))

    assert reranker.queries == ["a question"]
    assert [chunk.text for chunk in chunks] == [
        f"chunk {index}"
        for index in range(
            Params.RERANK_CANDIDATES - 1,
            Params.RERANK_CANDIDATES - 1 - Params.TOP_K,
            -1,
        )
    ]


def test_retrieve_carries_rerank_scores_on_the_same_provenance(monkeypatch):
    pipeline = build_rerank(monkeypatch, Params.RERANK_CANDIDATES, ReversingReranker())

    chunk = asyncio.run(pipeline.retrieve("a question"))[0]

    assert chunk.text == f"chunk {Params.RERANK_CANDIDATES - 1}"
    assert chunk.provenance.node_id == f"node-{Params.RERANK_CANDIDATES - 1}"
    assert chunk.score == pytest.approx(float(Params.RERANK_CANDIDATES - 1))


def test_empty_candidates_return_nothing_without_touching_the_reranker(monkeypatch):
    pipeline = build_rerank(monkeypatch, 0)

    async def refuse(query, documents):
        raise AssertionError("the reranker was asked about nothing")

    pipeline._reranker = SimpleNamespace(rerank=refuse)

    assert asyncio.run(pipeline.retrieve("a question")) == []


def test_rerank_chunks_keeps_text_and_provenance_and_keeps_only_the_depth():
    candidates = [
        Chunk(
            text=f"chunk {index}",
            score=float(index),
            provenance=Provenance(node_id=f"node-{index}"),
        )
        for index in range(Params.RERANK_CANDIDATES)
    ]

    ordered = rerank_chunks("a question", candidates, ReversingReranker(), Params.TOP_K)

    assert len(ordered) == Params.TOP_K
    assert [chunk.provenance.node_id for chunk in ordered] == [
        f"node-{index}"
        for index in range(
            Params.RERANK_CANDIDATES - 1,
            Params.RERANK_CANDIDATES - 1 - Params.TOP_K,
            -1,
        )
    ]
    assert all(
        chunk.text == f"chunk {chunk.provenance.node_id.split('-')[1]}"
        for chunk in ordered
    )


def test_a_rerank_run_holds_the_same_chunks_as_hybrid_in_a_different_order(monkeypatch):
    """The same store content through both pipelines: the membership matches, the order does not.

    This holds the pool at the committed depth, which is the one case where a wider fetch
    cannot promote anything outside hybrid's reach: with a full wider pool the reranker may
    also promote chunks hybrid's top would have cut off, and the wider tests above pin that.
    What never varies is the chunk identity itself, text and provenance carried through.
    """
    hybrid = asyncio.run(build_hybrid(monkeypatch, Params.TOP_K).retrieve("a question"))
    reranked = asyncio.run(
        build_rerank(monkeypatch, Params.TOP_K, ReversingReranker()).retrieve(
            "a question"
        )
    )

    assert len(hybrid) == len(reranked) == Params.TOP_K
    assert {chunk.provenance.node_id for chunk in reranked} == {
        chunk.provenance.node_id for chunk in hybrid
    }
    assert [chunk.provenance.node_id for chunk in reranked] != [
        chunk.provenance.node_id for chunk in hybrid
    ]
    assert all(isinstance(chunk, Chunk) for chunk in reranked)


def test_the_wider_candidate_set_is_recorded_in_the_run_configuration():
    configuration = frozen_configuration()

    assert configuration.retrieval_depth == Params.TOP_K
    assert configuration.rerank_candidates == Params.RERANK_CANDIDATES
    assert configuration.rerank_model.name == Params.RERANK_MODEL


def test_a_rerank_run_file_records_the_candidate_width(tmp_path, monkeypatch, offline):
    """The configuration a rerank run writes is the frozen one, width included."""
    from evaluation.run import run_evaluation

    monkeypatch.setattr(
        RerankRAG, "retrieve", _retrieve_from(found_everywhere, "papers")
    )

    run = asyncio.run(
        run_evaluation(
            "rerank",
            "papers",
            results_dir=tmp_path / "results",
            cache_dir=tmp_path / "cache",
        )
    )

    assert run.configuration.rerank_candidates == Params.RERANK_CANDIDATES
    assert run.architecture == "rerank"
    assert run.collection == "rag_rerank"


def test_rerank_is_reported_pooled_and_per_domain_alongside_the_others(
    tmp_path, monkeypatch, offline
):
    """One rerank run per domain lands in the summary next to every other architecture."""
    from evaluation.run import run_evaluation
    from evaluation.set import load_evaluation_set
    from evaluation.summary import load_runs, render_markdown, summarize

    sets = {domain: load_evaluation_set(domain) for domain in ("papers",)}

    def asked_of_the_sets(query: str, top_k: int = 5) -> list[Chunk]:
        for evaluation_set in sets.values():
            for question in evaluation_set.questions:
                if question.question == query:
                    return found_everywhere(question, top_k)[:top_k]
        raise AssertionError(f"the run asked a question no set holds: {query}")

    async def retrieve(self, query: str, top_k: int = 5) -> list[Chunk]:
        return asked_of_the_sets(query, top_k)

    for domain in ("papers",):
        monkeypatch.setattr(RerankRAG, "retrieve", retrieve)
        asyncio.run(
            run_evaluation(
                "rerank",
                domain,
                results_dir=tmp_path / "runs",
                cache_dir=tmp_path / "cache",
            )
        )

    summary = summarize(load_runs(tmp_path / "runs"))
    scopes = {row.scope for row in summary.rows if row.architecture == "rerank"}

    assert scopes == {
        "papers",
        "papers:identifier_heavy",
        "papers:paraphrase",
        "pooled",
        "pooled:identifier_heavy",
        "pooled:paraphrase",
    }
    assert render_markdown(summary).count("| rerank |") == 6


def test_rerank_summarizes_alongside_hybrid_without_an_isolation_refusal(
    tmp_path, monkeypatch, offline
):
    """Hybrid and rerank runs blend into one table because the width is frozen, not per-run.

    The candidate width is recorded on every run's configuration at the same value, so the
    isolation check that refuses mixed setups has nothing to refuse: a width that varied
    per architecture would stop the summary instead of blending it.
    """
    from evaluation.run import run_evaluation
    from evaluation.set import load_evaluation_set
    from evaluation.summary import load_runs, summarize

    sets = {domain: load_evaluation_set(domain) for domain in ("papers",)}

    def asked_of_the_sets(query: str, top_k: int = 5) -> list[Chunk]:
        for evaluation_set in sets.values():
            for question in evaluation_set.questions:
                if question.question == query:
                    return found_everywhere(question, top_k)[:top_k]
        raise AssertionError(f"the run asked a question no set holds: {query}")

    async def retrieve(self, query: str, top_k: int = 5) -> list[Chunk]:
        return asked_of_the_sets(query, top_k)

    for architecture in ("hybrid", "rerank"):
        monkeypatch.setattr(
            registry.architecture(architecture).pipeline, "retrieve", retrieve
        )
        for domain in ("papers",):
            asyncio.run(
                run_evaluation(
                    architecture,
                    domain,
                    results_dir=tmp_path / "runs",
                    cache_dir=tmp_path / "cache",
                )
            )

    summary = summarize(load_runs(tmp_path / "runs"))

    assert {row.architecture for row in summary.rows} == {"hybrid", "rerank"}
    for architecture in ("hybrid", "rerank"):
        assert {
            row.scope for row in summary.rows if row.architecture == architecture
        } == {
            "papers",
            "papers:identifier_heavy",
            "papers:paraphrase",
            "pooled",
            "pooled:identifier_heavy",
            "pooled:paraphrase",
        }


def _retrieve_from(chunks_for, domain):
    """A retriever answering one domain's questions with the passages a test names."""
    from evaluation.set import load_evaluation_set

    evaluation_set = load_evaluation_set(domain)

    def asked_of_the_set(query: str, top_k: int = 5) -> list[Chunk]:
        question = next(
            one for one in evaluation_set.questions if one.question == query
        )
        return chunks_for(question, top_k)[:top_k]

    async def retrieve(self, query: str, top_k: int = 5) -> list[Chunk]:
        return asked_of_the_set(query, top_k)

    return retrieve
