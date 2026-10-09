"""The chunking comparison: fixed-size, semantic, and hierarchical on retrieval only.

One axis varies, the chunker, and everything else is frozen: the same dense
retrieval, the same depth, the same prompt, models, and temperature. A
hierarchical strategy indexes whole sections alongside their slices and semantic
splits vary in length, so the generator would not read the same context and answer
quality is not compared. The token budget is measured and the table says whether
it came out matched.
"""

import asyncio
import sys

import pytest

from config.configuration import Chunker
from config.params import CHUNKING_STRATEGIES
from core import registry
from core.chunk import Chunk
from tests.fast.test_evaluation_run import found_everywhere, offline  # noqa: F401

CHUNKING = ("chunk-fixed", "chunk-semantic", "chunk-hierarchical")


def test_all_three_chunking_strategies_are_registered():
    assert {name for name in CHUNKING} <= {a.name for a in registry.all_architectures()}


def test_each_chunking_architecture_gets_a_collection_of_its_own():
    declared = {a.name: a for a in registry.all_architectures() if a.name in CHUNKING}

    assert set(declared) == set(CHUNKING)
    assert len({a.collection for a in declared.values()}) == 3
    for architecture in declared.values():
        assert architecture.vectors == (registry.DENSE,)


def test_chunking_retrieval_is_dense_only_like_naive(monkeypatch):
    """The axis under test is the chunker, so retrieval itself must not vary."""
    from architectures.naive import NaiveRAG

    monkeypatch.setattr(registry, "get_generation_llm", lambda: "llm")
    monkeypatch.setattr(registry, "get_embed_model", lambda: "encoder")
    naive = registry.architecture("naive").build()

    for name in CHUNKING:
        pipeline = registry.architecture(name).build()
        assert isinstance(pipeline, NaiveRAG)
        assert pipeline.collection_name == registry.architecture(name).collection
        assert pipeline.collection_name != naive.collection_name


def test_chunking_ingestion_needs_no_dispatcher_edit(monkeypatch):
    """Ingestion reads the chunker the architecture declares, like vectors."""
    ingested = []
    monkeypatch.setattr(
        "build_index.build_index",
        lambda collection, vectors=(registry.DENSE,), **kwargs: ingested.append(
            (collection, vectors, kwargs)
        ),
    )

    for name in CHUNKING:
        registry.architecture(name).ingest()

    assert len(ingested) == 3
    strategies = set()
    for collection, vectors, kwargs in ingested:
        assert vectors == (registry.DENSE,)
        assert kwargs.get("chunker")
        strategies.add(kwargs["chunker"].strategy)
    assert strategies == {"fixed", "semantic", "hierarchical"}


def test_build_index_hands_each_strategy_to_the_splitter(monkeypatch):
    """The ingestion path: each named strategy reaches the splitter it names."""
    import build_index

    seen = {}

    class RecordingSplitter:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def process(self, elements):
            return []

    class FakeLoader:
        def __init__(self, *args, **kwargs):
            pass

        def load(self, path):
            return []

    class FakeClient:
        def upsert(self, **kwargs):
            pass

    monkeypatch.setattr(build_index, "PDFSplitter", RecordingSplitter)
    monkeypatch.setattr(build_index, "PDFLoader", FakeLoader)
    monkeypatch.setattr(build_index, "get_embed_model", lambda: None)
    monkeypatch.setattr(build_index, "get_sparse_embed_model", lambda: None)
    monkeypatch.setattr(build_index, "get_qdrant_client", lambda: FakeClient())
    monkeypatch.setattr(build_index, "reset_vector_store", lambda collection: None)
    monkeypatch.setattr(
        build_index, "create_collection", lambda client, name, vectors: None
    )
    monkeypatch.setattr(build_index, "glob", lambda pattern, recursive: [])

    for strategy in CHUNKING_STRATEGIES:
        seen.clear()
        build_index.build_index(
            "rag_test", vectors=("dense",), chunker=Chunker.variant(strategy, 512, 128)
        )
        assert seen.get("chunking_strategy") == strategy
        assert seen.get("chunk_size") == 512


def test_a_chunking_run_records_the_chunker_it_was_measured_at(
    tmp_path, monkeypatch, offline
):
    from evaluation.run import run_evaluation

    for name in CHUNKING:
        monkeypatch.setattr(
            registry.architecture(name).pipeline,
            "retrieve",
            _retrieve_from(found_everywhere),
        )
        run = asyncio.run(
            run_evaluation(
                name,
                "papers",
                results_dir=tmp_path / "runs",
                cache_dir=tmp_path / "cache",
            )
        )
        declared = registry.architecture(name).resolved_chunker()
        assert run.architecture == name
        assert run.configuration.chunker == declared
        assert run.configuration.chunker.is_committed is False


def test_a_hierarchical_run_records_the_section_size_it_indexed_parents_at(
    tmp_path, monkeypatch, offline
):
    """The chunk size column is the one number a reader judges budget matching by, so it has
    to be the size that actually cut the corpus: a hierarchical strategy indexing whole
    sections alongside their slices indexes them four times larger."""
    from evaluation.run import run_evaluation

    monkeypatch.setattr(
        registry.architecture("chunk-hierarchical").pipeline,
        "retrieve",
        _retrieve_from(found_everywhere),
    )
    run = asyncio.run(
        run_evaluation(
            "chunk-hierarchical",
            "papers",
            results_dir=tmp_path / "runs",
            cache_dir=tmp_path / "cache",
        )
    )

    assert run.configuration.chunker.parent_size == 4 * run.configuration.chunker.size


def test_the_splitter_cuts_a_hierarchical_corpus_at_the_size_the_run_records(
    monkeypatch,
):
    """The recorded parent size has to be the one the splitter uses, or the run file is a claim
    the ingestion never acted on."""
    import data.splitter

    monkeypatch.setattr(
        data.splitter.HuggingFaceEmbedding, "__init__", lambda self, **kwargs: None
    )
    splitter = data.splitter.PDFSplitter(
        chunking_strategy="hierarchical", chunk_size=512, chunk_overlap=128
    )

    assert splitter.node_parser.chunk_sizes == [2048, 512, 128]
    assert splitter.hierarchical_sizes() == [2048, 512, 128]


def test_chunking_runs_are_compared_on_retrieval_metrics_only(
    tmp_path, monkeypatch, offline
):
    """No citation or exact-match column: answer quality cannot hold context constant."""
    from evaluation.chunking import (
        load_chunking_runs,
        render_chunking_markdown,
        summarize_chunking,
    )

    runs_dir = _run_chunking_over_papers(tmp_path, monkeypatch)
    table = render_chunking_markdown(summarize_chunking(load_chunking_runs(runs_dir)))

    for metric in (
        "recall@1",
        "recall@3",
        "recall@5",
        "ndcg@1",
        "ndcg@3",
        "ndcg@5",
        "mrr",
    ):
        assert metric in table
    assert "cite" not in table
    assert "exact_match" not in table


def test_chunking_table_states_the_token_budget_and_its_gaps(
    tmp_path, monkeypatch, offline
):
    """A matched budget is stated, and where a strategy cannot match it, so is that."""
    from evaluation.chunking import (
        load_chunking_runs,
        render_chunking_markdown,
        summarize_chunking,
    )

    runs_dir = _run_chunking_over_papers(tmp_path, monkeypatch)
    table = render_chunking_markdown(summarize_chunking(load_chunking_runs(runs_dir)))

    assert "token budget" in table.lower()
    assert "retrieval only" in table.lower() or "retrieval-only" in table.lower()
    for name in CHUNKING:
        assert name in table


def test_the_chunking_table_excludes_runs_measured_at_the_committed_chunker(
    tmp_path, monkeypatch, offline
):
    """Both kinds of run live in one directory. A retriever row in a table framed as one axis
    varying would be comparing a retriever against a chunker."""
    from evaluation.chunking import (
        load_chunking_runs,
        render_chunking_markdown,
        summarize_chunking,
    )
    from evaluation.run import run_evaluation
    from tests.fast.test_evaluation_run import found_nothing

    monkeypatch.setattr(
        registry.architecture("naive").pipeline,
        "retrieve",
        _retrieve_from(found_nothing),
    )
    asyncio.run(
        run_evaluation(
            "naive",
            "papers",
            results_dir=tmp_path / "runs",
            cache_dir=tmp_path / "cache",
        )
    )
    runs_dir = _run_chunking_over_papers(tmp_path, monkeypatch)

    table = render_chunking_markdown(summarize_chunking(load_chunking_runs(runs_dir)))

    assert "| naive |" not in table
    assert {run.architecture for run in load_chunking_runs(runs_dir)} == set(CHUNKING)


def test_the_chunking_table_says_whether_the_budget_came_out_matched(
    tmp_path, monkeypatch, offline
):
    """The claim is measured off the rows, not asserted: a budget that did not match is
    reported as unmatched rather than described as matched."""
    from evaluation.chunking import (
        load_chunking_runs,
        render_chunking_markdown,
        summarize_chunking,
    )

    runs_dir = _run_chunking_over_papers(tmp_path, monkeypatch)
    summary = summarize_chunking(load_chunking_runs(runs_dir))
    table = render_chunking_markdown(summary)

    pooled = [
        row.retrieved_words_mean
        for row in summary.rows
        if row.scope == "pooled" and row.retrieved_words_mean is not None
    ]
    spread = max(pooled) - min(pooled)
    verdict = (
        "matched across strategies"
        if spread <= 0.1 * max(pooled)
        else "NOT matched across strategies"
    )

    assert summary.budget_matched is (spread <= 0.1 * max(pooled))
    assert f"budget is {verdict}" in table
    for measured in pooled:
        assert f"{measured:.0f} words" in table


def test_chunking_table_is_pooled_and_per_domain(tmp_path, monkeypatch, offline):
    from evaluation.chunking import load_chunking_runs, summarize_chunking

    summary = summarize_chunking(
        load_chunking_runs(_run_chunking_over_papers(tmp_path, monkeypatch))
    )

    for name in CHUNKING:
        assert {row.scope for row in summary.rows if row.architecture == name} == {
            "papers",
            "pooled",
        }


def test_chunking_summary_refuses_runs_that_differ_beyond_the_chunker(
    tmp_path, monkeypatch, offline
):
    """One axis varies. A second varying axis stops the summary instead of blending."""
    import json

    from evaluation.chunking import load_chunking_runs, summarize_chunking

    runs_dir = _run_chunking_over_papers(tmp_path, monkeypatch)
    target = next(runs_dir.rglob("chunk-fixed-*.json"))
    recorded = json.loads(target.read_text())
    recorded["configuration"]["retrieval_depth"] = 10
    target.write_text(json.dumps(recorded))

    with pytest.raises(ValueError, match="retrieval_depth"):
        summarize_chunking(load_chunking_runs(runs_dir))


def test_the_main_summary_leaves_chunking_runs_out_rather_than_blending_them(
    tmp_path, monkeypatch, offline
):
    """The main table is the retrieval comparison at one chunker. A variant is left out of it
    rather than refused, because a chunking run is a different measurement, not a broken one."""
    from evaluation.run import run_evaluation
    from evaluation.summary import load_committed_runs, summarize
    from tests.fast.test_evaluation_run import found_nothing

    monkeypatch.setattr(
        registry.architecture("naive").pipeline,
        "retrieve",
        _retrieve_from(found_everywhere),
    )
    asyncio.run(
        run_evaluation(
            "naive",
            "papers",
            results_dir=tmp_path / "runs",
            cache_dir=tmp_path / "cache",
        )
    )
    runs_dir = _run_chunking_over_papers(tmp_path, monkeypatch)

    summary = summarize(load_committed_runs(runs_dir))

    assert {row.architecture for row in summary.rows} == {"naive"}


def test_the_chunking_command_renders_the_retrieval_only_table(
    tmp_path, monkeypatch, capsys, offline
):
    from main import main as cli

    _run_chunking_over_papers(tmp_path, monkeypatch, out=tmp_path / "runs")
    out = tmp_path / "chunking.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rag-analysis",
            "summarize-chunking",
            "--results-dir",
            str(tmp_path / "runs"),
            "--out",
            str(out),
        ],
    )

    asyncio.run(cli())
    reported = capsys.readouterr().out
    table = out.read_text()

    assert "Chunking" in table
    assert "cite" not in table
    assert table.strip() in reported.strip() or table in reported


def _retrieve_from(chunks_for):
    from evaluation.set import load_evaluation_set

    sets = {domain: load_evaluation_set(domain) for domain in ("papers",)}

    def asked_of_the_sets(query: str, top_k: int = 5) -> list[Chunk]:
        for evaluation_set in sets.values():
            for question in evaluation_set.questions:
                if question.question == query:
                    return chunks_for(question, top_k)[:top_k]
        raise AssertionError(f"the run asked a question no set holds: {query}")

    async def retrieve(self, query: str, top_k: int = 5) -> list[Chunk]:
        return asked_of_the_sets(query, top_k)

    return retrieve


def _run_chunking_over_papers(tmp_path, monkeypatch, out=None):
    from evaluation.run import run_evaluation

    results_dir = out or (tmp_path / "runs")
    for name in CHUNKING:
        monkeypatch.setattr(
            registry.architecture(name).pipeline,
            "retrieve",
            _retrieve_from(found_everywhere),
        )
        for domain in ("papers",):
            asyncio.run(
                run_evaluation(
                    name, domain, results_dir=results_dir, cache_dir=tmp_path / "cache"
                )
            )
    return results_dir
