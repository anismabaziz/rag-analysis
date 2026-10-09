"""Latency percentiles and generation token cost, reported per run.

Every run times retrieval and generation separately, because they have different
causes and a reader comparing architectures needs to know which one moved. The
per-question numbers live in the run file next to the scores, and the medians
and 95th percentiles in the aggregates are computed from that run's own
questions rather than across runs. Token cost is counted deterministically from
the rendered prompt and the answer, so the same run always reports the same
cost without calling anything.
"""

import asyncio
import json

from core import registry
from evaluation.run import run_evaluation
from test_evaluation_run import found_everywhere, found_nothing, offline  # noqa: F401
from test_evaluation_run import retrieve_from, run_over


def test_median_and_p95_retrieval_latency_are_reported(tmp_path, monkeypatch, offline):
    run = run_over(tmp_path, monkeypatch, "naive", "papers", found_everywhere)

    assert run.aggregates.retrieval_latency_p50_s >= 0.0
    assert run.aggregates.retrieval_latency_p95_s >= 0.0
    assert (
        run.aggregates.retrieval_latency_p50_s <= run.aggregates.retrieval_latency_p95_s
    )


def test_retrieval_and_generation_latency_are_reported_separately(
    tmp_path, monkeypatch, offline
):
    run = run_over(tmp_path, monkeypatch, "naive", "papers", found_everywhere)

    for result in run.results:
        assert result.retrieval_latency_s >= 0.0
        assert result.generation_latency_s is not None
        assert result.generation_latency_s >= 0.0

    assert run.aggregates.generation_latency_p50_s is not None
    assert run.aggregates.generation_latency_p95_s is not None


def test_empty_retrieval_never_reaches_generation_and_carries_no_generation_latency(
    tmp_path, monkeypatch, offline
):
    run = run_over(tmp_path, monkeypatch, "naive", "papers", found_nothing)

    for result in run.results:
        assert result.retrieval_latency_s >= 0.0
        assert result.generation_latency_s is None

    assert run.aggregates.generation_latency_p50_s is None
    assert run.aggregates.generation_latency_p95_s is None


def test_generation_token_cost_is_reported(tmp_path, monkeypatch, offline):
    run = run_over(tmp_path, monkeypatch, "naive", "papers", found_everywhere)

    for result in run.results:
        assert result.prompt_tokens is not None and result.prompt_tokens > 0
        assert result.completion_tokens is not None
        assert result.total_tokens == result.prompt_tokens + result.completion_tokens

    assert run.aggregates.total_tokens > 0
    assert run.aggregates.generation_count == len(run.results)
    assert run.aggregates.total_tokens_mean > 0


def test_refusals_carry_no_token_cost(tmp_path, monkeypatch, offline):
    run = run_over(tmp_path, monkeypatch, "naive", "papers", found_nothing)

    for result in run.results:
        assert result.prompt_tokens is None
        assert result.completion_tokens is None
        assert result.total_tokens is None

    assert run.aggregates.total_tokens == 0
    assert run.aggregates.generation_count == 0


def test_latency_percentiles_are_computed_from_this_runs_questions(
    tmp_path, monkeypatch, offline
):
    run = run_over(tmp_path, monkeypatch, "naive", "papers", found_everywhere)

    retrieval = sorted(result.retrieval_latency_s for result in run.results)
    generation = sorted(
        result.generation_latency_s
        for result in run.results
        if result.generation_latency_s is not None
    )

    assert run.aggregates.retrieval_latency_p50_s == _percentile(retrieval, 50)
    assert run.aggregates.retrieval_latency_p95_s == _percentile(retrieval, 95)
    assert run.aggregates.generation_latency_p50_s == _percentile(generation, 50)
    assert run.aggregates.generation_latency_p95_s == _percentile(generation, 95)


def _percentile(ordered: list[float], rank: float) -> float:
    """The rank percentile of already-sorted values, interpolated between closest ranks.

    An independent reading of the run's own stored questions, so the test checks the
    aggregates were computed from this run rather than trusting the helper they share.
    """
    position = (len(ordered) - 1) * rank / 100
    low = int(position)
    high = low + (1 if position != low else 0)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def test_latency_and_cost_appear_in_the_run_file_not_only_in_printed_output(
    tmp_path, monkeypatch, offline
):
    run = run_over(tmp_path, monkeypatch, "naive", "papers", found_everywhere)
    written = sorted((tmp_path / "results").rglob("*.json"))[0]
    recorded = json.loads(written.read_text())

    for key in (
        "retrieval_latency_p50_s",
        "retrieval_latency_p95_s",
        "generation_latency_p50_s",
        "generation_latency_p95_s",
        "total_tokens",
    ):
        assert key in recorded["aggregates"], key

    first = recorded["results"][0]
    for key in (
        "retrieval_latency_s",
        "generation_latency_s",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    ):
        assert key in first, key

    assert (
        recorded["aggregates"]["retrieval_latency_p50_s"]
        == run.aggregates.retrieval_latency_p50_s
    )
