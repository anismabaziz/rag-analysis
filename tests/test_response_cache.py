"""Model responses cached on disk, so a repeated run is free and identical.

A hosted model is not deterministic even at temperature zero and it gets deprecated, so
without this the published numbers silently change or become unreproducible the moment the
pinned model is withdrawn. The cache lets the README claim its numbers are exactly
reproducible: a second identical run is served entirely from cache and issues no model
calls, while a change to the prompt, model, or temperature invalidates the entry.

These tests drive the cache through the evaluation entry point, the same call a reader
makes to regenerate the results table. No test calls a model or a store.
"""

import asyncio
import gzip
import json
import subprocess
from pathlib import Path

from core import registry
from evaluation.cache import CACHE_DIR, CacheKey
from evaluation.run import run_evaluation
from test_evaluation_run import found_everywhere, found_nothing, offline  # noqa: F401
from test_evaluation_run import retrieve_from


class CountingLlm:
	"""Stands in for the hosted model, counting how often a run reaches it."""

	def __init__(self):
		self.calls = 0

	async def apredict(self, template, **kwargs):
		self.calls += 1
		return f"an answer for {kwargs.get('query', '')}"


def answering_with(llm, monkeypatch):
	"""The same offline wiring as every other run test, with a model the test can count."""
	monkeypatch.setattr(registry, "get_generation_llm", lambda: llm)
	return llm


def run_with_cache(tmp_path, monkeypatch, architecture, domain, chunks_for, cache_dir, **kwargs):
	"""A run of `architecture` over `domain` with responses cached under `cache_dir`."""
	monkeypatch.setattr(
		registry.architecture(architecture).pipeline, "retrieve", retrieve_from(chunks_for)
	)

	return asyncio.run(
		run_evaluation(
			architecture, domain, results_dir=tmp_path / "results", cache_dir=cache_dir, **kwargs
		)
	)


def test_a_second_identical_run_is_served_entirely_from_cache(tmp_path, monkeypatch, offline):
	"""A repeated run issues no model calls and answers identically."""
	first_llm = answering_with(CountingLlm(), monkeypatch)
	cache_dir = tmp_path / "cache"

	first = run_with_cache(tmp_path, monkeypatch, "naive", "papers", found_everywhere, cache_dir)

	assert first_llm.calls > 0
	assert sorted(cache_dir.rglob("*.json.gz")), "a miss writes the response to the cache"

	class RefusingLlm:
		async def apredict(self, template, **kwargs):
			raise AssertionError("the second run reached the model instead of the cache")

	answering_with(RefusingLlm(), monkeypatch)

	second = run_with_cache(tmp_path, monkeypatch, "naive", "papers", found_everywhere, cache_dir)

	assert [result.answer for result in second.results] == [
		result.answer for result in first.results
	]


def test_a_miss_calls_the_model_and_a_hit_does_not(tmp_path, monkeypatch, offline):
	"""The first run for a question reaches the model; the rerun for it does not."""
	first_llm = answering_with(CountingLlm(), monkeypatch)
	cache_dir = tmp_path / "cache"

	run_with_cache(tmp_path, monkeypatch, "naive", "papers", found_everywhere, cache_dir)
	misses = first_llm.calls

	assert misses > 0

	second_llm = answering_with(CountingLlm(), monkeypatch)

	run_with_cache(tmp_path, monkeypatch, "naive", "papers", found_everywhere, cache_dir)

	assert second_llm.calls == 0
	assert first_llm.calls == misses


def test_changing_the_prompt_invalidates_the_entry(tmp_path, monkeypatch, offline):
	"""A different prompt is a miss, even for the same query, context, and model."""
	answering_with(CountingLlm(), monkeypatch)
	cache_dir = tmp_path / "cache"

	run_with_cache(tmp_path, monkeypatch, "naive", "papers", found_everywhere, cache_dir)

	monkeypatch.setattr("config.configuration.answer_prompt", lambda: "a different prompt")
	monkeypatch.setattr("config.configuration.prompt_fingerprint", lambda: "sha256:other")
	second_llm = answering_with(CountingLlm(), monkeypatch)

	run_with_cache(tmp_path, monkeypatch, "naive", "papers", found_everywhere, cache_dir)

	assert second_llm.calls > 0


def test_changing_the_model_invalidates_the_entry(tmp_path, monkeypatch, offline):
	"""A different model is a miss, even for the same prompt, context, and query."""
	answering_with(CountingLlm(), monkeypatch)
	cache_dir = tmp_path / "cache"

	run_with_cache(tmp_path, monkeypatch, "naive", "papers", found_everywhere, cache_dir)

	monkeypatch.setattr("config.params.Params.GENERATION_MODEL", "another-model")
	second_llm = answering_with(CountingLlm(), monkeypatch)

	run_with_cache(tmp_path, monkeypatch, "naive", "papers", found_everywhere, cache_dir)

	assert second_llm.calls > 0


def test_changing_the_temperature_invalidates_the_entry(tmp_path, monkeypatch, offline):
	"""A different temperature is a miss, even for the same prompt, model, and query."""
	answering_with(CountingLlm(), monkeypatch)
	cache_dir = tmp_path / "cache"

	run_with_cache(tmp_path, monkeypatch, "naive", "papers", found_everywhere, cache_dir)

	monkeypatch.setattr("config.params.Params.GENERATION_TEMPERATURE", 0.7)
	second_llm = answering_with(CountingLlm(), monkeypatch)

	run_with_cache(tmp_path, monkeypatch, "naive", "papers", found_everywhere, cache_dir)

	assert second_llm.calls > 0


def test_retrieving_different_context_invalidates_the_entry(tmp_path, monkeypatch, offline):
	"""The same question retrieving different chunks is a miss, not the earlier answer.

	The context is part of the key precisely because two architectures retrieve different
	chunks for the same question. Without it, one architecture's answer would be replayed as
	the other's.
	"""
	answering_with(CountingLlm(), monkeypatch)
	cache_dir = tmp_path / "cache"

	run_with_cache(tmp_path, monkeypatch, "naive", "papers", found_everywhere, cache_dir)
	second_llm = answering_with(CountingLlm(), monkeypatch)

	run_with_cache(tmp_path, monkeypatch, "naive", "papers", answering_from_other_text, cache_dir)

	assert second_llm.calls > 0


def answering_from_other_text(question, top_k):
	"""Retrieval that answers the same questions out of different text, changing the context."""
	from dataclasses import replace

	from test_evaluation_run import a_chunk

	if question.document is None:
		return [a_chunk("attention-is-all-you-need", "3.2.2 Multi-Head Attention")]

	return [replace(a_chunk(question.document, question.section), text="a different span")]


def test_two_questions_retrieving_the_same_context_get_their_own_answers(tmp_path, monkeypatch, offline):
	"""The query is part of the key, so identical context does not replay one answer twice.

	Without it, a retriever that returns the same passage for several questions would serve
	every one of them the first question's answer.
	"""
	llm = answering_with(CountingLlm(), monkeypatch)
	cache_dir = tmp_path / "cache"

	run = run_with_cache(tmp_path, monkeypatch, "naive", "papers", same_text_for_everything, cache_dir)

	assert llm.calls == len(run.results)
	assert len({result.answer for result in run.results}) == len(run.results)


def same_text_for_everything(question, top_k):
	"""Retrieval that returns the same passage for every question, whatever was asked."""
	from test_evaluation_run import a_chunk

	return [a_chunk("attention-is-all-you-need", "3.2.2 Multi-Head Attention")]


def test_empty_retrieval_never_touches_the_cache_or_the_model(tmp_path, monkeypatch, offline):
	"""A question with nothing retrieved is refused without reading or writing the cache."""
	llm = answering_with(CountingLlm(), monkeypatch)
	cache_dir = tmp_path / "cache"

	run = run_with_cache(tmp_path, monkeypatch, "naive", "papers", found_nothing, cache_dir)

	assert llm.calls == 0
	assert list(cache_dir.rglob("*.json.gz")) == []
	assert all(result.answer for result in run.results)


def test_cache_entries_are_compressed_json_named_after_the_key(tmp_path, monkeypatch, offline):
	"""Each entry is a small gzipped record whose file name is the digest of its key."""
	answering_with(CountingLlm(), monkeypatch)
	cache_dir = tmp_path / "cache"

	run = run_with_cache(tmp_path, monkeypatch, "naive", "papers", found_everywhere, cache_dir)

	entries = sorted(cache_dir.rglob("*.json.gz"))
	assert entries

	first = entries[0]
	with gzip.open(first, "rt", encoding="utf-8") as handle:
		recorded = json.load(handle)

	assert set(recorded) == {"prompt", "model", "temperature", "query", "context", "response"}
	assert recorded["response"] in [result.answer for result in run.results]

	key = CacheKey(
		prompt=recorded["prompt"],
		model=recorded["model"],
		temperature=recorded["temperature"],
		query=recorded["query"],
		context=recorded["context"],
	)
	assert key.digest() == first.name.removesuffix(".json.gz")


def test_the_cache_does_not_dominate_the_repository(tmp_path, monkeypatch, offline):
	"""A whole domain's responses stay small enough to commit beside the results they back."""
	answering_with(CountingLlm(), monkeypatch)
	cache_dir = tmp_path / "cache"

	run_with_cache(tmp_path, monkeypatch, "naive", "papers", found_everywhere, cache_dir)

	entries = sorted(cache_dir.rglob("*.json.gz"))
	compressed = sum(path.stat().st_size for path in entries)

	with gzip.open(entries[0], "rt", encoding="utf-8") as handle:
		plain = len(handle.read())

	assert compressed < 128 * 1024, f"the papers responses weigh {compressed} bytes compressed"
	assert plain > 0
	assert entries[0].stat().st_size < plain, "a compressed entry is smaller than its JSON"


def test_recording_the_same_response_twice_leaves_the_bytes_unchanged(tmp_path, monkeypatch, offline):
	"""Regenerating a run must not churn the cache, or every rerun is a diff in git."""
	answering_with(CountingLlm(), monkeypatch)
	cache_dir = tmp_path / "cache"

	run_with_cache(tmp_path, monkeypatch, "naive", "papers", found_everywhere, cache_dir)
	before = {path.name: path.read_bytes() for path in cache_dir.rglob("*.json.gz")}

	# Point the cache at fresh files so the rerun is a miss and rewrites every entry.
	fresh = tmp_path / "fresh-cache"
	run_with_cache(tmp_path, monkeypatch, "naive", "papers", found_everywhere, fresh)
	rerun = {path.name: path.read_bytes() for path in fresh.rglob("*.json.gz")}

	assert rerun == before


def test_the_default_cache_lives_beside_the_runs_and_is_not_ignored():
	"""The committed cache is the default one, so a reader reproducing the table reuses it."""
	assert CACHE_DIR == Path("results/cache")

	ignored = subprocess.run(
		["git", "check-ignore", "-q", str(CACHE_DIR / "example.json.gz")],
		capture_output=True,
	)

	assert ignored.returncode != 0, "the response cache must be committable, not ignored"


def test_the_run_command_writes_the_cache_where_it_was_asked_to(tmp_path, monkeypatch, offline):
	"""A reader can point a run at a cache of their own, rather than only the committed one."""
	import sys

	from main import main as cli

	answering_with(CountingLlm(), monkeypatch)
	cache_dir = tmp_path / "cache"
	monkeypatch.setattr(
		registry.architecture("naive").pipeline, "retrieve", retrieve_from(found_everywhere)
	)
	monkeypatch.setattr(
		sys,
		"argv",
		[
			"rag-analysis",
			"run",
			"naive",
			"--domain",
			"papers",
			"--out",
			str(tmp_path / "results"),
			"--cache-dir",
			str(cache_dir),
		],
	)

	asyncio.run(cli())

	assert sorted(cache_dir.rglob("*.json.gz")), "the run wrote its responses where it was told to"
	assert list((tmp_path / "results").rglob("*.json"))
