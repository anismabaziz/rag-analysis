"""The summary table over run files: pooled and per domain, generated rather than written.

The table is where the project's claim is read, so what it shows has to be traceable to the
files it was generated from: every number in it is a mean over stored per-question results, the
depths are the ones the runs recorded rather than a flag on the command, and runs measured at
different configurations are refused instead of blended into one row.
"""

import asyncio
import json
import sys
from pathlib import Path

import pytest

from core import registry
from core.chunk import Chunk
from evaluation.run import load_run_file, run_evaluation
from evaluation.set import load_evaluation_set
from test_evaluation_run import found_everywhere, found_nothing, offline  # noqa: F401


def retrieve_from_both(chunks_for):
	"""A retriever that answers each question of either committed set with the passages named."""

	sets = {domain: load_evaluation_set(domain) for domain in ("papers", "manuals")}

	def asked_of_the_sets(query: str, top_k: int = 5) -> list[Chunk]:
		for evaluation_set in sets.values():
			for question in evaluation_set.questions:
				if question.question == query:
					return chunks_for(question, top_k)[:top_k]

		raise AssertionError(f"the run asked a question no set holds: {query}")

	async def retrieve(self, query: str, top_k: int = 5) -> list[Chunk]:
		return asked_of_the_sets(query, top_k)

	return retrieve


def runs_dir(tmp_path):
	"""Where this test's run files live, laid out the way runs are written."""
	return tmp_path / "runs"


def run_files(runs):
	"""Every run file below the directory, in a stable order."""
	return sorted(runs.rglob("*.json"))


def run_over_results(tmp_path, monkeypatch, architecture, domain, chunks_for):
	"""A run of `architecture` over `domain`, written below this test's runs directory."""
	monkeypatch.setattr(
		registry.architecture(architecture).pipeline, "retrieve", retrieve_from_both(chunks_for)
	)

	return asyncio.run(
		run_evaluation(architecture, domain, results_dir=runs_dir(tmp_path), cache_dir=tmp_path / "cache")
	)


def retrieval_architectures():
	"""The architectures the main table compares: the ones measured at the committed chunker.

	Chunking variants declare their own chunker, so they are compared in the
	retrieval-only chunking table instead of appearing here.
	"""
	return [registered for registered in registry.all_architectures() if registered.chunker is None]


def every_architecture_two_domains(tmp_path, monkeypatch):
	"""One run of each retrieval architecture over each domain, written where runs are written.

	Dense finds the passages and the rest find nothing, so the rows differ rather than repeating.
	"""
	finders = {"naive": found_everywhere}

	for domain in ("papers", "manuals"):
		for registered in retrieval_architectures():
			run_over_results(
				tmp_path, monkeypatch, registered.name, domain, finders.get(registered.name, found_nothing)
			)

	return runs_dir(tmp_path)


def test_every_architecture_is_compared_in_one_table(tmp_path, monkeypatch, offline):
	"""The finding is the comparison, so one table holds every architecture side by side.

	The sparse architecture is in the table because without it a gain over dense could be the
	sparse signal alone, and no row would say which of the two parts did the work.
	"""
	from evaluation.summary import load_committed_runs, render_markdown, summarize

	registered_names = {registered.name for registered in retrieval_architectures()}
	table = render_markdown(summarize(load_committed_runs(every_architecture_two_domains(tmp_path, monkeypatch))))

	assert "sparse" in registered_names
	for name in registered_names:
		assert table.count(f"| {name} |") == 3, name


def test_the_sparse_architecture_is_reported_pooled_and_per_domain(tmp_path, monkeypatch, offline):
	from evaluation.summary import load_committed_runs, summarize

	summary = summarize(load_committed_runs(every_architecture_two_domains(tmp_path, monkeypatch)))
	scopes = {row.scope for row in summary.rows if row.architecture == "sparse"}

	assert scopes == {"papers", "manuals", "pooled"}


def test_every_metric_is_reported_pooled_and_per_domain(tmp_path, monkeypatch, offline):
	"""A pooled average alone would hide the boundary the claim is about, so each domain gets
	its own row and the pool gets one computed over every scored question."""
	from evaluation.summary import load_committed_runs, render_markdown, summarize

	runs = every_architecture_two_domains(tmp_path, monkeypatch)
	table = render_markdown(summarize(load_committed_runs(runs)))

	assert "| papers |" in table
	assert "| manuals |" in table
	assert "| pooled |" in table
	for metric in ("recall@1", "recall@3", "recall@5", "ndcg@1", "ndcg@3", "ndcg@5"):
		assert metric in table


def test_the_pooled_row_is_the_mean_over_every_scored_question(tmp_path, monkeypatch, offline):
	"""The pool is questions, not a mean of domain means: every scored per-question result
	counts once, however uneven the domains are."""
	from evaluation.summary import load_committed_runs, summarize

	runs = every_architecture_two_domains(tmp_path, monkeypatch)
	summary = summarize(load_committed_runs(runs))

	naive_files = [load_run_file(path) for path in run_files(runs) if path.name.startswith("naive-")]
	scored = [result for run in naive_files for result in run.results if result.scored]
	expected = sum(result.recall_at["1"] for result in scored) / len(scored)

	pooled = next(row for row in summary.rows if row.architecture == "naive" and row.scope == "pooled")

	assert pooled.questions == sum(run.aggregates.questions for run in naive_files)
	assert pooled.recall_at["1"] == pytest.approx(expected)
	assert pooled.recall_at["5"] == pytest.approx(27.5 / 28)


def test_the_table_is_generated_from_the_files_not_written_by_hand(tmp_path, monkeypatch, offline):
	"""Changing what a run file records changes what the table shows, because the table is a
	reading of the files rather than a document of its own."""
	from evaluation.summary import load_committed_runs, summarize

	runs = every_architecture_two_domains(tmp_path, monkeypatch)

	def pooled_recall_at_1():
		summary = summarize(load_committed_runs(runs))
		return next(
			row for row in summary.rows if row.architecture == "naive" and row.scope == "pooled"
		)

	before = pooled_recall_at_1().recall_at["1"]
	target = next(path for path in run_files(runs) if path.name.startswith("naive-"))
	recorded = json.loads(target.read_text())
	perfect = next(
		result for result in recorded["results"] if result["scored"] and result["recall_at"]["1"] == 1.0
	)
	perfect["recall_at"]["1"] = 0.0
	target.write_text(json.dumps(recorded))

	files = [load_run_file(path) for path in run_files(runs) if path.name.startswith("naive-")]
	scored = sum(run.aggregates.scored for run in files)

	assert pooled_recall_at_1().recall_at["1"] == pytest.approx(before - 1 / scored)


def test_runs_measured_at_different_configurations_are_refused(tmp_path, monkeypatch, offline):
	"""Blending two configurations into one row would compare two setups instead of two
	retrievers, so the summary stops rather than averaging over the difference."""
	from evaluation.summary import load_committed_runs, summarize

	runs = every_architecture_two_domains(tmp_path, monkeypatch)
	target = next(path for path in run_files(runs) if path.name.startswith("hybrid-"))
	recorded = json.loads(target.read_text())
	recorded["configuration"]["retrieval_depth"] = 10
	target.write_text(json.dumps(recorded))

	with pytest.raises(ValueError, match="retrieval_depth"):
		summarize(load_committed_runs(runs))


def test_summarizing_with_no_run_files_says_so(tmp_path):
	"""An empty table would read as a measured zero, so no files is an error instead."""
	from evaluation.summary import load_committed_runs

	with pytest.raises(FileNotFoundError, match="no run files"):
		load_committed_runs(tmp_path / "empty")


def test_the_summarize_command_writes_the_table_a_reader_asked_for(tmp_path, monkeypatch, capsys, offline):
	"""One command regenerates the table from the run files, to stdout and to a file."""
	from main import main as cli

	runs = every_architecture_two_domains(tmp_path, monkeypatch)
	out = tmp_path / "table.md"
	monkeypatch.setattr(
		sys, "argv", ["rag-analysis", "summarize", "--results-dir", str(runs), "--out", str(out)]
	)

	import asyncio

	asyncio.run(cli())
	reported = capsys.readouterr().out

	table = Path(out).read_text()

	assert table.startswith("# Retrieval results")
	assert table in reported
	for name in (registered.name for registered in retrieval_architectures()):
		assert f"| {name} |" in table
	assert "| pooled |" in table
