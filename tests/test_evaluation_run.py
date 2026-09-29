"""A run of one architecture over one domain's evaluation set, and the file it writes.

The run is where this project produces its numbers: everything a reader would check about a
visible in the file it leaves behind, so that is what these tests read. Nothing here reaches a
store, a model, or a network. The store is never asked, because retrieval is answered directly,
and the encoders and the hosted model are replaced where the registry builds them, so a run here
is the same run a reader would get from a populated corpus except that the retrieved passages are
made up.

The properties that matter are the ones a number in a results table rests on: that the evaluation
set is checked before a single question is asked, that a hit is credited by the document a
question names rather than by the luck of the ranking, that a question with nowhere to look is
recorded without being scored as a miss, that the file says which configuration, commit, and
corpus produced it, and that a second run at the same commit and corpus is comparable to the
first.
"""

import asyncio
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from core import registry
from core.chunk import Chunk, Provenance
from corpus.manifest import load_manifest
from evaluation.run import load_run_file, run_evaluation
from evaluation.set import Stratum, load_evaluation_set

# A repository of its own, so a test can put the run at a commit of its choosing and then dirty it.
CLEAN_TREE_MESSAGE = "clean"


class RecordingLlm:
	"""Stands in for the hosted model, which no test may call."""

	async def apredict(self, template, **kwargs):
		return "an answer"


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


@pytest.fixture
def offline(monkeypatch):
	"""The models and the stores replaced, so a run in a test asks nothing of anything."""
	monkeypatch.setattr(registry, "get_generation_llm", lambda: RecordingLlm())
	monkeypatch.setattr(registry, "get_embed_model", lambda: DenseEmbedModel())

	for registered in registry.all_architectures():
		try:
			module = importlib.import_module(registered.pipeline.__module__)
		except ModuleNotFoundError:
			# A declaration another test file left behind after its throwaway package was
			# deleted. Nothing of this run can reach it, so there is nothing to replace.
			continue
		for name, replacement in (
			("get_qdrant_client", lambda: None),
			("get_sparse_embed_model", lambda: SparseEmbedModel()),
		):
			if hasattr(module, name):
				monkeypatch.setattr(module, name, replacement)


def a_chunk(document_id, section):
	"""One retrieved chunk, shaped as the store hands it back and pointing at a real document."""
	document = next(one for one in load_manifest().documents if one.id == document_id)

	return Chunk(
		text="a span",
		score=0.9,
		provenance=Provenance(
			source=f"./{document.target(Path('documents'))}",
			section=section,
			page=4,
			node_id="node-1",
		),
	)


def found_everywhere(question, top_k):
	"""Retrieval that answers a question out of the document the question names.

	An unanswerable question names no document, so the retriever answers it out of one anyway,
	which is the behaviour the unanswerable stratum exists to measure.
	"""
	if question.document is None:
		return [a_chunk("attention-is-all-you-need", "3.2.2 Multi-Head Attention")]

	return [a_chunk(question.document, question.section)]


def finds_only_the_first_location(question, top_k):
	"""A retriever that turns up one of a multi-hop question's two passages and not the other."""
	if question.stratum is not Stratum.MULTI_HOP:
		return found_everywhere(question, top_k)

	first = question.locations[0]

	return [a_chunk(first.document, first.section)]


def found_nothing(question, top_k):
	return []


def finds_at_rank_five(question, top_k):
	"""A retriever that buries the answer at the bottom of five, beneath four misses.

	Depths are read off one retrieval by truncation, so this is what separates a recall at one
	from a recall at five: the same ranking scores zero shallow and a hit deep.
	"""
	# A document of the other domain, so no question of this run's set can name it: every one
	# of these four is a miss wherever the run is pointed.
	misses = [a_chunk("postgresql-16-documentation", "Some other section") for _ in range(4)]

	return (misses + found_everywhere(question, top_k))[:top_k]


def retrieve_from(chunks_for):
	"""A retriever that answers each question of the committed set with the passages a test names."""

	evaluation_set = load_evaluation_set("papers")

	def asked_of_the_set(query: str, top_k: int = 5) -> list[Chunk]:
		question = next(one for one in evaluation_set.questions if one.question == query)
		return chunks_for(question, top_k)[:top_k]

	async def retrieve(self, query: str, top_k: int = 5) -> list[Chunk]:
		return asked_of_the_set(query, top_k)

	return retrieve


def run_over(tmp_path, monkeypatch, architecture, domain, chunks_for, **kwargs):
	"""A run of `architecture` over `domain`, with retrieval answered by `chunks_for`."""
	monkeypatch.setattr(registry.architecture(architecture).pipeline, "retrieve", retrieve_from(chunks_for))
	kwargs.setdefault("cache_dir", tmp_path / "cache")

	return asyncio.run(run_evaluation(architecture, domain, results_dir=tmp_path / "results", **kwargs))


def written_files(tmp_path):
	"""Every result file a run wrote below this test's directory."""
	return sorted((tmp_path / "results").rglob("*.json"))


def a_repository(path: Path) -> str:
	"""A git repository with one commit in it, and the commit it is at."""
	run_git(path, "init", "--quiet")
	run_git(path, "add", ".")
	run_git(path, "-c", "user.email=run@example.org", "-c", "user.name=Runs", "commit", "--quiet", "-m", "a commit")

	return run_git(path, "rev-parse", "HEAD")


def run_git(path: Path, *arguments: str) -> str:
	return subprocess.run(
		["git", *arguments],
		cwd=path,
		capture_output=True,
		text=True,
		check=True,
	).stdout.strip()


def test_a_run_asks_every_question_in_the_evaluation_set(tmp_path, monkeypatch, offline):
	evaluation_set = load_evaluation_set("papers")
	asked = []

	def record(question, top_k):
		asked.append(question.id)
		return found_everywhere(question, top_k)

	run_over(tmp_path, monkeypatch, "naive", "papers", record)

	assert asked == [question.id for question in evaluation_set.questions]


def test_a_run_reports_recall_and_reciprocal_rank_for_the_questions_it_can(tmp_path, monkeypatch, offline):
	run = run_over(tmp_path, monkeypatch, "naive", "papers", finds_only_the_first_location)

	assert run.aggregates.questions == 32
	assert run.aggregates.scored == 28
	assert run.aggregates.unscored == 4
	assert run.aggregates.recall == pytest.approx(27.5 / 28)
	assert 0 < run.aggregates.reciprocal_rank <= 1


def test_a_run_scores_recall_and_discounted_gain_at_three_depths(tmp_path, monkeypatch, offline):
	"""The full retrieval set lives on the run file: recall at one, three, and five, and the
	discounted gain beside it, all read off the one retrieval the run made."""
	run = run_over(tmp_path, monkeypatch, "naive", "papers", found_everywhere)

	assert set(run.aggregates.recall_at) == {"1", "3", "5"}
	assert set(run.aggregates.ndcg_at) == {"1", "3", "5"}
	assert run.aggregates.recall_at["5"] == pytest.approx(run.aggregates.recall)
	assert run.aggregates.recall_at["1"] <= run.aggregates.recall_at["3"] <= run.aggregates.recall_at["5"]
	for depth in ("1", "3", "5"):
		assert 0 < run.aggregates.ndcg_at[depth] <= 1


def test_depth_scores_are_read_off_one_retrieval_by_truncation(tmp_path, monkeypatch, offline):
	"""An answer buried at rank five is invisible shallow and a hit deep, from the same ranking."""
	run = run_over(tmp_path, monkeypatch, "naive", "papers", finds_at_rank_five)

	assert run.aggregates.recall_at["1"] == 0.0
	assert run.aggregates.recall_at["3"] == 0.0
	assert run.aggregates.recall_at["5"] == pytest.approx(run.aggregates.recall)
	assert run.aggregates.ndcg_at["1"] == 0.0
	assert run.aggregates.ndcg_at["3"] == 0.0

	single = next(
		result
		for result in run.results
		if result.scored and len(result.locations) == 1
	)
	assert single.recall_at == {"1": 0.0, "3": 0.0, "5": 1.0}
	assert single.ndcg_at["5"] == pytest.approx(0.38685280723454163)


def test_per_question_results_carry_depth_scores_and_unscored_carry_none(tmp_path, monkeypatch, offline):
	"""A surprising aggregate is traced to its questions, so the depths live there too."""
	run = run_over(tmp_path, monkeypatch, "naive", "papers", found_everywhere)

	for result in run.results:
		if result.scored:
			assert set(result.recall_at) == {"1", "3", "5"}
			assert set(result.ndcg_at) == {"1", "3", "5"}
		else:
			assert result.recall_at is None
			assert result.ndcg_at is None


def test_a_question_with_nothing_to_look_in_is_recorded_but_not_scored_as_a_miss(tmp_path, monkeypatch, offline):
	"""An unanswerable question names no place in the corpus, so recall has nothing to miss."""
	run = run_over(tmp_path, monkeypatch, "naive", "papers", found_nothing)

	unanswerable = [result for result in run.results if result.stratum is Stratum.UNANSWERABLE]

	assert len(unanswerable) == 4
	for result in unanswerable:
		assert result.scored is False
		assert result.recall is None
		assert result.reciprocal_rank is None
		assert result.unscored_reason


def test_a_question_whose_passages_were_not_retrieved_scores_zero_rather_than_being_dropped(tmp_path, monkeypatch, offline):
	run = run_over(tmp_path, monkeypatch, "naive", "papers", found_nothing)

	scored = [result for result in run.results if result.scored]

	assert len(scored) == 28
	assert all(result.recall == 0.0 and result.reciprocal_rank == 0.0 for result in scored)


def test_a_multi_hop_question_is_scored_on_every_location_it_names(tmp_path, monkeypatch, offline):
	"""Finding one of a question's two passages has not answered a question that needs both."""
	run = run_over(tmp_path, monkeypatch, "naive", "papers", finds_only_the_first_location)
	multi_hop = [result for result in run.results if result.stratum is Stratum.MULTI_HOP]

	crossing = [result for result in multi_hop if len({one.document for one in result.locations}) > 1]

	assert len(multi_hop) == 4
	assert crossing, "the set holds a question that reaches into a second document"
	for result in crossing:
		assert result.recall == pytest.approx(0.5)
		assert [location.rank for location in result.locations] == [1, None]


def test_the_pieces_a_retriever_returns_are_kept_with_the_result_they_were_scored_on(tmp_path, monkeypatch, offline):
	"""A surprising aggregate has to be traceable to the chunks behind it, not just to a number."""
	run = run_over(tmp_path, monkeypatch, "naive", "papers", found_everywhere)
	first = run.results[0]

	assert first.retrieved[0].document == first.locations[0].document
	assert first.retrieved[0].rank == 1
	assert first.retrieved[0].page == 4


def test_the_run_file_records_the_configuration_the_run_was_measured_at(tmp_path, monkeypatch, offline):
	configuration = run_over(tmp_path, monkeypatch, "naive", "papers", found_everywhere).configuration

	assert configuration.retrieval_depth == 5
	assert configuration.generation_temperature == 0.0
	assert configuration.generation_model
	assert "Use ONLY the context below" in configuration.prompt
	assert configuration.prompt_fingerprint.startswith("sha256:")
	assert configuration.chunker.strategy and configuration.chunker.size
	for encoder in (configuration.dense_encoder, configuration.sparse_encoder):
		assert encoder.name


def test_the_configuration_records_the_encoders_the_run_named_and_the_commits_they_were_read_at(
	tmp_path, monkeypatch, offline
):
	"""A result file that names one encoder while another produced the vectors is worse than none."""
	from config.params import Params

	cache = tmp_path / "model-cache"
	repository = cache / "models--qdrant--all-MiniLM-L6-v2-onnx" / "refs"
	repository.mkdir(parents=True)
	(repository / "main").write_text("c0ffee1234567890")
	monkeypatch.setattr("data.embed.model_cache_dir", lambda: cache)

	configuration = run_over(tmp_path, monkeypatch, "naive", "papers", found_everywhere).configuration

	assert configuration.dense_encoder.name == Params.DENSE_EMBEDDING_MODEL
	assert configuration.dense_encoder.revision == "c0ffee1234567890"
	assert configuration.sparse_encoder.name == Params.SPARSE_EMBEDDING_MODEL


def test_an_encoder_the_run_never_downloaded_records_no_revision_rather_than_a_guess(tmp_path, monkeypatch, offline):
	monkeypatch.setattr("data.embed.model_cache_dir", lambda: tmp_path / "no-models-here")

	configuration = run_over(tmp_path, monkeypatch, "naive", "papers", found_everywhere).configuration

	assert configuration.dense_encoder.revision is None


def test_the_run_file_records_the_commit_and_the_corpus_it_was_measured_against(tmp_path, monkeypatch, offline):
	from corpus.manifest import corpus_identifier

	run = run_over(tmp_path, monkeypatch, "naive", "papers", found_everywhere)

	assert run.commit.sha and len(run.commit.sha) == 40
	assert run.corpus.identifier == corpus_identifier(load_manifest())
	assert run.corpus.domain == "papers"
	assert run.corpus.documents == [document.id for document in load_manifest().in_domain("papers")]


def test_a_run_is_recorded_at_the_commit_it_was_made_at(tmp_path, monkeypatch, offline):
	repository = tmp_path / "checkout"
	repository.mkdir()
	(repository / "a-file").write_text("the state of the tree")
	commit = a_repository(repository)

	run = run_over(tmp_path, monkeypatch, "naive", "papers", found_everywhere, root=repository)

	assert run.commit.sha == commit
	assert run.commit.dirty is False


def test_a_run_made_on_top_of_uncommitted_work_says_so(tmp_path, monkeypatch, offline):
	"""The commit does not explain the run in that case, and a reader has to be told which is why."""
	repository = tmp_path / "checkout"
	repository.mkdir()
	(repository / "a-file").write_text("the state of the tree")
	a_repository(repository)
	(repository / "a-file").write_text("a change the commit does not have")

	run = run_over(tmp_path, monkeypatch, "naive", "papers", found_everywhere, root=repository)

	assert run.commit.dirty is True


def test_a_run_made_outside_a_repository_records_no_commit_rather_than_a_guess(tmp_path, monkeypatch, offline):
	empty = tmp_path / "not-a-repository"
	empty.mkdir()

	run = run_over(tmp_path, monkeypatch, "naive", "papers", found_everywhere, root=empty)

	assert run.commit.sha is None
	assert run.commit.dirty is None


def test_the_run_file_holds_every_question_result_next_to_the_aggregates(tmp_path, monkeypatch, offline):
	run = run_over(tmp_path, monkeypatch, "naive", "papers", found_everywhere)
	evaluation_set = load_evaluation_set("papers")

	assert [result.question for result in run.results] == [question.id for question in evaluation_set.questions]
	assert run.aggregates.recall == pytest.approx(
		sum(result.recall for result in run.results if result.scored) / run.aggregates.scored
	)


def test_a_run_writes_a_file_named_after_the_architecture_the_domain_the_commit_and_the_corpus(tmp_path, monkeypatch, offline):
	"""The commit and the corpus are both in the name, because together they are what makes two
	runs different files: a rerun at the same commit against the same corpus lands on the same
	path, and anything else lands beside it instead of overwriting it."""
	run = run_over(tmp_path, monkeypatch, "naive", "papers", found_everywhere)

	assert [path.name for path in written_files(tmp_path)] == [
		f"naive-{run.commit.sha[:12]}-{run.corpus.identifier.split(':')[1][:12]}.json"
	]
	assert written_files(tmp_path)[0].parent.name == "papers"


def test_a_second_run_at_the_same_commit_and_corpus_writes_a_comparable_file(tmp_path, monkeypatch, offline):
	first = run_over(tmp_path, monkeypatch, "naive", "papers", found_everywhere)
	second = run_over(tmp_path, monkeypatch, "naive", "papers", found_everywhere)

	# Latency is timed per run, so two runs never agree on it to the microsecond: what makes
	# the second comparable to the first is everything else, measured the same way twice.
	assert _without_latency(second) == _without_latency(first)
	assert len(written_files(tmp_path)) == 1
	assert _without_latency(load_run_file(written_files(tmp_path)[0])) == _without_latency(first)


def _without_latency(run):
	"""A run as a plain record with its timed fields dropped, for comparing two runs."""
	return run.model_dump(
		exclude={
			"results": {"__all__": {"retrieval_latency_s", "generation_latency_s"}},
			"aggregates": {
				"retrieval_latency_p50_s",
				"retrieval_latency_p95_s",
				"generation_latency_p50_s",
				"generation_latency_p95_s",
			},
		}
	)


def test_a_run_at_another_commit_is_written_beside_the_first_rather_than_over_it(tmp_path, monkeypatch, offline):
	first = tmp_path / "first"
	second = tmp_path / "second"

	run_over(first, monkeypatch, "naive", "papers", found_everywhere)
	monkeypatch.setattr("evaluation.run.current_revision", lambda root: _another_commit())
	run_over(second, monkeypatch, "naive", "papers", found_everywhere)

	assert [path.name for path in written_files(second)] != [path.name for path in written_files(first)]
	assert written_files(first) and written_files(second)


def _another_commit():
	from core.commit import Revision

	return Revision(sha="0" * 40, dirty=False)


def test_each_architecture_is_run_against_the_same_evaluation_set_and_writes_its_own_file(tmp_path, monkeypatch, offline):
	"""Sparse finds the passages and the other architectures find nothing, so a run that recorded
	one architecture's retrieval for another's would be caught here rather than in a table."""
	runs = {
		registered.name: run_over(
			tmp_path,
			monkeypatch,
			registered.name,
			"papers",
			found_everywhere if registered.name == "sparse" else found_nothing,
		)
		for registered in registry.all_architectures()
	}
	written = [path.name for path in written_files(tmp_path)]

	assert runs["sparse"].collection == "rag_sparse"
	assert {name: run.collection for name, run in runs.items()} == {
		registered.name: registered.collection for registered in registry.all_architectures()
	}
	assert len(set(run.collection for run in runs.values())) == len(runs)
	for name, run in runs.items():
		assert any(file_name.startswith(f"{name}-") for file_name in written), name
		assert run.aggregates.questions == 32, name
	assert runs["sparse"].aggregates.recall > 0.0
	assert all(
		run.aggregates.recall == 0.0 for name, run in runs.items() if name != "sparse"
	)


def test_the_run_file_is_json_readable_back_into_the_shape_it_was_written_in(tmp_path, monkeypatch, offline):
	run = run_over(tmp_path, monkeypatch, "naive", "papers", found_everywhere)
	written = written_files(tmp_path)[0]

	recorded = json.loads(written.read_text())

	assert set(recorded) == {
		"architecture",
		"collection",
		"domain",
		"commit",
		"corpus",
		"configuration",
		"results",
		"aggregates",
	}
	assert load_run_file(written) == run


def test_an_evaluation_set_whose_labels_do_not_hold_stops_the_run_before_a_question_is_asked(tmp_path, monkeypatch, offline):
	"""A set naming a document the corpus does not hold would otherwise score every question wrong."""
	import evaluation.run
	import evaluation.set

	broken = tmp_path / "sets"
	broken.mkdir()
	recorded = json.loads(Path("evaluation/sets/papers.json").read_text())
	recorded["questions"][0]["document"] = "a-paper-the-corpus-does-not-hold"
	(broken / "papers.json").write_text(json.dumps(recorded))
	monkeypatch.setattr(evaluation.set, "SETS_DIR", broken)

	async def refuse(self, query, top_k=5):
		raise AssertionError("the run asked a question it should not have asked")

	monkeypatch.setattr(registry.architecture("naive").pipeline, "retrieve", refuse)

	with pytest.raises(ValueError, match="names documents the corpus does not hold"):
		asyncio.run(
			run_evaluation("naive", "papers", results_dir=tmp_path / "results", cache_dir=tmp_path / "cache")
		)


def test_the_run_command_asks_for_a_domain_that_has_an_evaluation_set(tmp_path, monkeypatch, capsys, offline):
	from main import main as cli

	monkeypatch.setattr(sys, "argv", ["rag-analysis", "run", "naive", "--domain", "recipes", "--out", str(tmp_path)])

	with pytest.raises(SystemExit) as exit_info:
		asyncio.run(cli())

	assert exit_info.value.code == 2
	assert "no evaluation set for recipes" in capsys.readouterr().err


def test_the_run_command_reports_an_unknown_architecture_before_it_asks_a_question(tmp_path, monkeypatch, capsys, offline):
	from main import main as cli

	async def refuse(self, query, top_k=5):
		raise AssertionError("the run asked a question it should not have asked")

	monkeypatch.setattr(registry.architecture("naive").pipeline, "retrieve", refuse)
	monkeypatch.setattr(sys, "argv", ["rag-analysis", "run", "graph", "--domain", "papers", "--out", str(tmp_path)])

	with pytest.raises(SystemExit) as exit_info:
		asyncio.run(cli())

	assert exit_info.value.code == 2
	assert "naive" in capsys.readouterr().err


def test_the_run_command_writes_the_file_a_reader_asked_for(tmp_path, monkeypatch, capsys, offline):
	from main import main as cli

	monkeypatch.setattr(registry.architecture("naive").pipeline, "retrieve", retrieve_from(found_everywhere))
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
			str(tmp_path),
			"--cache-dir",
			str(tmp_path / "cache"),
		],
	)

	asyncio.run(cli())
	reported = capsys.readouterr().out

	written = list(tmp_path.rglob("*.json"))
	assert len(written) == 1
	assert load_run_file(written[0]).architecture == "naive"
	assert "recall@5" in reported
	assert "wrote" in reported
