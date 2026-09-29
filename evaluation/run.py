"""One architecture, one domain's evaluation set, and the result file it leaves behind.

This is where the project stops being plumbing and starts producing numbers. A run reads the
evaluation set, checks it against the corpus before asking a single question, retrieves for every
question in it, scores what came back, and writes one file per run recording everything a reader
would need to trust the numbers: the configuration the run was measured at, the commit, the
corpus it was measured against, the result for every question, and the aggregates over them.

The file name carries the architecture, the domain, the corpus digest, and the commit, so a
directory of runs is readable without opening one, and a rerun at the same commit against the
same corpus lands on the same file rather than beside it: that second run is the same run, and
the file it writes has to be comparable to the first rather than a second copy of it. The scores
themselves may move a little between two such runs, because the store returns its own ranking
and a hosted model answers its own way, but the configuration, the commit, and the corpus the
file records are what two runs are compared on. Git keeps the earlier file whenever the name is
different.

The unanswerable questions are recorded and not scored. They name no place in the corpus, so there
is nothing for recall or reciprocal rank to find or miss, and counting them as misses would drag
every architecture's retrieval score down by a failure that has nothing to do with retrieval. What
a system does with them is a question of its own, and the retrieved chunks are kept in the file
so it can be asked of the same run.
"""

from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field

from config.configuration import Configuration, frozen_configuration
from core.commit import current_revision
from core.registry import architecture
from corpus.manifest import Manifest, corpus_identifier, load_manifest
from evaluation.metrics import Retrieved, locate, recall, reciprocal_rank, retrieved_from
from evaluation.set import EvaluationSet, Stratum, load_evaluation_set

# Where run files are written. One directory per domain, because results are reported per domain
# and a reader looking for the manuals numbers should not have to read the papers ones to find them.
RESULTS_DIR = Path("results/runs")


class LocationResult(BaseModel):
	"""One place a question is answered from, and where in the results it turned up."""

	model_config = ConfigDict(extra="forbid")

	document: str
	section: str
	rank: int | None = Field(description="Where it was retrieved, or nothing if it was not.")


class QuestionResult(BaseModel):
	"""What one question retrieved and what that is worth.

	A question with no locations is recorded with its results and no scores. It is not a zero,
	because a zero would say the retriever missed when there was nothing to miss.
	"""

	model_config = ConfigDict(extra="forbid")

	question: str
	stratum: Stratum
	scored: bool
	unscored_reason: str | None = None
	recall: float | None = None
	reciprocal_rank: float | None = None
	locations: list[LocationResult] = Field(default_factory=list)
	retrieved: list[Retrieved] = Field(default_factory=list)


class Aggregates(BaseModel):
	"""The numbers over the questions a run could score, and how many it could not.

	Reciprocal rank is the mean of each question's reciprocal rank on its own first location, the
	place the question is named after, which is what keeps a second passage found earlier from
	counting as an answer to where the question is answered.
	"""

	model_config = ConfigDict(extra="forbid")

	questions: int
	scored: int
	unscored: int
	recall: float
	reciprocal_rank: float


class CorpusRecord(BaseModel):
	"""The corpus a run was measured against, by digest rather than by directory."""

	model_config = ConfigDict(extra="forbid")

	identifier: str
	domain: str
	documents: list[str]

	def short(self) -> str:
		"""The digest as it appears in a file's name, or a word when there is no corpus."""
		return self.identifier.split(":", 1)[1][:12] if ":" in self.identifier else "uncorpus"


class RevisionRecord(BaseModel):
	"""The commit a run was made at, and whether the tree was clean when it was."""

	model_config = ConfigDict(extra="forbid")

	sha: str | None
	dirty: bool | None

	def short(self) -> str:
		"""The commit as it appears in a file's name, or a word when there is no commit."""
		return self.sha[:12] if self.sha else "uncommitted"


class RunFile(BaseModel):
	"""One run, written to disk: what was asked, what was measured, and under what configuration."""

	model_config = ConfigDict(extra="forbid")

	architecture: str
	collection: str
	domain: str
	commit: RevisionRecord
	corpus: CorpusRecord
	configuration: Configuration
	results: list[QuestionResult]
	aggregates: Aggregates


async def run_evaluation(
	architecture_name: str,
	domain: str,
	results_dir: Path = RESULTS_DIR,
	root: Path = Path("."),
) -> RunFile:
	"""Run one architecture over one domain's evaluation set, and write the file recording it.

	The architecture is resolved and the evaluation set is read and checked before anything is
	built or contacted, so a name that does not exist or a set whose labels do not hold costs a
	run nothing.
	"""
	registered = architecture(architecture_name)
	manifest = load_manifest()
	evaluation_set = load_evaluation_set(domain, manifest=manifest)
	configuration = frozen_configuration()
	revision = current_revision(root)

	pipeline = registered.build()
	results = await _ask_every_question(pipeline, evaluation_set, manifest, configuration)

	run = RunFile(
		architecture=registered.name,
		collection=registered.collection,
		domain=domain,
		commit=RevisionRecord(sha=revision.sha, dirty=revision.dirty),
		corpus=CorpusRecord(
			identifier=corpus_identifier(manifest),
			domain=domain,
			documents=[document.id for document in manifest.in_domain(domain)],
		),
		configuration=configuration,
		results=results,
		aggregates=_aggregates(results),
	)

	path = run_path(results_dir, run)
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(run.model_dump_json(indent=2) + "\n")

	report(run, path)

	return run


async def _ask_every_question(
	pipeline,
	evaluation_set: EvaluationSet,
	manifest: Manifest,
	configuration: Configuration,
) -> list[QuestionResult]:
	"""Retrieve for every question in the set, and score what came back against its labels."""
	results = []
	total = len(evaluation_set.questions)

	for number, question in enumerate(evaluation_set.questions, start=1):
		chunks = await pipeline.retrieve(question.question, top_k=configuration.retrieval_depth)
		retrieved = retrieved_from(chunks, manifest)
		locations = question.all_locations()

		result = QuestionResult(
			question=question.id,
			stratum=question.stratum,
			scored=bool(locations),
			unscored_reason=None if locations else "names no place in the corpus to look in",
			recall=recall(retrieved, locations) if locations else None,
			reciprocal_rank=reciprocal_rank(retrieved, locations[0]) if locations else None,
			locations=[
				LocationResult(
					document=location.document,
					section=location.section,
					rank=locate(retrieved, location),
				)
				for location in locations
			],
			retrieved=retrieved,
		)

		results.append(result)
		print(f"[RUN] {number}/{total} {question.id} {_score_line(result)}")

	return results


def _score_line(result: QuestionResult) -> str:
	"""How one question scored, for the line the run prints as it goes."""
	if not result.scored:
		return "not scored on retrieval"

	return f"recall={result.recall:.2f} reciprocal_rank={result.reciprocal_rank:.2f}"


def _aggregates(results: list[QuestionResult]) -> Aggregates:
	"""The numbers over the questions that could be scored, and the count that could not.

	Both means are over the scored questions only, and the counts are recorded beside them,
	because a mean over a pool that silently included the unscored would read as a retrieval
	score a reader could not account for.
	"""
	scored = [result for result in results if result.scored]
	count = len(scored)

	return Aggregates(
		questions=len(results),
		scored=count,
		unscored=len(results) - count,
		recall=_mean(result.recall for result in scored),
		reciprocal_rank=_mean(result.reciprocal_rank for result in scored),
	)


def _mean(scores: Iterable[float | None]) -> float:
	"""The mean of the scores, and zero when a run had none to average.

	A set of nothing but unanswerable questions is possible, and a run that had nothing to score
	has a mean of zero rather than no result file at all. A `None` is skipped rather than counted
	as a miss, for the same reason an unscored question is not a miss.
	"""
	measured = [score for score in scores if score is not None]

	return sum(measured) / len(measured) if measured else 0.0


def run_path(results_dir: Path, run: RunFile) -> Path:
	"""Where a run's file lives, named after the architecture, the domain, the corpus, and the commit.

	The commit and the corpus are both in the name because together they are what makes two runs
	different files. A re-run at the same commit against the same corpus is the same run, and it
	lands on the same path; anything else lands beside it instead of overwriting it.
	"""
	name = f"{run.architecture}-{run.commit.short()}-{run.corpus.short()}.json"

	return Path(results_dir) / run.domain / name


def load_run_file(path: Path) -> RunFile:
	"""A run file read back, checked against the same shape it was written in."""
	return RunFile.model_validate_json(Path(path).read_text())


def report(run: RunFile, path: Path) -> None:
	"""Say what the run found and where it put it."""
	depth = run.configuration.retrieval_depth
	aggregates = run.aggregates

	print(f"[RUN] {run.architecture} on {run.domain}: {aggregates.questions} questions, {aggregates.scored} scored")
	print(f"[RUN] recall@{depth} {aggregates.recall:.4f}")
	print(f"[RUN] reciprocal rank@{depth} {aggregates.reciprocal_rank:.4f}")
	if aggregates.unscored:
		print(f"[RUN] {aggregates.unscored} questions name no place to look in and are not scored on retrieval")
	if run.commit.sha is None:
		print("[RUN] no commit was found for this working tree, so this run cannot be traced to one")
	elif run.commit.dirty:
		print(f"[RUN] the tree had uncommitted changes at {run.commit.sha}, so the commit alone does not reproduce it")
	print(f"[RUN] wrote {path}")
