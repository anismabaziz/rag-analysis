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

from pydantic import BaseModel, ConfigDict, Field

from config.configuration import Configuration, frozen_configuration
from core.chunk import REFUSAL, build_context
from core.commit import current_revision
from core.registry import architecture
from corpus.manifest import Manifest, corpus_identifier, load_manifest
from evaluation.citations import citation_accuracy, claim_supported, split_claims
from evaluation.metrics import (
	REPORT_DEPTHS,
	Retrieved,
	locate,
	mean,
	ndcg_at,
	recall,
	recall_at,
	reciprocal_rank,
	retrieved_from,
)
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

	The answer is generated from the retrieved chunks and every claim in it is checked
	against those same chunks, so citation accuracy says whether the answer came from the
	context it was given. An answer with no claims, which is what a refusal is, carries
	no citation score rather than a zero.
	"""

	model_config = ConfigDict(extra="forbid")

	question: str
	stratum: Stratum
	scored: bool
	unscored_reason: str | None = None
	recall: float | None = None
	reciprocal_rank: float | None = None
	recall_at: dict[str, float] | None = Field(
		default=None,
		description="Recall read off the one retrieval at each reported depth, keyed by depth.",
	)
	ndcg_at: dict[str, float] | None = Field(
		default=None,
		description="Discounted gain read off the one retrieval at each reported depth, keyed by depth.",
	)
	locations: list[LocationResult] = Field(default_factory=list)
	retrieved: list[Retrieved] = Field(default_factory=list)
	answer: str = ""
	claims: list[str] = Field(default_factory=list)
	claims_supported: int = 0
	claims_total: int = 0
	citation_accuracy: float | None = Field(
		default=None,
		description="Share of the answer's claims one retrieved chunk each supports, or nothing when it makes none.",
	)


class Aggregates(BaseModel):
	"""The numbers over the questions a run could score, and how many it could not.

	Reciprocal rank is the mean of each question's reciprocal rank on its own first location, the
	place the question is named after, which is what keeps a second passage found earlier from
	counting as an answer to where the question is answered.

	Citation accuracy is the mean over the questions whose answers made claims, including the
	unanswerable stratum, because answering a question the corpus never covers is the failure
	the metric exists to show. The unanswerable slice is reported beside the answerable one so
	a system that answers regardless reads visibly worse. Questions whose answers made no
	claims, which is what a refusal does, are counted beside the means rather than averaged
	as zeroes.
	"""

	model_config = ConfigDict(extra="forbid")

	questions: int
	scored: int
	unscored: int
	recall: float
	reciprocal_rank: float
	recall_at: dict[str, float] = Field(
		description="Mean recall at each reported depth, over the scored questions only.",
	)
	ndcg_at: dict[str, float] = Field(
		description="Mean discounted gain at each reported depth, over the scored questions only.",
	)
	citation_accuracy: float | None = Field(
		default=None,
		description="Mean share of supported claims, over the questions whose answers made claims.",
	)
	citation_accuracy_answerable: float | None = Field(
		default=None,
		description="The same mean over every stratum except unanswerable.",
	)
	citation_accuracy_unanswerable: float | None = Field(
		default=None,
		description="The same mean over the unanswerable stratum alone.",
	)
	citation_scored: int = Field(
		default=0,
		description="Questions whose answers made claims and carry a citation score.",
	)
	citation_unscored: int = Field(
		default=0,
		description="Questions whose answers made no claims and carry none.",
	)


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
	"""Retrieve for every question in the set, answer from what came back, and score both.

	Retrieval is scored against the question's labels and the answer is scored against the
	chunks it was answered with: each claim counts only when one retrieved chunk holds a
	supporting span for it. Empty retrieval is never sent to the model; the refusal stands
	in as the answer, produces no claims, and carries no citation score.
	"""
	results = []
	total = len(evaluation_set.questions)

	for number, question in enumerate(evaluation_set.questions, start=1):
		chunks = await pipeline.retrieve(question.question, top_k=configuration.retrieval_depth)
		retrieved = retrieved_from(chunks, manifest)
		locations = question.all_locations()
		depths = [depth for depth in REPORT_DEPTHS if depth <= configuration.retrieval_depth]
		answer = REFUSAL if not chunks else await pipeline.generate(question.question, build_context(chunks))
		claims = split_claims(answer)
		supported = sum(1 for claim in claims if claim_supported(claim, chunks))

		result = QuestionResult(
			question=question.id,
			stratum=question.stratum,
			scored=bool(locations),
			unscored_reason=None if locations else "names no place in the corpus to look in",
			recall=recall(retrieved, locations) if locations else None,
			reciprocal_rank=reciprocal_rank(retrieved, locations[0]) if locations else None,
			recall_at={str(depth): recall_at(retrieved, locations, depth) for depth in depths}
			if locations
			else None,
			ndcg_at={str(depth): ndcg_at(retrieved, locations, depth) for depth in depths}
			if locations
			else None,
			locations=[
				LocationResult(
					document=location.document,
					section=location.section,
					rank=locate(retrieved, location),
				)
				for location in locations
			],
			retrieved=retrieved,
			answer=answer,
			claims=claims,
			claims_supported=supported,
			claims_total=len(claims),
			citation_accuracy=citation_accuracy(answer, chunks),
		)

		results.append(result)
		print(f"[RUN] {number}/{total} {question.id} {_score_line(result)}")

	return results


def _score_line(result: QuestionResult) -> str:
	"""How one question scored, for the line the run prints as it goes."""
	if result.citation_accuracy is None:
		cited = "citation=n/a"
	else:
		cited = f"citation={result.citation_accuracy:.2f}"
	if not result.scored:
		return f"not scored on retrieval {cited}"

	return f"recall={result.recall:.2f} reciprocal_rank={result.reciprocal_rank:.2f} {cited}"


def _aggregates(results: list[QuestionResult]) -> Aggregates:
	"""The numbers over the questions that could be scored, and the count that could not.

	Both means are over the scored questions only, and the counts are recorded beside them,
	because a mean over a pool that silently included the unscored would read as a retrieval
	score a reader could not account for. Citation accuracy is averaged the same way, over
	the questions whose answers made claims, with the unanswerable stratum kept as its own
	slice beside the answerable one.
	"""
	scored = [result for result in results if result.scored]
	count = len(scored)
	with_claims = [result for result in results if result.citation_accuracy is not None]
	answerable = [result for result in with_claims if result.stratum is not Stratum.UNANSWERABLE]
	unanswerable = [result for result in with_claims if result.stratum is Stratum.UNANSWERABLE]

	def _mean_or_nothing(values: list[float | None]) -> float | None:
		measured = [value for value in values if value is not None]
		return sum(measured) / len(measured) if measured else None

	return Aggregates(
		questions=len(results),
		scored=count,
		unscored=len(results) - count,
		recall=mean(result.recall for result in scored),
		reciprocal_rank=mean(result.reciprocal_rank for result in scored),
		recall_at=_depth_means(scored, "recall_at"),
		ndcg_at=_depth_means(scored, "ndcg_at"),
		citation_accuracy=_mean_or_nothing([result.citation_accuracy for result in with_claims]),
		citation_accuracy_answerable=_mean_or_nothing([result.citation_accuracy for result in answerable]),
		citation_accuracy_unanswerable=_mean_or_nothing([result.citation_accuracy for result in unanswerable]),
		citation_scored=len(with_claims),
		citation_unscored=len(results) - len(with_claims),
	)


def _depth_means(scored: list[QuestionResult], field: str) -> dict[str, float]:
	"""The mean at each reported depth, over the questions that carry that depth.

	The depths are the ones the runs recorded, not a list kept here, so a summary built from
	run files and this aggregate can never disagree about which depths exist.
	"""
	depths: list[str] = []
	for result in scored:
		for depth in getattr(result, field) or {}:
			if depth not in depths:
				depths.append(depth)

	return {
		depth: mean(getattr(result, field).get(depth) for result in scored if getattr(result, field))
		for depth in depths
	}


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
	for reported in sorted(aggregates.recall_at, key=int):
		print(f"[RUN] recall@{reported} {aggregates.recall_at[reported]:.4f}")
	for reported in sorted(aggregates.ndcg_at, key=int):
		print(f"[RUN] ndcg@{reported} {aggregates.ndcg_at[reported]:.4f}")
	print(f"[RUN] reciprocal rank@{depth} {aggregates.reciprocal_rank:.4f}")
	print(f"[RUN] {_citation_line(aggregates)}")
	if aggregates.unscored:
		print(f"[RUN] {aggregates.unscored} questions name no place to look in and are not scored on retrieval")
	if aggregates.citation_unscored:
		print(f"[RUN] {aggregates.citation_unscored} questions made no claims and are not scored on citation")
	if run.commit.sha is None:
		print("[RUN] no commit was found for this working tree, so this run cannot be traced to one")
	elif run.commit.dirty:
		print(f"[RUN] the tree had uncommitted changes at {run.commit.sha}, so the commit alone does not reproduce it")
	print(f"[RUN] wrote {path}")


def _citation_line(aggregates: Aggregates) -> str:
	"""One line for the citation means, with the unanswerable slice beside the rest."""
	def _cell(value: float | None) -> str:
		return f"{value:.4f}" if value is not None else "n/a"

	return (
		f"citation {_cell(aggregates.citation_accuracy)} "
		f"(answerable {_cell(aggregates.citation_accuracy_answerable)}, "
		f"unanswerable {_cell(aggregates.citation_accuracy_unanswerable)}) "
		f"over {aggregates.citation_scored} questions with claims"
	)
