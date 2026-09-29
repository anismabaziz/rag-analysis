"""One table over every run file, pooled and per domain, generated rather than written.

A run file holds one architecture on one domain, so the comparison the project claims lives
nowhere until the files are read together. This module reads them: one row per architecture
per domain, and one pooled row per architecture whose every number is the mean over the
concatenated scored per-question results of its domains, so a domain with more questions moves
the pool more than one with fewer.

Two things the table refuses to do. It never names a depth of its own: the columns are the
depths the run files recorded, which were read off the committed retrieval depth by
truncation. And it never blends runs measured at different configurations into one row: every
file's frozen configuration is compared before anything is averaged, and a difference stops
the summary, because a row over two setups would compare everything except retrieval.
"""

from dataclasses import dataclass, field
from pathlib import Path

from core.chunk import Chunk, Provenance
from evaluation.citations import citation_accuracy
from evaluation.exact_match import exact_match_score
from evaluation.metrics import mean
from evaluation.run import RESULTS_DIR, QuestionResult, RunFile, load_run_file

# The scope of a row built from every domain's scored questions rather than one domain's.
POOLED_SCOPE = "pooled"


@dataclass(frozen=True)
class SummaryRow:
	"""One line of the table: an architecture measured at one scope.

	`scope` is a domain, or the pooled scope when the numbers are the mean over every scored
	question of the architecture's domains rather than one domain's. Citation accuracy is the
	mean over the questions whose answers made claims, with the unanswerable slice beside
	the answerable one, because answering regardless is the failure that stratum exists to show.
	Exact match is the mean over the extractive questions, with citation accuracy over those
	same questions beside it, so a reader can see whether the deterministic support check
	and the objective span check agree.
	"""

	architecture: str
	scope: str
	questions: int
	scored: int
	unscored: int
	recall: float
	reciprocal_rank: float
	recall_at: dict[str, float] = field(default_factory=dict)
	ndcg_at: dict[str, float] = field(default_factory=dict)
	citation_accuracy: float | None = None
	citation_accuracy_answerable: float | None = None
	citation_accuracy_unanswerable: float | None = None
	citation_scored: int = 0
	citation_unscored: int = 0
	exact_match: float | None = None
	exact_match_scored: int = 0
	exact_match_unscored: int = 0
	citation_accuracy_extractive: float | None = None
	citation_extractive_scored: int = 0


@dataclass(frozen=True)
class Summary:
	"""The whole table: every row, the depths its columns were read at, and what it was read from."""

	rows: tuple[SummaryRow, ...]
	depths: tuple[str, ...]
	commits: tuple[str, ...]
	corpora: tuple[str, ...]
	prompt_fingerprint: str
	generation_model: str
	retrieval_depth: int
	files: tuple[str, ...]


def load_runs(results_dir: Path = RESULTS_DIR) -> list[RunFile]:
	"""Every run file below the directory, in a stable order.

	An empty directory is an error rather than an empty table, because a table with no rows
	would read as a measurement in which every architecture scored nothing.
	"""
	paths = sorted(Path(results_dir).rglob("*.json"))
	if not paths:
		raise FileNotFoundError(f"no run files in {results_dir}: run an architecture first")

	return [load_run_file(path) for path in paths]


def summarize(runs: list[RunFile]) -> Summary:
	"""The pooled and per-domain rows over the runs, after checking they are comparable.

	Runs group by architecture, commit, and corpus: a second run at the same commit against
	the same corpus is the same measurement and lands in the same group, while anything else
	gets its own rows rather than being averaged into them.
	"""
	if not runs:
		raise ValueError("no runs to summarize")

	_check_isolation(runs)

	groups: dict[tuple[str, str | None, str], list[RunFile]] = {}
	for run in runs:
		groups.setdefault((run.architecture, run.commit.sha, run.corpus.identifier), []).append(run)

	rows = []
	for key in sorted(groups, key=lambda group: str(group[0])):
		group = groups[key]
		for run in sorted(group, key=lambda run: run.domain):
			rows.append(_domain_row(run))
		rows.append(_pooled_row(group))

	first = runs[0]
	depths = sorted(
		{depth for run in runs for depth in (*run.aggregates.recall_at, *run.aggregates.ndcg_at)},
		key=int,
	)

	return Summary(
		rows=tuple(rows),
		depths=tuple(depths),
		commits=tuple(sorted({run.commit.short() for run in runs})),
		corpora=tuple(sorted({run.corpus.identifier for run in runs})),
		prompt_fingerprint=first.configuration.prompt_fingerprint,
		generation_model=first.configuration.generation_model,
		retrieval_depth=first.configuration.retrieval_depth,
		files=tuple(sorted(f"{run.architecture} on {run.domain}" for run in runs)),
	)


def render_markdown(summary: Summary) -> str:
	"""The summary as a markdown table, with the commit, corpus, and configuration it was read from.

	The header names what the numbers trace back to, so a reader can check the frozen
	configuration from the table rather than taking the project's word for it.
	"""
	from evaluation.set import Stratum

	recall_columns = [f"recall@{depth}" for depth in summary.depths]
	ndcg_columns = [f"ndcg@{depth}" for depth in summary.depths]
	header = [
		"architecture",
		"scope",
		"questions",
		"scored",
		*recall_columns,
		*ndcg_columns,
		"mrr",
		"cite",
		"cite_answerable",
		"cite_unanswerable",
		"exact_match",
		"exact_match_n",
		"cite_extractive",
	]

	lines = [
		"# Retrieval results",
		"",
		f"{len(summary.files)} runs at commit {', '.join(summary.commits)} "
		f"against corpus {', '.join(summary.corpora)}.",
		f"Frozen configuration: prompt {summary.prompt_fingerprint}, "
		f"{summary.generation_model} at temperature 0.0, depth {summary.retrieval_depth}.",
		"",
		"| " + " | ".join(header) + " |",
		"| " + " | ".join("---" for _ in header) + " |",
	]

	for row in summary.rows:
		cells = [
			row.architecture,
			row.scope,
			str(row.questions),
			str(row.scored),
			*(_cell(row.recall_at.get(depth)) for depth in summary.depths),
			*(_cell(row.ndcg_at.get(depth)) for depth in summary.depths),
			f"{row.reciprocal_rank:.4f}",
			_cell(row.citation_accuracy),
			_cell(row.citation_accuracy_answerable),
			_cell(row.citation_accuracy_unanswerable),
			_cell(row.exact_match),
			str(row.exact_match_scored),
			_cell(row.citation_accuracy_extractive),
		]
		lines.append("| " + " | ".join(cells) + " |")

	lines.extend(
		[
			"",
			"cite is the share of answer claims one retrieved chunk each supports; "
			f"cite_unanswerable is the same share over the {Stratum.UNANSWERABLE.value} stratum alone.",
			"exact_match is the share of extractive questions whose answer matches the gold span, "
			"tolerating surrounding whitespace and punctuation only; "
			"cite_extractive is citation accuracy over those same questions.",
		]
	)

	return "\n".join(lines) + "\n"


def _domain_row(run: RunFile) -> SummaryRow:
	"""One run's row, recomputed from its stored per-question results rather than its aggregates.

	The aggregates travel in the file, but the row does not trust them: a hand-edited aggregate
	next to untouched questions would otherwise survive into the table, and the table would no
	longer be a reading of what the run measured.
	"""
	scored = [result for result in run.results if result.scored]

	return SummaryRow(
		architecture=run.architecture,
		scope=run.domain,
		questions=len(run.results),
		scored=len(scored),
		unscored=len(run.results) - len(scored),
		recall=mean(result.recall for result in scored),
		reciprocal_rank=mean(result.reciprocal_rank for result in scored),
		recall_at=_question_means(scored, "recall_at"),
		ndcg_at=_question_means(scored, "ndcg_at"),
		citation_accuracy=_citation_mean(run.results, None),
		citation_accuracy_answerable=_citation_mean(run.results, False),
		citation_accuracy_unanswerable=_citation_mean(run.results, True),
		citation_scored=len([result for result in run.results if result.citation_accuracy is not None]),
		citation_unscored=len([result for result in run.results if result.citation_accuracy is None]),
		**_extractive_cells(run.results),
	)


def _pooled_row(group: list[RunFile]) -> SummaryRow:
	"""One architecture's domains pooled: every scored per-question result counted once.

	The pool is recomputed from the stored per-question results rather than averaged from the
	domain rows, so a domain with more scored questions moves the pool more than one with
	fewer, and a hand-edited aggregate cannot survive next to the questions behind it.
	"""
	scored = [result for run in group for result in run.results if result.scored]
	pooled = [result for run in group for result in run.results]

	return SummaryRow(
		architecture=group[0].architecture,
		scope=POOLED_SCOPE,
		questions=sum(len(run.results) for run in group),
		scored=len(scored),
		unscored=sum(len(run.results) for run in group) - len(scored),
		recall=mean(result.recall for result in scored),
		reciprocal_rank=mean(result.reciprocal_rank for result in scored),
		recall_at=_question_means(scored, "recall_at"),
		ndcg_at=_question_means(scored, "ndcg_at"),
		citation_accuracy=_citation_mean(pooled, None),
		citation_accuracy_answerable=_citation_mean(pooled, False),
		citation_accuracy_unanswerable=_citation_mean(pooled, True),
		citation_scored=len([result for result in pooled if result.citation_accuracy is not None]),
		citation_unscored=len([result for result in pooled if result.citation_accuracy is None]),
		**_extractive_cells(pooled),
	)


def _extractive_cells(results: list[QuestionResult]) -> dict:
	"""The exact-match columns over the extractive questions, recomputed from what is stored.

	One helper serves the domain row and the pooled row, so the two can never disagree
	about which questions count as extractive: the flag the run stored from the evaluation
	set, which is what identifies an extractive question.
	"""
	extractive = [result for result in results if result.extractive]

	return {
		"exact_match": _exact_mean(results),
		"exact_match_scored": len([result for result in extractive if exact_match_score(result.answer, result.gold_answer, True) is not None]),
		"exact_match_unscored": len(results) - len(extractive),
		"citation_accuracy_extractive": _citation_mean_extractive(results),
		"citation_extractive_scored": len(
			[result for result in extractive if result.citation_accuracy is not None]
		),
	}


def _citation_mean(results: list[QuestionResult], unanswerable: bool | None) -> float | None:
	"""The mean citation accuracy over the questions that made claims, optionally sliced.

	`None` slices nothing, `True` keeps the unanswerable stratum alone, and `False` keeps
	everything except it, because answering regardless is the failure the slice exists to show.
	`None` when no question in the slice made a claim, so a refusal reads as unscored.

	Recomputed from each stored answer and the texts of the chunks it was answered with,
	rather than trusted from the run's aggregates, so a hand-edited aggregate cannot survive
	next to the questions behind it.
	"""
	from evaluation.set import Stratum

	if unanswerable is True:
		kept = [result for result in results if result.stratum is Stratum.UNANSWERABLE]
	elif unanswerable is False:
		kept = [result for result in results if result.stratum is not Stratum.UNANSWERABLE]
	else:
		kept = list(results)

	measured = []
	for result in kept:
		chunks = [Chunk(text=item.text, score=item.score, provenance=Provenance()) for item in result.retrieved]
		score = citation_accuracy(result.answer, chunks)
		if score is not None:
			measured.append(score)

	return sum(measured) / len(measured) if measured else None


def _exact_mean(results: list[QuestionResult]) -> float | None:
	"""The mean exact match over the extractive questions, recomputed from what is stored.

	Each stored answer is compared to its stored gold span rather than trusting the run's
	aggregates, so a hand-edited aggregate cannot survive next to the questions behind it.
	`None` when no question in the pool is extractive, and a refusal on an extractive
	question counts as a miss because the span was there to name.
	"""
	measured = []
	for result in results:
		score = exact_match_score(result.answer, result.gold_answer, result.extractive)
		if score is not None:
			measured.append(score)

	return sum(measured) / len(measured) if measured else None


def _citation_mean_extractive(results: list[QuestionResult]) -> float | None:
	"""Citation accuracy over the extractive questions alone, recomputed from what is stored.

	The same recomputation as the overall citation mean, sliced to the questions the
	evaluation set marked extractive, because that slice is what sits beside exact match
	in the table.
	"""
	kept = [result for result in results if result.extractive]

	measured = []
	for result in kept:
		chunks = [Chunk(text=item.text, score=item.score, provenance=Provenance()) for item in result.retrieved]
		score = citation_accuracy(result.answer, chunks)
		if score is not None:
			measured.append(score)

	return sum(measured) / len(measured) if measured else None


def _question_means(scored: list[QuestionResult], field: str) -> dict[str, float]:
	"""The mean at each recorded depth, over the scored questions that carry that depth.

	The depths are the ones the questions recorded, not a list kept here, so a summary built
	from run files can never disagree with the files about which depths exist.
	"""
	depths = sorted(
		{depth for result in scored for depth in (getattr(result, field) or {})},
		key=int,
	)

	return {
		depth: mean(getattr(result, field).get(depth) for result in scored if getattr(result, field))
		for depth in depths
	}


def _check_isolation(runs: list[RunFile]) -> None:
	"""Refuse runs whose frozen configurations disagree about anything.

	Every field the configuration records is held constant across architectures by design: a
	difference in any of them means two rows would compare setups rather than retrievers, and
	the summary stops instead of averaging over the difference.
	"""
	first = runs[0].configuration.model_dump()

	for run in runs[1:]:
		differences = _differences(first, run.configuration.model_dump())
		if differences:
			raise ValueError(
				f"{run.architecture} on {run.domain} was measured at a different configuration: "
				f"{', '.join(differences)}. Re-run every architecture at the committed configuration."
			)


def _differences(first: dict, second: dict, prefix: str = "") -> list[str]:
	"""The dotted paths where two recorded configurations disagree."""
	found = []

	for key in sorted(set(first) | set(second)):
		path = f"{prefix}{key}"
		left, right = first.get(key), second.get(key)
		if isinstance(left, dict) and isinstance(right, dict):
			found.extend(_differences(left, right, f"{path}."))
		elif left != right:
			found.append(path)

	return found


def _cell(value: float | None) -> str:
	"""One metric as the table shows it, or a dash when the run never recorded that depth."""
	return f"{value:.4f}" if value is not None else "-"
