"""The chunking comparison: one axis varying, reported on retrieval only.

Fixed-size, semantic, and hierarchical chunkers are compared with everything else
frozen: the same dense retrieval, the same depth, the same prompt, models, and
temperature. A hierarchical strategy indexes whole sections alongside their slices
and semantic splits vary in length, so the generator would not read the same context
and answer quality is not compared. Where the token budget is not matched, the table
says so and shows the budget as measured rather than as nominal.
"""

from dataclasses import dataclass
from pathlib import Path

from config.params import HIERARCHICAL_PARENT_FACTOR, Params
from evaluation.metrics import mean
from evaluation.run import RESULTS_DIR, QuestionResult, RunFile
from evaluation.table import cell, cell_int, question_means, retrieved_words_mean
from evaluation.summary import POOLED_SCOPE, configuration_differences, load_runs

# How far two strategies' measured budgets may differ before the table calls the budget
# unmatched. Retrieval returns a fixed number of chunks, so a strategy cutting smaller
# pieces hands the generator less text; anything past this is a difference in budget
# rather than noise in it.
BUDGET_TOLERANCE = 0.1


@dataclass(frozen=True)
class ChunkingRow:
	"""One line of the chunking table: an architecture measured at one scope.

	Only retrieval numbers appear here. Citation accuracy and exact match are left
	out on purpose: whole sections and variable semantic lengths mean the generator
	would not read the same context, so a quality figure would measure how much text
	retrieval handed over rather than how the corpus was cut.
	"""

	architecture: str
	scope: str
	questions: int
	scored: int
	chunker: str
	chunk_size: int
	chunk_overlap: int
	parent_size: int | None
	retrieved_words_mean: float | None
	recall_at: dict[str, float]
	ndcg_at: dict[str, float]
	reciprocal_rank: float


@dataclass(frozen=True)
class ChunkingSummary:
	"""The whole chunking table: every row, the depths, and what it was read from."""

	rows: tuple[ChunkingRow, ...]
	depths: tuple[str, ...]
	strategies: tuple[str, ...]
	budget_matched: bool
	prompt_fingerprint: str
	generation_model: str
	retrieval_depth: int
	files: tuple[str, ...]


def load_chunking_runs(results_dir: Path = RESULTS_DIR) -> list[RunFile]:
	"""The runs measured against a chunking variant, in a stable order.

	Every run file is read, because a chunking run and a retrieval run live side by side
	in the same directory. Only the variants belong in this table: a run at the committed
	chunker is the retrieval comparison, and letting it in would make a table framed as
	one axis varying actually span a retriever and a chunker.
	"""
	variants = [run for run in load_runs(results_dir) if not run.configuration.chunker.is_committed]
	if not variants:
		raise ValueError(
			f"no run in {results_dir} was measured against a chunking variant, "
			"so there is nothing to compare"
		)

	return variants


def summarize_chunking(runs: list[RunFile]) -> ChunkingSummary:
	"""The pooled and per-domain retrieval rows over runs that differ only in chunker.

	Every frozen value except the chunker must agree across runs: a second varying axis
	would stop this being a chunking comparison. The chunker is the only field allowed to
	differ, and the table states whether the token budget it produced came out matched.
	"""
	if not runs:
		raise ValueError("no runs to summarize")

	_check_chunking_isolation(runs)

	groups: dict[str, list[RunFile]] = {}
	for run in runs:
		groups.setdefault(run.architecture, []).append(run)

	rows = []
	for name in sorted(groups):
		group = sorted(groups[name], key=lambda run: run.domain)
		for run in group:
			rows.append(_domain_row(run))
		rows.append(_pooled_row(group))

	first = runs[0]
	depths = sorted(
		{depth for run in runs for depth in (*run.aggregates.recall_at, *run.aggregates.ndcg_at)},
		key=int,
	)

	return ChunkingSummary(
		rows=tuple(rows),
		depths=tuple(depths),
		strategies=tuple(sorted({run.configuration.chunker.strategy for run in runs})),
		budget_matched=_budget_matched(rows),
		prompt_fingerprint=first.configuration.prompt_fingerprint,
		generation_model=first.configuration.generation_model,
		retrieval_depth=first.configuration.retrieval_depth,
		files=tuple(sorted(f"{run.architecture} on {run.domain}" for run in runs)),
	)


def render_chunking_markdown(summary: ChunkingSummary) -> str:
	"""The chunking comparison as a markdown table, retrieval metrics only.

	The header states the frozen axis and whether the token budget came out matched,
	so a reader can see the isolation and the budget from the table itself. No
	answer-quality column appears: such a figure would measure how much text
	retrieval returned rather than how the corpus was cut.
	"""
	recall_columns = [f"recall@{depth}" for depth in summary.depths]
	ndcg_columns = [f"ndcg@{depth}" for depth in summary.depths]
	header = [
		"architecture",
		"scope",
		"questions",
		"scored",
		"chunker",
		"chunk_size",
		"chunk_overlap",
		"parent_size",
		"retrieved_words_mean",
		*recall_columns,
		*ndcg_columns,
		"mrr",
	]

	lines = [
		"# Chunking comparison (retrieval only)",
		"",
		f"{len(summary.files)} runs over {', '.join(summary.strategies)} at depth "
		f"{summary.retrieval_depth}.",
		"One axis varies: the chunker. Everything else is frozen: prompt "
		f"{summary.prompt_fingerprint}, {summary.generation_model} at temperature 0.0, "
		f"dense retrieval at depth {summary.retrieval_depth}.",
		f"Token budget: {_budget_statement(summary)}",
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
			row.chunker,
			str(row.chunk_size),
			str(row.chunk_overlap),
			cell_int(row.parent_size),
			cell(row.retrieved_words_mean),
			*(cell(row.recall_at.get(depth)) for depth in summary.depths),
			*(cell(row.ndcg_at.get(depth)) for depth in summary.depths),
			f"{row.reciprocal_rank:.4f}",
		]
		lines.append("| " + " | ".join(cells) + " |")

	lines.extend(
		[
			"",
			"Answer quality is not compared across chunking strategies. A hierarchical strategy "
			"indexes whole sections alongside their slices, so a retrieved chunk can be a section "
			"where another strategy returns a slice, and semantic splitting produces different "
			"chunk sizes than fixed-size splitting. The generator would not read the same context "
			"either way, so a quality figure here would measure how much text retrieval returned "
			"rather than how the corpus was cut. Any answer-quality figure reported across "
			"strategies states its token matching explicitly.",
			"retrieved_words_mean is the mean words retrieval returned per scored question, counted "
			"as whitespace-separated words. It is measured rather than nominal, which is what makes "
			"the token budget claim checkable.",
		]
	)

	return "\n".join(lines) + "\n"


def _domain_row(run: RunFile) -> ChunkingRow:
	"""One run's retrieval row, recomputed from its stored per-question results."""
	scored = [result for result in run.results if result.scored]

	return _row(run.architecture, run.domain, run.results, scored, run.configuration.chunker)


def _pooled_row(group: list[RunFile]) -> ChunkingRow:
	"""One architecture's domains pooled: every scored per-question result counted once."""
	scored = [result for run in group for result in run.results if result.scored]
	every = [result for run in group for result in run.results]

	return _row(group[0].architecture, POOLED_SCOPE, every, scored, group[0].configuration.chunker)


def _row(architecture: str, scope: str, results: list[QuestionResult], scored, chunker) -> ChunkingRow:
	"""One row of the table, from the per-question results rather than the stored aggregates.

	The domain row and the pooled row are built here so the two can never disagree about
	which questions count: the row is a reading of the results, not of the file's summary.
	"""
	return ChunkingRow(
		architecture=architecture,
		scope=scope,
		questions=len(results),
		scored=len(scored),
		chunker=chunker.strategy,
		chunk_size=chunker.size,
		chunk_overlap=chunker.overlap,
		parent_size=chunker.parent_size,
		retrieved_words_mean=retrieved_words_mean(scored),
		recall_at=question_means(scored, "recall_at"),
		ndcg_at=question_means(scored, "ndcg_at"),
		reciprocal_rank=mean(result.reciprocal_rank for result in scored),
	)


def _budget_matched(rows: list[ChunkingRow]) -> bool:
	"""Whether the strategies returned about the same amount of text per question.

	Read off the measured budgets of the pooled rows, which is the comparison the
	reader makes. Two strategies that cut differently sized pieces hand the generator
	more or less text, and a quality figure across them would be reporting that
	difference under another name.
	"""
	measured = [
		row.retrieved_words_mean
		for row in rows
		if row.scope == POOLED_SCOPE and row.retrieved_words_mean is not None
	]
	if len(measured) < 2:
		return False

	smallest, largest = min(measured), max(measured)

	return largest - smallest <= BUDGET_TOLERANCE * largest


def _budget_statement(summary: ChunkingSummary) -> str:
	"""What the token budget did, stated in the table's own numbers.

	Measured rather than declared, because a claim that a budget was matched is worth
	nothing if it was not checked. A hierarchical strategy indexes whole sections
	alongside their slices, so a budget is rarely matched across all three, and saying so
	is the point of stating it.
	"""
	pools = [
		row for row in summary.rows if row.scope == POOLED_SCOPE and row.retrieved_words_mean is not None
	]
	if not pools:
		return "no strategy returned text to measure."

	by_strategy = {row.chunker: row.retrieved_words_mean for row in pools}
	per_strategy = ", ".join(
		f"{strategy} {words:.0f} words" for strategy, words in sorted(by_strategy.items())
	)
	verdict = "matched across strategies" if summary.budget_matched else "NOT matched across strategies"

	return (
		f"measured at {per_strategy} per scored question, so the budget is {verdict}. "
		f"Fixed-size cuts {Params.CHUNK_SIZE} words with {Params.CHUNK_OVERLAP} overlap; semantic "
		"splits vary in length around that nominal size; a hierarchical strategy indexes whole "
		"sections at "
		f"{HIERARCHICAL_PARENT_FACTOR}x the child size alongside their slices. "
		"retrieved_words_mean counts whitespace-separated words, the same unit the cost column uses."
	)


def _check_chunking_isolation(runs: list[RunFile]) -> None:
	"""Refuse runs whose frozen configurations disagree about anything but the chunker.

	Every field except the chunker is held constant by design: a difference in any of
	them means two rows would compare setups rather than chunkers, and the summary stops
	instead of averaging over the difference.
	"""
	first = runs[0].configuration.model_dump()
	first.pop("chunker", None)

	for run in runs[1:]:
		recorded = run.configuration.model_dump()
		recorded.pop("chunker", None)
		differences = configuration_differences(first, recorded)
		if differences:
			raise ValueError(
				f"{run.architecture} on {run.domain} was measured at a different configuration: "
				f"{', '.join(differences)}. Re-run every architecture at the committed configuration."
			)
