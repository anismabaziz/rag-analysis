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
from evaluation.metrics import mean, mean_or_nothing, percentile
from evaluation.run import RESULTS_DIR, QuestionResult, RunFile, load_run_file
from evaluation.set import Stratum
from evaluation.table import cell, configuration_differences, question_means

# The scope of a row built from every domain's scored questions rather than one domain's.
POOLED_SCOPE = "pooled"

# The strata that get their own rows, because the claim under test is a difference between
# them: hybrid retrieval is expected to beat dense-only on identifier-heavy questions and lose
# on paraphrased ones. Multi-hop and unanswerable answer different questions (partial recall,
# refusal), so they stay out of the retrieval comparison rather than being averaged into it.
STRATUM_SCOPES = (Stratum.IDENTIFIER_HEAVY, Stratum.PARAPHRASE)


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
    retrieval_latency_p50_s: float = 0.0
    retrieval_latency_p95_s: float = 0.0
    generation_latency_p50_s: float | None = None
    generation_latency_p95_s: float | None = None
    prompt_tokens_mean: float | None = None
    completion_tokens_mean: float | None = None
    total_tokens_mean: float | None = None
    total_tokens: int = 0
    generation_count: int = 0


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
        raise FileNotFoundError(
            f"no run files in {results_dir}: run an architecture first"
        )

    return [load_run_file(path) for path in paths]


def load_committed_runs(results_dir: Path = RESULTS_DIR) -> list[RunFile]:
    """The run files measured at the committed configuration, in a stable order.

    A run against a chunking variant is left out rather than refused, because the two
    kinds of run answer different questions and both belong in the repository. This table
    is the retrieval comparison at one chunker, so a variant row in it would be
    comparing a retriever against a chunker.
    """
    paths = sorted(Path(results_dir).rglob("*.json"))
    if not paths:
        raise FileNotFoundError(
            f"no run files in {results_dir}: run an architecture first"
        )

    committed = [load_run_file(path) for path in paths]
    kept = [run for run in committed if run.configuration.chunker.is_committed]
    if not kept:
        raise ValueError(
            f"every run in {results_dir} was measured against a chunking variant, "
            "and none against the committed configuration"
        )

    return kept


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
        groups.setdefault(
            (run.architecture, run.commit.sha, run.corpus.identifier), []
        ).append(run)

    rows = []
    for key in sorted(groups, key=lambda group: str(group[0])):
        group = groups[key]
        for run in sorted(group, key=lambda run: run.domain):
            rows.append(_domain_row(run))
            for stratum in STRATUM_SCOPES:
                rows.append(_stratum_row(run, stratum))
        rows.append(_pooled_row(group))
        for stratum in STRATUM_SCOPES:
            rows.append(_pooled_stratum_row(group, stratum))

    first = runs[0]
    depths = sorted(
        {
            depth
            for run in runs
            for depth in (*run.aggregates.recall_at, *run.aggregates.ndcg_at)
        },
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
        "retrieval_p50_s",
        "retrieval_p95_s",
        "generation_p50_s",
        "generation_p95_s",
        "tokens_total",
        "tokens_mean",
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
            *(cell(row.recall_at.get(depth)) for depth in summary.depths),
            *(cell(row.ndcg_at.get(depth)) for depth in summary.depths),
            f"{row.reciprocal_rank:.4f}",
            cell(row.citation_accuracy),
            cell(row.citation_accuracy_answerable),
            cell(row.citation_accuracy_unanswerable),
            cell(row.exact_match),
            str(row.exact_match_scored),
            cell(row.citation_accuracy_extractive),
            f"{row.retrieval_latency_p50_s:.4f}",
            f"{row.retrieval_latency_p95_s:.4f}",
            cell(row.generation_latency_p50_s),
            cell(row.generation_latency_p95_s),
            str(row.total_tokens),
            cell(row.total_tokens_mean),
        ]
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "A scope of papers:identifier_heavy slices a domain to one stratum; "
            "pooled:identifier_heavy pools that slice across domains. "
            "The claim is a difference between those slices, so they are rows rather than a footnote.",
            "cite is the share of answer claims one retrieved chunk each supports; "
            f"cite_unanswerable is the same share over the {Stratum.UNANSWERABLE.value} stratum alone.",
            "exact_match is the share of extractive questions whose answer matches the gold span, "
            "tolerating surrounding whitespace and punctuation only; "
            "cite_extractive is citation accuracy over those same questions.",
            "retrieval_p50_s and retrieval_p95_s are the median and 95th percentile retrieval seconds "
            "over the run's questions; generation_p50_s and generation_p95_s are the same over the "
            "generated answers alone. tokens_total is prompt plus completion words summed over the "
            "generated answers, counted as whitespace-separated words.",
        ]
    )

    return "\n".join(lines) + "\n"


def _domain_row(run: RunFile) -> SummaryRow:
    """One run's row, recomputed from its stored per-question results rather than its aggregates.

    The aggregates travel in the file, but the row does not trust them: a hand-edited aggregate
    next to untouched questions would otherwise survive into the table, and the table would no
    longer be a reading of what the run measured.
    """
    return _row_for(run.architecture, run.domain, run.results)


def _stratum_row(run: RunFile, stratum: Stratum) -> SummaryRow:
    """One run's row sliced to one stratum, recomputed from the stored per-question results.

    The slice keeps the questions of that stratum only, so the identifier-heavy numbers sit
    beside the paraphrase ones instead of being averaged away. The scope names the domain and
    the stratum together, because the same stratum of another domain is a different slice.
    """
    kept = [result for result in run.results if result.stratum is stratum]

    return _row_for(run.architecture, f"{run.domain}:{stratum.value}", kept)


def _pooled_row(group: list[RunFile]) -> SummaryRow:
    """One architecture's domains pooled: every scored per-question result counted once.

    The pool is recomputed from the stored per-question results rather than averaged from the
    domain rows, so a domain with more scored questions moves the pool more than one with
    fewer, and a hand-edited aggregate cannot survive next to the questions behind it.
    """
    pooled = [result for run in group for result in run.results]

    return _row_for(group[0].architecture, POOLED_SCOPE, pooled)


def _pooled_stratum_row(group: list[RunFile], stratum: Stratum) -> SummaryRow:
    """One architecture's stratum pooled across its domains: every such result counted once.

    Recomputed from the stored per-question results rather than averaged from the domain
    stratum rows, for the same reason the pool is recomputed from the domain rows.
    """
    kept = [result for run in group for result in run.results if result.stratum is stratum]

    return _row_for(group[0].architecture, f"{POOLED_SCOPE}:{stratum.value}", kept)


def _row_for(
    architecture: str, scope: str, results: list[QuestionResult]
) -> SummaryRow:
    """One row over the given questions, recomputed from what the run files stored.

    One constructor serves domain, stratum, and pooled rows, so the three can never disagree
    about what a mean is: scored means over the questions that named a place to look, citation
    over the ones whose answers made claims, exact match over the extractive ones.
    """
    scored = [result for result in results if result.scored]

    return SummaryRow(
        architecture=architecture,
        scope=scope,
        questions=len(results),
        scored=len(scored),
        unscored=len(results) - len(scored),
        recall=mean(result.recall for result in scored),
        reciprocal_rank=mean(result.reciprocal_rank for result in scored),
        recall_at=_question_means(scored, "recall_at"),
        ndcg_at=_question_means(scored, "ndcg_at"),
        citation_accuracy=_citation_mean(results, None),
        citation_accuracy_answerable=_citation_mean(results, False),
        citation_accuracy_unanswerable=_citation_mean(results, True),
        citation_scored=len(
            [result for result in results if result.citation_accuracy is not None]
        ),
        citation_unscored=len(
            [result for result in results if result.citation_accuracy is None]
        ),
        **_extractive_cells(results),
        **_latency_cells(results),
        **_cost_cells(results),
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
        "exact_match_scored": len(
            [
                result
                for result in extractive
                if exact_match_score(result.answer, result.gold_answer, True)
                is not None
            ]
        ),
        "exact_match_unscored": len(results) - len(extractive),
        "citation_accuracy_extractive": _citation_mean_extractive(results),
        "citation_extractive_scored": len(
            [result for result in extractive if result.citation_accuracy is not None]
        ),
    }


def _latency_cells(results: list[QuestionResult]) -> dict:
    """The latency percentile columns, recomputed from the stored per-question timings.

    One helper serves the domain row and the pooled row, so the pooled percentiles are
    percentiles over the pooled questions rather than means of domain medians. Retrieval
    covers every stored question and generation covers only the answers the model was
    asked for, because empty retrieval never reaches it.
    """
    retrieval = [result.retrieval_latency_s for result in results]
    generation = [
        result.generation_latency_s
        for result in results
        if result.generation_latency_s is not None
    ]

    return {
        "retrieval_latency_p50_s": percentile(retrieval, 50),
        "retrieval_latency_p95_s": percentile(retrieval, 95),
        "generation_latency_p50_s": percentile(generation, 50) if generation else None,
        "generation_latency_p95_s": percentile(generation, 95) if generation else None,
    }


def _cost_cells(results: list[QuestionResult]) -> dict:
    """The token cost columns, recomputed from the stored per-question counts.

    One helper serves the domain row and the pooled row, so the two can never disagree
    about what counts as generated: a question whose retrieval never reached the model
    carries no tokens and drops out of the means instead of dragging them down.
    """
    generated = [result for result in results if result.total_tokens is not None]

    return {
        "prompt_tokens_mean": mean_or_nothing(
            [result.prompt_tokens for result in generated]
        ),
        "completion_tokens_mean": mean_or_nothing(
            [result.completion_tokens for result in generated]
        ),
        "total_tokens_mean": mean_or_nothing(
            [result.total_tokens for result in generated]
        ),
        "total_tokens": sum(
            result.total_tokens
            for result in generated
            if result.total_tokens is not None
        ),
        "generation_count": len(generated),
    }


def _citation_mean(
    results: list[QuestionResult], unanswerable: bool | None
) -> float | None:
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
        kept = [
            result for result in results if result.stratum is not Stratum.UNANSWERABLE
        ]
    else:
        kept = list(results)

    measured = []
    for result in kept:
        chunks = [
            Chunk(text=item.text, score=item.score, provenance=Provenance())
            for item in result.retrieved
        ]
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
        chunks = [
            Chunk(text=item.text, score=item.score, provenance=Provenance())
            for item in result.retrieved
        ]
        score = citation_accuracy(result.answer, chunks)
        if score is not None:
            measured.append(score)

    return sum(measured) / len(measured) if measured else None


def _question_means(scored: list[QuestionResult], field: str) -> dict[str, float]:
    """The mean at each recorded depth, over the scored questions that carry that depth.

    The depths are the ones the questions recorded, not a list kept here, so a summary built
    from run files can never disagree with the files about which depths exist.
    """
    return question_means(scored, field)


def _check_isolation(runs: list[RunFile]) -> None:
    """Refuse runs whose frozen configurations disagree about anything.

    Every field the configuration records is held constant across architectures by design: a
    difference in any of them means two rows would compare setups rather than retrievers, and
    the summary stops instead of averaging over the difference.
    """
    first = runs[0].configuration.model_dump()

    for run in runs[1:]:
        differences = configuration_differences(first, run.configuration.model_dump())
        if differences:
            raise ValueError(
                f"{run.architecture} on {run.domain} was measured at a different configuration: "
                f"{', '.join(differences)}. Re-run every architecture at the committed configuration."
            )
