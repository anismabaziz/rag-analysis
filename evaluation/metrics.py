"""What a retrieved chunk is, in the terms a scorer can judge it by.

Retrieval hands back chunks carrying a score and a provenance handle, and this module is where
those become a ranked list a metric can read: each result is numbered from one, names the corpus
document it came from, and keeps the heading and page so a stricter question can be asked of the
same stored results later.

Two numbers come out of that list. Recall is the share of the places a question names that
retrieval turned up, which is what a reader checks first because it says whether the answer was
on the table at all. Reciprocal rank is the reciprocal of the rank of the first hit, which is what
distinguishes a relevant passage in first place from one buried at the bottom of five.

Recall is read at three depths rather than one, and normalized discounted cumulative gain sits
beside it, because a retriever that finds everything at rank five is not the retriever that finds
it first. Every depth is read off the same retrieval by truncation: the run retrieves once at the
committed depth and the shallower numbers are what the first one or three results were worth, so
no architecture is ever measured at a depth of its own choosing.

A hit is a chunk from the document the label names, not one from the section. The section is
stored on every result and the label names one, but the document label is the fallback the
evaluation set was built with, on the grounds that a chunk from the right paper under the wrong
heading is a retrieval success and only a loader that files a whole paper under one heading should
cost a domain its score. Scoring the section strictly is a later question, answerable from the
same per-question results without rerunning anything.
"""

import math
from dataclasses import dataclass
from typing import Iterable

from core.chunk import Chunk
from corpus.manifest import Manifest
from evaluation.set import Location

# The depths recall and discounted gain are reported at. Every one of them is at or below the
# retrieval depth the committed configuration names, because a depth is read off a run by
# truncation and a run cannot report a depth it never retrieved.
REPORT_DEPTHS: tuple[int, ...] = (1, 3, 5)


@dataclass(frozen=True)
class Retrieved:
    """One retrieved chunk, numbered and named, as a metric reads it."""

    rank: int
    document: str | None
    section: str | None
    page: int | None
    score: float
    node_id: str | None
    text: str = ""


def retrieved_from(chunks: list[Chunk], manifest: Manifest) -> list[Retrieved]:
    """The retrieved chunks as a numbered list, each naming the corpus document it came from.

    Provenance carries the path ingestion globbed, and the manifest names its documents by file
    name, so the path is reduced to its file name before the lookup: a run from a different
    directory, or a repository path recorded with or without its leading `./`, resolves to the
    same document. A path the manifest does not hold keeps its rank and is reported with no
    document rather than dropped, because dropping it would quietly change the depth every metric
    was computed at.
    """
    results = []

    for rank, chunk in enumerate(chunks, start=1):
        provenance = chunk.provenance
        results.append(
            Retrieved(
                rank=rank,
                document=_document_named_by(provenance.source, manifest),
                section=provenance.section,
                page=provenance.page,
                score=chunk.score,
                node_id=provenance.node_id,
                text=chunk.text,
            )
        )

    return results


def locate(results: list[Retrieved], location: Location) -> int | None:
    """The rank of the first result from the document `location` names, or nothing.

    A chunk is a hit on the document alone. The section the label names is kept on the result for
    a later, stricter reading, and the document label is the fallback that keeps one loader's
    heading detection from zeroing a domain.
    """
    for result in results:
        if result.document == location.document:
            return result.rank

    return None


def recall(results: list[Retrieved], locations: list[Location]) -> float:
    """The share of the places a question names that retrieval turned up.

    Measured over every location rather than the first, so a multi-hop question that found only
    one of its two passages scores as a partial hit instead of a full one. A question with no
    locations has nothing to find and scores zero rather than raising.
    """
    if not locations:
        return 0.0

    return sum(
        1 for location in locations if locate(results, location) is not None
    ) / len(locations)


def recall_at(results: list[Retrieved], locations: list[Location], depth: int) -> float:
    """The share of the places a question names that the first `depth` results turned up.

    The depth is a truncation of the one retrieval the run made, not a second retrieval at a
    shallow depth, so two depths of the same question are two readings of the same ranking.
    """
    if not locations:
        return 0.0

    top = results[:depth]

    return sum(1 for location in locations if locate(top, location) is not None) / len(
        locations
    )


def ndcg_at(results: list[Retrieved], locations: list[Location], depth: int) -> float:
    """How high the places a question names ranked, discounted by rank and normalized.

    Relevance is binary on the document: a result counts when it comes from a document the
    question names. Each named document counts once, at the rank it first appears, so five
    chunks from one relevant document earn one discounted gain rather than five: without that,
    a retriever returning the same document five times would score above the ideal ranking.
    The ideal ranking is each distinct named document in turn at the top, so a multi-hop
    question whose two passages share one document is ideally answered by that document once
    rather than twice. A question with no locations has nothing to rank and scores zero
    rather than raising.
    """
    if not locations:
        return 0.0

    relevant = {location.document for location in locations}
    seen: set[str | None] = set()
    discounted = 0.0
    for result in results[:depth]:
        if result.document in relevant and result.document not in seen:
            seen.add(result.document)
            discounted += 1.0 / math.log2(result.rank + 1)
    ideal = sum(
        1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), depth) + 1)
    )

    return discounted / ideal if ideal else 0.0


def reciprocal_rank(results: list[Retrieved], location: Location) -> float:
    """The reciprocal of the rank of the first hit, and zero when there was none.

    Reciprocal rather than plain rank, so being two places down costs a quarter of the credit of
    being first and a run that finds everything late still scores far below one that finds it
    first. A multi-hop question is ranked on its own first location, the place it is named after,
    because a second passage found earlier is not an answer to where the question is answered.
    """
    rank = locate(results, location)

    return 0.0 if rank is None else 1.0 / rank


def mean(scores: Iterable[float | None]) -> float:
    """The mean of the measured scores, and zero when there was nothing to average.

    A `None` is skipped rather than counted as a miss, for the same reason an unscored
    question is not a miss. Shared by the run that averages its questions and the summary
    that averages across runs, so the two can never disagree about what a mean is.
    """
    measured = [score for score in scores if score is not None]

    return sum(measured) / len(measured) if measured else 0.0


def percentile(values: Iterable[float | None], rank: float) -> float:
    """The value below which `rank` percent of the measured values fall.

    A `None` is skipped rather than counted as zero, for the same reason an unscored
    question is not a miss. Linear interpolation between closest ranks, so the 50th
    percentile of an even count is the mean of the two middle values. Zero when there
    was nothing to measure. Shared by the run and the summary so the two can never
    disagree about what a percentile is.
    """
    measured = sorted(value for value in values if value is not None)
    if not measured:
        return 0.0

    position = (len(measured) - 1) * rank / 100
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return measured[low]

    weight = position - low
    return measured[low] * (1 - weight) + measured[high] * weight


def mean_or_nothing(values: Iterable[float | None]) -> float | None:
    """The mean of the measured values, and nothing when there was nothing to average.

    A `None` is skipped rather than counted as a miss, for the same reason an unscored
    question is not a miss. Shared by the run that averages its generated answers and
    the summary that averages the same stored counts, so the two can never disagree
    about what a mean over tokens is.
    """
    measured = [value for value in values if value is not None]

    return sum(measured) / len(measured) if measured else None


def _document_named_by(source: str | None, manifest: Manifest) -> str | None:
    """The manifest document a retrieved chunk's source path names, if the corpus holds it."""
    if not source:
        return None

    return manifest.document_named_by_file(source)
