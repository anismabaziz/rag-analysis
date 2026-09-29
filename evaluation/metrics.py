"""What a retrieved chunk is, in the terms a scorer can judge it by.

Retrieval hands back chunks carrying a score and a provenance handle, and this module is where
those become a ranked list a metric can read: each result is numbered from one, names the corpus
document it came from, and keeps the heading and page so a stricter question can be asked of the
same stored results later.

Two numbers come out of that list. Recall is the share of the places a question names that
retrieval turned up, which is what a reader checks first because it says whether the answer was
on the table at all. Reciprocal rank is the reciprocal of the rank of the first hit, which is what
distinguishes a relevant passage in first place from one buried at the bottom of five.

A hit is a chunk from the document the label names, not one from the section. The section is
stored on every result and the label names one, but the document label is the fallback the
evaluation set was built with, on the grounds that a chunk from the right paper under the wrong
heading is a retrieval success and only a loader that files a whole paper under one heading should
cost a domain its score. Scoring the section strictly is a later question, answerable from the
same per-question results without rerunning anything.
"""

from dataclasses import dataclass

from core.chunk import Chunk
from corpus.manifest import Manifest
from evaluation.set import Location


@dataclass(frozen=True)
class Retrieved:
	"""One retrieved chunk, numbered and named, as a metric reads it."""

	rank: int
	document: str | None
	section: str | None
	page: int | None
	score: float
	node_id: str | None


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

	return sum(1 for location in locations if locate(results, location) is not None) / len(locations)


def reciprocal_rank(results: list[Retrieved], location: Location) -> float:
	"""The reciprocal of the rank of the first hit, and zero when there was none.

	Reciprocal rather than plain rank, so being two places down costs a quarter of the credit of
	being first and a run that finds everything late still scores far below one that finds it
	first. A multi-hop question is ranked on its own first location, the place it is named after,
	because a second passage found earlier is not an answer to where the question is answered.
	"""
	rank = locate(results, location)

	return 0.0 if rank is None else 1.0 / rank


def _document_named_by(source: str | None, manifest: Manifest) -> str | None:
	"""The manifest document a retrieved chunk's source path names, if the corpus holds it."""
	if not source:
		return None

	return manifest.document_named_by_file(source)
