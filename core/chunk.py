"""What every architecture hands back, and how it becomes a prompt.

Retrieval used to return bare strings, which carried no score and no handle back to the
source. Nothing could be scored, reranked, or cited without both, so retrieval now returns
a chunk: reading store results into chunks is the one job of `vector.store`, and assembling
those chunks into a prompt is the one job of this module. No architecture formats a context
of its own, so a difference in results is a difference in retrieval.
"""

from dataclasses import dataclass

# Stands in for the context of an empty retrieval, and is returned as the answer in its
# place, so the model is never asked to invent one.
REFUSAL = "I cannot answer this from the indexed documents, because retrieval found nothing relevant."

UNKNOWN_SOURCE = "unknown source"


@dataclass(frozen=True)
class Provenance:
	"""Where a chunk came from, enough to point a reader at the source.

	A chunk can span pages, so `page` is the first page it appears on.
	"""

	source: str | None = None
	section: str | None = None
	page: int | None = None
	node_id: str | None = None


@dataclass(frozen=True)
class Chunk:
	"""A retrieved span of text with the score that selected it and where it lives."""

	text: str
	score: float
	provenance: Provenance

	def citation(self) -> str:
		"""A one-line handle for this chunk, for prompts and for reporting."""
		source = self.provenance.source or UNKNOWN_SOURCE
		locator = ", ".join(
			part
			for part in (
				f"section {self.provenance.section}" if self.provenance.section else None,
				f"page {self.provenance.page}" if self.provenance.page is not None else None,
			)
			if part
		)
		return f"{source} ({locator})" if locator else source

	def report_line(self) -> str:
		"""How this chunk appears in a dumped query, score and provenance first."""
		return f"score={self.score:.4f} {self.citation()} node={self.provenance.node_id}"


def build_context(chunks: list[Chunk]) -> str:
	"""Assemble retrieved chunks into the single context block every architecture prompts with.

	Empty retrieval produces the refusal rather than an empty string, so a run that retrieved
	nothing is visible in its own output instead of looking like a run whose question happened
	to need no context.
	"""
	if not chunks:
		return REFUSAL

	blocks = [
		f"[{index}] ({chunk.citation()})\n{chunk.text}"
		for index, chunk in enumerate(chunks, start=1)
	]
	return "\n\n".join(blocks)
