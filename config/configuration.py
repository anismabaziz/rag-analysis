"""The values a run is measured at, written down where a result file can carry them.

Architectures are only comparable because everything except the one axis under test is held still.
That is a claim, and this is the claim's evidence: the prompt, the models behind the encoders, the
retrieval depth, the temperature, and the chunker are read from one place and recorded on every
run, so a reader can check the isolation from a result file instead of taking the project's word
for it.

Nothing here is chosen per run. A configuration that a run could vary would make two results
incomparable, and the run would be recording its own isolation rather than demonstrating it.
"""

from pydantic import BaseModel, ConfigDict

from config.params import HIERARCHICAL, HIERARCHICAL_PARENT_FACTOR, Params
from core.prompt import answer_prompt, prompt_fingerprint
from data.embed import dense_encoder, rerank_encoder, sparse_encoder


class Encoder(BaseModel):
	"""One encoder: the name it is called by, and the commit of the weights behind that name.

	The revision is the one the local model cache resolved the name to, read after the run has
	loaded the encoder, and it is `None` on a machine where the encoder was never downloaded. A
	revision nobody can check is worse than an honest gap, because it looks like a guarantee.
	"""

	model_config = ConfigDict(extra="forbid")

	name: str
	revision: str | None


class Chunker(BaseModel):
	"""How the corpus was cut into chunks, which decides what retrieval can return at all.

	`parent_size` is the size a hierarchical strategy indexed whole sections at, and is
	nothing for the strategies that cut into one size. It is recorded because a strategy
	indexing parents alongside children can return a whole section where another returns
	a slice, and that is the difference an answer-quality figure would have to state.

	`is_committed` says whether this is the chunker every run is measured at. A run file
	recorded against a variant says so, which is what keeps the two kinds of run from
	being compared as if they shared a setup.
	"""

	model_config = ConfigDict(extra="forbid")

	strategy: str
	size: int
	overlap: int
	parent_size: int | None = None
	is_committed: bool = True

	@classmethod
	def committed(cls) -> "Chunker":
		"""The chunker every run is measured at, read from the one place it is named."""
		return cls(
			strategy=Params.CHUNKER_STRATEGY,
			size=Params.CHUNK_SIZE,
			overlap=Params.CHUNK_OVERLAP,
		)

	@classmethod
	def variant(cls, strategy: str, size: int, overlap: int) -> "Chunker":
		"""A chunker that replaces the committed one, as a chunking comparison needs."""
		return cls(
			strategy=strategy,
			size=size,
			overlap=overlap,
			parent_size=HIERARCHICAL_PARENT_FACTOR * size if strategy == HIERARCHICAL else None,
			is_committed=False,
		)


class Configuration(BaseModel):
	"""Everything a run holds still, as a result file records it.

	The generation model is recorded by name alone, with no revision beside it, because a hosted
	model alias publishes none: the model id is the whole pin. That is the weaker half of this
	record, and the response cache is what will make a run against it reproducible.
	"""

	model_config = ConfigDict(extra="forbid")

	prompt: str
	prompt_fingerprint: str
	generation_model: str
	generation_temperature: float
	dense_encoder: Encoder
	sparse_encoder: Encoder
	rerank_model: Encoder
	retrieval_depth: int
	rerank_candidates: int
	chunker: Chunker


def frozen_configuration(chunker: Chunker | None = None) -> Configuration:
	"""The configuration every run is measured at, read from the one place each value is named.

	A chunker passed in is what the run records, because a chunking variant is measured at
	the strategy it declares rather than the committed one. Left out, the committed chunker
	is recorded, which is what every retrieval architecture is measured at.
	"""
	dense, sparse = dense_encoder(), sparse_encoder()
	rerank = rerank_encoder()

	return Configuration(
		prompt=answer_prompt(),
		prompt_fingerprint=prompt_fingerprint(),
		generation_model=Params.GENERATION_MODEL,
		generation_temperature=Params.GENERATION_TEMPERATURE,
		dense_encoder=Encoder(name=dense.name, revision=dense.revision),
		sparse_encoder=Encoder(name=sparse.name, revision=sparse.revision),
		rerank_model=Encoder(name=rerank.name, revision=rerank.revision),
		retrieval_depth=Params.TOP_K,
		rerank_candidates=Params.RERANK_CANDIDATES,
		chunker=chunker or Chunker.committed(),
	)
