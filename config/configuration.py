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

from config.params import Params
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
	"""How the corpus was cut into chunks, which decides what retrieval can return at all."""

	model_config = ConfigDict(extra="forbid")

	strategy: str
	size: int
	overlap: int


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


def frozen_configuration() -> Configuration:
	"""The configuration every run is measured at, read from the one place each value is named."""
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
		chunker=Chunker(
			strategy=Params.CHUNKER_STRATEGY,
			size=Params.CHUNK_SIZE,
			overlap=Params.CHUNK_OVERLAP,
		),
	)
