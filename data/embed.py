"""The models that turn text into vectors, named once.

Both encoders are named here rather than at each use, because a result file that records one
model's name while the run used another is worse than no result file: the number survives and the
reason for it does not. The revision of each is read from the model cache the run loaded it out
of, so what a run file records is the commit of the weights that answered it rather than whatever
the model repository's default branch happens to point at today.
"""

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from config.params import Params
from fastembed import SparseTextEmbedding, TextEmbedding

# The name fastembed serves each of the project's models from, which is not always the name the
# run refers to it by: a model published for one library is often re-hosted for another.
DENSE_SOURCE_REPOSITORY = "qdrant/all-MiniLM-L6-v2-onnx"
SPARSE_SOURCE_REPOSITORY = "Qdrant/all_miniLM_L6_v2_with_attentions"


@dataclass(frozen=True)
class Encoder:
	"""One encoder: what it is called, and the commit of the weights behind the name."""

	name: str
	revision: str | None


def get_embed_model():
  return TextEmbedding(model_name=Params.DENSE_EMBEDDING_MODEL)


def get_sparse_embed_model():
  return SparseTextEmbedding(model_name=Params.SPARSE_EMBEDDING_MODEL)


def dense_encoder() -> Encoder:
	"""The dense encoder, as a run file records it."""
	return _encoder(Params.DENSE_EMBEDDING_MODEL, DENSE_SOURCE_REPOSITORY)


def sparse_encoder() -> Encoder:
	"""The sparse encoder, as a run file records it."""
	return _encoder(Params.SPARSE_EMBEDDING_MODEL, SPARSE_SOURCE_REPOSITORY)


def _encoder(name: str, repository: str) -> Encoder:
	"""The encoder of that name, with the revision the local model cache resolved it to.

	The revision is read from disk rather than from the network, so recording it costs a run
	nothing and a run on a machine where the model was never downloaded records `None` instead of
	a guess. That is the honest entry: a revision nobody can verify is decoration.
	"""
	ref = Path(model_cache_dir()) / f"models--{repository.replace('/', '--')}" / "refs" / "main"

	return Encoder(name=name, revision=ref.read_text().strip() if ref.is_file() else None)


def model_cache_dir() -> Path:
	"""Where the encoders are cached, which is where a run reads their revisions from.

	This mirrors the path the model library picks, because it is the one place a downloaded
	encoder's commit is written down. `FASTEMBED_CACHE_PATH` wins, as it does for the library.
	"""
	return Path(os.getenv("FASTEMBED_CACHE_PATH") or Path(tempfile.gettempdir()) / "fastembed_cache")
