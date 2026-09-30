"""Every architecture the harness can run, declared once and found by name.

Adding an architecture used to mean a pipeline class, a wiring module, and a branch in the
command dispatcher. Only the first said anything true about the architecture; the other two
were lists of what existed, and a list written in three places is a list that eventually
disagrees with itself. An architecture is now a single declaration, and one registry built
from those declarations drives ingestion and querying alike, so an architecture cannot be
queryable without also being ingestible. Nothing else in the project lists architectures, so
nothing else has to be edited to add one: a new architecture is a new module in the
architecture package, and that module is the only file that changes.
"""

import importlib
import pkgutil
from dataclasses import dataclass
from typing import Callable

from config.configuration import Chunker
from core.base import get_generation_llm
from data.embed import get_embed_model

# The package whose modules declare architectures. Discovery reads every module in it, so
# nothing registers a new architecture by hand.
ARCHITECTURE_PACKAGE = "architectures"

# The vectors a chunk can be indexed under, named the way the store names them.
DENSE = "dense"
SPARSE = "sparse"


class UnknownArchitecture(LookupError):
	"""A name no architecture is registered under."""


@dataclass(frozen=True)
class Architecture:
	"""One retrieval architecture: what it is called, what it indexes, and what answers.

	`vectors` is what ingestion writes per chunk and `pipeline` is what reads it back. Both
	live in one declaration because a collection indexed one way and queried another way is a
	comparison measuring nothing. The chunker lives here too, for the same reason: a
	collection cut one way and recorded another way would leave the run file describing a
	corpus nobody measured. A chunker left out means the committed one, so the retrieval
	architectures keep sharing it while a chunking variant declares its own strategy.
	"""

	name: str
	description: str
	collection: str
	pipeline: type
	vectors: tuple[str, ...] = (DENSE,)
	chunker: Chunker | None = None

	def resolved_chunker(self) -> Chunker:
		"""The chunker this architecture is ingested and measured at.

		One named here rather than resolved at each use, so ingestion, the run file, and
		the table over run files cannot disagree about how a collection was cut.
		"""
		return self.chunker or Chunker.committed()

	def ingest(self) -> list:
		"""Index the corpus into this architecture's own collection, replacing what was there."""
		# Imported here because ingestion reaches back for the registry to resolve a name.
		from build_index import build_index

		return build_index(self.collection, vectors=self.vectors, chunker=self.resolved_chunker())

	def build(self):
		"""The pipeline this architecture queries with, wired to its own collection."""
		return self.pipeline(get_generation_llm(), get_embed_model(), self.collection)


_REGISTERED: dict[str, Architecture] = {}


def register(
	name: str,
	description: str,
	collection: str,
	vectors: tuple[str, ...] = (DENSE,),
	chunker: Chunker | None = None,
) -> Callable[[type], type]:
	"""Declare an architecture by decorating the pipeline that implements it.

	The decorated class is returned unchanged, so declaring an architecture costs the pipeline
	one line and introduces no other name. A chunker named here is what ingestion cuts the
	corpus with and what the run file records; left out, both use the committed one.
	"""

	def declare(pipeline: type) -> type:
		_REGISTERED[name] = Architecture(
			name=name,
			description=description,
			collection=collection,
			pipeline=pipeline,
			vectors=vectors,
			chunker=chunker,
		)
		return pipeline

	return declare


def load(package: str = ARCHITECTURE_PACKAGE) -> None:
	"""Import every module of the architecture package, so the declarations in them take effect.

	Registering on import is what keeps adding an architecture a one-file change. Importing is
	idempotent, so this is safe to call before every lookup.
	"""
	modules = importlib.import_module(package)
	for module in pkgutil.iter_modules(modules.__path__):
		importlib.import_module(f"{package}.{module.name}")


def architecture(name: str) -> Architecture:
	"""The architecture registered under this name.

	Names are resolved before anything is built, indexed, or queried, so a misspelled
	architecture cannot leave a half-ingested collection behind.
	"""
	load()
	try:
		return _REGISTERED[name]
	except KeyError:
		raise UnknownArchitecture(unknown_message(name)) from None


def all_architectures() -> list[Architecture]:
	"""Every registered architecture, in name order."""
	load()
	return [_REGISTERED[name] for name in sorted(_REGISTERED)]


def unknown_message(name: str) -> str:
	"""What a rejected name says, naming what is on offer instead of only what is wrong."""
	load()
	known = ", ".join(sorted(_REGISTERED)) or "none"
	return f"unknown architecture '{name}'. Registered architectures: {known}"
