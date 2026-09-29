"""What the harness can run, and how it finds out.

An architecture used to be written down in three places at once: a pipeline class, a wiring
module, and a branch in the command dispatcher. These tests pin the replacement. One
declaration is read by ingestion and by querying alike, the command line lists what exists,
and a new architecture is a new file rather than an edit to a list of everything that already
exists.
"""

import asyncio
import importlib
import sys
import tomllib
from pathlib import Path

import pytest

from architectures.hybrid import HybridRAG
from architectures.naive import NaiveRAG
from architectures.sparse import SparseRAG
from core import registry

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"

THROWAWAY_PACKAGE = "throwaway_architectures"

# A whole architecture, in the one file that exists to declare it: a pipeline, and the
# collection it reads. Nothing outside this file mentions it.
THROWAWAY_DECLARATION = '''
from core.base import BaseRAG
from core.registry import SPARSE, register


@register(
\tname="word-frequency",
\tdescription="Answers from a word frequency table.",
\tcollection="rag_word_frequency",
\tvectors=(SPARSE,),
)
class WordFrequencyRAG(BaseRAG):
\t"""Answers from a word frequency table."""

\tasync def retrieve(self, query: str):
\t\treturn []
'''


def run_cli(*argv):
	"""Run the CLI the way a reader would, with the command line given as arguments."""
	saved = sys.argv
	sys.argv = ["rag-analysis", *argv]
	try:
		asyncio.run(importlib.import_module("main").main())
	finally:
		sys.argv = saved


@pytest.fixture
def declared_architecture(tmp_path, monkeypatch):
	"""An architecture declared in a package of its own, on the import path and nowhere else."""
	package = tmp_path / THROWAWAY_PACKAGE
	package.mkdir()
	(package / "__init__.py").write_text("")
	(package / "word_frequency.py").write_text(THROWAWAY_DECLARATION)

	monkeypatch.syspath_prepend(str(tmp_path))
	importlib.invalidate_caches()
	# Registering on import is what makes adding an architecture a one-file change, and the
	# cost is that a declaration outlives the package it came from. Swapping the registry for a
	# copy puts the committed architectures back when the package goes.
	monkeypatch.setattr(registry, "_REGISTERED", dict(registry._REGISTERED))

	yield f"{THROWAWAY_PACKAGE}.word_frequency"

	for name in [name for name in sys.modules if name.startswith(THROWAWAY_PACKAGE)]:
		del sys.modules[name]


@pytest.fixture
def no_store(monkeypatch):
	"""Records anything that would reach the store, so a rejected name can be shown to stop short."""
	ingested = []
	monkeypatch.setattr("build_index.build_index", lambda collection, vectors: ingested.append((collection, vectors)))
	return ingested


def test_every_architecture_is_registered_by_declaration():
	"""Nothing lists architectures anymore, so these are the only place the names exist."""
	assert {architecture.name for architecture in registry.all_architectures()} == {"naive", "sparse", "hybrid"}


def test_listing_prints_every_registered_architecture(capsys):
	run_cli("architectures")

	lines = capsys.readouterr().out.splitlines()
	for architecture in registry.all_architectures():
		assert any(architecture.name in line and architecture.collection in line for line in lines), architecture.name


def test_listing_includes_an_architecture_that_was_only_just_declared(declared_architecture, capsys):
	registry.load(THROWAWAY_PACKAGE)

	run_cli("architectures")

	assert "word-frequency" in capsys.readouterr().out


def test_the_harness_is_invokable_as_a_console_command(capsys):
	"""A reader should not need to know the module layout to run the thing."""
	module_name, _, attribute = (
		tomllib.loads(PYPROJECT.read_text())["project"]["scripts"]["rag-analysis"].partition(":")
	)
	entry_point = getattr(importlib.import_module(module_name), attribute)

	saved = sys.argv
	sys.argv = ["rag-analysis", "architectures"]
	try:
		entry_point()
	finally:
		sys.argv = saved

	assert "naive" in capsys.readouterr().out


def test_ingestion_reads_the_collection_the_architecture_declares(no_store):
	for architecture in registry.all_architectures():
		architecture.ingest()
		assert no_store[-1] == (architecture.collection, architecture.vectors)


def test_querying_reads_the_pipeline_the_architecture_declares(monkeypatch):
	monkeypatch.setattr(registry, "get_generation_llm", lambda: "llm")
	monkeypatch.setattr(registry, "get_embed_model", lambda: "encoder")
	pipelines = (("naive", NaiveRAG), ("sparse", SparseRAG), ("hybrid", HybridRAG))

	assert {name for name, _ in pipelines} == {a.name for a in registry.all_architectures()}
	for architecture, expected in pipelines:
		pipeline = registry.architecture(architecture).build()
		assert isinstance(pipeline, expected)
		assert pipeline.collection_name == registry.architecture(architecture).collection


def test_every_architecture_is_ingested_into_a_collection_of_its_own():
	"""A rerun must not mix corpora, and a sparse architecture judged against a collection that
	also holds dense points would be judging a setup rather than a retriever. Ingestion is what
	could break this, and the test above pins that it reads the declaration."""
	declared = {a.name: (a.collection, a.vectors) for a in registry.all_architectures()}

	assert declared["sparse"] == ("rag_sparse", (registry.SPARSE,))
	assert len({collection for collection, _ in declared.values()}) == len(declared)


def test_the_cli_ingests_through_the_same_declaration_ingestion_reads(no_store):
	run_cli("ingest", "hybrid")

	assert no_store == [("rag_hybrid", (registry.DENSE, registry.SPARSE))]


def test_the_cli_queries_the_architecture_it_is_given(monkeypatch):
	asked = []

	async def record(self, query):
		asked.append((self.collection_name, query))

	monkeypatch.setattr(registry, "get_generation_llm", lambda: "llm")
	monkeypatch.setattr(registry, "get_embed_model", lambda: "encoder")
	monkeypatch.setattr(NaiveRAG, "answer", record)

	run_cli("query", "naive", "how are positional encodings scaled?")

	assert asked == [("rag_naive", "how are positional encodings scaled?")]


def test_an_unknown_architecture_is_rejected_before_the_store_is_contacted(no_store):
	with pytest.raises(SystemExit) as exit_info:
		run_cli("ingest", "graph")

	assert exit_info.value.code == 2
	assert no_store == []


def test_an_unknown_architecture_cannot_be_queried_either(monkeypatch):
	asked = []

	async def record(self, query):
		asked.append(query)

	monkeypatch.setattr(NaiveRAG, "answer", record)

	with pytest.raises(SystemExit) as exit_info:
		run_cli("query", "graph", "a question")

	assert exit_info.value.code == 2
	assert asked == []


def test_a_rejected_name_says_what_is_registered_instead(capsys):
	with pytest.raises(SystemExit):
		run_cli("ingest", "graph")

	assert "naive" in capsys.readouterr().err


def test_a_throwaway_architecture_works_without_being_wired_anywhere(declared_architecture, monkeypatch, no_store):
	"""Adding an architecture is one new file, so a fourth and fifth cost the same as a second."""
	monkeypatch.setattr(registry, "get_generation_llm", lambda: "llm")
	monkeypatch.setattr(registry, "get_embed_model", lambda: "encoder")
	registry.load(THROWAWAY_PACKAGE)

	declared = registry.architecture("word-frequency")
	declared.ingest()
	pipeline = declared.build()

	assert no_store == [("rag_word_frequency", (registry.SPARSE,))]
	assert isinstance(pipeline, importlib.import_module(declared_architecture).WordFrequencyRAG)
