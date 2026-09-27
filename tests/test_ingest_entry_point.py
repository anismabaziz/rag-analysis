"""The ingest commands a reader is told to run must work without a live store."""

import asyncio
import importlib
import sys

import pytest


class CliUnderTest:
	"""Runs a CLI with ingestion recorded rather than performed."""

	def __init__(self, module, recorded):
		self._module = module
		self._recorded = recorded

	def run(self, *argv):
		saved = sys.argv
		sys.argv = [self._module.__name__, *argv]
		try:
			asyncio.run(self._module.main())
		finally:
			sys.argv = saved
		return self._recorded


@pytest.fixture
def ingested(monkeypatch):
	"""Every ingestion any command asked for, in the form it asked for it."""
	recorded = []
	monkeypatch.setattr("build_index.build_index", lambda collection, vectors: recorded.append((collection, vectors)))
	return recorded


@pytest.fixture
def cli(ingested):
	return CliUnderTest(importlib.import_module("main"), ingested)


def test_naive_ingest_uses_the_dense_only_collection(cli):
	assert cli.run("ingest", "naive") == [("rag_naive", ("dense",))]


def test_hybrid_ingest_uses_the_hybrid_collection(cli):
	assert cli.run("ingest", "hybrid") == [("rag_hybrid", ("dense", "sparse"))]


def test_unknown_architecture_ingests_nothing(cli, ingested, capsys):
	"""A name no architecture answers to used to be reported and then ignored."""
	with pytest.raises(SystemExit) as exit_info:
		cli.run("ingest", "graph")

	assert exit_info.value.code == 2
	assert ingested == []


def test_running_the_ingest_module_directly_needs_an_architecture(monkeypatch):
	"""`python build_index.py` used to call build_index() with no arguments."""
	module = importlib.import_module("build_index")
	monkeypatch.setattr(sys, "argv", ["build_index.py"])

	with pytest.raises(SystemExit) as exit_info:
		module.main()

	assert exit_info.value.code == 2


def test_running_the_ingest_module_directly_ingests_the_named_architecture(monkeypatch, ingested):
	module = importlib.import_module("build_index")
	monkeypatch.setattr(sys, "argv", ["build_index.py", "--architecture", "hybrid"])

	module.main()

	assert ingested == [("rag_hybrid", ("dense", "sparse"))]
