"""The ingest commands a reader is told to run must work without a live store."""

import asyncio
import importlib
import sys

import pytest


class CliUnderTest:
	"""Runs the CLI with ingestion recorded rather than performed."""

	def __init__(self, module, recorded, monkeypatch):
		self._module = module
		self._recorded = recorded
		self._monkeypatch = monkeypatch

	def ingest(self, *argv):
		self._monkeypatch.setattr(sys, "argv", ["main.py", *argv])
		asyncio.run(self._module.main())
		return self._recorded


@pytest.fixture
def cli(monkeypatch):
	module = importlib.import_module("main")

	recorded = []
	monkeypatch.setattr(module, "build_index", lambda name, enable_hybrid: recorded.append((name, enable_hybrid)))

	return CliUnderTest(module, recorded, monkeypatch)


def test_naive_ingest_uses_the_dense_only_collection(cli):
	assert cli.ingest("ingest", "naive") == [("rag_naive", False)]


def test_hybrid_ingest_uses_the_hybrid_collection(cli):
	assert cli.ingest("ingest", "hybrid") == [("rag_hybrid", True)]


def test_unknown_architecture_ingests_nothing(cli):
	assert cli.ingest("ingest", "graph") == []


def test_running_the_ingest_module_directly_needs_a_collection(monkeypatch):
	"""`python build_index.py` used to call build_index() with no arguments."""
	module = importlib.import_module("build_index")
	monkeypatch.setattr(sys, "argv", ["build_index.py"])

	with pytest.raises(SystemExit) as exit_info:
		module.main()

	assert exit_info.value.code == 2


def test_running_the_ingest_module_directly_forwards_its_arguments(monkeypatch):
	module = importlib.import_module("build_index")

	recorded = []
	monkeypatch.setattr(module, "build_index", lambda name, enable_hybrid: recorded.append((name, enable_hybrid)))
	monkeypatch.setattr(sys, "argv", ["build_index.py", "--collection", "rag_hybrid", "--hybrid"])

	module.main()

	assert recorded == [("rag_hybrid", True)]
