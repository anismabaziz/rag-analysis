"""What retrieval hands back is the raw material every later metric is computed from.

A score and a handle to the source are what make scoring, reranking, and citation accuracy
possible, so these tests pin the returned chunks, the context they are assembled into, and
the refusal that stands in for an empty retrieval.
"""

import asyncio
import importlib

from qdrant_client import models

from architectures.hybrid import HybridRAG
from architectures.naive import NaiveRAG
from core.chunk import REFUSAL, Chunk, Provenance, build_context


def indexed_point(score, text="A span of retrieved text.", **metadata):
	"""A scored store point shaped like the ones ingestion writes."""
	return models.ScoredPoint(
		id="node-1",
		version=0,
		score=score,
		payload={"text": text, "metadata": metadata},
	)


class RecordingLlm:
	"""Reports the prompts it was asked to answer."""

	def __init__(self):
		self.prompts = []

	async def apredict(self, template, **kwargs):
		self.prompts.append(kwargs)
		return "an answer"


class DenseEmbedModel:
	"""Stands in for the encoder, which no test may load."""

	def embed(self, texts):
		return [[0.0] * 384 for _ in texts]


class SparseValues(list):
	"""The sparse encoder returns arrays; the architecture only ever calls tolist on them."""

	def tolist(self):
		return list(self)


class SparseVector:
	"""The shape the sparse encoder returns, without loading a model."""

	indices = SparseValues([0])
	values = SparseValues([1.0])


class SparseEmbedModel:
	"""Stands in for the sparse encoder, which no test may download."""

	def embed(self, texts):
		return [SparseVector() for _ in texts]


class FixedStore:
	"""A store that answers every query with the same scored points."""

	def __init__(self, points):
		self._points = points

	def query_points(self, **kwargs):
		return type("Result", (), {"points": self._points})()


def architecture_with_points(architecture_class, points, monkeypatch):
	"""An architecture reading a fixed set of scored points instead of a live store."""
	module = importlib.import_module(architecture_class.__module__)
	monkeypatch.setattr(module, "get_qdrant_client", lambda: FixedStore(points))
	monkeypatch.setattr(module, "get_sparse_embed_model", lambda: SparseEmbedModel(), raising=False)

	return architecture_class(RecordingLlm(), DenseEmbedModel(), "rag_test")


def test_dense_retrieval_returns_the_text_score_and_provenance_of_each_chunk(monkeypatch):
	point = indexed_point(
		0.82,
		source_file="./documents/paper.pdf",
		section="Evaluation",
		pages=[7],
	)
	architecture = architecture_with_points(NaiveRAG, [point], monkeypatch)

	chunks = asyncio.run(architecture.retrieve("how is retrieval evaluated?"))

	assert chunks == [
		Chunk(
			text="A span of retrieved text.",
			score=0.82,
			provenance=Provenance(
				source="./documents/paper.pdf",
				section="Evaluation",
				page=7,
				node_id="node-1",
			),
		)
	]


def test_fused_retrieval_returns_chunks_rather_than_bare_strings(monkeypatch):
	point = indexed_point(0.5, source_file="./documents/manual.pdf", section="Setup", pages=[2])
	architecture = architecture_with_points(HybridRAG, [point], monkeypatch)

	chunks = asyncio.run(architecture.retrieve("how do i set this up?"))

	assert all(isinstance(chunk, Chunk) for chunk in chunks)
	assert chunks[0].provenance.section == "Setup"


def test_retrieval_preserves_the_order_the_store_ranked_the_chunks(monkeypatch):
	points = [indexed_point(0.1, text="lower ranked"), indexed_point(0.9, text="top ranked")]
	architecture = architecture_with_points(NaiveRAG, points, monkeypatch)

	chunks = asyncio.run(architecture.retrieve("anything"))

	assert [chunk.text for chunk in chunks] == ["lower ranked", "top ranked"]


def test_a_chunk_cites_its_source_section_and_page():
	chunk = Chunk(
		text="text",
		score=1.0,
		provenance=Provenance(source="./documents/paper.pdf", section="Results", page=12),
	)

	assert chunk.citation() == "./documents/paper.pdf (section Results, page 12)"


def test_a_chunk_with_nothing_to_cite_still_names_its_source():
	chunk = Chunk(text="text", score=1.0, provenance=Provenance())

	assert chunk.citation() == "unknown source"


def test_every_architecture_hands_the_generator_the_same_assembled_context(monkeypatch):
	"""Architectures are only comparable if prompt formatting is not one of the differences."""
	points = [indexed_point(0.4, text="shared span", source_file="./documents/paper.pdf", section="Intro", pages=[1])]
	seen = []

	for architecture_class in (NaiveRAG, HybridRAG):
		architecture = architecture_with_points(architecture_class, points, monkeypatch)
		asyncio.run(architecture.answer("what is introduced?"))
		seen.append(architecture.llm.prompts[0]["context"])

	assert seen[0] == seen[1]
	assert "shared span" in seen[0]


def test_a_dumped_query_shows_the_score_and_provenance_of_every_chunk(monkeypatch, capsys):
	point = indexed_point(0.42, text="the dumped span", source_file="./documents/paper.pdf", section="Intro", pages=[3])
	architecture = architecture_with_points(NaiveRAG, [point], monkeypatch)

	asyncio.run(architecture.answer("what is introduced?"))
	dump = capsys.readouterr().out

	assert "score=0.4200" in dump
	assert "./documents/paper.pdf (section Intro, page 3)" in dump
	assert "node=node-1" in dump
	assert "the dumped span" in dump


def test_an_empty_retrieval_refuses_instead_of_asking_the_model(monkeypatch):
	architecture = architecture_with_points(NaiveRAG, [], monkeypatch)

	answer = asyncio.run(architecture.answer("what is the answer to everything?"))

	assert answer == REFUSAL
	assert architecture.llm.prompts == []


def test_empty_retrieval_assembles_to_the_refusal_rather_than_an_empty_string():
	assert build_context([]) == REFUSAL
	assert REFUSAL.strip() != ""
