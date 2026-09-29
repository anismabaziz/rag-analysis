"""Citation accuracy: does each claim in an answer come from retrieved context.

Each claim is checked against the chunks that were actually retrieved, and counts as
supported only when one retrieved chunk contains a supporting span. The check is
deterministic string matching and uses no judge model, so the same answer and the same
chunks always give the same number.

The unanswerable stratum is included on purpose: a system that answers regardless names
strings the corpus never prints, so its claims land unsupported and the stratum scores
visibly worse. A system that refuses when retrieval is empty produces no claims and is
not scored as a miss.
"""

from core.chunk import Chunk, Provenance, REFUSAL


def a_chunk(text="the base Transformer uses h = 8 parallel attention layers"):
	return Chunk(text=text, score=0.9, provenance=Provenance(source="./documents/papers/x.pdf"))


def test_claims_split_on_sentence_boundaries():
	from evaluation.citations import split_claims

	claims = split_claims("The base model uses eight heads. It trains quickly!")

	assert claims == ["The base model uses eight heads.", "It trains quickly!"]


def test_empty_and_refusal_answers_produce_no_claims():
	from evaluation.citations import split_claims

	assert split_claims("") == []
	assert split_claims("   ") == []
	assert split_claims(REFUSAL) == []


def test_a_claim_quoting_retrieved_text_is_supported():
	from evaluation.citations import claim_supported

	chunk = a_chunk("the base Transformer uses h = 8 parallel attention layers")

	assert claim_supported("The base Transformer uses h = 8 parallel attention layers.", [chunk]) is True


def test_a_claim_naming_a_string_no_chunk_prints_is_unsupported():
	from evaluation.citations import claim_supported

	chunk = a_chunk("the base Transformer uses h = 8 parallel attention layers")

	assert claim_supported("The benchmark suite is MTEB.", [chunk]) is False


def test_support_needs_one_chunk_not_the_pool():
	"""A claim pieced together from two chunks is not supported by either."""
	from evaluation.citations import claim_supported

	first = a_chunk("the base Transformer uses eight heads")
	second = a_chunk("training takes three days on eight GPUs")

	assert claim_supported("The base Transformer uses eight heads trained on eight GPUs.", [first, second]) is False


def test_citation_accuracy_is_the_share_of_supported_claims():
	from evaluation.citations import citation_accuracy

	chunks = [a_chunk("the base Transformer uses h = 8 parallel attention layers")]

	assert citation_accuracy("The base Transformer uses h = 8 parallel attention layers. The benchmark is MTEB.", chunks) == 0.5


def test_citation_accuracy_with_no_claims_is_nothing_to_average():
	"""A refusal produces no claims, so there is no share to report rather than a zero."""
	from evaluation.citations import citation_accuracy

	assert citation_accuracy(REFUSAL, []) is None
	assert citation_accuracy("", [a_chunk()]) is None


def test_the_check_is_deterministic_and_uses_no_model():
	"""The same answer and chunks always give the same number without calling anything."""
	from evaluation.citations import citation_accuracy

	chunks = [a_chunk()]
	answer = "The base Transformer uses h = 8 parallel attention layers."

	assert citation_accuracy(answer, chunks) == citation_accuracy(answer, chunks) == 1.0


def _quoting_chunks(question, top_k):
	"""Retrieval that answers out of a passage quoting the answer the generator will give."""
	from test_evaluation_run import a_chunk as real_chunk

	if question.document is None:
		return [real_chunk("attention-is-all-you-need", "3.2.2 Multi-Head Attention")]

	return [real_chunk(question.document, question.section)]


async def _quoting_generate(self, query, context):
	"""A generator that quotes the retrieved context for answerable questions and names an
	absent probe for unanswerable ones, which is what answering regardless looks like."""
	from evaluation.set import Stratum, load_evaluation_set

	asked = next(one for one in load_evaluation_set("papers").questions if one.question == query)
	if asked.stratum is Stratum.UNANSWERABLE:
		return "The benchmark suite is MTEB."

	first_line = context.splitlines()[1] if len(context.splitlines()) > 1 else context
	return first_line


def _run_over(tmp_path, monkeypatch, chunks_for, generate):
	import asyncio
	import importlib

	from core import registry
	from evaluation.run import run_evaluation
	from test_evaluation_run import RecordingLlm, retrieve_from

	monkeypatch.setattr(registry, "get_generation_llm", lambda: RecordingLlm())
	monkeypatch.setattr(registry, "get_embed_model", lambda: _embed_model())

	for registered in registry.all_architectures():
		try:
			module = importlib.import_module(registered.pipeline.__module__)
		except ModuleNotFoundError:
			continue
		for name, replacement in (("get_qdrant_client", lambda: None),):
			if hasattr(module, name):
				monkeypatch.setattr(module, name, replacement)

	pipeline = registry.architecture("naive").pipeline
	monkeypatch.setattr(pipeline, "retrieve", retrieve_from(chunks_for))
	monkeypatch.setattr(pipeline, "generate", generate)

	return asyncio.run(run_evaluation("naive", "papers", results_dir=tmp_path / "results"))


class _embed_model:
	def embed(self, texts):
		return [[0.0] * 384 for _ in texts]


def test_each_claim_is_checked_against_the_chunks_actually_retrieved(tmp_path, monkeypatch):
	"""The per-question score is a reading of the stored answer and stored chunks."""
	from core.chunk import Chunk, Provenance
	from evaluation.citations import citation_accuracy
	from evaluation.run import load_run_file

	run = _run_over(tmp_path, monkeypatch, _quoting_chunks, _quoting_generate)

	checked = 0
	for result in run.results:
		chunks = [Chunk(text=item.text, score=item.score, provenance=Provenance()) for item in result.retrieved]
		assert result.citation_accuracy == citation_accuracy(result.answer, chunks)
		assert result.claims_total == len(result.claims)
		assert result.claims_supported <= result.claims_total
		if result.citation_accuracy is not None:
			checked += 1

	assert checked > 0
	written = sorted((tmp_path / "results").rglob("*.json"))
	assert load_run_file(written[0]) == run


def test_citation_is_reported_on_the_unanswerable_stratum_separately(tmp_path, monkeypatch):
	run = _run_over(tmp_path, monkeypatch, _quoting_chunks, _quoting_generate)

	assert run.aggregates.citation_accuracy is not None
	assert run.aggregates.citation_accuracy_unanswerable is not None
	assert run.aggregates.citation_accuracy_answerable is not None


def test_citation_on_unanswerable_is_worse_than_on_answerable(tmp_path, monkeypatch):
	"""Answering regardless names strings the corpus never prints, so the claims land unsupported."""
	run = _run_over(tmp_path, monkeypatch, _quoting_chunks, _quoting_generate)

	assert run.aggregates.citation_accuracy_unanswerable < run.aggregates.citation_accuracy_answerable


def test_a_refusal_when_retrieval_is_empty_carries_no_citation_score(tmp_path, monkeypatch):
	"""Producing no claims is not a miss: the question drops out of the mean instead."""
	from test_evaluation_run import found_nothing

	async def refuse_generate(self, query, context):
		raise AssertionError("empty retrieval must never reach the model")

	run = _run_over(tmp_path, monkeypatch, found_nothing, refuse_generate)

	assert all(result.citation_accuracy is None for result in run.results)
	assert run.aggregates.citation_accuracy is None
	assert run.aggregates.citation_scored == 0
	assert run.aggregates.citation_unscored == len(run.results)
