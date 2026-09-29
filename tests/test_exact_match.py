"""Exact match on the extractive stratum: one objective number beside citation accuracy.

Extractive questions carry a short span copied from the source as their gold answer.
Each is scored by comparing the generated answer to that span, tolerating
surrounding whitespace and punctuation only. The score is reported pooled and
per domain, alongside citation accuracy over the same questions, so a reader
can see whether the two agree.
"""

from core.chunk import Chunk, Provenance


def a_chunk(document_id, section, text="a span"):
	from corpus.manifest import load_manifest
	from pathlib import Path

	document = next(one for one in load_manifest().documents if one.id == document_id)
	return Chunk(
		text=text,
		score=0.9,
		provenance=Provenance(
			source=f"./{document.target(Path('documents'))}",
			section=section,
			page=4,
			node_id="node-1",
		),
	)


def test_normalize_tolerates_surrounding_whitespace_and_punctuation_only():
	from evaluation.exact_match import normalize_answer

	assert normalize_answer("  h = 8 parallel attention layers.  ") == "h = 8 parallel attention layers"
	assert normalize_answer('"h = 8 parallel attention layers"') == "h = 8 parallel attention layers"
	assert normalize_answer("  'warmup_steps = 4000!' ") == "warmup_steps = 4000"


def test_normalize_keeps_interior_and_case_intact():
	from evaluation.exact_match import normalize_answer

	assert normalize_answer("H = 8 parallel attention layers") != normalize_answer(
		"h = 8 parallel attention layers"
	)
	assert normalize_answer("h =  8 parallel attention layers") != normalize_answer(
		"h = 8 parallel attention layers"
	)
	assert normalize_answer("h = 8, parallel attention layers") != normalize_answer(
		"h = 8 parallel attention layers"
	)


def test_exact_match_scores_an_answer_against_its_gold_span():
	from evaluation.exact_match import exact_match

	assert exact_match("h = 8 parallel attention layers.", "h = 8 parallel attention layers") is True
	assert exact_match("h = 8 parallel attention layers", "warmup_steps = 4000") is False


def test_a_run_scores_each_extractive_question_by_exact_match(tmp_path, monkeypatch):
	import asyncio

	from core import registry
	from evaluation.run import run_evaluation
	from evaluation.set import load_evaluation_set
	from test_evaluation_run import RecordingLlm, found_everywhere, offline, retrieve_from  # noqa: F401

	evaluation_set = load_evaluation_set("papers")
	extractive_ids = {q.id for q in evaluation_set.questions if q.extractive}
	assert extractive_ids, "the committed set holds no extractive questions to score"

	by_question = {q.id: q for q in evaluation_set.questions}

	async def exact_generate(self, query, context):
		asked = next(one for one in evaluation_set.questions if one.question == query)
		if asked.extractive:
			return (asked.gold_answer or "") + "."
		return "an answer that names nothing exact"

	monkeypatch.setattr(registry.architecture("naive").pipeline, "retrieve", retrieve_from(found_everywhere))

	async def _generate(self, query, context):
		return await exact_generate(self, query, context)

	monkeypatch.setattr(registry.architecture("naive").pipeline, "generate", _generate)

	run = asyncio.run(run_evaluation("naive", "papers", results_dir=tmp_path / "results"))

	for result in run.results:
		question = by_question[result.question]
		if question.extractive:
			assert result.exact_match == 1.0, result.question
			assert result.gold_answer == question.gold_answer
		else:
			assert result.exact_match is None, result.question

	assert run.aggregates.exact_match == 1.0
	assert run.aggregates.exact_match_scored == len(extractive_ids)


def test_a_wrong_answer_scores_zero_and_a_refusal_scores_zero(tmp_path, monkeypatch):
	import asyncio

	from core import registry
	from evaluation.run import run_evaluation
	from evaluation.set import load_evaluation_set
	from test_evaluation_run import RecordingLlm, found_everywhere, offline, retrieve_from  # noqa: F401

	async def wrong_generate(self, query, context):
		return "something else entirely"

	monkeypatch.setattr(registry.architecture("naive").pipeline, "retrieve", retrieve_from(found_everywhere))
	monkeypatch.setattr(registry.architecture("naive").pipeline, "generate", wrong_generate)

	run = asyncio.run(run_evaluation("naive", "papers", results_dir=tmp_path / "results"))

	for result in run.results:
		if result.extractive:
			assert result.exact_match == 0.0

	assert run.aggregates.exact_match == 0.0


def test_exact_match_is_reported_alongside_citation_for_the_same_questions(tmp_path, monkeypatch):
	import asyncio

	from core import registry
	from evaluation.run import run_evaluation
	from evaluation.set import load_evaluation_set
	from test_evaluation_run import offline, retrieve_from  # noqa: F401

	evaluation_set = load_evaluation_set("papers")

	def chunks_for(question, top_k):
		from test_evaluation_run import found_everywhere

		chunks = found_everywhere(question, top_k)
		if question.extractive and question.gold_answer:
			chunks[0] = a_chunk(question.document, question.section, text=question.gold_answer)
		return chunks

	async def quoting_generate(self, query, context):
		asked = next(one for one in evaluation_set.questions if one.question == query)
		if asked.extractive:
			return asked.gold_answer or ""
		return "an answer that names nothing exact"

	monkeypatch.setattr(registry.architecture("naive").pipeline, "retrieve", retrieve_from(chunks_for))
	monkeypatch.setattr(registry.architecture("naive").pipeline, "generate", quoting_generate)

	run = asyncio.run(run_evaluation("naive", "papers", results_dir=tmp_path / "results"))

	assert run.aggregates.exact_match == 1.0
	assert run.aggregates.citation_accuracy_extractive == 1.0


def test_summary_reports_exact_match_pooled_and_per_domain(tmp_path, monkeypatch):
	import asyncio

	from core import registry
	from evaluation.run import run_evaluation
	from evaluation.set import load_evaluation_set
	from test_evaluation_run import offline  # noqa: F401

	sets = {domain: load_evaluation_set(domain) for domain in ("papers", "manuals")}

	def chunks_for(question, top_k):
		from test_evaluation_run import found_everywhere

		return found_everywhere(question, top_k)

	def asked_of_the_sets(query: str, top_k: int = 5):
		from core.chunk import Chunk as _Chunk

		for evaluation_set in sets.values():
			for question in evaluation_set.questions:
				if question.question == query:
					return chunks_for(question, top_k)[:top_k]
		raise AssertionError(f"the run asked a question no set holds: {query}")

	async def retrieve(self, query: str, top_k: int = 5):
		return asked_of_the_sets(query, top_k)

	async def exact_generate(self, query, context):
		for evaluation_set in sets.values():
			for asked in evaluation_set.questions:
				if asked.question == query:
					if asked.extractive:
						return (asked.gold_answer or "") + "."
					return "an answer that names nothing exact"
		raise AssertionError(f"unknown query: {query}")

	for architecture in ("naive", "hybrid"):
		for domain in ("papers", "manuals"):
			monkeypatch.setattr(registry.architecture(architecture).pipeline, "retrieve", retrieve)
			monkeypatch.setattr(registry.architecture(architecture).pipeline, "generate", exact_generate)
			asyncio.run(run_evaluation(architecture, domain, results_dir=tmp_path / "runs"))

	from evaluation.summary import load_runs, render_markdown, summarize

	summary = summarize(load_runs(tmp_path / "runs"))
	table = render_markdown(summary)

	assert "exact_match" in table or "exact" in table
	scopes = {(row.architecture, row.scope) for row in summary.rows}
	assert ("naive", "papers") in scopes
	assert ("naive", "manuals") in scopes
	assert ("naive", "pooled") in scopes

	pooled = next(row for row in summary.rows if row.architecture == "naive" and row.scope == "pooled")
	assert pooled.exact_match == 1.0
	assert pooled.citation_accuracy_extractive is not None
