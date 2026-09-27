"""The hand-written questions a run is scored against, and the labels they carry.

The committed set is the evidence the whole project rests on, so the tests here assert its
properties directly rather than trusting the reader that enforces them: the labels name documents
the corpus holds, the headings are strings the source actually prints, the counts and shares are
the ones recorded in the file, and the answers are spans of the source. Where the corpus is not on
disk, the checks that need it say they are being skipped rather than passing quietly.

The rejection cases are written the way an author would hit them, a stray document id or a
question added without a gold answer, because those are the mistakes that would otherwise make a
whole domain's numbers meaningless while the file still parses.
"""

import json
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from corpus.manifest import Manifest, load_manifest
from evaluation.set import (
	PROPORTION_TOLERANCE,
	Stratum,
	domains,
	load_evaluation_set,
)

# Where the corpus lands, the same root the fetch command installs it into. Only the checks that
# read the sources themselves need it.
CORPUS_ROOT = Path(__file__).parents[1] / "documents"

# The counts of the papers set as it is committed. A count is a recorded result, so it is asserted
# rather than recomputed: adding a question has to be a deliberate change to this test, not a side
# effect of appending to a file.
PAPERS_TOTAL = 24
PAPERS_BY_STRATUM = {
	Stratum.IDENTIFIER_HEAVY: 14,
	Stratum.PARAPHRASE: 10,
}


def write_set(root: Path, recorded: dict) -> None:
	"""A set file written to disk and read back, so a test goes through the real reader."""
	root.mkdir(parents=True, exist_ok=True)
	(root / "papers.json").write_text(json.dumps(recorded))


def committed_set() -> dict:
	"""The committed papers set as plain data, for a test that changes one thing about it."""
	path = Path(__file__).parents[1] / "evaluation" / "sets" / "papers.json"
	return json.loads(path.read_text())


def collapsed(text: str) -> str:
	"""Text as a comparison sees it, with the PDF's line breaks and soft hyphens flattened.

	Only whitespace is touched. Accents, ligatures, and the like are left alone, because a label
	that only matches after a looser comparison is one the scoring code would have to be told about.
	"""
	return re.sub(r"\s+", " ", text)


def sources_for(document_ids) -> dict[str, str]:
	"""The text of each named corpus document, or a skip when any of them is not on disk."""
	manifest = load_manifest()
	sources = {document_id: source_text(manifest, document_id) for document_id in document_ids}

	missing = sorted(document_id for document_id, text in sources.items() if text is None)
	if missing:
		pytest.skip(f"the corpus is not on disk for {', '.join(missing)}; fetch it to check the labels")

	return sources


def a_question(**overrides) -> dict:
	"""One valid question, so a test states only the field it is breaking."""
	question = {
		"id": "papers-example-01",
		"question": "How many heads does the base Transformer use?",
		"stratum": "identifier_heavy",
		"document": "attention-is-all-you-need",
		"section": "3.2.2 Multi-Head Attention",
		"extractive": True,
		"gold_answer": "h = 8 parallel attention layers",
	}
	question.update(overrides)
	return question


def source_text(manifest: Manifest, document_id: str) -> str | None:
	"""The pages of one corpus document as the PDF prints them, or nothing when it is not on disk.

	Read straight from the PDF rather than through the loader, because a label checked against the
	loader would be checked against the code the label is supposed to be independent of.
	"""
	document = next(entry for entry in manifest.documents if entry.id == document_id)
	path = document.target(CORPUS_ROOT)
	if not path.is_file():
		return None

	from pypdf import PdfReader

	reader = PdfReader(str(path))
	return "\n".join(page.extract_text() or "" for page in reader.pages)


def test_the_papers_set_holds_the_counts_it_was_built_to():
	evaluation_set = load_evaluation_set("papers")

	assert len(evaluation_set.questions) == PAPERS_TOTAL
	assert dict(evaluation_set.strata_counts()) == PAPERS_BY_STRATUM


def test_both_strata_the_claim_turns_on_are_present_and_declared():
	evaluation_set = load_evaluation_set("papers")

	answered = {stratum: len(evaluation_set.for_stratum(stratum)) for stratum in PAPERS_BY_STRATUM}

	assert all(count > 0 for count in answered.values())
	assert sum(entry.share for entry in evaluation_set.strata.values()) == pytest.approx(1.0)


def test_every_question_is_labeled_with_a_document_and_a_section_the_corpus_holds():
	evaluation_set = load_evaluation_set("papers")
	papers = {document.id for document in load_manifest().in_domain("papers")}

	for question in evaluation_set.questions:
		assert question.document in papers
		assert question.section.strip() == question.section
		assert question.section

	assert evaluation_set.documents() <= papers


def test_every_question_in_the_papers_set_is_about_a_different_place_in_the_corpus():
	"""A set that only ever asked about one document would be scored by document alone.

	The section label is the finer one, so the set has to spread across documents and across
	headings inside them, otherwise the finer label is decoration.
	"""
	evaluation_set = load_evaluation_set("papers")

	assert len(evaluation_set.documents()) >= 4
	assert len({question.section for question in evaluation_set.questions}) >= 12


def test_every_extractive_question_carries_a_gold_answer_and_nothing_else_does():
	evaluation_set = load_evaluation_set("papers")

	for question in evaluation_set.questions:
		if question.extractive:
			assert question.gold_answer
		else:
			assert question.gold_answer is None


def test_the_shares_of_a_set_are_the_ones_it_declares():
	evaluation_set = load_evaluation_set("papers")
	total = len(evaluation_set.questions)

	for stratum, entry in evaluation_set.strata.items():
		share = evaluation_set.strata_counts()[stratum] / total
		assert entry.count == evaluation_set.strata_counts()[stratum]
		assert abs(share - entry.share) <= PROPORTION_TOLERANCE


def test_every_section_label_is_a_heading_the_source_prints():
	"""A label the source does not print cannot be matched against anything a chunk carries.

	Read from the PDF directly, because a label checked through the loader would be checked against
	the code under test. Skipped when the corpus is not on disk, since a test that quietly passes
	without reading a source is worse than one that says it read nothing.
	"""
	sources = sources_for(load_evaluation_set("papers").documents())

	for question in load_evaluation_set("papers").questions:
		assert collapsed(question.section) in collapsed(sources[question.document])


def test_every_gold_answer_is_a_span_the_source_prints():
	"""Exact match can only mean something if the answer is a span the system can retrieve."""
	evaluation_set = load_evaluation_set("papers")
	extractives = [question for question in evaluation_set.questions if question.gold_answer]

	sources = sources_for({question.document for question in extractives})

	for question in extractives:
		assert collapsed(question.gold_answer) in collapsed(sources[question.document])


def test_the_answers_a_second_reader_checked_are_recorded_with_their_name():
	evaluation_set = load_evaluation_set("papers")
	answered = {question.id for question in evaluation_set.questions if question.gold_answer}

	assert evaluation_set.spot_check.by
	assert evaluation_set.spot_check.on
	assert set(evaluation_set.spot_check.answers) <= answered
	assert len(evaluation_set.spot_check.answers) >= 2


def test_papers_is_a_domain_a_set_exists_for():
	assert "papers" in domains()


def test_a_domain_with_no_set_says_so_rather_than_scoring_nothing(tmp_path):
	with pytest.raises(FileNotFoundError, match="no evaluation set for manuals"):
		load_evaluation_set("manuals", root=tmp_path)


def test_a_label_naming_a_document_the_corpus_does_not_hold_is_refused(tmp_path):
	recorded = committed_set()
	recorded["questions"][0]["document"] = "some-other-paper"
	write_set(tmp_path, recorded)

	with pytest.raises(ValueError, match="names documents the corpus does not hold: some-other-paper"):
		load_evaluation_set("papers", root=tmp_path)


def test_a_label_naming_a_document_from_the_other_domain_is_refused(tmp_path):
	recorded = committed_set()
	recorded["questions"][0]["document"] = "postgresql-16-documentation"
	write_set(tmp_path, recorded)

	with pytest.raises(ValueError, match="names documents the corpus does not hold"):
		load_evaluation_set("papers", root=tmp_path)


def test_a_question_with_no_section_is_refused(tmp_path):
	recorded = committed_set()
	recorded["questions"].append(a_question(id="papers-no-section-01", section="  "))
	write_set(tmp_path, recorded)

	with pytest.raises(ValidationError):
		load_evaluation_set("papers", root=tmp_path)


def test_an_extractive_question_with_no_gold_answer_is_refused(tmp_path):
	recorded = committed_set()
	recorded["questions"].append(a_question(id="papers-no-answer-01", gold_answer=None))
	write_set(tmp_path, recorded)

	with pytest.raises(ValidationError, match="extractive but carries no gold answer"):
		load_evaluation_set("papers", root=tmp_path)


def test_a_gold_answer_on_a_question_that_never_claimed_to_be_extractive_is_refused(tmp_path):
	recorded = committed_set()
	recorded["questions"].append(a_question(id="papers-unchecked-01", extractive=False))
	write_set(tmp_path, recorded)

	with pytest.raises(ValidationError, match="not extractive but carries a gold answer"):
		load_evaluation_set("papers", root=tmp_path)


def test_a_set_whose_questions_drifted_from_its_declared_shares_is_refused(tmp_path):
	recorded = committed_set()
	recorded["strata"]["identifier_heavy"]["share"] = 0.9
	recorded["strata"]["paraphrase"]["share"] = 0.1
	write_set(tmp_path, recorded)

	with pytest.raises(ValueError, match="against a declared"):
		load_evaluation_set("papers", root=tmp_path)


def test_a_set_whose_strata_do_not_add_up_is_refused(tmp_path):
	recorded = committed_set()
	recorded["strata"]["identifier_heavy"]["share"] = 0.5
	write_set(tmp_path, recorded)

	with pytest.raises(ValueError, match="do not add up to 1"):
		load_evaluation_set("papers", root=tmp_path)


def test_a_set_with_no_questions_is_refused(tmp_path):
	recorded = committed_set()
	recorded["questions"] = []
	write_set(tmp_path, recorded)

	with pytest.raises(ValueError, match="has no questions"):
		load_evaluation_set("papers", root=tmp_path)


def test_a_recorded_count_the_questions_disagree_with_is_refused(tmp_path):
	recorded = committed_set()
	recorded["strata"]["identifier_heavy"]["count"] = 15
	write_set(tmp_path, recorded)

	with pytest.raises(ValueError, match="records 15 identifier_heavy questions and holds 14"):
		load_evaluation_set("papers", root=tmp_path)


def test_a_spot_check_naming_a_question_the_set_does_not_hold_is_refused(tmp_path):
	recorded = committed_set()
	recorded["spot_check"]["answers"].append("papers-not-a-question")
	write_set(tmp_path, recorded)

	with pytest.raises(ValueError, match="spot check names questions that are not in the set"):
		load_evaluation_set("papers", root=tmp_path)


def test_a_set_nobody_read_a_second_time_is_refused(tmp_path):
	recorded = committed_set()
	del recorded["spot_check"]
	write_set(tmp_path, recorded)

	with pytest.raises(ValidationError):
		load_evaluation_set("papers", root=tmp_path)


def test_two_questions_sharing_an_id_are_refused(tmp_path):
	recorded = committed_set()
	recorded["questions"].append(a_question(id=recorded["questions"][0]["id"]))
	write_set(tmp_path, recorded)

	with pytest.raises(ValueError, match="reuses ids"):
		load_evaluation_set("papers", root=tmp_path)


def test_a_field_nobody_reads_is_refused_rather_than_ignored(tmp_path):
	recorded = committed_set()
	recorded["questions"].append(a_question(id="papers-extra-field-01", notes="checked by hand"))
	write_set(tmp_path, recorded)

	with pytest.raises(ValidationError):
		load_evaluation_set("papers", root=tmp_path)
