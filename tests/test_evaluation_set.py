"""The hand-written questions a run is scored against, and the labels they carry.

The committed set is the evidence the whole project rests on, so the tests here assert its
properties directly rather than trusting the reader that enforces them: the labels name documents
the corpus holds, the headings are strings the source actually prints, the counts and shares are
the ones recorded in the file, and the answers are spans of the source. Where the corpus is not on
disk, the checks that need it say they are being skipped rather than passing quietly.

The rejection cases are written the way an author would hit them, a stray document id or a
question added without a gold answer, because those are the mistakes that would otherwise make a
whole domain's numbers meaningless while the file still parses. The two strata that cannot be
scored like the others are rejected for their own reasons: a multi-hop question that does not say
which locations to combine, and an unanswerable one that points at a place to look.
"""

import json
import re
from dataclasses import dataclass
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

# The counts of each set as it is committed. A count is a recorded result, so it is asserted rather
# than recomputed: adding a question has to be a deliberate change to this test, not a side effect
# of appending to a file. Both domains are held to the same split on purpose, which is what lets the
# results table compare them; the test below says so out loud.
SPLIT = {
    Stratum.IDENTIFIER_HEAVY: 14,
    Stratum.PARAPHRASE: 10,
    Stratum.MULTI_HOP: 4,
    Stratum.UNANSWERABLE: 4,
}


@dataclass(frozen=True)
class Expected:
    """What a committed set of one domain is recorded to hold."""

    domain: str
    total: int
    by_stratum: dict[Stratum, int]
    documents: int


DOMAINS = {
    domain: Expected(domain=domain, total=32, by_stratum=SPLIT, documents=documents)
    for domain, documents in {"papers": 6, "manuals": 2}.items()
}


def letters(text: str) -> str:
    """Text reduced to the letters and digits a reader would search for.

    An unanswerable question is only unanswerable if nothing in the corpus supports it, and the
    check that establishes that has to survive a PDF printing a name as `psgs_w100` in one place and
    `psgs w100` in another. Everything that is not alphanumeric is dropped on both sides, and case
    is ignored, so a probe cannot be absent merely because the typesetting differs.
    """
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def set_path(domain: str) -> Path:
    """Where the committed set of a domain lives, named after the domain itself."""
    return Path(__file__).parents[1] / "evaluation" / "sets" / f"{domain}.json"


def write_set(root: Path, recorded: dict, domain: str) -> None:
    """A set file written to disk and read back, so a test goes through the real reader."""
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{domain}.json").write_text(json.dumps(recorded))


def committed_set(domain: str) -> dict:
    """The committed set of a domain as plain data, for a test that changes one thing about it."""
    return json.loads(set_path(domain).read_text())


def collapsed(text: str) -> str:
    """Text as a comparison sees it, with the PDF's line breaks and soft hyphens flattened.

    Only whitespace is touched. Accents, ligatures, and the like are left alone, because a label
    that only matches after a looser comparison is one the scoring code would have to be told about.
    """
    return re.sub(r"\s+", " ", text)


def sources_for(document_ids) -> dict[str, str]:
    """The text of each named corpus document, or a skip when any of them is not on disk."""
    manifest = load_manifest()
    sources = {
        document_id: source_text(manifest, document_id) for document_id in document_ids
    }

    missing = sorted(
        document_id for document_id, text in sources.items() if text is None
    )
    if missing:
        pytest.skip(
            f"the corpus is not on disk for {', '.join(missing)}; fetch it to check the labels"
        )

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


@pytest.mark.parametrize("domain", sorted(DOMAINS))
def test_a_set_holds_the_counts_it_was_built_to(domain):
    evaluation_set = load_evaluation_set(domain)
    expected = DOMAINS[domain]

    assert len(evaluation_set.questions) == expected.total
    assert dict(evaluation_set.strata_counts()) == expected.by_stratum


@pytest.mark.parametrize("domain", sorted(DOMAINS))
def test_every_declared_stratum_is_present_and_the_shares_add_up(domain):
    evaluation_set = load_evaluation_set(domain)

    answered = {
        stratum: len(evaluation_set.for_stratum(stratum))
        for stratum in DOMAINS[domain].by_stratum
    }

    assert all(count > 0 for count in answered.values())
    assert sum(
        entry.share for entry in evaluation_set.strata.values()
    ) == pytest.approx(1.0)


@pytest.mark.parametrize("domain", sorted(DOMAINS))
def test_every_answerable_question_is_labeled_with_a_document_and_a_section_the_corpus_holds(
    domain,
):
    """An unanswerable question is the one question with nowhere to point, so it carries no label.

    Everything else names a document the domain holds and a heading inside it, and the heading is
    checked against the source itself further down.
    """
    evaluation_set = load_evaluation_set(domain)
    available = {document.id for document in load_manifest().in_domain(domain)}
    answerable = placed(evaluation_set.questions)

    for question in answerable:
        assert question.document in available
        assert question.section.strip() == question.section
        assert question.section

    assert evaluation_set.documents() <= available


@pytest.mark.parametrize("domain", sorted(DOMAINS))
def test_every_question_in_a_set_is_about_a_different_place_in_the_corpus(domain):
    """Two questions about the same place would let one passage answer both, and the domain would
    be scored on fewer locations than the question count makes it look.

    Every document the domain holds has to be asked about, and no two questions may share a
    section, so the score is spread as widely as the set says it is. Together with the check below
    on a multi-hop question's extra locations, that gives every (document, section) pair in the
    domain to exactly one question.
    """
    evaluation_set = load_evaluation_set(domain)
    held = {document.id for document in load_manifest().in_domain(domain)}
    answerable = placed(evaluation_set.questions)

    assert evaluation_set.documents() == held
    assert len({question.section for question in answerable}) == len(answerable)


@pytest.mark.parametrize("domain", sorted(DOMAINS))
def test_a_multi_hop_question_says_which_locations_have_to_be_combined(domain):
    """A question that needs two passages is only measurable if it says which two.

    The first location is the question's own place, so the document-level fallback a retrieved
    chunk is judged on still points somewhere real, and the rest are what has to be retrieved
    alongside it.
    """
    evaluation_set = load_evaluation_set(domain)
    multi_hop = evaluation_set.for_stratum(Stratum.MULTI_HOP)

    for question in multi_hop:
        assert question.locations is not None
        assert len(question.locations) >= 2
        assert question.locations[0].document == question.document
        assert question.locations[0].section == question.section
        assert len(
            {(location.document, location.section) for location in question.locations}
        ) == len(question.locations)


@pytest.mark.parametrize("domain", sorted(DOMAINS))
def test_no_extra_location_a_multi_hop_question_needs_is_another_question_s_own_place(
    domain,
):
    """A passage that answers one question outright and is part of another's answer is counted twice.

    Retrieval has to turn up every location a multi-hop question names, so a second location that
    is some other question's whole answer would let one hit stand in for two questions.
    """
    evaluation_set = load_evaluation_set(domain)
    extra = {
        (location.document, location.section)
        for question in evaluation_set.for_stratum(Stratum.MULTI_HOP)
        for location in question.locations[1:]
    }

    assert extra
    assert not extra & anchors(evaluation_set.questions)


@pytest.mark.parametrize("domain", sorted(DOMAINS))
def test_an_unanswerable_question_names_no_place_to_look_and_says_what_to_do_instead(
    domain,
):
    """Pointing at a section would make the question answerable and the stratum empty.

    What a correct system does is part of the label: a system that declines is right here and the
    only way to score that is to have said so before the run.
    """
    evaluation_set = load_evaluation_set(domain)
    unanswerable = evaluation_set.for_stratum(Stratum.UNANSWERABLE)

    for question in unanswerable:
        assert question.document is None
        assert question.section is None
        assert question.locations is None
        assert question.gold_answer is None
        assert question.extractive is False
        assert question.expected_behavior
        assert question.absent_probe


@pytest.mark.parametrize("domain", sorted(DOMAINS))
def test_no_unanswerable_question_is_supported_anywhere_in_its_domain(domain):
    """The stratum is worth nothing unless the corpus really holds no answer.

    Each question carries the string a correct answer would have to name, and the check is that
    the string is in no document of the domain. Read from the PDFs directly, and skipped, loudly,
    when the corpus is not on disk.
    """
    evaluation_set = load_evaluation_set(domain)
    unanswerable = evaluation_set.for_stratum(Stratum.UNANSWERABLE)
    manifest = load_manifest()
    sources = sources_for(document.id for document in manifest.in_domain(domain))

    # Reduced once per document rather than once per probe: the manuals run to several thousand
    # pages each, and re-reducing one of them for every question is minutes of nothing.
    corpus = {document_id: letters(text) for document_id, text in sources.items()}

    for question in unanswerable:
        probe = letters(question.absent_probe)
        for document_id, text in corpus.items():
            assert probe not in text, f"{question.id} is supported by {document_id}"


@pytest.mark.parametrize("domain", sorted(DOMAINS))
def test_every_extractive_question_carries_a_gold_answer_and_nothing_else_does(domain):
    evaluation_set = load_evaluation_set(domain)

    for question in evaluation_set.questions:
        if question.extractive:
            assert question.gold_answer
        else:
            assert question.gold_answer is None


@pytest.mark.parametrize("domain", sorted(DOMAINS))
def test_the_shares_of_a_set_are_the_ones_it_declares(domain):
    evaluation_set = load_evaluation_set(domain)
    total = len(evaluation_set.questions)

    for stratum, entry in evaluation_set.strata.items():
        share = evaluation_set.strata_counts()[stratum] / total
        assert entry.count == evaluation_set.strata_counts()[stratum]
        assert abs(share - entry.share) <= PROPORTION_TOLERANCE


def test_the_two_domains_are_split_the_same_way_so_the_contrast_is_readable():
    """The boundary condition is the difference between the domains, not a difference in how they
    were built. Two sets written to different shares would show a register effect and a sampling
    effect at the same time, and nothing in the results would separate the two.
    """
    papers = load_evaluation_set("papers")
    manuals = load_evaluation_set("manuals")

    assert {stratum: entry.share for stratum, entry in papers.strata.items()} == {
        stratum: entry.share for stratum, entry in manuals.strata.items()
    }


@pytest.mark.parametrize("domain", sorted(DOMAINS))
def test_every_section_label_is_a_heading_the_source_prints(domain):
    """A label the source does not print cannot be matched against anything a chunk carries.

    Read from the PDF directly, because a label checked through the loader would be checked against
    the code under test. Skipped when the corpus is not on disk, since a test that quietly passes
    without reading a source is worse than one that says it read nothing.
    """
    evaluation_set = load_evaluation_set(domain)
    sources = sources_for(evaluation_set.documents())

    for question in evaluation_set.questions:
        for location in question.all_locations():
            assert collapsed(location.section) in collapsed(sources[location.document])


@pytest.mark.parametrize("domain", sorted(DOMAINS))
def test_every_gold_answer_is_a_span_the_source_prints(domain):
    """Exact match can only mean something if the answer is a span the system can retrieve."""
    evaluation_set = load_evaluation_set(domain)
    extractives = [
        question for question in evaluation_set.questions if question.gold_answer
    ]

    sources = sources_for({question.document for question in extractives})

    for question in extractives:
        assert collapsed(question.gold_answer) in collapsed(sources[question.document])


@pytest.mark.parametrize("domain", sorted(DOMAINS))
def test_the_answers_a_second_reader_checked_are_recorded_with_their_name(domain):
    evaluation_set = load_evaluation_set(domain)
    answered = {
        question.id for question in evaluation_set.questions if question.gold_answer
    }

    assert evaluation_set.spot_check.by
    assert evaluation_set.spot_check.on
    assert set(evaluation_set.spot_check.answers) <= answered
    assert len(evaluation_set.spot_check.answers) >= 2


@pytest.mark.parametrize("domain", sorted(DOMAINS))
def test_a_domain_is_named_by_a_set_that_exists(domain):
    assert domain in domains()


def test_a_domain_with_no_set_says_so_rather_than_scoring_nothing(tmp_path):
    with pytest.raises(FileNotFoundError, match="no evaluation set for recipes"):
        load_evaluation_set("recipes", root=tmp_path)


def test_a_label_naming_a_document_the_corpus_does_not_hold_is_refused(tmp_path):
    recorded = committed_set("papers")
    recorded["questions"][0]["document"] = "some-other-paper"
    write_set(tmp_path, recorded, "papers")

    with pytest.raises(
        ValueError, match="names documents the corpus does not hold: some-other-paper"
    ):
        load_evaluation_set("papers", root=tmp_path)


def test_a_label_naming_a_document_from_the_other_domain_is_refused(tmp_path):
    recorded = committed_set("papers")
    recorded["questions"][0]["document"] = "postgresql-16-documentation"
    write_set(tmp_path, recorded, "papers")

    with pytest.raises(ValueError, match="names documents the corpus does not hold"):
        load_evaluation_set("papers", root=tmp_path)


def test_a_question_with_no_section_is_refused(tmp_path):
    recorded = committed_set("papers")
    recorded["questions"].append(a_question(id="papers-no-section-01", section="  "))
    write_set(tmp_path, recorded, "papers")

    with pytest.raises(ValidationError):
        load_evaluation_set("papers", root=tmp_path)


def test_an_extractive_question_with_no_gold_answer_is_refused(tmp_path):
    recorded = committed_set("papers")
    recorded["questions"].append(a_question(id="papers-no-answer-01", gold_answer=None))
    write_set(tmp_path, recorded, "papers")

    with pytest.raises(ValidationError, match="extractive but carries no gold answer"):
        load_evaluation_set("papers", root=tmp_path)


def test_a_multi_hop_question_that_does_not_say_which_locations_to_combine_is_refused(
    tmp_path,
):
    recorded = committed_set("papers")
    recorded["questions"].append(
        a_question(
            id="papers-one-place-01",
            stratum="multi_hop",
            extractive=False,
            gold_answer=None,
        )
    )
    write_set(tmp_path, recorded, "papers")

    with pytest.raises(ValidationError, match="multi_hop but names no locations"):
        load_evaluation_set("papers", root=tmp_path)


def test_a_multi_hop_question_naming_a_single_location_is_refused(tmp_path):
    recorded = committed_set("papers")
    recorded["questions"].append(
        a_question(
            id="papers-one-hop-01",
            stratum="multi_hop",
            extractive=False,
            gold_answer=None,
            locations=[
                {
                    "document": "attention-is-all-you-need",
                    "section": "3.1 Encoder and Decoder Stacks",
                }
            ],
        )
    )
    write_set(tmp_path, recorded, "papers")

    with pytest.raises(ValidationError, match="names one location"):
        load_evaluation_set("papers", root=tmp_path)


def test_a_multi_hop_question_whose_first_location_is_not_its_own_place_is_refused(
    tmp_path,
):
    recorded = committed_set("papers")
    recorded["questions"].append(
        a_question(
            id="papers-mismatched-01",
            stratum="multi_hop",
            extractive=False,
            gold_answer=None,
            locations=[
                {
                    "document": "attention-is-all-you-need",
                    "section": "3.1 Encoder and Decoder Stacks",
                },
                {
                    "document": "attention-is-all-you-need",
                    "section": "6.2 Model Variations",
                },
            ],
        )
    )
    write_set(tmp_path, recorded, "papers")

    with pytest.raises(
        ValidationError, match="does not start at its own document and section"
    ):
        load_evaluation_set("papers", root=tmp_path)


def question_named(recorded: dict, question_id: str) -> dict:
    """The one question a committed set files under an id, so a test can change a single field."""
    return next(
        question for question in recorded["questions"] if question["id"] == question_id
    )


def placed(questions) -> list:
    """The questions that name a place to look, which is every question but an unanswerable one."""
    return [question for question in questions if question.section]


def anchors(questions) -> set:
    """Where each placed question says its answer is, as (document, section) pairs."""
    return {(question.document, question.section) for question in placed(questions)}


def test_a_question_naming_a_location_the_corpus_does_not_hold_is_refused(tmp_path):
    recorded = committed_set("papers")
    question_named(recorded, "papers-multi-hop-01")["locations"][1]["document"] = (
        "some-other-paper"
    )
    write_set(tmp_path, recorded, "papers")

    with pytest.raises(
        ValueError, match="names documents the corpus does not hold: some-other-paper"
    ):
        load_evaluation_set("papers", root=tmp_path)


def an_unanswerable_question(**overrides) -> dict:
    """One question with no answer in the corpus, so a test states only the field it is breaking."""
    question = {
        "id": "papers-unanswerable-example-01",
        "question": "Which of the papers in this collection reports on retrieving photographs by caption?",
        "stratum": "unanswerable",
        "extractive": False,
        "expected_behavior": "decline, and name no document and no section",
        "absent_probe": "Flickr30k",
    }
    question.update(overrides)
    return question


def an_answerable_question_claiming_the_unanswerable_labels(**overrides) -> dict:
    """A question that can be answered, wearing the labels that say it cannot be."""
    return a_question(
        **{
            "expected_behavior": "decline, and name no document and no section",
            "absent_probe": "Flickr30k",
            **overrides,
        }
    )


def test_an_unanswerable_question_that_points_at_a_document_is_refused(tmp_path):
    recorded = committed_set("papers")
    recorded["questions"].append(
        an_unanswerable_question(document="attention-is-all-you-need")
    )
    write_set(tmp_path, recorded, "papers")

    with pytest.raises(ValidationError, match="unanswerable but names a document"):
        load_evaluation_set("papers", root=tmp_path)


def test_an_unanswerable_question_that_does_not_say_what_to_do_instead_is_refused(
    tmp_path,
):
    recorded = committed_set("papers")
    recorded["questions"].append(an_unanswerable_question(expected_behavior=None))
    write_set(tmp_path, recorded, "papers")

    with pytest.raises(
        ValidationError, match="says nothing about what a correct system should do"
    ):
        load_evaluation_set("papers", root=tmp_path)


def test_an_unanswerable_question_with_nothing_to_check_against_is_refused(tmp_path):
    recorded = committed_set("papers")
    recorded["questions"].append(an_unanswerable_question(absent_probe=None))
    write_set(tmp_path, recorded, "papers")

    with pytest.raises(ValidationError, match="no string to check the corpus against"):
        load_evaluation_set("papers", root=tmp_path)


def test_an_answerable_question_that_claims_to_be_unanswerable_is_refused(tmp_path):
    recorded = committed_set("papers")
    recorded["questions"].append(
        an_answerable_question_claiming_the_unanswerable_labels(
            id="papers-false-refusal-01"
        )
    )
    write_set(tmp_path, recorded, "papers")

    with pytest.raises(ValidationError, match="is not one"):
        load_evaluation_set("papers", root=tmp_path)


def test_a_gold_answer_on_a_question_that_never_claimed_to_be_extractive_is_refused(
    tmp_path,
):
    recorded = committed_set("papers")
    recorded["questions"].append(a_question(id="papers-unchecked-01", extractive=False))
    write_set(tmp_path, recorded, "papers")

    with pytest.raises(
        ValidationError, match="not extractive but carries a gold answer"
    ):
        load_evaluation_set("papers", root=tmp_path)


def test_a_set_whose_questions_drifted_from_its_declared_shares_is_refused(tmp_path):
    recorded = committed_set("papers")
    recorded["strata"]["identifier_heavy"]["share"] = 0.9
    recorded["strata"]["paraphrase"]["share"] = 0.1
    recorded["strata"]["multi_hop"]["share"] = 0.0
    recorded["strata"]["unanswerable"]["share"] = 0.0
    write_set(tmp_path, recorded, "papers")

    with pytest.raises(ValueError, match="against a declared"):
        load_evaluation_set("papers", root=tmp_path)


def test_a_set_whose_strata_do_not_add_up_is_refused(tmp_path):
    recorded = committed_set("papers")
    recorded["strata"]["identifier_heavy"]["share"] = 0.5
    write_set(tmp_path, recorded, "papers")

    with pytest.raises(ValueError, match="do not add up to 1"):
        load_evaluation_set("papers", root=tmp_path)


def test_a_set_with_no_questions_is_refused(tmp_path):
    recorded = committed_set("papers")
    recorded["questions"] = []
    write_set(tmp_path, recorded, "papers")

    with pytest.raises(ValueError, match="has no questions"):
        load_evaluation_set("papers", root=tmp_path)


def test_a_recorded_count_the_questions_disagree_with_is_refused(tmp_path):
    recorded = committed_set("papers")
    recorded["strata"]["identifier_heavy"]["count"] = 15
    write_set(tmp_path, recorded, "papers")

    with pytest.raises(
        ValueError, match="records 15 identifier_heavy questions and holds 14"
    ):
        load_evaluation_set("papers", root=tmp_path)


def test_a_spot_check_naming_a_question_the_set_does_not_hold_is_refused(tmp_path):
    recorded = committed_set("papers")
    recorded["spot_check"]["answers"].append("papers-not-a-question")
    write_set(tmp_path, recorded, "papers")

    with pytest.raises(
        ValueError, match="spot check names questions that are not in the set"
    ):
        load_evaluation_set("papers", root=tmp_path)


def test_a_set_nobody_read_a_second_time_is_refused(tmp_path):
    recorded = committed_set("papers")
    del recorded["spot_check"]
    write_set(tmp_path, recorded, "papers")

    with pytest.raises(ValidationError):
        load_evaluation_set("papers", root=tmp_path)


def test_two_questions_sharing_an_id_are_refused(tmp_path):
    recorded = committed_set("papers")
    recorded["questions"].append(a_question(id=recorded["questions"][0]["id"]))
    write_set(tmp_path, recorded, "papers")

    with pytest.raises(ValueError, match="reuses ids"):
        load_evaluation_set("papers", root=tmp_path)


def test_a_field_nobody_reads_is_refused_rather_than_ignored(tmp_path):
    recorded = committed_set("papers")
    recorded["questions"].append(
        a_question(id="papers-extra-field-01", notes="checked by hand")
    )
    write_set(tmp_path, recorded, "papers")

    with pytest.raises(ValidationError):
        load_evaluation_set("papers", root=tmp_path)
