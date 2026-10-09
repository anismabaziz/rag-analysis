"""Whether a retrieved chunk is the place a question's label points at.

Recall and reciprocal rank are the two numbers the first real run reports, and both are decided
by one question: does this chunk come from the document the label names. The section is recorded
alongside every result but deliberately does not gate a hit, because the evaluation set names a
document on its own as a fallback: a chunk from the right paper under the wrong heading is a
retrieval success, and only a loader that filed the whole paper under one heading should cost a
domain its score. A stricter question about the section can be asked of the same stored results
later without rerunning anything.
"""

import pytest

from core.chunk import Chunk, Provenance
from corpus.manifest import load_manifest
from evaluation.metrics import (
    locate,
    ndcg_at,
    recall,
    recall_at,
    reciprocal_rank,
    retrieved_from,
)
from evaluation.set import Location

DPR = "./documents/papers/dpr.pdf"
LOST = "./documents/papers/lost-in-the-middle.pdf"


def a_chunk(
    score=0.5, source=DPR, section="3 Dense Passage Retriever (DPR)", page=3, node="n1"
):
    return Chunk(
        text="a span",
        score=score,
        provenance=Provenance(source=source, section=section, page=page, node_id=node),
    )


def dpr():
    return Location(
        document="dense-passage-retrieval", section="3 Dense Passage Retriever (DPR)"
    )


def bge():
    return Location(document="bge-m3", section="4.3 Multilingual Long-Doc Retrieval")


def test_a_chunk_is_read_as_a_ranked_result_naming_its_own_place():
    """Provenance is the only handle a scorer has on a chunk, so it is resolved once, here."""
    results = retrieved_from(
        [a_chunk(score=0.9), a_chunk(score=0.4, node="n2")], load_manifest()
    )

    assert [
        (result.rank, result.document, result.section, result.node_id)
        for result in results
    ] == [
        (1, "dense-passage-retrieval", "3 Dense Passage Retriever (DPR)", "n1"),
        (2, "dense-passage-retrieval", "3 Dense Passage Retriever (DPR)", "n2"),
    ]


def test_a_chunk_from_a_document_the_manifest_does_not_hold_is_kept_and_reported_as_unknown():
    """A path the manifest cannot name is still a retrieved chunk, and dropping it would quietly
    change the depth the metrics were computed at."""
    results = retrieved_from([a_chunk(source="./documents/stray.pdf")], load_manifest())

    assert results[0].document is None
    assert results[0].section == "3 Dense Passage Retriever (DPR)"


def test_a_chunk_whose_manifest_document_is_known_by_a_path_the_reader_ran_from_is_matched():
    """Ingestion records whatever path it globbed, so the same document arrives under two names."""
    results = retrieved_from(
        [a_chunk(source="documents/papers/dpr.pdf")], load_manifest()
    )

    assert results[0].document == "dense-passage-retrieval"


def test_a_chunk_from_the_document_a_question_names_is_a_hit():
    results = retrieved_from([a_chunk()], load_manifest())

    assert locate(results, dpr()) == 1


def test_a_chunk_from_the_right_document_under_the_wrong_heading_is_still_a_hit():
    """The document label is the fallback, so one loader bug cannot zero a whole domain."""
    results = retrieved_from([a_chunk(section="Appendix C")], load_manifest())

    assert locate(results, dpr()) == 1


def test_a_chunk_from_another_document_is_not_a_hit():
    results = retrieved_from([a_chunk(source=LOST)], load_manifest())

    assert locate(results, dpr()) is None


def test_a_hit_is_credited_only_to_the_questions_that_named_that_document():
    results = retrieved_from([a_chunk()], load_manifest())

    assert locate(results, bge()) is None


def test_recall_is_the_share_of_the_places_a_question_names_that_were_retrieved():
    results = retrieved_from([a_chunk()], load_manifest())

    assert recall(results, [dpr(), bge()]) == pytest.approx(0.5)


def test_recall_of_a_question_with_nothing_to_find_is_zero():
    results = retrieved_from([a_chunk(source=LOST)], load_manifest())

    assert recall(results, [dpr()]) == 0.0


def test_the_first_hit_decides_the_reciprocal_rank():
    results = retrieved_from(
        [a_chunk(source=LOST, node="n0"), a_chunk(), a_chunk()], load_manifest()
    )

    assert reciprocal_rank(results, dpr()) == pytest.approx(1 / 2)


def test_a_question_nothing_retrieved_for_has_a_reciprocal_rank_of_zero():
    results = retrieved_from([a_chunk()], load_manifest())

    assert reciprocal_rank(results, bge()) == 0.0


def test_an_empty_retrieval_scores_as_a_miss_rather_than_raising():
    """A run over a collection holding nothing still has to produce a number for every question."""
    results = retrieved_from([], load_manifest())

    assert recall(results, [dpr()]) == 0.0
    assert reciprocal_rank(results, dpr()) == 0.0


def test_recall_at_one_counts_only_a_first_place_hit():
    """A hit buried below first place is a miss at depth one, not a near hit."""
    results = retrieved_from(
        [a_chunk(source=LOST, node="n0"), a_chunk()], load_manifest()
    )

    assert recall_at(results, [dpr()], 1) == 0.0
    assert recall_at(results, [dpr()], 3) == pytest.approx(1.0)


def test_recall_beyond_the_depth_is_not_counted():
    """Depths are read off one retrieval by truncation, so a fourth-place hit is invisible at three."""
    chunks = [a_chunk(source=LOST, node=f"n{i}") for i in range(4)] + [
        a_chunk(node="hit")
    ]
    results = retrieved_from(chunks, load_manifest())

    assert recall_at(results, [dpr()], 3) == 0.0
    assert recall_at(results, [dpr()], 5) == pytest.approx(1.0)


def test_recall_at_is_partial_over_a_multi_hop_questions_locations():
    """Finding one of two passages is half the recall at whatever depth found it."""
    results = retrieved_from([a_chunk()], load_manifest())

    assert recall_at(results, [dpr(), bge()], 5) == pytest.approx(0.5)
    assert recall_at(results, [dpr(), bge()], 1) == pytest.approx(0.5)


def test_recall_at_with_nothing_to_find_is_zero():
    results = retrieved_from([a_chunk(source=LOST)], load_manifest())

    assert recall_at(results, [], 5) == 0.0
    assert recall_at(results, [dpr()], 5) == 0.0


def test_ndcg_of_a_first_place_hit_is_one():
    results = retrieved_from([a_chunk()], load_manifest())

    assert ndcg_at(results, [dpr()], 5) == pytest.approx(1.0)


def test_ndcg_of_a_second_place_hit_is_discounted_by_rank():
    """Second place earns one over log-two of three, which is what separates it from first."""
    results = retrieved_from(
        [a_chunk(source=LOST, node="n0"), a_chunk()], load_manifest()
    )

    assert ndcg_at(results, [dpr()], 5) == pytest.approx(0.6309297535714575)


def test_ndcg_beyond_the_depth_is_zero():
    """A hit below the depth is a miss at that depth, however discounted it would have been."""
    chunks = [a_chunk(source=LOST, node=f"n{i}") for i in range(4)] + [
        a_chunk(node="hit")
    ]
    results = retrieved_from(chunks, load_manifest())

    assert ndcg_at(results, [dpr()], 3) == 0.0
    assert ndcg_at(results, [dpr()], 5) == pytest.approx(1.0 / 2.584962500721156)


def test_ndcg_of_a_miss_is_zero():
    results = retrieved_from([a_chunk(source=LOST)], load_manifest())

    assert ndcg_at(results, [dpr()], 5) == 0.0


def test_ndcg_of_two_passages_found_first_is_one():
    """Two locations in two documents, both at the top, is the ideal ranking."""
    bge_chunk = Chunk(
        text="a span",
        score=0.4,
        provenance=Provenance(
            source="./documents/papers/bge-m3.pdf",
            section="4.3 Multilingual Long-Doc Retrieval",
            page=5,
            node_id="n2",
        ),
    )
    results = retrieved_from([a_chunk(), bge_chunk], load_manifest())

    assert ndcg_at(results, [dpr(), bge()], 5) == pytest.approx(1.0)


def test_ndcg_counts_each_relevant_document_once():
    """Five chunks from one relevant document are one hit, not five: without that, returning
    the same document five times would score above the ideal ranking it is normalized by."""
    results = retrieved_from([a_chunk(node=f"n{i}") for i in range(5)], load_manifest())

    assert ndcg_at(results, [dpr()], 5) == pytest.approx(1.0)


def test_ndcg_with_nothing_to_find_is_zero():
    results = retrieved_from([a_chunk()], load_manifest())

    assert ndcg_at(results, [], 5) == 0.0
    assert ndcg_at([], [dpr()], 5) == 0.0
