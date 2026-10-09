"""What the loader hands the rest of the project: the text of a document, with the things a
PDF puts there by accident taken out, and every element filed under the section it was
read from.

Retrieval scores are computed from these elements, so the cleanup is pinned against a
checked-in document that carries the structures which break real parsing. Reading a PDF into
elements takes a layout model that is slow and downloads weights, so it is replaced by a
stand-in that reads the same document from the same file. The subject of these tests is what
the loader does to the elements it is given, so nothing here reaches a network.
"""

import importlib
import re
from pathlib import Path

import pytest
from pypdf import PdfReader
from unstructured.documents.elements import ElementMetadata, NarrativeText

from fixtures.build_messy_layout import (
    BODY,
    CAPTION,
    HEADING,
    RUNNING_FOOTER,
    RUNNING_HEADER,
    TABLE_COLUMNS,
    TITLE,
)

loader_module = importlib.import_module("data.loader")
PDFLoader = loader_module.PDFLoader

FIXTURE = str(Path(__file__).parent / "fixtures" / "messy_layout.pdf")

DRAFT_STAMP = "Draft, do not cite"
LETTER_STAMP = "A b c D e f G h"
ARXIV_STAMP = "g u A 2 ] L C . s c [ 7 v 2 6 7 3 0 . 6 0 7"
ARXIV_VERTICAL_STAMP = "arXiv:2401.00001v1 [cs.CL] 3 Feb 2024"

TOP_MARGIN, BOTTOM_MARGIN = 720, 70
TOUCHING_LINES = 18


class PlacedLine:
    """One run of text on a page, with the position and size it was drawn at."""

    def __init__(self, page, y, x, size, text):
        self.page = page
        self.y = y
        self.x = x
        self.size = size
        self.text = text


class Block:
    """The lines a reader would take as one paragraph, one row of a table, or one heading."""

    def __init__(self, rows):
        self.rows = rows
        self.top = rows[0][0]
        self.size = max(line.size for _, cells in rows for line in cells)
        self.columns = max(len(cells) for _, cells in rows)

    @property
    def text(self):
        return " ".join(" ".join(line.text for line in cells) for _, cells in self.rows)


def lines_of_page(page, number):
    """Every run of text on a page, in the order it was drawn."""
    lines = []

    def collect(text, current_matrix, text_matrix, font, size):
        if text.strip():
            lines.append(
                PlacedLine(number, text_matrix[5], text_matrix[4], size, text.strip())
            )

    page.extract_text(visitor_text=collect)

    return lines


def blocks_of_page(lines):
    """Group the runs of a page into blocks, a new block wherever the lines jump down."""
    rows = {}
    for line in lines:
        rows.setdefault(line.y, []).append(line)

    block = []
    for y in sorted(rows, reverse=True):
        if block and block[-1][0] - y > TOUCHING_LINES:
            yield Block(block)
            block = []
        block.append((y, sorted(rows[y], key=lambda line: line.x)))

    if block:
        yield Block(block)


def category_of(block):
    """The element type a layout model would give a block that looks like this one."""
    if block.top >= TOP_MARGIN:
        return "Header"
    if block.top <= BOTTOM_MARGIN:
        return "Footer"
    if block.size >= TITLE:
        return "Title"
    if block.size >= HEADING:
        return "Section-header"
    if block.columns >= TABLE_COLUMNS:
        return "Table"
    if block.text.startswith("Figure "):
        return "FigureCaption"
    if re.match(r"^\d+[ .]", block.text):
        return "Footnote"

    return "NarrativeText"


def table_html(block):
    """The first row of a table as headings and the rest as cells."""
    rows = []
    for position, (_, cells) in enumerate(block.rows):
        tag = "th" if position == 0 else "td"
        rows.append(
            "<tr>" + "".join(f"<{tag}>{line.text}</{tag}>" for line in cells) + "</tr>"
        )

    return "<table>" + "".join(rows) + "</table>"


def elements_of_page(number, blocks):
    """The elements a layout model would report for one page of the document."""
    for position, block in enumerate(blocks, start=1):
        element = NarrativeText(block.text)
        element.category = category_of(block)
        element.element_id = f"p{number}-{position}"
        element.metadata = ElementMetadata(page_number=number, filename=FIXTURE)
        if element.category == "Table":
            element.metadata.text_as_html = table_html(block)
        yield element


class LayoutModel:
    """Stands in for the layout model, reading the document it is pointed at."""

    def __call__(
        self,
        filename,
        strategy=None,
        infer_table_structure=None,
        extract_images_in_pdf=None,
    ):
        return [
            element
            for number, page in enumerate(PdfReader(filename).pages, start=1)
            for element in elements_of_page(
                number, blocks_of_page(lines_of_page(page, number))
            )
        ]


@pytest.fixture
def loader(monkeypatch):
    """A loader whose partition step reads the fixture document instead of running a model."""
    monkeypatch.setattr(loader_module, "partition_pdf", LayoutModel())

    return PDFLoader()


@pytest.fixture
def elements(loader):
    return loader.load(FIXTURE)


def of_type(elements, category):
    """Every element of one type, in the order the document was read."""
    return [element for element in elements if element["type"] == category]


def only_element(elements, category):
    """The one element of a type a document holds a single copy of."""
    matches = of_type(elements, category)

    assert len(matches) == 1, f"expected one {category}, found {len(matches)}"

    return matches[0]


def document_lines():
    """The runs of text the fixture document is drawn with, one list per page."""
    return [
        lines_of_page(page, number)
        for number, page in enumerate(PdfReader(FIXTURE).pages, start=1)
    ]


def test_the_fixture_document_carries_the_structures_that_break_parsing():
    pages = document_lines()

    assert [lines[0].text for lines in pages] == [DRAFT_STAMP] + [RUNNING_HEADER] * 4
    assert [lines[-1].text for lines in pages] == [RUNNING_FOOTER] * 5

    drawn = [line for lines in pages for line in lines]
    assert [line.text for line in drawn if line.text == DRAFT_STAMP] == [
        DRAFT_STAMP
    ] * 5
    assert any(line.text.endswith("mea-") for line in drawn)
    assert any(line.text.startswith("Figure 1:") for line in drawn)
    assert any(line.text == "Architecture" for line in drawn)
    assert any(line.text.startswith("1 Correspondence") for line in drawn)
    assert [line.text for line in drawn if line.text.startswith("arXiv:")] == [
        ARXIV_VERTICAL_STAMP
    ]
    assert [
        line.text for line in drawn if line.text in (LETTER_STAMP, ARXIV_STAMP)
    ] == [LETTER_STAMP, ARXIV_STAMP]


def test_reading_the_same_document_twice_gives_the_same_elements(loader):
    assert loader.load(FIXTURE) == loader.load(FIXTURE)


def test_a_missing_document_is_reported_before_anything_is_parsed(loader):
    with pytest.raises(FileNotFoundError):
        loader.load("./documents/not-here.pdf")


def test_running_matter_repeated_on_every_page_is_not_part_of_the_document(elements):
    texts = [element["text"] for element in elements]

    assert not [
        text for text in texts if RUNNING_HEADER in text or RUNNING_FOOTER in text
    ]
    assert DRAFT_STAMP not in texts
    assert ARXIV_VERTICAL_STAMP not in texts
    assert {element["metadata"]["page"] for element in elements} == {1, 2, 3, 4, 5}


def test_a_word_split_across_two_lines_comes_back_as_one_word(elements):
    paragraphs = [
        element
        for element in elements
        if element["text"].startswith("Text taken from a PDF")
    ]

    assert len(paragraphs) == 1
    assert paragraphs[0]["text"].endswith(
        "the score is measured by the annotators and not by the parser."
    )


def test_a_table_comes_back_as_a_table_element(elements):
    table = only_element(elements, "Table")

    assert table["text"].startswith("<table>")
    assert (
        "<th>Architecture</th><th>Dense</th><th>Sparse</th><th>Hybrid</th>"
        in table["text"]
    )
    assert "<td>0.74</td>" in table["text"]


def test_a_figure_caption_is_kept_as_its_own_element(elements):
    captions = of_type(elements, "FigureCaption")

    assert [caption["text"] for caption in captions] == [
        "Figure 1: Mean recall at five for each architecture on domain one."
    ]


def test_a_block_of_notes_is_kept_as_its_own_element(elements):
    notes = only_element(elements, "Footnote")

    assert notes["text"] == (
        "1 Correspondence to retrieval.study@example.org "
        "2 The loader is unchanged between the two runs reported here."
    )
    assert notes["metadata"]["page"] == 3


def test_every_element_is_filed_under_the_section_it_was_read_from(elements):
    by_page = {}
    for element in elements:
        by_page.setdefault(element["metadata"]["page"], []).append(
            element["current_section"]
        )

    assert by_page == {
        1: ["How Retrieval Quality Depends on Document Ingestion"] * 2
        + ["1 Introduction"] * 5,
        2: ["2 Ingestion and Cleaning"] * 4,
        3: ["3 Results and Findings"] * 3,
        4: ["3 Results and Findings"] * 2,
        5: ["3 Results and Findings"],
    }


def test_a_heading_draws_the_boundary_that_follows_it(elements):
    boundaries = [
        element["text"] for element in elements if element["is_section_boundary"]
    ]

    assert boundaries == [
        "How Retrieval Quality Depends on Document Ingestion",
        "1 Introduction",
        "2 Ingestion and Cleaning",
        "3 Results and Findings",
    ]


def test_a_section_name_is_normalized_however_wide_the_heading_was_drawn(elements):
    names = {element["current_section"] for element in elements} - {None}

    assert names == {
        "How Retrieval Quality Depends on Document Ingestion",
        "1 Introduction",
        "2 Ingestion and Cleaning",
        "3 Results and Findings",
    }
    assert all(name == " ".join(name.split()) for name in names)


def test_a_margin_stamp_read_as_a_heading_does_not_move_the_section(elements):
    texts = [element["text"] for element in elements]

    assert [stamp for stamp in (LETTER_STAMP, ARXIV_STAMP) if stamp in texts] == []
    assert [
        element["current_section"]
        for element in elements
        if element["metadata"]["page"] == 2
    ] == ["2 Ingestion and Cleaning"] * 4
    assert [
        element["current_section"]
        for element in elements
        if element["metadata"]["page"] == 4
    ] == ["3 Results and Findings"] * 2


def test_a_sentence_kept_from_two_pages_stays_kept_on_both(elements):
    kept = [
        element
        for element in elements
        if element["text"]
        == "A chunk is the unit of retrieval that the generator reads."
    ]

    assert [element["metadata"]["page"] for element in kept] == [1, 2]
    assert len({element["index"] for element in kept}) == 2


def test_every_element_is_numbered_from_the_start_and_remembers_its_page(elements):
    assert [element["index"] for element in elements] == list(range(len(elements)))
    assert [element["index"] for element in elements if element["page_changed"]] == [
        0,
        7,
        11,
        14,
        16,
    ]
    assert {element["metadata"]["source"] for element in elements} == {"pdf"}


def test_the_whole_document_comes_back_in_reading_order(elements):
    assert [(element["type"], element["text"]) for element in elements] == [
        ("Title", "How Retrieval Quality Depends on Document Ingestion"),
        ("NarrativeText", "Abdelaziz Example and Rowan Sample"),
        ("Section-header", "1 Introduction"),
        (
            "NarrativeText",
            "Retrieval quality is decided twice: once while the document is read and once while the index "
            "is searched. A chunk cut in the wrong place cannot be recovered by a better ranker, so the "
            "ingestion stage is part of the result.",
        ),
        ("NarrativeText", "A chunk is the unit of retrieval that the generator reads."),
        (
            "FigureCaption",
            "Figure 1: Mean recall at five for each architecture on domain one.",
        ),
        (
            "Table",
            "<table><tr><th>Architecture</th><th>Dense</th><th>Sparse</th><th>Hybrid</th></tr>"
            "<tr><td>Dense</td><td>0.61</td><td>0.00</td><td>0.68</td></tr>"
            "<tr><td>Sparse</td><td>0.34</td><td>0.72</td><td>0.70</td></tr>"
            "<tr><td>Rerank</td><td>0.55</td><td>0.69</td><td>0.74</td></tr></table>",
        ),
        ("Section-header", "2 Ingestion and Cleaning"),
        (
            "NarrativeText",
            "Text taken from a PDF arrives in pieces. A word split across two lines arrives as a hyphen at "
            "the end of one line and a fragment at the start of the next, and the word has to be put back "
            "together before the reader can judge it: the score is measured by the annotators and not by "
            "the parser.",
        ),
        ("NarrativeText", "A chunk is the unit of retrieval that the generator reads."),
        (
            "NarrativeText",
            "The stamp above is not a heading. A layout model sometimes reads the margin as one, and the "
            "text under it still belongs to the section above it.",
        ),
        ("Section-header", "3 Results and Findings"),
        (
            "NarrativeText",
            "Hybrid retrieval came first on both domains, and the gap over dense retrieval was wider on the "
            "domain with the more unusual vocabulary. Reranking on top of hybrid retrieval added little, "
            "which is reported here rather than left out.",
        ),
        (
            "Footnote",
            "1 Correspondence to retrieval.study@example.org 2 The loader is unchanged between the two "
            "runs reported here.",
        ),
        (
            "NarrativeText",
            "The stamp on this page is the sideways identifier the typesetter ran into the margin. It is "
            "drawn as letters with a space between each of them, and it is not a section, so the text under "
            "it stays in the section above.",
        ),
        (
            "NarrativeText",
            "Reading order is what a layout model guesses at and what the rest of the project then depends "
            "on, so the guess is worth pinning down.",
        ),
        (
            "NarrativeText",
            "Nothing on this page starts a new section, which is the point of it: the last section of the "
            "document carries on to the end, and a page boundary in the middle of it is not a section "
            "boundary.",
        ),
    ]
