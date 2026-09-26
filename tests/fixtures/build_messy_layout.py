"""Writes the fixture document the loader tests read.

The document is a short paper whose layout contains the things that break text
extraction in practice: a running header on every page but the first, a draft
stamp and a running footer on all of them, a word split across two lines, a
figure caption, a four column table, numbered notes, and two margin stamps that
a layout model is liable to read as headings.

The corpus this project measures is downloaded and described by a manifest
rather than kept in the repository, because it runs to tens of megabytes. This
file is a few kilobytes, and committing it is what lets the tests read the same
bytes on every run. The tests read the generated PDF and never this file, so the
PDF is the one definition of the fixture. Run this only to change the fixture:

    uv run python tests/fixtures/build_messy_layout.py
"""

from pathlib import Path

FIXTURE = Path(__file__).parent / "messy_layout.pdf"

HEADER_Y, DRAFT_Y, FOOTER_Y = 760, 740, 60

RUNNING_HEADER = "Journal of Reproducible Retrieval Studies, Volume 4, Issue 2"
DRAFT_STAMP = "Draft, do not cite"
RUNNING_FOOTER = "Preprint. Under review. Do not redistribute."

BODY, TITLE, HEADING, CAPTION = 10, 18, 13, 9

TABLE_COLUMNS = 4
COLUMN_X = [172 + 100 * i for i in range(TABLE_COLUMNS)]


def line(size, y, text, x=72):
	return (size, x, y, text)


def running_matter(y, text):
	return line(BODY, y, text)


def table_row(y, first, *values):
	cells = [line(CAPTION, y, first)] + [line(CAPTION, y, value, x) for x, value in zip(COLUMN_X, values)]

	return cells


PAGE_ONE = [
	running_matter(DRAFT_Y, DRAFT_STAMP),
	line(TITLE, 710, "How Retrieval Quality Depends on Document Ingestion"),
	line(BODY, 686, "Abdelaziz Example and Rowan Sample"),
	line(HEADING, 662, "1 Introduction"),
	line(BODY, 638, "Retrieval quality is decided twice: once while the document is read and"),
	line(BODY, 624, "once while the index is searched. A chunk cut in the wrong place cannot be"),
	line(BODY, 610, "recovered by a better ranker, so the ingestion stage is part of the result."),
	line(BODY, 578, "A chunk is the unit of retrieval that the generator reads."),
	line(CAPTION, 550, "Figure 1: Mean recall at five for each architecture on domain one."),
	*table_row(200, "Architecture", "Dense", "Sparse", "Hybrid"),
	*table_row(188, "Dense", "0.61", "0.00", "0.68"),
	*table_row(176, "Sparse", "0.34", "0.72", "0.70"),
	*table_row(164, "Rerank", "0.55", "0.69", "0.74"),
	running_matter(FOOTER_Y, RUNNING_FOOTER),
]

PAGE_TWO = [
	running_matter(HEADER_Y, RUNNING_HEADER),
	running_matter(DRAFT_Y, DRAFT_STAMP),
	line(HEADING, 710, "2   Ingestion   and   Cleaning"),
	line(BODY, 682, "Text taken from a PDF arrives in pieces. A word split across two lines arrives"),
	line(BODY, 668, "as a hyphen at the end of one line and a fragment at the start of the next, and"),
	line(BODY, 654, "the word has to be put back together before the reader can judge it: the score is"),
	line(BODY, 640, "mea-"),
	line(BODY, 626, "sured by the annotators and not by the parser."),
	line(BODY, 598, "A chunk is the unit of retrieval that the generator reads."),
	line(HEADING, 570, "A b c D e f G h"),
	line(BODY, 542, "The stamp above is not a heading. A layout model sometimes reads the margin as"),
	line(BODY, 528, "one, and the text under it still belongs to the section above it."),
	running_matter(FOOTER_Y, RUNNING_FOOTER),
]

PAGE_THREE = [
	running_matter(HEADER_Y, RUNNING_HEADER),
	running_matter(DRAFT_Y, DRAFT_STAMP),
	line(HEADING, 710, "3   Results   and   Findings"),
	line(BODY, 682, "Hybrid retrieval came first on both domains, and the gap over dense retrieval"),
	line(BODY, 668, "was wider on the domain with the more unusual vocabulary. Reranking on top of"),
	line(BODY, 654, "hybrid retrieval added little, which is reported here rather than left out."),
	line(BODY, 626, "1 Correspondence to retrieval.study@example.org"),
	line(BODY, 614, "2 The loader is unchanged between the two runs reported here."),
	running_matter(FOOTER_Y, RUNNING_FOOTER),
]

PAGE_FOUR = [
	running_matter(HEADER_Y, RUNNING_HEADER),
	running_matter(DRAFT_Y, DRAFT_STAMP),
	line(BODY, 682, "The stamp on this page is the sideways identifier the typesetter ran into the"),
	line(BODY, 668, "margin. It is drawn as letters with a space between each of them, and it is not"),
	line(BODY, 654, "a section, so the text under it stays in the section above."),
	line(HEADING, 626, "g u A 2 ] L C . s c [ 7 v 2 6 7 3 0 . 6 0 7"),
	line(BODY, 598, "Reading order is what a layout model guesses at and what the rest of the"),
	line(BODY, 584, "project then depends on, so the guess is worth pinning down."),
	running_matter(FOOTER_Y, RUNNING_FOOTER),
]

PAGE_FIVE = [
	running_matter(HEADER_Y, RUNNING_HEADER),
	running_matter(DRAFT_Y, DRAFT_STAMP),
	line(CAPTION, 630, "arXiv:2401.00001v1 [cs.CL] 3 Feb 2024"),
	line(BODY, 682, "Nothing on this page starts a new section, which is the point of it: the last"),
	line(BODY, 668, "section of the document carries on to the end, and a page boundary in the"),
	line(BODY, 654, "middle of it is not a section boundary."),
	running_matter(FOOTER_Y, RUNNING_FOOTER),
]

PAGES = [PAGE_ONE, PAGE_TWO, PAGE_THREE, PAGE_FOUR, PAGE_FIVE]


def content_stream(lines):
	"""The page description operators that draw the given lines."""
	drawing = ["BT"]
	for size, x, y, text in lines:
		escaped = text.replace("\\", "\\\\").replace("(", r"\(").replace(")", r"\)")
		drawing.append(f"/F1 {size} Tf 1 0 0 1 {x} {y} Tm ({escaped}) Tj")
	drawing.append("ET")

	return "\n".join(drawing).encode("latin-1")


def build(path, pages):
	"""Write a PDF with one page per entry, each page drawing its own lines."""
	pages_id, font_id = 2, 3
	page_ids = [font_id + 1 + 2 * i for i in range(len(pages))]
	objects = [
		b"<< /Type /Catalog /Pages 2 0 R >>",
		(
			"<< /Type /Pages /Kids ["
			+ " ".join(f"{page_id} 0 R" for page_id in page_ids)
			+ f"] /Count {len(pages)} >>"
		).encode(),
		b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
	]

	for number, lines in enumerate(pages):
		contents_id = page_ids[number] + 1
		objects.append(
			f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 612 792] "
			f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
			f"/Contents {contents_id} 0 R >>".encode()
		)
		stream = content_stream(lines)
		objects.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")

	document = bytearray(b"%PDF-1.4\n")
	offsets = []
	for number, body in enumerate(objects, start=1):
		offsets.append(len(document))
		document += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"

	start_xref = len(document)
	document += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
	for offset in offsets:
		document += f"{offset:010d} 00000 n \n".encode()
	document += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{start_xref}\n%%EOF\n".encode()

	path.write_bytes(bytes(document))


if __name__ == "__main__":
	build(FIXTURE, PAGES)
	print(f"wrote {FIXTURE}")
