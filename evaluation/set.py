"""The questions a run is scored against, and the rules their labels have to satisfy.

An evaluation set is only worth measuring against if the labels are better than the thing being
measured. Every question here is written by hand and every label is a source document plus a
section heading read off the source by a person, because a label produced by the loader's own
section detection would move wherever the loader moves, and the score of every architecture would
then be a measure of the loader rather than of retrieval.

Two things follow from that, and both are enforced here rather than left to the author. Every
question names a document that the corpus manifest actually holds, so a label cannot point at a
paper the run never ingested. Every question also names a document on its own, so a chunk with the
wrong section is still counted as a hit, which is what keeps one loader bug from zeroing a domain.

The strata are declared before any question is written, in the set file, together with the share
and the count of questions each one should hold. The claim under test is that retrieval signals
separate on these two question types, so the split is the measurement and not a by-product of how
the questions were drafted. A file whose questions have drifted away from what it declares is
rejected.

The answers are spans copied out of the sources, and the file records who read a source a second
time to check them. A label nobody else has looked at is one person's reading of a paper, and
saying so is the difference between a spot-checked label and an assumed one.
"""

import json
from collections import Counter
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from corpus.manifest import Manifest, load_manifest

SETS_DIR = Path(__file__).with_name("sets")

# The set files are one per domain, named after it, which is the same name the manifest and the
# results table use, so a reader can go from a number to its questions without a lookup table.
SET_FILE_SUFFIX = ".json"

# How far a set's questions may sit from the shares it declares. A single question is worth about
# 4% of a 24-question domain, so this is roughly one question either way and no more.
PROPORTION_TOLERANCE = 0.05


class Stratum(StrEnum):
	"""What kind of question a question is, which is what the claim is about.

	A question that quotes a rare identifier from the source and one that asks about the same
	content in the asker's own words retrieve differently, and the project's claim is that the
	difference is not in the average. The strata are therefore written down before any question is,
	all four of them, including the two this set does not use yet: a set drafted without the boundary
	in mind would blur it, and a boundary that was never defined cannot be reported as measured.
	"""

	IDENTIFIER_HEAVY = "identifier_heavy"
	"""Names a rare string from the source: a model, a dataset, a function, a number, a symbol.

	Dense retrieval has to have seen the identifier in a similar context to place it, so these are
	the questions where lexical matching is expected to earn its place.
	"""

	PARAPHRASE = "paraphrase"
	"""Asks about content without reusing the source's own wording.

	The dense side is expected to do better here, and the boundary between the two strata is
	reported rather than averaged over.
	"""

	MULTI_HOP = "multi_hop"
	"""Needs more than one location combined before it can be answered."""

	UNANSWERABLE = "unanswerable"
	"""Has no correct answer anywhere in the domain, so answering it at all is the failure."""


class StratumShare(BaseModel):
	"""One stratum's slice of a set: what it means, how much of the set it holds, and how many.

	The count and the share are both written down rather than one being derived from the other,
	because a result table quotes a count and a design decision is a share, and a reader should be
	able to check each against the questions without counting them.
	"""

	model_config = ConfigDict(extra="forbid")

	definition: str
	share: float = Field(ge=0.0, le=1.0)
	count: int = Field(ge=0)


class SpotCheck(BaseModel):
	"""Who read the source a second time, and which answers they read it for.

	An answer copied out of a source by the same person who wrote the question is one guess with
	extra steps, so who checked it is part of what a label is worth. It is recorded here, in the set
	itself, because a note in a commit message is not something a reader of the results can find.
	"""

	model_config = ConfigDict(extra="forbid")

	by: str = Field(min_length=1)
	on: str = Field(min_length=1)
	answers: list[str] = Field(min_length=1)


class Question(BaseModel):
	"""One question with the ground truth a scorer needs to check an answer against.

	`document` is the document-level label and `section` is the finer one. Both are read off the
	source by a person, and the document label is always present, so a retrieved chunk is judged
	correct on the document when its section does not match rather than being called wrong.

	`gold_answer` is a short span copied from the source, present on every extractive question and
	on no other. Exact match is only meaningful where there is a span to match, and a question that
	cannot be answered with one says so rather than carrying an answer nobody checked.
	"""

	model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

	id: str = Field(min_length=1)
	question: str = Field(min_length=1)
	stratum: Stratum
	document: str = Field(min_length=1)
	section: str = Field(min_length=1)
	extractive: bool
	gold_answer: str | None = None

	@model_validator(mode="after")
	def check_answer_presence(self) -> "Question":
		"""Refuse a question whose answer is missing or invented.

		A gold answer on a non-extractive question is a claim that something short was checked
		against the source when it was not, which is the one mistake the exact-match score cannot
		be trusted to reveal later.
		"""
		if self.extractive and not self.gold_answer:
			raise ValueError(f"{self.id} is extractive but carries no gold answer")

		if not self.extractive and self.gold_answer:
			raise ValueError(f"{self.id} is not extractive but carries a gold answer")

		return self


class EvaluationSet(BaseModel):
	"""Every question one domain is scored on, with the strata it declares them under."""

	model_config = ConfigDict(extra="forbid")

	domain: str
	strata: dict[Stratum, StratumShare]
	spot_check: SpotCheck
	questions: list[Question]

	@model_validator(mode="after")
	def check_questions(self) -> "EvaluationSet":
		"""Refuse a set that is empty, or that asks the same thing twice under one name.

		Two questions sharing an id would be scored as one, quietly dropping a data point from
		whichever stratum it was filed under, which is the kind of loss a result table cannot show.
		"""
		if not self.questions:
			raise ValueError(f"the evaluation set for {self.domain} has no questions")

		names = [question.id for question in self.questions]
		duplicated = sorted({name for name in names if names.count(name) > 1})
		if duplicated:
			raise ValueError(f"the evaluation set for {self.domain} reuses ids: {', '.join(duplicated)}")

		unknown = sorted(set(self.spot_check.answers) - set(names))
		if unknown:
			raise ValueError(f"the spot check names questions that are not in the set: {', '.join(unknown)}")

		return self

	@model_validator(mode="after")
	def check_shares(self) -> "EvaluationSet":
		"""Refuse a set whose questions no longer match the shares and counts it declares.

		The shares are the whole point of stratifying: a set that drifted to three quarters one
		stratum would still produce a score, and the score would describe a different run than the
		one the file describes.
		"""
		if abs(sum(entry.share for entry in self.strata.values()) - 1.0) > 1e-9:
			raise ValueError(f"the declared strata of {self.domain} do not add up to 1")

		total = len(self.questions)
		counted = self.strata_counts()

		for stratum, declared in self.strata.items():
			share = counted[stratum] / total

			if counted[stratum] != declared.count:
				raise ValueError(
					f"{self.domain} records {declared.count} {stratum} questions "
					f"and holds {counted[stratum]}"
				)

			if abs(share - declared.share) > PROPORTION_TOLERANCE:
				raise ValueError(
					f"{self.domain} holds {counted[stratum]} of {total} {stratum} questions, "
					f"which is {share:.2f} against a declared {declared.share:.2f}"
				)

		return self

	def strata_counts(self) -> Counter[Stratum]:
		"""How many questions each stratum holds, which is the count a result table quotes."""
		return Counter(question.stratum for question in self.questions)

	def documents(self) -> set[str]:
		"""The corpus documents this set draws its questions from."""
		return {question.document for question in self.questions}

	def for_stratum(self, stratum: Stratum) -> list[Question]:
		"""The questions of one stratum, in the order the set files them."""
		return [question for question in self.questions if question.stratum == stratum]


def load_evaluation_set(domain: str, root: Path = SETS_DIR, manifest: Manifest | None = None) -> EvaluationSet:
	"""Read the questions of one domain and check them against its own rules and the corpus.

	Every label is checked against the manifest, so a set naming a document the corpus does not
	hold fails to load rather than scoring every question in it wrong. The manifest defaults to
	the committed one, because a set is only meaningful against the corpus it was written for.
	"""
	path = Path(root) / f"{domain}{SET_FILE_SUFFIX}"
	if not path.is_file():
		raise FileNotFoundError(f"no evaluation set for {domain} at {path}")

	evaluation_set = EvaluationSet.model_validate(json.loads(path.read_text()))

	if evaluation_set.domain != domain:
		raise ValueError(f"{path} holds the {evaluation_set.domain} set, not {domain}")

	if manifest is None:
		manifest = load_manifest()

	_check_documents(evaluation_set, manifest)

	return evaluation_set


def domains() -> tuple[str, ...]:
	"""Every domain a set exists for, in the order the files are named."""
	return tuple(sorted(path.stem for path in Path(SETS_DIR).glob(f"*{SET_FILE_SUFFIX}")))


def _check_documents(evaluation_set: EvaluationSet, manifest: Manifest) -> None:
	"""Refuse labels naming a document the domain's part of the corpus does not hold."""
	available = {document.id for document in manifest.in_domain(evaluation_set.domain)}

	unknown = sorted(evaluation_set.documents() - available)
	if unknown:
		raise ValueError(
			f"the evaluation set for {evaluation_set.domain} names documents the corpus "
			f"does not hold: {', '.join(unknown)}"
		)
