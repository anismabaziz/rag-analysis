"""Whether each claim in an answer comes from the chunks it was answered with.

A claim counts as supported only when one retrieved chunk contains a supporting span
for the whole claim, not when the words are scattered across the pool. The check is
plain string matching over normalized text and uses no judge model, so the same answer
and the same chunks always give the same number.

Normalization drops case and every non-alphanumeric character before comparing, so a
claim is not marked unsupported merely because the source set it in small caps or with
different punctuation. Common function words are dropped from the token comparison, so
a claim is not marked supported merely because it shares a "the" and an "is" with a
chunk: what has to overlap is the content. A refusal produces no claims rather than a
zero, so a system that declines when retrieval is empty is not scored as a miss.
"""

import re

from core.chunk import REFUSAL, Chunk

# Sentence boundaries: a run of . ! ? followed by whitespace. Kept deterministic on
# purpose: no model, no punkt download, the same answer always splits the same way.
_SENTENCE_END = re.compile(r"(?<=[.!?])[ \t\n\r]+")

# Non-alphanumeric runs become one space on both sides before comparing, so casing and
# punctuation never decide support.
_NON_ALNUM = re.compile(r"[^a-z0-9]+")

# Function words that carry no factual load. A claim sharing only these with a chunk
# is not supported by it.
STOPWORDS = frozenset(
	{
		"a",
		"an",
		"and",
		"are",
		"as",
		"at",
		"be",
		"been",
		"by",
		"can",
		"did",
		"do",
		"does",
		"for",
		"from",
		"how",
		"in",
		"is",
		"it",
		"its",
		"of",
		"on",
		"or",
		"that",
		"the",
		"these",
		"this",
		"those",
		"to",
		"was",
		"were",
		"what",
		"when",
		"where",
		"which",
		"who",
		"why",
		"with",
	}
)


def normalize(text: str) -> str:
	"""Lowercase with every non-alphanumeric run as one space, collapsed and stripped."""
	return _NON_ALNUM.sub(" ", text.lower()).strip()


def content_tokens(text: str) -> list[str]:
	"""The normalized words of `text` that carry factual load."""
	return [token for token in normalize(text).split() if token not in STOPWORDS]


def is_refusal(answer: str) -> bool:
	"""Whether `answer` is the refusal rather than a system output to score."""
	return normalize(answer) == normalize(REFUSAL)


def split_claims(answer: str) -> list[str]:
	"""The sentences of an answer, each one claim to check.

	A refusal or an empty answer produces no claims rather than one empty claim, so a
	system that declines is not scored as a miss.
	"""
	if not answer or not answer.strip() or is_refusal(answer):
		return []

	return [part.strip() for part in _SENTENCE_END.split(answer.strip()) if part.strip()]


def claim_supported(claim: str, chunks: list[Chunk]) -> bool:
	"""Whether one retrieved chunk contains a supporting span for the whole claim.

	Support needs a single chunk holding every content token of the claim. Words
	scattered across the pool do not count: a claim pieced together from two passages
	is not supported by either. A claim with no content tokens falls back to a
	verbatim check, so hedging without facts is not scored as supported.
	"""
	needed = content_tokens(claim)
	if not needed:
		return any(normalize(claim) in normalize(chunk.text) for chunk in chunks if chunk.text)

	required = set(needed)
	for chunk in chunks:
		if required.issubset(set(normalize(chunk.text).split())):
			return True

	return False


def citation_accuracy(answer: str, chunks: list[Chunk]) -> float | None:
	"""The share of the answer's claims one retrieved chunk each supports.

	`None` when the answer makes no claims, which is what a refusal does, so the
	question drops out of the mean instead of dragging it to zero.
	"""
	claims = split_claims(answer)
	if not claims:
		return None

	return sum(1 for claim in claims if claim_supported(claim, chunks)) / len(claims)
