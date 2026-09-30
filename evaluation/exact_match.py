"""Whether a generated answer says exactly the span the source prints.

Extractive questions carry a short span copied from the source as their gold
answer. Each is scored by comparing the generated answer to that span, with
nothing but the edges forgiven: surrounding whitespace and surrounding
punctuation are stripped from both sides before comparing, and everything else
has to match. Case is kept, interior spacing is kept, and interior punctuation
is kept, so a paraphrase of the span does not count.

A question that is not extractive carries no span to match and scores nothing
rather than a zero, so it drops out of the mean instead of dragging it down.
An extractive question always scores, even when the system refused: a refusal
names no span and is a miss, not a question with nothing to compare.
"""

import string

# ASCII punctuation stripped from both ends before comparing, together with
# surrounding whitespace. Interior runs are never touched: only the edges are
# forgiven, which is what keeps a paraphrase from counting as a match.
_EDGE_PUNCTUATION = string.punctuation


def normalize_answer(text: str) -> str:
	"""Both edges stripped of whitespace and punctuation, and nothing else.

	The strip repeats until the text stops moving, because punctuation and
	whitespace interleave at the edges: a quoted answer ends in `.'␣␣`, and
	one strip of each leaves the other behind.
	"""
	current = text.strip()
	previous: str | None = None
	while current != previous:
		previous = current
		current = current.strip(_EDGE_PUNCTUATION).strip()

	return current


def exact_match(answer: str, gold: str) -> bool:
	"""Whether `answer` says exactly the span `gold` prints, edges forgiven.

	Both sides are normalized the same way, so a trailing period on a
	generated answer does not fail a span copied without one, while a
	difference in case or a difference inside the span does.
	"""
	return normalize_answer(answer) == normalize_answer(gold)


def exact_match_score(answer: str, gold: str | None, extractive: bool) -> float | None:
	"""One extractive question scored: one for a match, zero for a miss.

	`None` when the question is not extractive or carries no span, which is
	what a non-extractive question does, so the question drops out of the
	mean instead of reading as a miss. A refusal on an extractive question
	is a miss rather than unscored, because the span was there to name.
	"""
	if not extractive or gold is None:
		return None

	return 1.0 if exact_match(answer, gold) else 0.0
