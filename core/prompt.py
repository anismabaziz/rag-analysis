"""The one prompt every architecture answers with, and its fingerprint.

The prompt lives here as a value rather than inside the answering method, for two reasons. It has
to be a value so a result file can record the exact text a run was measured with, rather than
pointing at code and asking a reader to find it; and it has to be one value so no architecture
can answer with its own wording, which would make a difference between two results impossible to
attribute to retrieval.

The fingerprint is the sha256 of that text. A run file carries the prompt and the fingerprint
together: the prompt is what a reader reads, and the fingerprint is what they compare when they
want to know whether two runs were answered the same way without reading it twice.
"""

import hashlib

from llama_index.core import PromptTemplate

ANSWER_PROMPT = PromptTemplate("""
You are a helpful assistant.

Use ONLY the context below to answer the question. If the context does not contain the
answer, say so instead of guessing.

Context:
{context}

Question:
{query}

Answer:
""")


def answer_prompt() -> str:
	"""The prompt as text, which is what a result file records."""
	return ANSWER_PROMPT.template


def prompt_fingerprint() -> str:
	"""The sha256 of the prompt, so two runs can be compared without reading it twice."""
	return "sha256:" + hashlib.sha256(answer_prompt().encode("utf-8")).hexdigest()
