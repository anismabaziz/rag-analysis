"""What one generated answer costs, counted without calling anything.

Token cost is the prompt the model was given plus the answer it gave back, both
counted as whitespace-separated words. That is an estimate, not the model's own
tokenizer, and it is one on purpose: the same prompt and answer always give the
same number, no network or model weights are needed to count it, and every
architecture is counted the same way, so a difference between two costs is a
difference in how much context retrieval gave the model and how much it wrote
back rather than in how it was counted.
"""


def estimate_tokens(text: str) -> int:
    """The tokens in `text`, counted as whitespace-separated words.

    Empty text costs nothing. The count is deterministic and needs nothing beyond
    the text itself, which is what keeps a run's cost reproducible from its file.
    """
    return len(text.split())


def rendered_prompt(prompt_template: str, context: str, query: str) -> str:
    """The prompt the model was given, with its context and question filled in.

    The template is the frozen one the run records, so the count is a reading of
    the configuration the run was measured at rather than a second copy of it.
    """
    return prompt_template.replace("{context}", context).replace("{query}", query)
