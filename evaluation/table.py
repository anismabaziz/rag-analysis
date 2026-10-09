"""How a table cell is written, and how a row's numbers are averaged.

The retrieval table and the chunking table are the same shape of thing over the same
stored results, so the parts they share live here rather than being written twice. A
mean computed two ways is two numbers that can drift apart, and the one thing a
results table cannot afford is a row that disagrees with another about what it measured.
"""

from evaluation.metrics import mean
from evaluation.run import QuestionResult


def cell(value: float | None) -> str:
    """One metric as a table shows it, or a dash when the run never recorded it."""
    return f"{value:.4f}" if value is not None else "-"


def cell_int(value: int | None) -> str:
    """One count as a table shows it, or a dash when there is none."""
    return str(value) if value is not None else "-"


def question_means(scored: list[QuestionResult], attribute: str) -> dict[str, float]:
    """The mean at each recorded depth, over the scored questions that carry that depth.

    The depths are the ones the questions recorded, not a list kept here, so a table
    built from run files can never disagree with the files about which depths exist.
    """
    depths = sorted(
        {depth for result in scored for depth in (getattr(result, attribute) or {})},
        key=int,
    )

    return {
        depth: mean(
            getattr(result, attribute).get(depth)
            for result in scored
            if getattr(result, attribute)
        )
        for depth in depths
    }


def configuration_differences(first: dict, second: dict, prefix: str = "") -> list[str]:
    """The dotted paths where two recorded configurations disagree."""
    found = []

    for key in sorted(set(first) | set(second)):
        path = f"{prefix}{key}"
        left, right = first.get(key), second.get(key)
        if isinstance(left, dict) and isinstance(right, dict):
            found.extend(configuration_differences(left, right, f"{path}."))
        elif left != right:
            found.append(path)

    return found


def retrieved_words_mean(scored: list[QuestionResult]) -> float | None:
    """The mean words retrieval returned per scored question.

    Counted from the stored retrieved texts, so the token budget is a reading of what
    the run handed the generator rather than a nominal size. Empty retrieval counts as
    zero words, because nothing was handed over. Nothing scored means no measurement.
    """
    if not scored:
        return None

    return sum(
        sum(len(item.text.split()) for item in result.retrieved) for result in scored
    ) / len(scored)
