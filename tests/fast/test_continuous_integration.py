"""The gates that run on every push, read as configuration rather than trusted.

A workflow file is only a promise: nothing in the suite runs a lint error through it or pushes a
real credential into it, so a step that quietly stopped gating anything would leave the suite
green. These tests read the workflows and hold them to the three things they exist for. A gate
that is renamed, dropped, or made advisory has to change one of these tests on purpose, which is
what a person does when they mean it.
"""

import re
import shlex
from pathlib import Path

WORKFLOWS = Path(__file__).resolve().parent.parent.parent / ".github" / "workflows"
CI = WORKFLOWS / "ci.yml"
SECRET_SCAN = WORKFLOWS / "secret-scan.yml"

SHA = re.compile(r"@[0-9a-f]{40}$")
DIGEST = re.compile(r"@sha256:[0-9a-f]{64}$")


def text_of(path: Path) -> str:
    """A workflow as text. Read rather than parsed, so a test reads the file a person edits."""
    return path.read_text()


def step_running(path: Path, command: str) -> str:
    """The step that runs the command, so a test reads what surrounds it rather than trusting
    that the command is mentioned somewhere in the file."""
    for step in re.finditer(r"(?ms)^[ ]*- name:.*?(?=^[ ]*- name:|\Z)", text_of(path)):
        if command in step.group(0):
            return step.group(0)

    raise AssertionError(f"no step in {path.name} runs {command!r}")


def run_command(path: Path, command: str) -> list[str]:
    """The step's `run:` line as arguments, so a test can read what is passed to the command
    rather than matching the line as written."""
    line = next(
        line
        for line in step_running(path, command).splitlines()
        if line.strip().startswith("run:")
    )

    return shlex.split(line.split("run:", 1)[1])


def triggers(path: Path) -> set[str]:
    """The events a workflow runs on, in either the block form or the list form."""
    lines = text_of(path).splitlines()
    start = next(number for number, line in enumerate(lines) if line.startswith("on:"))
    events = set(re.findall(r"[a-z_]+", lines[start].split(":", 1)[1]))

    for line in lines[start + 1 :]:
        if not line.strip() or not line.startswith(" "):
            break
        events.add(line.strip().rstrip(":"))

    return events


def asks_only_to_read(path: Path) -> bool:
    """Whether the workflow's whole token is read access to the repository."""
    return (
        re.search(r"^permissions:\n[ ]+contents: read$", text_of(path), re.MULTILINE)
        is not None
    )


def test_both_gates_are_in_the_repository():
    """A test that reads a workflow nobody wrote is a test of nothing."""
    assert CI.is_file(), "the lint and test gate is missing"
    assert SECRET_SCAN.is_file(), "the secret scan is missing"


def test_a_push_runs_the_suite():
    """Otherwise nothing says a broken commit before a reviewer reads it."""
    assert "push" in triggers(CI)
    assert "pull_request" in triggers(CI)
    assert "push" in triggers(SECRET_SCAN)
    assert "pull_request" in triggers(SECRET_SCAN)


def test_a_push_runs_the_linter():
    """The lint gate is a step whose failure fails the job, which is what makes it a gate."""
    assert run_command(CI, "ruff check") == ["uv", "run", "ruff", "check", "."]


def test_a_push_runs_the_whole_suite():
    """Every suite directory has a step that runs it, so a new test file lands in a suite
    that runs and no step can quietly select a subset. Flags are fine; they name no subset."""
    commands = [
        run_command(CI, f"tests/{suite}") for suite in ("fast", "slow", "integration")
    ]

    for command in commands:
        assert command[:3] == ["uv", "run", "pytest"]

    named = {
        word
        for command in commands
        for word in command[3:]
        if word.startswith("tests/")
    }
    assert named == {"tests/fast", "tests/slow", "tests/integration"}

    for command in commands:
        assert [
            word
            for word in command[3:]
            if not word.startswith("-") and not word.startswith("tests/")
        ] == []


def test_a_failing_gate_fails_the_run():
    """`continue-on-error` on the step, or on the job that holds it, turns the gate into advice."""
    for path in (CI, SECRET_SCAN):
        assert "continue-on-error" not in text_of(path), (
            f"{path.name} makes a gate advisory"
        )


def test_dependencies_are_installed_from_the_lockfile_as_committed():
    """`uv sync` alone re-resolves and re-locks, which is how a build starts testing a version
    nobody reviewed."""
    assert "uv sync --frozen" in text_of(CI)


def test_every_action_and_image_is_pinned_to_one_revision():
    """A tag moves under the workflow, so what ran last month is not what runs today."""
    for path in sorted(WORKFLOWS.glob("*.yml")):
        for reference in re.findall(r"uses:\s*(\S+)", text_of(path)):
            assert SHA.search(reference), f"{path.name} uses {reference} unpinned"

    image = re.search(r"ghcr\.io/gitleaks/gitleaks:\S+", text_of(SECRET_SCAN))
    assert image, "the scan no longer names the image it runs"
    assert DIGEST.search(image.group(0)), (
        f"the scan image is pinned to a tag, not a digest: {image.group(0)}"
    )


def test_a_committed_credential_fails_a_push():
    """The scan reads the committed history, so a credential deleted in a later commit is still
    found, and it is run on pushes rather than only on pull requests."""
    scan = step_running(SECRET_SCAN, "gitleaks")

    assert "fetch-depth: 0" in text_of(SECRET_SCAN)
    assert "--redact" in scan


def test_no_gate_needs_a_service_or_a_credential():
    """A job that needs a running store or a real provider key is a job nobody can run on a fork,
    and one that stops running the day a key expires."""
    forbidden = (
        "services:",
        "secrets.",
        "GROQ_API_KEY",
        "QDRANT_URL",
        "NEO4J_PASSWORD",
        "localhost:",
        "127.0.0.1:",
    )

    for path in sorted(WORKFLOWS.glob("*.yml")):
        for reference in forbidden:
            assert reference not in text_of(path), f"{path.name} refers to {reference}"


def test_the_gates_ask_for_nothing_they_do_not_need():
    """Read-only is enough to check out, install, lint, and test."""
    assert asks_only_to_read(CI)
    assert asks_only_to_read(SECRET_SCAN)


def test_the_corpus_the_labels_are_checked_against_is_fetched_before_the_suite_runs():
    """The gold answers are checked against the documents they were copied from, and those checks
    skip when the corpus is absent. Fetching it turns a skip into a check. The papers are
    fetched: their digests are pinned to immutable arXiv PDFs, so a build never fails
    because upstream regenerated a file."""
    workflow = text_of(CI)
    fetched_at = workflow.find("fetch-corpus")
    tested_at = workflow.find("uv run pytest")

    assert run_command(CI, "fetch-corpus") == [
        "uv",
        "run",
        "rag-analysis",
        "fetch-corpus",
        "--domain",
        "papers",
    ]
    assert -1 not in (fetched_at, tested_at), (
        "the corpus is never fetched, or the suite never runs"
    )
    assert fetched_at < tested_at
