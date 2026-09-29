"""Which commit a run was made at, and whether the tree it was made in was clean.

A result file is worth what its commit is worth, and a commit is worth nothing for a run made on
top of uncommitted work: the number survives, the code that produced it does not. So a run records
both, and a reader who sees a dirty tree knows the run is not reproducible from the commit alone
and why.

Nothing here is a guess when git cannot answer. A checkout distributed as an archive has no
repository to ask, and a fabricated revision would be worse than a missing one, so the answer is
`None` and the run says so out loud.
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path

# How long to wait for git before concluding there is no repository to ask. A run is not worth
# hanging on, and a missing repository answers immediately on every platform that has git.
GIT_TIMEOUT_SECONDS = 5


@dataclass(frozen=True)
class Revision:
	"""Where a run was made: the commit, and whether the tree had changes the commit did not."""

	sha: str | None
	dirty: bool | None


def current_revision(root: Path = Path(".")) -> Revision:
	"""The commit the working tree is at, and whether it has changes that commit does not have."""
	sha = _git(root, "rev-parse", "HEAD")

	if sha is None:
		return Revision(sha=None, dirty=None)

	changed = _git(root, "status", "--porcelain")

	return Revision(sha=sha, dirty=bool(changed))


def _git(root: Path, *arguments: str) -> str | None:
	"""One git answer, trimmed, or nothing when git could not give one."""
	try:
		completed = subprocess.run(
			["git", *arguments],
			cwd=root,
			capture_output=True,
			text=True,
			timeout=GIT_TIMEOUT_SECONDS,
			check=True,
		)
	except (OSError, subprocess.SubprocessError):
		return None

	return completed.stdout.strip() or None
