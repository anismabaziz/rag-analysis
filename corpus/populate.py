"""Fetching the corpus, and proving afterwards that what is on disk is what the manifest says.

Downloading a document and trusting it is the one way this corpus could quietly change between
two runs, so nothing is installed before its bytes have been checked against the recorded
digest, and a mismatch leaves no file behind. A document already on disk is verified rather
than fetched again, which is what makes the fetch command cheap to re-run and what lets a
reader check the corpus without a second copy of it.
"""

import urllib.request
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Callable

from corpus.manifest import READ_BLOCK, Document, Manifest, sha256_of

TIMEOUT_SECONDS = 120

# Written next to the document and renamed only once the digest matches, so an interrupted or
# tampered fetch is never left where ingestion would read it.
PARTIAL_SUFFIX = ".part"


class Action(StrEnum):
    """What happened to one document of the corpus, as the output names it."""

    FETCHED = "fetched"
    ALREADY_PRESENT = "already present"
    REPLACED = "replaced"
    UNREACHABLE = "unreachable"
    REJECTED = "rejected"
    VERIFIED = "verified"
    MISSING = "missing"
    ALTERED = "altered"


@dataclass(frozen=True)
class Outcome:
    """What happened to one document of the corpus, in the terms a reader of the output needs."""

    document: Document
    path: Path
    action: Action
    ok: bool
    message: str = ""


def download_document(url: str, destination: Path) -> None:
    """Fetch one document to `destination`. Raising means the document was not retrieved.

    Written in blocks rather than read into memory, because a manual in this corpus is tens of
    megabytes and a reader may point this at a larger corpus tomorrow.
    """
    request = urllib.request.Request(
        url, headers={"User-Agent": "rag-analysis/0.1 corpus fetch"}
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        with open(destination, "wb") as handle:
            for block in iter(lambda: response.read(READ_BLOCK), b""):
                handle.write(block)


def populate(
    manifest: Manifest,
    root: Path,
    download: Callable[[str, Path], None] | None = None,
    domains: list[str] | None = None,
) -> list[Outcome]:
    """Install every document of the named domains under `root`, checking each digest on the way in.

    A document already on disk whose bytes match is left alone, one whose bytes do not match is
    fetched again, and a document that cannot be installed is reported rather than raising, so a
    single dead URL does not cost the reader the rest of the corpus.
    """
    if download is None:
        download = download_document

    return [
        _populate_document(document, Path(root), download)
        for document in manifest.in_domains(domains)
    ]


def verify_corpus(
    manifest: Manifest, root: Path, domains: list[str] | None = None
) -> list[Outcome]:
    """Check the documents of the named domains on disk against the manifest, fetching nothing."""
    return [
        _verify_document(document, Path(root))
        for document in manifest.in_domains(domains)
    ]


def digest_mismatch(document: Document, path: Path) -> str:
    """Why the bytes at `path` are not the document the manifest describes, or nothing if they are.

    One comparison serves both the fetch and the verify path, so a document is held to the same
    standard whichever command found it.
    """
    actual = sha256_of(path)
    if actual == document.sha256:
        return ""

    return f"digest mismatch: the manifest records {document.sha256}, {path} hashes to {actual}"


def _populate_document(
    document: Document, root: Path, download: Callable[[str, Path], None]
) -> Outcome:
    target = document.target(root)

    if target.is_file() and document.matches(target):
        return Outcome(document, target, Action.ALREADY_PRESENT, True)

    action = Action.REPLACED if target.exists() else Action.FETCHED
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + PARTIAL_SUFFIX)

    try:
        download(document.url, partial)
    except Exception as error:
        partial.unlink(missing_ok=True)
        return Outcome(
            document,
            target,
            Action.UNREACHABLE,
            False,
            f"{type(error).__name__}: {error}",
        )

    mismatch = digest_mismatch(document, partial)
    if mismatch:
        partial.unlink(missing_ok=True)
        return Outcome(document, target, Action.REJECTED, False, mismatch)

    partial.replace(target)
    return Outcome(document, target, action, True)


def _verify_document(document: Document, root: Path) -> Outcome:
    target = document.target(root)

    if not target.is_file():
        return Outcome(
            document,
            target,
            Action.MISSING,
            False,
            f"{target} is not on disk; run the fetch command",
        )

    mismatch = digest_mismatch(document, target)
    if mismatch:
        return Outcome(document, target, Action.ALTERED, False, mismatch)

    return Outcome(document, target, Action.VERIFIED, True)
