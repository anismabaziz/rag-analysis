"""The corpus is described by a committed manifest rather than by committed PDFs, so the tests
here pin the manifest itself and the two behaviours that make it trustworthy: a document is
only installed once its bytes match the digest recorded for it, and a document already on disk
is checked against that digest instead of downloaded again.

Nothing here reaches a network or a real corpus. Downloads are handed a stand-in that serves
the bytes a document is supposed to have, including one served the wrong bytes, so a digest
mismatch is something these tests cause on purpose.
"""

import asyncio
import hashlib
import importlib
import json
import subprocess
from pathlib import Path

import pytest

from corpus.manifest import MANIFEST_PATH, corpus_identifier, load_manifest
from corpus.populate import Action, populate, verify_corpus

PAPER_IDS = {
    "attention-is-all-you-need",
    "rag-for-knowledge-intensive-nlp-tasks",
    "dense-passage-retrieval",
    "lexically-accelerated-dense-retrieval",
    "lost-in-the-middle",
    "bge-m3",
}

class FakeDownloader:
    """Serves the bytes a document should have, and records the URLs it was asked for."""

    def __init__(self, payloads, corrupt=()):
        self._payloads = payloads
        self._corrupt = set(corrupt)
        self.requested = []

    def __call__(self, url, destination):
        self.requested.append(url)
        payload = self._payloads[url]
        if url in self._corrupt:
            payload += b" tampered with in flight"
        Path(destination).write_bytes(payload)


def unreachable(url, destination):
    """Stands in for a fetch, so a test that reaches the network fails loudly."""
    raise AssertionError("the test reached the network")


def manifest_of(tmp_path, entries):
    """A manifest written to disk and read back, so every test goes through the real reader."""
    path = tmp_path / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"documents": entries}))
    return load_manifest(path)


def entry(
    document_id, domain="papers", url=None, payload=b"document bytes", filename=None
):
    """One manifest entry whose digest matches the bytes it is paired with in a test."""
    return {
        "id": document_id,
        "domain": domain,
        "title": f"Title of {document_id}",
        "url": url or f"https://example.org/{document_id}.pdf",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "filename": filename or f"{document_id}.pdf",
        "pages": 1,
        "bytes": len(payload),
    }


def run_cli(monkeypatch, root, *argv):
    """Runs the CLI with the corpus installed under `root` rather than in the working tree."""
    module = importlib.import_module("main")
    monkeypatch.setattr(module, "CORPUS_DIR", root)
    monkeypatch.setattr("sys.argv", ["main.py", *argv])
    asyncio.run(module.main())


def test_the_manifest_lists_every_document_of_the_corpus():
    manifest = load_manifest()

    assert set(manifest.domains) == {"papers"}
    assert {document.id for document in manifest.in_domain("papers")} == PAPER_IDS


def test_every_manifest_document_carries_a_https_url_and_a_digest():
    manifest = load_manifest()

    for document in manifest.documents:
        assert document.url.startswith("https://"), document.id
        assert document.title.strip(), document.id
        assert len(document.sha256) == 64, document.id
        assert set(document.sha256) <= set("0123456789abcdef"), document.id
        assert document.filename.endswith(".pdf"), document.id
        assert document.pages > 0, document.id
        assert document.bytes > 0, document.id


def test_every_manifest_document_is_distinct():
    """Two entries landing on one file would make the digest check ambiguous."""
    manifest = load_manifest()

    ids = [document.id for document in manifest.documents]
    targets = [
        str(document.target(Path("documents"))) for document in manifest.documents
    ]

    assert len(set(ids)) == len(ids)
    assert len(set(targets)) == len(targets)


def test_no_corpus_pdf_is_committed():
    """The manifest is committed; the documents it points at are fetched, not committed."""
    tracked = subprocess.run(
        ["git", "ls-files", "documents"],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).parent.parent.parent,
    )

    assert [line for line in tracked.stdout.splitlines() if line.endswith(".pdf")] == []


def test_the_same_corpus_always_gets_the_same_identifier():
    assert corpus_identifier(load_manifest()) == corpus_identifier(load_manifest())


def test_the_identifier_moves_when_a_document_s_bytes_do(tmp_path):
    """The corpus changes without the commit changing, so the identifier is what a result cites."""
    entries = [entry("a-paper", payload=b"paper bytes")]
    recording = manifest_of(tmp_path, entries)

    entries[0]["sha256"] = hashlib.sha256(b"a different paper").hexdigest()
    corrected = manifest_of(tmp_path, entries)

    assert corpus_identifier(recording) != corpus_identifier(corrected)


def test_the_identifier_does_not_depend_on_where_the_corpus_was_populated(tmp_path):
    """A reader who put the corpus somewhere else has the same corpus and must get the same name."""
    entries = [entry("a-paper"), entry("a-manual", domain="manuals")]

    assert corpus_identifier(manifest_of(tmp_path, entries)) == corpus_identifier(
        manifest_of(tmp_path / "elsewhere", entries)
    )


def test_a_path_is_matched_to_a_document_by_its_file_name():
    """A retrieved chunk records the path ingestion globbed, which depends on where the run started."""
    manifest = load_manifest()

    for source in (
        "./documents/papers/dpr.pdf",
        "documents/papers/dpr.pdf",
        "/somewhere/else/documents/papers/dpr.pdf",
    ):
        assert manifest.document_named_by_file(source) == "dense-passage-retrieval"


def test_a_path_the_corpus_does_not_hold_names_no_document():
    assert (
        load_manifest().document_named_by_file("./documents/papers/stray.pdf") is None
    )


def test_populating_the_corpus_installs_every_document_where_ingestion_looks_for_it(
    tmp_path,
):
    entries = [
        entry("a-paper", payload=b"paper bytes"),
        entry("a-manual", domain="manuals", payload=b"manual bytes"),
    ]
    manifest = manifest_of(tmp_path, entries)
    downloader = FakeDownloader(
        {
            "https://example.org/a-paper.pdf": b"paper bytes",
            "https://example.org/a-manual.pdf": b"manual bytes",
        }
    )

    results = populate(manifest, tmp_path, download=downloader)

    assert [result.ok for result in results] == [True, True]
    assert (tmp_path / "papers" / "a-paper.pdf").read_bytes() == b"paper bytes"
    assert (tmp_path / "manuals" / "a-manual.pdf").read_bytes() == b"manual bytes"
    assert downloader.requested == [item["url"] for item in entries]


def test_a_document_whose_bytes_do_not_match_its_digest_is_not_installed(tmp_path):
    manifest = manifest_of(tmp_path, [entry("a-paper", payload=b"paper bytes")])
    downloader = FakeDownloader(
        {"https://example.org/a-paper.pdf": b"paper bytes"},
        corrupt={"https://example.org/a-paper.pdf"},
    )

    results = populate(manifest, tmp_path, download=downloader)

    assert [result.ok for result in results] == [False]
    assert hashlib.sha256(b"paper bytes").hexdigest() in results[0].message
    assert not (tmp_path / "papers" / "a-paper.pdf").exists()
    assert list((tmp_path / "papers").iterdir()) == []


def test_a_document_already_on_disk_is_verified_instead_of_downloaded(tmp_path):
    manifest = manifest_of(tmp_path, [entry("a-paper", payload=b"paper bytes")])
    (tmp_path / "papers").mkdir(parents=True)
    (tmp_path / "papers" / "a-paper.pdf").write_bytes(b"paper bytes")

    results = populate(manifest, tmp_path, download=FakeDownloader({}))

    assert [(result.action, result.ok) for result in results] == [
        (Action.ALREADY_PRESENT, True)
    ]


def test_a_document_on_disk_that_no_longer_matches_its_digest_is_replaced(tmp_path):
    manifest = manifest_of(tmp_path, [entry("a-paper", payload=b"paper bytes")])
    (tmp_path / "papers").mkdir(parents=True)
    (tmp_path / "papers" / "a-paper.pdf").write_bytes(b"a truncated download")

    results = populate(
        manifest,
        tmp_path,
        download=FakeDownloader({"https://example.org/a-paper.pdf": b"paper bytes"}),
    )

    assert [(result.action, result.ok) for result in results] == [
        (Action.REPLACED, True)
    ]
    assert (tmp_path / "papers" / "a-paper.pdf").read_bytes() == b"paper bytes"


def test_a_document_that_cannot_be_fetched_is_reported_and_the_rest_still_installed(
    tmp_path,
):
    entries = [
        entry("a-paper", payload=b"paper bytes"),
        entry("b-paper", payload=b"second paper"),
    ]
    manifest = manifest_of(tmp_path, entries)

    def download(url, destination):
        if url.endswith("a-paper.pdf"):
            raise OSError("the host did not answer")
        Path(destination).write_bytes(b"second paper")

    results = populate(manifest, tmp_path, download=download)

    assert [(result.document.id, result.ok) for result in results] == [
        ("a-paper", False),
        ("b-paper", True),
    ]
    assert "the host did not answer" in results[0].message
    assert (tmp_path / "papers" / "b-paper.pdf").exists()


def test_populating_one_domain_leaves_the_other_alone(tmp_path):
    entries = [
        entry("a-paper", payload=b"paper bytes"),
        entry("a-manual", domain="manuals", payload=b"manual bytes"),
    ]
    manifest = manifest_of(tmp_path, entries)
    downloader = FakeDownloader(
        {
            "https://example.org/a-paper.pdf": b"paper bytes",
            "https://example.org/a-manual.pdf": b"manual bytes",
        }
    )

    results = populate(manifest, tmp_path, download=downloader, domains=["papers"])

    assert [result.document.id for result in results] == ["a-paper"]
    assert not (tmp_path / "manuals").exists()


def test_verifying_the_corpus_checks_the_bytes_on_disk_without_downloading_anything(
    tmp_path, monkeypatch
):
    manifest = manifest_of(tmp_path, [entry("a-paper", payload=b"paper bytes")])
    (tmp_path / "papers").mkdir(parents=True)
    (tmp_path / "papers" / "a-paper.pdf").write_bytes(b"paper bytes")

    monkeypatch.setattr("corpus.populate.download_document", unreachable)

    assert [
        (result.action, result.ok) for result in verify_corpus(manifest, tmp_path)
    ] == [(Action.VERIFIED, True)]


def test_verifying_names_the_document_whose_bytes_were_altered(tmp_path):
    manifest = manifest_of(tmp_path, [entry("a-paper", payload=b"paper bytes")])
    (tmp_path / "papers").mkdir(parents=True)
    (tmp_path / "papers" / "a-paper.pdf").write_bytes(b"something else entirely")

    results = verify_corpus(manifest, tmp_path)

    assert [result.ok for result in results] == [False]
    assert "a-paper" in results[0].message
    assert hashlib.sha256(b"paper bytes").hexdigest() in results[0].message


def test_verifying_reports_a_document_that_was_never_fetched(tmp_path):
    manifest = manifest_of(tmp_path, [entry("a-paper", payload=b"paper bytes")])

    results = verify_corpus(manifest, tmp_path)

    assert [(result.action, result.ok) for result in results] == [
        (Action.MISSING, False)
    ]
    assert "not on disk" in results[0].message


def test_the_reader_rejects_a_document_recorded_without_a_digest(tmp_path):
    entries = [entry("a-paper")]
    del entries[0]["sha256"]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"documents": entries}))

    with pytest.raises(ValueError, match="sha256"):
        load_manifest(manifest_path)


def test_the_reader_rejects_an_entry_missing_a_field_it_cannot_do_without(tmp_path):
    entries = [entry("a-paper")]
    del entries[0]["url"]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"documents": entries}))

    with pytest.raises(ValueError, match="url"):
        load_manifest(manifest_path)


def test_the_reader_rejects_two_documents_landing_on_one_file(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "documents": [
                    entry("a-paper", filename="shared.pdf", payload=b"one"),
                    entry("b-paper", filename="shared.pdf", payload=b"two"),
                ]
            }
        )
    )

    with pytest.raises(ValueError, match="shared.pdf"):
        load_manifest(path)


def test_the_fetch_command_installs_the_corpus_and_reports_every_document(
    monkeypatch, tmp_path, capsys
):
    entries = [entry("a-paper", payload=b"paper bytes")]
    manifest = manifest_of(tmp_path, entries)
    downloader = FakeDownloader({"https://example.org/a-paper.pdf": b"paper bytes"})

    module = importlib.import_module("main")
    monkeypatch.setattr(module, "load_manifest", lambda: manifest)
    monkeypatch.setattr("corpus.populate.download_document", downloader)
    run_cli(monkeypatch, tmp_path, "fetch-corpus")

    reported = capsys.readouterr().out
    assert "a-paper" in reported
    assert "fetched" in reported
    assert (tmp_path / "papers" / "a-paper.pdf").read_bytes() == b"paper bytes"


def test_the_fetch_command_fails_when_a_document_does_not_match_its_digest(
    monkeypatch, tmp_path
):
    manifest = manifest_of(tmp_path, [entry("a-paper", payload=b"paper bytes")])
    module = importlib.import_module("main")
    monkeypatch.setattr(module, "load_manifest", lambda: manifest)
    monkeypatch.setattr(
        "corpus.populate.download_document",
        FakeDownloader(
            {"https://example.org/a-paper.pdf": b"paper bytes"},
            corrupt={"https://example.org/a-paper.pdf"},
        ),
    )

    with pytest.raises(SystemExit) as exit_info:
        run_cli(monkeypatch, tmp_path, "fetch-corpus")

    assert exit_info.value.code == 1


def test_the_verify_command_reports_a_clean_corpus_without_fetching_anything(
    monkeypatch, tmp_path, capsys
):
    manifest = manifest_of(tmp_path, [entry("a-paper", payload=b"paper bytes")])
    (tmp_path / "papers").mkdir(parents=True)
    (tmp_path / "papers" / "a-paper.pdf").write_bytes(b"paper bytes")
    module = importlib.import_module("main")
    monkeypatch.setattr(module, "load_manifest", lambda: manifest)

    monkeypatch.setattr("corpus.populate.download_document", unreachable)
    run_cli(monkeypatch, tmp_path, "verify-corpus")

    assert "a-paper" in capsys.readouterr().out


def test_both_corpus_commands_take_one_domain_at_a_time(monkeypatch, tmp_path, capsys):
    """A reader who fetched one domain must be able to check that domain without failing on the rest."""
    entries = [
        entry("a-paper", payload=b"paper bytes"),
        entry("a-manual", domain="manuals", payload=b"manual bytes"),
    ]
    manifest = manifest_of(tmp_path, entries)
    module = importlib.import_module("main")
    monkeypatch.setattr(module, "load_manifest", lambda: manifest)
    monkeypatch.setattr(
        "corpus.populate.download_document",
        FakeDownloader(
            {
                "https://example.org/a-paper.pdf": b"paper bytes",
                "https://example.org/a-manual.pdf": b"manual bytes",
            }
        ),
    )

    run_cli(monkeypatch, tmp_path, "fetch-corpus", "--domain", "papers")
    run_cli(monkeypatch, tmp_path, "verify-corpus", "--domain", "papers")

    assert "1 of 1 documents verified" in capsys.readouterr().out


def test_the_verify_command_fails_when_a_document_is_altered(monkeypatch, tmp_path):
    manifest = manifest_of(tmp_path, [entry("a-paper", payload=b"paper bytes")])
    (tmp_path / "papers").mkdir(parents=True)
    (tmp_path / "papers" / "a-paper.pdf").write_bytes(b"something else entirely")
    module = importlib.import_module("main")
    monkeypatch.setattr(module, "load_manifest", lambda: manifest)

    with pytest.raises(SystemExit) as exit_info:
        run_cli(monkeypatch, tmp_path, "verify-corpus")

    assert exit_info.value.code == 1
