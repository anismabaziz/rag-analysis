"""The corpus is a committed manifest, not a committed pile of PDFs.

Six papers and two database manuals are 60 MB of PDF, which is too much to ask of anyone who
cloned the repository to look at a results table, so what is committed is the list of
documents: where each one is fetched from, how big it is, and the sha256 of its bytes. This
module is the one place that list is read and shaped, and the one place that computes a digest,
so a document is checked the same way whether it was just fetched or has been on disk since the
last run.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

MANIFEST_PATH = Path(__file__).with_name("manifest.json")

DIGEST_LENGTH = 64

# Read in blocks, so verifying a several hundred megabyte manual does not need it in memory.
READ_BLOCK = 1024 * 1024


@dataclass(frozen=True)
class Document:
	"""One document of the corpus, as recorded in the manifest."""

	id: str
	domain: str
	title: str
	url: str
	sha256: str
	filename: str
	pages: int
	bytes: int

	def target(self, root: Path) -> Path:
		"""Where this document lives on disk: one directory per domain, under the corpus root.

		Ingestion walks `./documents/**/*.pdf`, so a domain is a directory rather than a field
		alone, which also keeps a document's domain visible in the path a chunk cites.
		"""
		return Path(root) / self.domain / self.filename

	def matches(self, path: Path) -> bool:
		"""Whether the bytes on disk are the ones this manifest entry records."""
		return Path(path).is_file() and sha256_of(path) == self.sha256


@dataclass(frozen=True)
class Manifest:
	"""Every document the corpus is made of."""

	documents: tuple[Document, ...]

	@property
	def domains(self) -> tuple[str, ...]:
		"""The domains in the corpus, in the order the manifest lists them first."""
		seen: list[str] = []
		for document in self.documents:
			if document.domain not in seen:
				seen.append(document.domain)
		return tuple(seen)

	def in_domain(self, domain: str) -> tuple[Document, ...]:
		"""The documents of one domain, which is the unit results are reported for."""
		return tuple(document for document in self.documents if document.domain == domain)

	def document_named_by_file(self, path: str | Path) -> str | None:
		"""The id of the document whose file this path is, or nothing when the corpus holds no such file.

		A retrieved chunk records the path ingestion globbed rather than a document id, so this is
		how a score is attached to provenance. The comparison is on the file name alone, because
		the directory a reader populated the corpus into is their choice and the path that reached
		the store differs by a leading `./` and by where the run was launched from.
		"""
		name = Path(path).name

		for document in self.documents:
			if document.filename == name:
				return document.id

		return None

	def in_domains(self, domains: list[str] | None) -> tuple[Document, ...]:
		"""The documents of the named domains, or all of them when no domain is named.

		An unknown domain is refused here rather than quietly fetching nothing, since a typo in
		`--domain` would otherwise look like a corpus with no documents in it.
		"""
		if domains is None:
			return self.documents

		unknown = sorted(set(domains) - set(self.domains))
		if unknown:
			raise ValueError(f"no such domain in the manifest: {', '.join(unknown)}")

		return tuple(document for document in self.documents if document.domain in domains)


def corpus_identifier(manifest: Manifest) -> str:
	"""One digest naming the exact bytes the corpus is made of.

	The corpus changes without the commit changing, because the loader is corrected more often
	than the manifest is edited and because a reader may have populated it from the manifest a
	while ago. A results file that carried only a commit would claim to describe a corpus nobody
	could check, so it carries this instead: a digest over every document id and the digest of its
	bytes, in the manifest's own order, so the identifier moves if and only if the corpus does.
	"""
	listing = "\n".join(f"{document.id}:{document.sha256}" for document in manifest.documents)

	return "sha256:" + hashlib.sha256(listing.encode("utf-8")).hexdigest()


def sha256_of(path: Path) -> str:
	"""The sha256 of a file, read in blocks so its size does not matter."""
	digest = hashlib.sha256()
	with open(path, "rb") as handle:
		for block in iter(lambda: handle.read(READ_BLOCK), b""):
			digest.update(block)
	return digest.hexdigest()


def load_manifest(path: Path = MANIFEST_PATH) -> Manifest:
	"""Read a manifest, rejecting an entry that could not be verified later.

	A missing or malformed digest is refused here rather than at fetch time, because the whole
	guarantee the project makes about its corpus rests on this file being complete.
	"""
	recorded = json.loads(Path(path).read_text())
	documents = tuple(_document_from(entry) for entry in recorded["documents"])

	seen: dict[tuple[str, str], str] = {}
	for document in documents:
		# Compared as domain and filename rather than as a full path, because the root a reader
		# populates the corpus into is their choice, not part of the manifest.
		landing = (document.domain, document.filename)
		if landing in seen:
			raise ValueError(f"two documents of the manifest land on the same file: {document.domain}/{document.filename}")
		seen[landing] = document.id

	return Manifest(documents=documents)


def _document_from(entry: dict) -> Document:
	"""Read one entry, naming the field and the document when the entry is unusable.

	The manifest is a file people edit by hand, so a missing field is a mistake worth pointing
	at rather than a traceback about a dict key.
	"""
	fields = ("id", "domain", "url", "filename", "pages", "bytes")
	missing = [field for field in fields if not entry.get(field)]
	if missing:
		raise ValueError(f"a manifest document is missing {', '.join(missing)}: {entry}")

	document = Document(
		id=entry["id"],
		domain=entry["domain"],
		title=entry.get("title", ""),
		url=entry["url"],
		sha256=entry.get("sha256", ""),
		filename=entry["filename"],
		pages=entry["pages"],
		bytes=entry["bytes"],
	)

	if len(document.sha256) != DIGEST_LENGTH:
		raise ValueError(f"{document.id} has no usable sha256 in the manifest")

	return document
