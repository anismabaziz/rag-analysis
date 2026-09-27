# The corpus

Eight documents in two domains, listed in `corpus/manifest.json`. Each entry records where the
document is fetched from, how many pages and bytes it is, and the sha256 of those bytes. The
PDFs themselves are not in the repository, and neither is a second copy of them anywhere: the
manifest is enough to rebuild the corpus and to prove that what is on disk is what a run was
measured against.

## The two domains

The corpus has to be heterogeneous, or the results flatter whichever architecture leans on
vocabulary similarity. A corpus of nothing but academic papers is trivially available, easy to
label, and shares one register: continuous argument, technical vocabulary, no short reference
entries. Dense retrieval does well on that register for reasons that have nothing to do with
the claim under test, so a single-domain corpus would quietly pick a winner before any
evaluation ran.

| Domain | Documents | Pages | Register |
| --- | --- | --- | --- |
| `papers` | 6 | 90 | Continuous argument, dense technical vocabulary, sectioned prose |
| `manuals` | 2 | 9,798 | Reference entries, option names, error codes, tables, one fact per paragraph |

Every entry records its own page count and size, so those figures can be checked against the
manifest rather than taken on trust.

The two registers are what the boundary condition is expected to sit between, so a result that
holds in one and not the other is a finding rather than noise.

`papers` is the retrieval literature itself, six arXiv papers on dense, sparse, and hybrid
retrieval. The questions a reader can most easily check an answer against are the ones about
these documents, so the domain is both heterogeneous against the manuals and easy to
hand-author ground truth for.

`manuals` is two database reference manuals, the PostgreSQL 16 documentation and the MySQL 9.3
reference manual. They are the opposite register. Almost every paragraph is a self-contained
entry about one option, function, or error code, and the rare strings in them are the thing
being asked about: `pg_stat_statements`, `innodb_flush_log_at_trx_commit`, `LATERAL`. A dense
encoder compresses "how do I make commits faster" and a BM25 index keeps `innodb_flush_log_at_trx_commit`
apart, which is exactly the axis the project claims a boundary condition sits on. The two
manuals are 9,798 pages between them, which is a deliberate cost: a domain that big is where a
retrieval result stops being about a document and starts being about a corpus.

Results are reported per domain, never as one pooled average, so a difference that only holds
in one register shows up as a difference rather than being averaged away.

## Fetching and verifying

```bash
uv run rag-analysis fetch-corpus               # both domains, 60 MB
uv run rag-analysis fetch-corpus --domain papers
uv run rag-analysis verify-corpus              # no network, checks the digests on disk
uv run rag-analysis verify-corpus --domain papers
```

`fetch-corpus` writes a document into `documents/<domain>/` only after its bytes hash to the
digest the manifest records. A mismatch leaves nothing behind and fails the command, so a
truncated download cannot become part of a corpus quietly. A document already on disk is
checked against the same digest and left alone when it matches, so a second run costs nothing.
One that does not match is fetched again and the output says `replaced`, because a half
downloaded manual is the common case and the reader asked for a working corpus.
`verify-corpus` is the same check with no network at all, and it reports instead of repairing,
which makes it the answer to "can I trust that these are the documents the results table was
produced from". It needs no backup of the corpus to answer that.

Both commands take `--domain`, so a corpus can be built and checked one domain at a time, and
both fail loudly rather than reporting a partial corpus as a whole one.

The three hosts are not equally stable. An arXiv URL serves the PDF of one paper and does not
change. The PostgreSQL and MySQL PDFs are regenerated when a point release ships, so their
digests will stop matching at some point, and each entry's title records the point release the
digest belongs to. When that happens the fetch fails with a digest mismatch, which is the
signal to read the new file, check it is the same documentation, and re-pin the digest in
`corpus/manifest.json`, where the change is a reviewed diff.

## What is not here

No HTML documentation and no pre-extracted text corpora. Both are easier to fetch, and both
skip the PDF ingestion path, which is the code this project is actually about. Every document
here is a PDF read by `data/loader.py`.
