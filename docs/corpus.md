# The corpus

Six documents in one domain, listed in `corpus/manifest.json`. Each entry records where the
document is fetched from, how many pages and bytes it is, and the sha256 of those bytes. The
PDFs themselves are not in the repository, and neither is a second copy of them anywhere: the
manifest is enough to rebuild the corpus and to prove that what is on disk is what a run was
measured against.

## The domain

| Domain | Documents | Pages | Register |
| --- | --- | --- | --- |
| `papers` | 6 | 94 | Continuous argument, dense technical vocabulary, sectioned prose |

Every entry records its own page count and size, so those figures can be checked against the
manifest rather than taken on trust.

`papers` is the retrieval literature itself, six arXiv papers on dense, sparse, and hybrid
retrieval: the Transformer, DPR, RAG, a study of where a model stops using a long prompt, a
multilingual embedding model, and a lexical-seeded acceleration of dense search. The questions a
reader can most easily check an answer against are the ones about these documents, so the domain is
easy to hand-author ground truth for.

A second domain of database reference manuals was removed: two manuals at nearly ten thousand
pages between them made ingestion too heavy to run locally, and a corpus nobody can ingest is
worse than a narrower one. The stratification inside `papers` (identifier-heavy vs paraphrase)
still carries the claim's boundary; see the decision record superseding the two-domain design.

## Fetching and verifying

```bash
uv run rag-analysis fetch-corpus               # the papers domain, a few MB
uv run rag-analysis fetch-corpus --domain papers
uv run rag-analysis verify-corpus              # no network, checks the digests on disk
uv run rag-analysis verify-corpus --domain papers
```

`fetch-corpus` writes a document into `documents/<domain>/` only after its bytes hash to the
digest the manifest records. A mismatch leaves nothing behind and fails the command, so a
truncated download cannot become part of a corpus quietly. A document already on disk is
checked against the same digest and left alone when it matches, so a second run costs nothing.
`verify-corpus` is the same check with no network at all, and it reports instead of repairing,
which makes it the answer to "can I trust that these are the documents the results table was
produced from". It needs no backup of the corpus to answer that.

Both commands take `--domain`, so a corpus can be built and checked one domain at a time, and
both fail loudly rather than reporting a partial corpus as a whole one.

An arXiv URL serves the PDF of one paper and does not change, so the pinned digests stay valid.

## What is not here

No HTML documentation and no pre-extracted text corpora. Both are easier to fetch, and both
skip the PDF ingestion path, which is the code this project is actually about. Every document
here is a PDF read by `data/loader.py`.
