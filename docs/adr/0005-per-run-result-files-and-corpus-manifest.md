# One result file per run, corpus identified by manifest, PDFs not committed

Each run writes its own result file recording the configuration, the commit SHA, a corpus
identifier, per-question results, and aggregates. The corpus itself is described by a committed
manifest of document URLs and sha256 digests rather than by committing the PDFs.

## Considered Options

A single accumulated results file that every run appends to was rejected. It cannot record the
configuration a given run used unless that discipline is maintained perfectly, and the result
is a file whose numbers cannot be traced to a run.

Committing the PDFs was rejected on size. Forty academic papers is roughly 80 MB, which makes
the repository unpleasant to clone for a reader who only came to look at the results table.

The corpus identifier is the load-bearing part of this decision. The corpus changes silently as
the loader is corrected, so a results table with no corpus fingerprint is unreproducible by
anyone, including the author.
