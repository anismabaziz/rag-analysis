# Two corpus domains, results reported per domain

> Superseded by 0007: the manuals domain was removed because a ten-thousand-page corpus
> could not be ingested on the hardware available. What follows is the original reasoning,
> kept as a record.

The corpus spans two heterogeneous domains, and every metric is reported as a pooled figure
plus a per-domain breakdown rather than a single average.

## Considered Options

A single-domain corpus of academic papers was rejected. It is trivially available and easy to
label, but its vocabulary is homogeneous enough that it flatters dense retrieval, which would
quietly bias the headline comparison in favor of one architecture. A single pooled average was
rejected for the same reason from the reporting side, since it hides the boundary conditions
where the claim flips between identifier-heavy questions and paraphrased ones.

Public text benchmarks were rejected because they ship pre-chunked, pre-extracted text, which
would bypass the PDF ingestion path that is the actual subject of this project and turn the
results into a benchmark of `unstructured` rather than of this code.

Per-domain reporting is what makes the boundary condition a finding rather than an
anecdote, and it is the cheapest available way to show the results are not an artifact of one
document type.
