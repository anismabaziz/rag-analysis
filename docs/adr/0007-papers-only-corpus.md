# Papers-only corpus

The corpus is six arXiv papers in one domain. The two-domain design in 0003 is superseded.

## Why

The manuals domain was two database reference manuals at nearly ten thousand pages between
them. Ingestion parses every PDF per architecture with Hi-Res layout analysis and embeds every
chunk twice, so seven architectures over that corpus needed hours and more memory than the
machine doing the work had. A corpus that cannot be ingested locally produces no runs and no
table, which is worse than a narrower corpus that does.

## What is kept

The `papers` evaluation set is stratified into identifier-heavy and paraphrase questions, so
the boundary the project claims still gets measured within the one domain. Per-domain and
pooled reporting stay as they are; with one domain the pool is the papers set.

## What is lost

The cross-register finding is gone: there is no longer a reference-manual register to show the
boundary holding outside continuous prose. If heavier hardware is available later, re-adding a
reference domain restores it.
