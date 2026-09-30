# Compare chunking strategies on retrieval metrics only

Fixed-size, semantic, and hierarchical chunkers are compared with one axis varying, the
chunker, and everything else frozen: the same dense retrieval at the same depth, the same
prompt, models, and temperature. The comparison reports retrieval metrics and nothing else.

## Context

`0001-freeze-the-generation-stage` freezes the context the generator reads, and names the
hierarchical splitter as the one case where context cannot be held constant. That decision
stands. A hierarchical strategy indexes whole sections alongside their slices, so a retrieved
chunk can be a whole section where another strategy returns a slice, and semantic splitting
produces different chunk sizes than fixed-size splitting, which makes a strategy comparison a
measurement of chunk size.

`0001` rejected letting the hierarchical splitter feed parent sections to the model. That
rejection still holds for the retrieval comparison, where every architecture answers from the
same depth of the same kind of chunk. This decision does not reopen it: the chunking arms are
retrieval arms, they are compared on retrieval, and no answer is generated across them.

## Consequences

Any answer-quality figure reported across strategies states its token matching explicitly.
Where the budget is not matched, the table says so rather than reporting the figure as if it
were. The budget is measured from the retrieved text rather than taken from the nominal chunk
size, because a claim that a budget was matched is worth nothing if nobody checked it.

A run file records the chunker it was measured at, and whether that chunker is the committed
one. The retrieval table takes the runs measured at the committed chunker and leaves the
variants out, so a table framed as one axis varying cannot silently span a retriever and a
chunker. The chunking table takes the variants and refuses runs that differ from each other in
anything but the chunker.

The cost is that a chunking strategy can look good on retrieval and still be the wrong choice
for an answering system, and this project will not say which. That question needs a matched
token budget, which is a separate piece of work rather than a footnote here.
