# Freeze the generation stage across all architectures

Every architecture shares one prompt, one generation model, one temperature (0), one `top_k` of
5, and one context assembly function. Each results row records the prompt hash, embedding model
and its revision, `top_k`, and temperature, so the isolation is verifiable from the output file
rather than asserted in prose.

## Considered Options

Tuning the prompt per architecture was rejected. It would likely raise every architecture's
score, and it would make the results table meaningless, because a reader could no longer tell
whether hybrid beat naive or whether the hybrid prompt was simply better. The same reasoning
applies to per-architecture `top_k` and to letting the hierarchical splitter feed parent
sections to the model, which is the one case where the context necessarily differs.

The consequence is accepted: some architectures score worse than they could. That is the
point. A project claiming to compare retrieval strategies cannot change the other half of the
pipeline mid-experiment.
