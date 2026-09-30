# Graph retrieval terminates in chunks, not triples

Entities and edges extracted from the corpus are persisted to Neo4j, and a query traverses the
graph to find relevant entities and edges. Those edges are then expanded back to the source
chunks they were extracted from, and those chunks are the retrieved units handed to the
generator, exactly as in every other architecture.

## Considered Options

Answering from the triples directly was rejected. The graph is derived text, so measuring
citation accuracy against it would be circular, and the answers would read like database output
rather than something a reader would recognize as sourced from the papers.

Using the graph to rewrite the query and then falling through to ordinary retrieval was also
rejected. It is cheaper, but it makes the graph a preprocessing step rather than a retrieval
strategy, so there is nothing left to compare against the other arms.

The design requirement this creates is a fallback path for queries that name no known entity,
and the unanswerable stratum of the evaluation set will exercise it constantly. The share of
queries that fall back is a reported result about the approach, not an implementation detail.
