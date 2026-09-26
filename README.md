# RAG analysis

A comparison harness for retrieval architectures. It runs the same corpus through several
retrieval strategies with the generation stage held fixed, so the differences between them are
attributable to retrieval and not to a prompt change.

## The claim

Hybrid retrieval (dense + sparse, fused with reciprocal rank fusion) beats dense-only retrieval on
technical questions containing rare identifiers, and loses to it on paraphrased questions.

That is the falsifiable part. A hybrid architecture that wins on both question types has not
demonstrated anything except that two retrieval signals were combined; where the hybrid advantage
stops holding is the more useful result than the average. The evaluation set is stratified to
measure that boundary rather than assume it.

The project exists to test that claim, and the numbers in the results table are not here yet. The
harness is being built in stages, and the README will carry the measured table once the evaluation
set exists. See the design decisions in `docs/adr/` for the reasoning behind how it is put
together.

## What is being compared

`naive` indexes and retrieves dense vectors only, using cosine similarity.

`hybrid` indexes a dense vector (all-MiniLM-L6-v2) alongside a BM42 sparse vector for the same
chunk, then fuses two ranked candidate lists with reciprocal rank fusion.

Both are ingested into their own Qdrant collection, so a rerun never mixes corpora and one
architecture can never retrieve the other's points. Generation is frozen to one model
(`llama-3.3-70b-versatile`) at temperature 0 with a single shared prompt, for the same reason.

## Setup

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), and Docker.

```bash
git clone <your fork url>
cd rag-analysis
uv sync --group dev
cp .env.example .env      # then put a GROQ_API_KEY in it
docker compose up -d
```

`docker compose up -d` starts Qdrant on port 6333, which is the only service the retrieval
architectures need. The compose file also brings up Neo4j, which no current architecture reads.
Image versions are pinned there, so a rerun does not silently change library versions underneath a
result.

The first ingest downloads the embedding models, the table layout model, and an English language
model, which takes a few minutes. Later runs reuse the cache.

## Running it

Put your PDFs under `./documents/`, then:

```bash
# index the corpus, once per architecture
uv run python main.py ingest naive
uv run python main.py ingest hybrid

# ask a question against one of them
uv run python main.py test-naive "how are the positional encodings scaled?"
uv run python main.py test-hybrid "how are the positional encodings scaled?"

# drop a collection and start over
uv run python main.py clear rag_naive
```

Ingestion can also be run on its own, without the top-level CLI:

```bash
uv run python build_index.py --collection rag_naive
uv run python build_index.py --collection rag_hybrid --hybrid
```

`ingest` replaces the contents of the target collection, so run it again after adding documents.

## Tests

```bash
uv run pytest
uv run ruff check .
```

No test needs a running service or network access. The vector store, the hosted model client, and
the embedding models are all swapped out inside the tests.

## Layout

| Path | What lives there |
| --- | --- |
| `main.py` | CLI: ingest, clear, test-naive, test-hybrid |
| `build_index.py` | PDF loading, chunking, embedding, indexing into Qdrant |
| `data/` | Loading, splitting, and embedding the corpus |
| `rag/` | The retrieval architectures |
| `runners/` | Wiring from the CLI to an architecture |
| `core/` | The shared pipeline and prompt |
| `vector/` | Qdrant access |
| `config/` | Configuration, read once at startup |
| `docs/adr/` | Recorded decisions and the options that lost |

## Configuration

Every variable the project reads is listed in `.env.example`. `GROQ_API_KEY` is the only value you
must supply. The rest have defaults, `QDRANT_URL` being `http://localhost:6333`.

## License

MIT. See [LICENSE](LICENSE).
