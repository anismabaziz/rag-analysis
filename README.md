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

`sparse` indexes a BM42 sparse vector per chunk and retrieves on it alone, so the sparse signal's
contribution can be told apart from the fusion. Without this row, a gain over dense-only could be
the sparse signal doing the work or the fusion doing it, and the table would not say which.

`hybrid` indexes a dense vector (all-MiniLM-L6-v2) alongside a BM42 sparse vector for the same
chunk, then fuses two ranked candidate lists with reciprocal rank fusion.

Each is ingested into its own Qdrant collection, so a rerun never mixes corpora and one
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

The harness is installed as a command, so none of this needs the module layout. The corpus is a
committed manifest of URLs and digests, not committed PDFs, so fetch it before the first
ingest:

```bash
uv run rag-analysis fetch-corpus              # 60 MB, both domains
uv run rag-analysis fetch-corpus --domain papers

# see what can be ingested and queried
uv run rag-analysis architectures

# index the corpus, once per architecture
uv run rag-analysis ingest naive
uv run rag-analysis ingest sparse
uv run rag-analysis ingest hybrid

# ask a question against one of them
uv run rag-analysis query naive "how are the positional encodings scaled?"
uv run rag-analysis query sparse "how are the positional encodings scaled?"
uv run rag-analysis query hybrid "how are the positional encodings scaled?"

# run one architecture over a domain's evaluation set and write the result file
uv run rag-analysis run naive --domain papers
uv run rag-analysis run sparse --domain papers
uv run rag-analysis run hybrid --domain manuals

# point a run at a cache of your own instead of the committed one
uv run rag-analysis run naive --domain papers --cache-dir /tmp/responses

# read every run file back as one pooled and per-domain retrieval table
uv run rag-analysis summarize
uv run rag-analysis summarize --out results/summary.md

# drop a collection and start over
uv run rag-analysis clear rag_naive

# check the documents on disk against the manifest, without any network
uv run rag-analysis verify-corpus
uv run rag-analysis verify-corpus --domain papers
```

A document is written to `documents/<domain>/` only once its bytes match the digest in
`corpus/manifest.json`, and a mismatch fails the command, so a truncated download cannot end up
inside a corpus. `verify-corpus` answers whether the corpus is unmodified, which is the check a
reader needs before trusting a number in the results table. The two domains and why they are
the two are described in [docs/corpus.md](docs/corpus.md). Ingestion reads every PDF under
`./documents`, and the manuals domain is 9,798 pages of it, so `--domain papers` is the fast
way in.

Ingestion can also be run on its own, without the rest of the CLI:

```bash
uv run python build_index.py --architecture naive
uv run python build_index.py --architecture sparse
uv run python build_index.py --architecture hybrid
```

`ingest` replaces the contents of the target collection, so run it again after adding documents.

## Reproducing the published numbers

A hosted model is not deterministic even at temperature zero, and hosted model aliases get
withdrawn. Both would quietly change the numbers in the results table after the fact, so every
generated answer is cached on disk under `results/cache/`, keyed by the prompt, the model, the
temperature, the query, and the retrieved context. Change any one of those and the entry is a
miss; keep all five and the answer is replayed without calling the model.

The cache is committed, so a reader regenerates the published table by running the evaluation
with no API key and no hosted model at all. It is gzipped JSON, one small file per question, so
it costs a few kilobytes per domain and a regenerated run rewrites only the entries that
actually changed. Answers for a run nobody has published do not need committing: point the run
at a scratch cache with `--cache-dir` instead.

## Adding an architecture

An architecture is one file: the pipeline class, decorated with a declaration of the collection
it reads and the vectors it indexes. `core/registry.py` discovers every module in
`architectures/` and registers what it finds, so ingestion and querying both read that
declaration and nothing else lists architectures. Drop a new module in and it can be ingested,
queried, and listed:

```python
# architectures/sparse.py
@register(
	name="sparse",
	description="BM42 sparse vectors only, ranked by term overlap.",
	collection="rag_sparse",
	vectors=(SPARSE,),
)
class SparseRAG(BaseRAG):
	async def retrieve(self, query: str) -> list[Chunk]:
		...
```

That is the whole of it. The name, the collection, and the vectors it indexes are what separate
one architecture from the next; nothing else in the project has to change.

## Tests

```bash
uv run pytest
uv run ruff check .
```

No test needs a running service or network access. The vector store, the hosted model client, and
the embedding models are all swapped out inside the tests. Runs in tests also get a scratch
cache, so a test run never writes into the committed one.

## Layout

| Path | What lives there |
| --- | --- |
| `main.py` | CLI: ingest, query, run, summarize, architectures, clear, fetch-corpus, verify-corpus |
| `build_index.py` | PDF loading, chunking, embedding, indexing into Qdrant |
| `corpus/` | The corpus manifest, and fetching and verifying the documents it names |
| `evaluation/` | The hand-written questions each domain is scored on, the rules their labels satisfy, and the run that scores an architecture against them |
| `data/` | Loading, splitting, and embedding the corpus |
| `architectures/` | One file per retrieval architecture: the pipeline, and the declaration that registers it |
| `core/` | The shared pipeline, prompt, and the architecture registry |
| `results/runs/` | One result file per run: the configuration, the commit, the corpus identifier, per-question results, and aggregates |
| `results/cache/` | The committed model responses the published numbers were measured from |
| `vector/` | Qdrant access |
| `config/` | Configuration, read once at startup |
| `docs/adr/` | Recorded decisions and the options that lost |

## Configuration

Every variable the project reads is listed in `.env.example`. `GROQ_API_KEY` is the only value you
must supply. The rest have defaults, `QDRANT_URL` being `http://localhost:6333`.

## License

MIT. See [LICENSE](LICENSE).
