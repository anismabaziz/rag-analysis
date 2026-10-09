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

`rerank` does the same over a wider candidate set and orders the survivors with a cross-encoder.

Each is ingested into its own Qdrant collection, so a rerun never mixes corpora and one
architecture can never retrieve another's points. Generation is frozen to one model
(`qwen/qwen3.8-27b`) at temperature 0 with a single shared prompt, for the same reason.

Three more, `chunk-fixed`, `chunk-semantic`, and `chunk-hierarchical`, are the chunking
comparison and are not part of the retrieval table. They differ from `naive` in one thing only:
how the corpus was cut before indexing.

## Chunking

How the corpus is cut decides what retrieval can return at all, so the three strategies are
compared on their own. `chunk-fixed` cuts sentence-sized pieces, `chunk-semantic` cuts at
meaning boundaries, and `chunk-hierarchical` indexes whole sections alongside their slices. All
three retrieve with the same dense retriever at the same depth, so the chunker is the only axis
that varies.

The comparison is on retrieval metrics only. A hierarchical chunk can be a whole section where
another strategy returns a slice, and semantic splits come out at different lengths than fixed
ones, so the generator would not read the same context either way. A single answer-quality figure
across the three would be a measurement of how much text retrieval returned, not of chunking. Any
answer-quality figure reported across strategies states its token matching explicitly.

```bash
uv run rag-analysis ingest chunk-fixed
uv run rag-analysis ingest chunk-semantic
uv run rag-analysis ingest chunk-hierarchical

uv run rag-analysis run chunk-fixed --domain papers
uv run rag-analysis run chunk-semantic --domain papers
uv run rag-analysis run chunk-hierarchical --domain papers

uv run rag-analysis summarize-chunking
uv run rag-analysis summarize-chunking --out results/chunking.md
```

The table states the token budget, measured from the retrieved text rather than taken from the
nominal chunk size, and says whether it came out matched. The main `summarize` table leaves these
runs out, because a row spanning two chunkers would be comparing setups rather than retrievers.
The reasoning is in [docs/adr/0006](docs/adr/0006-compare-chunking-on-retrieval-only.md).

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
uv run rag-analysis fetch-corpus              # a few MB, the papers domain
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

# run one architecture over the evaluation set and write the result file
uv run rag-analysis run naive --domain papers
uv run rag-analysis run sparse --domain papers
uv run rag-analysis run hybrid --domain papers

# point a run at a cache of your own instead of the committed one
uv run rag-analysis run naive --domain papers --cache-dir /tmp/responses

# read every run file back as one pooled and per-domain retrieval table
uv run rag-analysis summarize
uv run rag-analysis summarize --out results/summary.md

# read the chunking comparison, retrieval metrics only
uv run rag-analysis summarize-chunking --out results/chunking.md

# drop a collection and start over
uv run rag-analysis clear rag_naive

# check the documents on disk against the manifest, without any network
uv run rag-analysis verify-corpus
uv run rag-analysis verify-corpus --domain papers
```

A document is written to `documents/<domain>/` only once its bytes match the digest in
`corpus/manifest.json`, and a mismatch fails the command, so a truncated download cannot end up
inside a corpus. `verify-corpus` answers whether the corpus is unmodified, which is the check a
reader needs before trusting a number in the results table. The domain is described in
[docs/corpus.md](docs/corpus.md). Ingestion reads every PDF under `./documents`, six papers
at 94 pages in total.

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
    async def retrieve(self, query: str) -> list[Chunk]: ...
```

That is the whole of it. The name, the collection, and the vectors it indexes are what separate
one architecture from the next; nothing else in the project has to change.

## Tests

```bash
uv run pytest tests/fast # every change: hermetic, about a minute
uv run pytest # before pushing: everything, several minutes
uv run ruff check .
```

`tests/fast` swaps out the vector store, the hosted model client, and the embedding
models, and runs in tests get a scratch cache, so it needs no service, no key, and no
network, and a run never writes into the committed cache. `tests/slow` checks the gold
answers against the real corpus PDFs, which takes minutes. `tests/integration` hits the
network and the git history to fetch and verify the corpus. CI runs all three, fetching
the papers corpus first so the slow checks read real documents.

## Layout

| Path | What lives there |
| --- | --- |
| `main.py` | CLI: ingest, query, run, summarize, summarize-chunking, architectures, clear, fetch-corpus, verify-corpus |
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
