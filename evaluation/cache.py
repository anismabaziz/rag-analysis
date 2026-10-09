"""Model responses cached on disk, so a repeated run is free and identical.

A hosted model is not deterministic even at temperature zero and it gets deprecated, so
without this the published numbers silently change or become unreproducible the moment the
pinned model is withdrawn. A response is keyed by the prompt, the model, the temperature,
the query, and the retrieved context, with the context beside the other four because the
same question retrieves different chunks under different architectures: without it, one
architecture's answer would stand in for another's. Changing any of the five is a miss.

Each entry is one gzipped JSON file named after its key, so the cache stays small and git
only rewrites the entries a run actually changed. The compression is written without a
timestamp, so re-recording the same response produces the same bytes rather than a diff on
every regeneration. Empty retrieval never reaches the cache, because the refusal stands in
for the answer without asking the model.
"""

import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from config.configuration import Configuration

# Where responses live. It sits beside the run files because it is their evidence: the answers
# a published table was measured from. Nothing ignores it, so the entries backing the README
# table are committed with the code that reads them.
CACHE_DIR = Path("results/cache")


@dataclass(frozen=True)
class CacheKey:
    """What a response is keyed by: how it was asked, and what it was asked about.

    The first three come from the run's frozen configuration, so a run whose prompt, model, or
    temperature changed cannot read an entry recorded under the old one. The last two are the
    question and the context retrieval gave the model for it.
    """

    prompt: str
    model: str
    temperature: float
    query: str
    context: str

    @classmethod
    def for_call(
        cls, configuration: Configuration, query: str, context: str
    ) -> "CacheKey":
        """The key one answered call is stored under, read off the run's configuration."""
        return cls(
            prompt=configuration.prompt,
            model=configuration.generation_model,
            temperature=configuration.generation_temperature,
            query=query,
            context=context,
        )

    def digest(self) -> str:
        """The key as hex, from a canonical record of its five fields.

        The fields are encoded as one JSON record rather than joined with a separator, so no
        two combinations of them can produce the same digest.
        """
        return hashlib.sha256(self.record().encode("utf-8")).hexdigest()

    def record(self, response: str | None = None) -> str:
        """The key as one canonical JSON line, optionally with a response beside it.

        The key is written the same way whether it is being digested, stored, or read back, so
        an entry can never be filed under a key that does not match its contents.
        """
        fields = {
            "prompt": self.prompt,
            "model": self.model,
            "temperature": self.temperature,
            "query": self.query,
            "context": self.context,
        }
        if response is not None:
            fields["response"] = response

        return json.dumps(
            fields, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )


def cache_path(cache_dir: Path, key: CacheKey) -> Path:
    """Where the entry for `key` lives."""
    return Path(cache_dir) / f"{key.digest()}.json.gz"


def lookup(cache_dir: Path, key: CacheKey) -> str | None:
    """The cached response for this key, or nothing when there is none.

    A missing file is a miss. A file that cannot be read is also a miss rather than a failed
    run: the model is still there to answer, and the next store overwrites the broken entry.
    """
    path = cache_path(cache_dir, key)

    if not path.is_file():
        return None

    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)["response"]
    except (OSError, ValueError, KeyError):
        return None


def store(cache_dir: Path, key: CacheKey, response: str) -> None:
    """Write `response` to the cache under `key`.

    The entry carries the key beside the response, so a reader can see which configuration
    produced an answer without reverse-engineering a filename. The compression is written
    without a timestamp, so recording the same response twice yields the same bytes and git
    records no change.
    """
    path = cache_path(cache_dir, key)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_bytes(gzip.compress(key.record(response).encode("utf-8"), mtime=0))


async def generate_with_cache(
    pipeline,
    configuration: Configuration,
    query: str,
    context: str,
    cache_dir: Path = CACHE_DIR,
) -> str:
    """Answer `query` from `context`, reading the cached response when there is one.

    A hit returns without touching the model. A miss asks the model once and writes what it
    said before returning it, so the next identical run is a hit.
    """
    key = CacheKey.for_call(configuration, query, context)

    cached = lookup(cache_dir, key)
    if cached is not None:
        return cached

    answer = await pipeline.generate(query, context)
    store(cache_dir, key, answer)

    return answer
