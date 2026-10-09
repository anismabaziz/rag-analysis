from dotenv import load_dotenv
import os

load_dotenv()


class Params:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    # One generation model answers every architecture's questions. It is named here rather than
    # in each architecture because a second model would mean a comparison of two setups instead
    # of a comparison of retrieval.
    GENERATION_MODEL = os.getenv("GENERATION_MODEL", "qwen/qwen3.8-27b")

    # Everything below is the frozen part of a run: the values held constant across every
    # architecture, so that a difference between two results is a difference in retrieval and
    # nothing else. They are deliberately not environment variables. A run that read its own
    # isolation from the environment could not be verified from the result file afterwards, and
    # the one value that is worth varying, the architecture, is the one the reader names.

    # One encoder for dense vectors and one for sparse, named once so ingestion, querying, and the
    # configuration recorded in a result file cannot disagree about which models produced them.
    DENSE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    SPARSE_EMBEDDING_MODEL = "Qdrant/bm42-all-minilm-l6-v2-attentions"

    # The generation stage answers at zero, so a rerun of the same run is not sampling.
    GENERATION_TEMPERATURE = 0.0

    # How deep every architecture retrieves. Varying it per architecture would raise some scores
    # and leave the comparison meaning nothing.
    TOP_K = 5

    # How wide the rerank architecture's first stage retrieves before the cross-encoder
    # narrows it back to the depth above. Wider than the committed depth on purpose: the
    # reranker can only promote what the first stage fetched. Recorded in the run
    # configuration like every other frozen value, so a reader can check it from a result file.
    RERANK_CANDIDATES = 20

    # The cross-encoder that orders the rerank architecture's candidates. Named once here,
    # like the dense and sparse encoders, so the run file cannot disagree about which model did it.
    RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"

    # How the corpus is cut into chunks, which decides what a retrieval can return at all.
    CHUNKER_STRATEGY = "semantic"
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 128


# The strategies a chunking comparison runs. Named once so a variant, the record of it, and
# the table over it cannot disagree about what the options are.
FIXED = "fixed"
SEMANTIC = "semantic"
HIERARCHICAL = "hierarchical"
CHUNKING_STRATEGIES = (FIXED, SEMANTIC, HIERARCHICAL)

# How much larger than its children a hierarchical strategy cuts a whole section. A parent
# chunked at four times the child size holds the section the child was cut from, and a run
# file records that size so the context a retrieved chunk can carry is visible.
HIERARCHICAL_PARENT_FACTOR = 4
