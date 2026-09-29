from dotenv import load_dotenv
import os

load_dotenv()

class Params:
  GROQ_API_KEY = os.getenv("GROQ_API_KEY")
  QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
  NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
  NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
  # One generation model answers every architecture's questions. It is named here rather than
  # in each architecture because a second model would mean a comparison of two setups instead
  # of a comparison of retrieval.
  GENERATION_MODEL = os.getenv("GENERATION_MODEL", "llama-3.3-70b-versatile")

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

  # How the corpus is cut into chunks, which decides what a retrieval can return at all.
  CHUNKER_STRATEGY = "semantic"
  CHUNK_SIZE = 512
  CHUNK_OVERLAP = 128
