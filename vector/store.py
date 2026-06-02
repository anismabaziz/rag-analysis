from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


def get_vector_store():
  client = QdrantClient(url="http://localhost:6333")

  return QdrantVectorStore(
    client=client,
    collection_name="rag_benchmark"
  )