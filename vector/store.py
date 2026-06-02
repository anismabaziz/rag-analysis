from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient




def get_qdrant_client():
  return QdrantClient(url="http://localhost:6333")


def get_vector_store():
  return QdrantVectorStore(
    client=get_qdrant_client(),
    collection_name="rag_benchmark"
  )



def reset_vector_store():
  client = get_qdrant_client()

  client.delete_collection("rag_benchmark")
  print(f"[INFO] removed collection")

