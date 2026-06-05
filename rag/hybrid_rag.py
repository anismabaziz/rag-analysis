from core.base import BaseRAG
from llama_index.core.vector_stores.types import VectorStoreQuery
from vector.store import get_vector_store


class HybridRAG(BaseRAG):
  """
  HybridRAG implements a hybrid search pipeline combining:
  1. Dense Vector Search (semantic similarity via embeddings).
  2. Sparse Search (keyword matching via BM25).
  Combined using Reciprocal Rank Fusion (RRF) to rank retrieved nodes.
  """

  def __init__(self, llm, embed_model):
    """
    Initializes the HybridRAG pipeline with the necessary search and LLM models.
    """
    super().__init__(llm, embed_model)
    self.vector_store = get_vector_store("rag_hybrid", enable_hybrid=True)


  async def retrieve(self, query: str):
    """
    Retrieves the top_k most relevant nodes using both dense and sparse search,
    fused together using Reciprocal Rank Fusion (RRF).
    """

    # embed query
    query_embedding = await self.embed_model.aget_query_embedding(query)

    # query qdrant with hybrid search mode
    results = self.vector_store.query(
      VectorStoreQuery(
        query_embedding=query_embedding,
        mode="hybrid",
        sparse_top_k=10,
        dense_top_k=10,
        similarity_top_k=5
      )
    )

    return results.nodes or []