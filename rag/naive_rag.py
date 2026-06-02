from core.base import BaseRAG
from llama_index.core.vector_stores.types import VectorStoreQuery



class NaiveRAG(BaseRAG):


  async def retrieve(self, query):
    query_embedding = await self.embed_model.aget_query_embedding(query)
    results = self.vector_store.query(
      VectorStoreQuery(
        query_embedding=query_embedding,
        similarity_top_k=5,
      )
    )
    return results.nodes or []
  

  async def build_context(self, nodes):
    return "\n\n".join([node.text for node in nodes])