from core.base import BaseRAG
from llama_index.core.vector_stores.types import VectorStoreQuery


class NaiveRAG(BaseRAG):
	"""
	NaiveRAG implements a standard dense retrieval RAG pipeline.
	It retrieves context documents based purely on cosine similarity of text embeddings.
	"""

	def __init__(self, llm, embed_model):
		super().__init__(llm, embed_model)
		self.vector_store = None

	async def retrieve(self, query):
		"""
		Retrieves the top 5 most semantically similar document nodes from Qdrant.
		"""

		# generate vector embedding for the query string
		query_embedding = await self.embed_model.aget_query_embedding(query)

		# query the Qdrant vector store using similarity top_k=5
		results = self.vector_store.query(
			VectorStoreQuery(
				query_embedding=query_embedding,
				similarity_top_k=5,
			)
		)

		print(f"[DEBUG] Results scores")
		if results.nodes:
				for node, score in zip(results.nodes, results.similarities):
						print(f"Score: {score:.4f} - {node.metadata["section"]}")
						
		# return the retrieved nodes list, defaulting to empty list if None
		return results.nodes or []