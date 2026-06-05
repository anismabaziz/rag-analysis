from core.base import BaseRAG
from llama_index.core.vector_stores.types import VectorStoreQuery
from vector.store import get_vector_store

class NaiveRAG(BaseRAG):
	"""
	NaiveRAG implements a standard dense retrieval RAG pipeline.
	It retrieves context documents based purely on cosine similarity of text embeddings.
	"""

	def __init__(self, llm, embed_model):
		super().__init__(llm, embed_model)
		self.vector_store = get_vector_store("rag_naive", enable_hybrid=False)

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

		# return the retrieved nodes list, defaulting to empty list if None
		return results.nodes or []

	async def build_context(self, nodes):
		"""
		Concatenates the text of retrieved nodes to form the context for the LLM.
		"""
		context_parts = []
		for i, node in enumerate(nodes, 1):
				# Add section info if available
				section = node.metadata.get('section', 'Unknown')
				context_parts.append(f"[{i}] ({section}): {node.text}")
		
		return "\n\n".join(context_parts)