from core.base import BaseRAG
from core.chunk import Chunk
from vector.store import chunks_from_result, get_qdrant_client


class NaiveRAG(BaseRAG):
	"""
	NaiveRAG implements a standard dense retrieval RAG pipeline.
	It retrieves context documents based purely on cosine similarity of text embeddings.
	"""

	def __init__(self, llm, embed_model, collection_name):
		super().__init__(llm, embed_model)
		self.vector_store = get_qdrant_client()
		self.collection_name = collection_name

	async def retrieve(self, query: str, top_k: int = 5) -> list[Chunk]:

		# generate vector embedding for the query string
		query_embedding = list(self.embed_model.embed([query]))[0]

		# query the Qdrant vector store using similarity top_k=5
		results = self.vector_store.query_points(
			collection_name=self.collection_name,
			query=query_embedding,
			using='dense',
			limit=top_k
		)

		return chunks_from_result(results)
