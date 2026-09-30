from config.params import Params
from core.base import BaseRAG
from core.chunk import Chunk
from core.registry import register
from vector.store import chunks_from_result, get_qdrant_client


@register(
	name="naive",
	description="Dense vectors only, ranked by cosine similarity.",
	collection="rag_naive",
)
class NaiveRAG(BaseRAG):
	"""
	NaiveRAG implements a standard dense retrieval RAG pipeline.
	It retrieves context documents based purely on cosine similarity of text embeddings.
	"""

	def __init__(self, llm, embed_model, collection_name):
		super().__init__(llm, embed_model, collection_name)
		self.vector_store = get_qdrant_client()

	async def retrieve(self, query: str, top_k: int = Params.TOP_K) -> list[Chunk]:

		# generate vector embedding for the query string
		query_embedding = list(self.embed_model.embed([query]))[0]

		# query the Qdrant vector store for as many chunks as the frozen depth allows
		results = self.vector_store.query_points(
			collection_name=self.collection_name,
			query=query_embedding,
			using='dense',
			limit=top_k
		)

		return chunks_from_result(results)
