from config.params import Params
from core.base import BaseRAG
from core.chunk import Chunk
from core.registry import DENSE, SPARSE, register
from qdrant_client import models
from vector.store import chunks_from_result, get_qdrant_client
from data.embed import get_sparse_embed_model


@register(
    name="hybrid",
    description="Dense and BM42 sparse vectors, fused with reciprocal rank fusion.",
    collection="rag_hybrid",
    vectors=(DENSE, SPARSE),
)
class HybridRAG(BaseRAG):
    def __init__(self, llm, embed_model, collection_name):
        super().__init__(llm, embed_model, collection_name)
        self.client = get_qdrant_client()
        self.sparse_model = get_sparse_embed_model()

    async def retrieve(self, query: str, top_k: int = Params.TOP_K) -> list[Chunk]:

        # generate dense embedding
        dense_query = list(self.embed_model.embed([query]))[0]

        # generate sparse embedding
        sparse_query = list(self.sparse_model.embed([query]))[0]

        # get results
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_query.indices.tolist(),
                        values=sparse_query.values.tolist(),
                    ),
                    using="sparse",
                    limit=top_k,
                ),
                models.Prefetch(query=dense_query, using="dense", limit=top_k),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
        )

        return chunks_from_result(results)
