from config.params import Params
from core.base import BaseRAG
from core.chunk import Chunk
from core.registry import SPARSE, register
from qdrant_client import models
from vector.store import chunks_from_result, get_qdrant_client
from data.embed import get_sparse_embed_model


@register(
    name="sparse",
    description="BM42 sparse vectors only, ranked by the sparse encoder's own scoring.",
    collection="rag_sparse",
    vectors=(SPARSE,),
)
class SparseRAG(BaseRAG):
    """Retrieval on the sparse signal alone, so its contribution can be told apart from the rest.

    Nothing here fuses a dense ranking in. A gain this architecture shows over dense-only is the
    sparse signal doing it on its own, which is the difference between a hybrid number and a claim
    about what caused it.
    """

    def __init__(self, llm, embed_model, collection_name):
        super().__init__(llm, embed_model, collection_name)
        self.client = get_qdrant_client()
        self.sparse_model = get_sparse_embed_model()

    async def retrieve(self, query: str, top_k: int = Params.TOP_K) -> list[Chunk]:

        # generate sparse embedding
        sparse_query = list(self.sparse_model.embed([query]))[0]

        # get results
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=models.SparseVector(
                indices=sparse_query.indices.tolist(),
                values=sparse_query.values.tolist(),
            ),
            using=SPARSE,
            limit=top_k,
        )

        return chunks_from_result(results)
