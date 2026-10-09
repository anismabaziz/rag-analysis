from config.params import Params
from core.base import BaseRAG
from core.chunk import Chunk
from core.registry import DENSE, SPARSE, register
from qdrant_client import models
from vector.store import chunks_from_result, get_qdrant_client
from data.embed import get_rerank_model, get_sparse_embed_model


@register(
    name="rerank",
    description="Hybrid retrieval over a wider candidate set, ordered by a cross-encoder.",
    collection="rag_rerank",
    vectors=(DENSE, SPARSE),
)
class RerankRAG(BaseRAG):
    """Hybrid first, cross-encoder second, so the comparison has a two-stage retriever in it.

    The first stage is the hybrid ranking over a wider candidate set than the committed
    depth. The second stage scores each candidate against the query with a cross-encoder
    and keeps the committed depth. What comes back is the same chunk type at the same
    depth as every other architecture, differing only in ordering, which is what makes
    a gain over hybrid attributable to the reranking rather than to a different setup.
    """

    def __init__(self, llm, embed_model, collection_name):
        super().__init__(llm, embed_model, collection_name)
        self.client = get_qdrant_client()
        self.sparse_model = get_sparse_embed_model()
        self._reranker = None

    def _reranker_model(self):
        """The cross-encoder, loaded on first use rather than at construction.

        Loading it here keeps building the pipeline free, so listing architectures or
        running another architecture never downloads a model only this one queries with.
        """
        if self._reranker is None:
            self._reranker = get_rerank_model()
        return self._reranker

    async def retrieve(self, query: str, top_k: int = Params.TOP_K) -> list[Chunk]:
        # the first stage fetches wider than the committed depth, so the reranker has
        # something to promote that the depth alone would have cut off
        candidates_limit = max(Params.RERANK_CANDIDATES, top_k)

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
                    limit=candidates_limit,
                ),
                models.Prefetch(
                    query=dense_query, using="dense", limit=candidates_limit
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
        )

        candidates = chunks_from_result(results)

        return rerank_chunks(query, candidates, self._reranker_model(), top_k)


def rerank_chunks(
    query: str, candidates: list[Chunk], reranker, top_k: int
) -> list[Chunk]:
    """Order candidates by cross-encoder score and keep the committed depth.

    Each returned chunk keeps its text and provenance and carries the rerank score,
    so what differs from the hybrid ranking is the ordering and nothing else.
    """
    if not candidates:
        return []

    scores = list(reranker.rerank(query, [chunk.text for chunk in candidates]))
    ordered = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)

    return [
        Chunk(text=chunk.text, score=float(score), provenance=chunk.provenance)
        for chunk, score in ordered[:top_k]
    ]
