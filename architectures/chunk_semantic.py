from architectures.naive import NaiveRAG
from config.configuration import Chunker
from core.registry import DENSE, register


@register(
    name="chunk-semantic",
    description="Dense retrieval over semantic chunks, for the chunking comparison.",
    collection="rag_chunk_semantic",
    vectors=(DENSE,),
    chunker=Chunker.variant("semantic", 512, 128),
)
class ChunkSemanticRAG(NaiveRAG):
    """Semantic arm of the chunking comparison: dense retrieval, breakpoint splitter.

    The size is the nominal token budget the fixed arm is cut at. Semantic splits vary
    in length around it, so the budget is not matched and the comparison says so.
    """
