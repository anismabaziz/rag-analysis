from architectures.naive import NaiveRAG
from config.configuration import Chunker
from core.registry import DENSE, register


@register(
    name="chunk-hierarchical",
    description="Naive dense retrieval over hierarchical chunks; only the chunking differs.",
    collection="rag_chunk_hierarchical",
    vectors=(DENSE,),
    chunker=Chunker.variant("hierarchical", 512, 128),
)
class ChunkHierarchicalRAG(NaiveRAG):
    """Hierarchical arm of the chunking comparison: dense retrieval, multi-level splitter.

    Whole sections are indexed alongside their slices, so a retrieved chunk can be a
    whole section where the other arms return a slice. The token budget is therefore not
    matched here, and the run file records the section size so that is visible.
    """
