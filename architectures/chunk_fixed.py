from architectures.naive import NaiveRAG
from config.configuration import Chunker
from core.registry import DENSE, register


@register(
	name="chunk-fixed",
	description="Dense retrieval over fixed-size sentence chunks, for the chunking comparison.",
	collection="rag_chunk_fixed",
	vectors=(DENSE,),
	chunker=Chunker.variant("fixed", 512, 128),
)
class ChunkFixedRAG(NaiveRAG):
	"""Fixed-size arm of the chunking comparison: dense retrieval, sentence splitter.

	The retrieval is naive dense at the committed depth, so the only axis varying
	against the other chunking arms is how the corpus was cut.
	"""
