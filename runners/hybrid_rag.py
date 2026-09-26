from rag.hybrid_rag import HybridRAG
from data.embed import get_embed_model
from config.params import Params
from llama_index.llms.groq import Groq




async def run_hybrid(query: str):
	"""
	Executes Hybrid RAG (dense + sparse search with RRF) for a given query and prints the response.
	"""
	# initialize the embedding model and vector store client
	embedding_model = get_embed_model()
	
	# set up the Groq LLM client
	llm = Groq(
		model='llama-3.3-70b-versatile',
		api_key=Params.GROQ_API_KEY
	)
	
	# instantiate the HybridRAG pipeline with BM42 sparse resources
	hybrid_rag = HybridRAG(llm, embedding_model, 'rag_hybrid')
	
	# run hybrid retrieval (dense + sparse fused via RRF) and generate answer
	print("[INFO] Running Hybrid RAG (Dense + Sparse Search)")
	await hybrid_rag.answer(query)