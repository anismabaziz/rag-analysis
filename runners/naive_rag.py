from data.embed import get_embed_model
from rag.naive_rag import NaiveRAG
from vector.store import get_vector_store
from config.params import Params
from llama_index.llms.groq import Groq


async def run_naive(query: str):
    """
    Executes Naive RAG (dense vector search only) for a given query and prints the response.
    """
    # initialize the embedding model and vector store client
    embedding_model = get_embed_model()
    vector_store = get_vector_store()
    
    # set up the Groq LLM client
    llm = Groq(
        model="llama-3.3-70b-versatile",
        api_key=Params.GROQ_API_KEY
    )
    
    # instantiate the NaiveRAG pipeline
    naive_rag = NaiveRAG(llm, embedding_model, vector_store)
    
    # run retrieval, build context, and generate response
    print(f"\n--- Running Naive RAG (Dense Search only) ---")
    await naive_rag.answer(query)

