from ..data.embed import get_embed_model
from ..data.splitter import split_documents
from ..data.loader import load_documents
from ..rag.hybrid_rag import HybridRAG
from ..vector.store import get_vector_store
from ..data.utils import get_deterministic_node_id
from ..config.params import Params
from llama_index.llms.groq import Groq
from rank_bm25 import BM25Okapi
import re




def init():
    """
    Initializes the in-memory resources needed for RAG search.
    Loads documents, splits them, assigns deterministic UUIDs,
    and initializes the BM25 index and a mapping of node IDs to node objects.
    """
    # load embedding model and vector store client
    embedding_model = get_embed_model()
    vector_store = get_vector_store()

    # load documents from the local documents directory
    print("[INIT] Loading documents from ./documents...")
    documents = load_documents("./documents")
    
    # split documents into smaller text chunks/nodes
    print("[INIT] Splitting documents into text nodes...")
    nodes = split_documents(documents)

    # assign deterministic UUIDs to nodes using our utility
    print("[INIT] Generating deterministic node IDs...")
    for node in nodes:
        file_path = node.metadata.get("file_path", "unknown_path")
        node.id_ = get_deterministic_node_id(file_path, node.text)

    # tokenize the nodes for the BM25 sparse search index
    tokenized_docs = [
        re.findall(r"\w+", node.text.lower())
        for node in nodes
    ]

    # initialize the BM25 search engine with the tokenized documents
    print("[INIT] Creating BM25 index...")
    bm25 = BM25Okapi(tokenized_docs)
    
    # create a map from node ID to node object for quick retrieval in RRF fusion
    doc_map = {node.id_: node for node in nodes}

    print(f"[INIT] Prepared {len(nodes)} nodes for BM25 and Hybrid mapping.")
    return {
        "vector_store": vector_store,
        "embedding_model": embedding_model,
        "bm25": bm25,
        "doc_map": doc_map,
        "nodes": nodes
    }

async def run_hybrid(query: str):
    """
    Executes Hybrid RAG (dense + sparse search with RRF) for a given query and prints the response.
    """

    # load and process documents to construct the BM25 index and mapping
    res = init()
    
    # set up the Groq LLM client
    llm = Groq(
        model="llama-3.3-70b-versatile",
        api_key=Params.GROQ_API_KEY
    )
    
    # instantiate the HybridRAG pipeline with BM25 resources
    hybrid_rag = HybridRAG(
        llm, 
        res["embedding_model"], 
        res["vector_store"], 
        res["bm25"],
        res["doc_map"],
        res["nodes"]
    )
    
    # run hybrid retrieval (dense + sparse fused via RRF) and generate answer
    print(f"\n--- Running Hybrid RAG (Dense + BM25 Sparse Search + RRF) ---")
    await hybrid_rag.answer(query)