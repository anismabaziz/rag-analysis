from core.base import BaseRAG
from llama_index.core.vector_stores.types import VectorStoreQuery
import re


class HybridRAG(BaseRAG):
  """
  HybridRAG implements a hybrid search pipeline combining:
  1. Dense Vector Search (semantic similarity via embeddings).
  2. Sparse Search (keyword matching via BM25).
  Combined using Reciprocal Rank Fusion (RRF) to rank retrieved nodes.
  """

  def __init__(self, llm, embed_model, vector_store, bm25, doc_map, nodes):
    """
    Initializes the HybridRAG pipeline with the necessary search and LLM models.
    """
    super().__init__(llm, embed_model, vector_store)
    self.bm25 = bm25
    self.doc_map = doc_map
    self.nodes = nodes

  async def retrieve(self, query, top_k=10):
    """
    Retrieves the top_k most relevant nodes using both dense and sparse search,
    fused together using Reciprocal Rank Fusion (RRF).
    """
    
    # run semantic dense search against Qdrant
    dense = await self.dense_search(query, top_k)
    
    # run keyword sparse search using BM25
    sparse = self.sparse_search(query, top_k)


    print("\n=== DENSE ===")
    for n in dense:
      print(n.id_, n.text[:100])

    print("\n=== SPARSE ===")
    for n in sparse:
      print(n.id_, n.text[:100])
    
    # combine and re-rank the retrieved nodes using RRF
    return self.rrf(dense, sparse, top_k=5)

  def sparse_search(self, query: str, top_k=10):
    """
    Runs a keyword-based sparse search against the document set using BM25.
    """
    
    # tokenize the query terms
    tokenized_query = self._tokenize(query)
    
    # calculate BM25 scores for all documents/nodes
    scores = self.bm25.get_scores(tokenized_query)

    # keep only nodes with score > 0 to avoid retrieving random irrelevant nodes
    ranked = [
      (i, score)
      for i, score in enumerate(scores)
      if score > 0
    ]
    
    # sort the matching nodes by score in descending order
    ranked = sorted(ranked, key=lambda x: x[1], reverse=True)

    # retrieve and return the top_k nodes
    return [
      self.nodes[i]
      for i, _ in ranked[:top_k]
    ]

  async def dense_search(self, query, top_k=10):
    """
    Runs a semantic dense search against the Qdrant vector store.
    """
    # generate vector embedding for the query string
    query_embedding = await self.embed_model.aget_query_embedding(query)

    # query Qdrant vector store using the query embedding
    result = self.vector_store.query(
      VectorStoreQuery(
          query_embedding=query_embedding,
          similarity_top_k=top_k,
      )
    )

    # normalize the returned nodes and match them to our in-memory doc_map
    fixed_nodes = []
    for n in result.nodes:
      # Extract the identifier of the node
      node_id = getattr(n, "id_", None) or getattr(n, "node_id", None)
            
      # Map back to the in-memory node reference to sync document payloads
      if node_id in self.doc_map:
        fixed_nodes.append(self.doc_map[node_id])

    return fixed_nodes

  def rrf(self, dense, sparse, top_k=5, k=60):
    """
    Combines results of dense and sparse search using Reciprocal Rank Fusion (RRF).
    RRF prioritizes documents that appear high in either or both of the ranked results.
    """
    # map to hold accumulated RRF scores for each node ID
    scores = {}

    # accumulate reciprocal rank scores from the dense search results
    for rank, node in enumerate(dense):
        scores[node.id_] = scores.get(node.id_, 0) + 1 / (k + rank)

    # accumulate reciprocal rank scores from the sparse search results
    for rank, node in enumerate(sparse):
        scores[node.id_] = scores.get(node.id_, 0) + 1 / (k + rank)

    # sort nodes based on their combined RRF scores in descending order
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # map the sorted node IDs back to node objects and return the top_k
    return [
        self.doc_map[node_id]
        for node_id, _ in ranked[:top_k]
        if node_id in self.doc_map
    ]

  def _tokenize(self, text):
      """
      Helper function to tokenize and lower-case text into alpha-numeric words.
      """
      return re.findall(r"\w+", text.lower())

  async def build_context(self, nodes):
      """
      Concatenates the text of retrieved nodes to form the context for the LLM.
      """
      return "\n\n".join(node.text for node in nodes)