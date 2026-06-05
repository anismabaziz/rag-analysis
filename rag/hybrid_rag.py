from core.base import BaseRAG
from qdrant_client import QdrantClient
from llama_index.core.schema import TextNode


class HybridRAG(BaseRAG):
    
    def __init__(self, llm, embed_model):
        super().__init__(llm, embed_model)
        self.client = QdrantClient(url="http://localhost:6333")
        self.collection_name = "rag_hybrid"

    async def retrieve(self, query: str, top_k: int = 5):
        # Generate dense embedding (this returns a list already)
        query_embedding = await self.embed_model.aget_query_embedding(query)
        
        # query_points is the correct method [citation:4]
        # No .tolist() needed - it's already a list
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,  # Just pass the list directly
            limit=top_k,
            with_payload=True,
        )
        
        print(f"[DEBUG] Hybrid search results")
        
        nodes = []
        for point in results.points:
            print(f"Score: {point.score:.4f}")
            
            node = TextNode(
                text=point.payload.get("text", ""),
                id_=str(point.id),
                metadata=point.payload.get("metadata", {}),
            )
            node.metadata["score"] = point.score
            nodes.append(node)
        
        return nodes