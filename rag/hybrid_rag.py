from core.base import BaseRAG


class HybridRAG(BaseRAG):


  async def retrieve(self, query):
    
    return await super().retrieve(query)