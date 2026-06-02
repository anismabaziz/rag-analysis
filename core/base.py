from abc import ABC, abstractmethod


class BaseRAG(ABC):

  def __init__(self, llm, embed_model, vector_store):
    self.llm = llm
    self.embed_model = embed_model
    self.vector_store = vector_store


  @abstractmethod
  async def retrieve(self, query: str):
    pass

  async def build_context(self, docs):
    return "\n\n".join([d.text for d in docs])
  
  async def generate(self, query: str, context: str):
    prompt = f"""
You are a helpful assistant.

Context:
{context}

Question:
{query}

Answer:
"""
    return await self.llm.apredict(prompt)
  

  async def answer(self, query: str):
    docs = await self.retrieve(query)
    context = await self.build_context(docs)
    return await self.generate(query, context)
