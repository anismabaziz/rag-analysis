from abc import ABC, abstractmethod
from llama_index.core import PromptTemplate

class BaseRAG(ABC):

  def __init__(self, llm, embed_model, vector_store):
    self.llm = llm
    self.embed_model = embed_model
    self.vector_store = vector_store


  @abstractmethod
  async def retrieve(self, query: str):
    pass

    
  @abstractmethod
  async def build_context(self, retrieved):
    pass


  async def generate(self, query: str, context: str):
    template = PromptTemplate("""
  You are a helpful assistant.

  Use ONLY the context below to answer the question.

  Context:
  {context}

  Question:
  {query}

  Answer:
  """)

    return await self.llm.apredict(
          template,
          context=context,
          query=query
    )

  async def answer(self, query: str):
    print(f"\n[QUERY] {query}")

    retrieved = await self.retrieve(query)
    print("\n[RETRIEVED]")
    for r in retrieved:
        print(r.text)

        
    context = await self.build_context(retrieved)
    print("\n[CONTEXT]\n", context)

    answer = await self.generate(query, context)
    print("\n[ANSWER]\n", answer)

    return answer