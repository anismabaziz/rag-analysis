from abc import ABC, abstractmethod
from llama_index.core import PromptTemplate


class BaseRAG(ABC):
  """
  Abstract base class defining the standard structure and lifecycle of a RAG pipeline.
  Concrete subclasses must implement the retrieve and build_context methods.
  """

  def __init__(self, llm, embed_model, vector_store):
    """
    Initializes the base RAG class with an LLM, embedding model, and vector store.
    """
    self.llm = llm
    self.embed_model = embed_model
    self.vector_store = vector_store

  @abstractmethod
  async def retrieve(self, query: str):
    """
    Retrieves relevant document nodes for the query.
    To be implemented by subclasses.
    """
    pass

  @abstractmethod
  async def build_context(self, retrieved):
    """
    Builds the context string from retrieved nodes.
    To be implemented by subclasses.
    """
    pass

  async def generate(self, query: str, context: str):
    """
    Queries the LLM using the retrieved context to answer the user question.
    """
    # Step 1: Define prompt template restricting answer to provided context
    template = PromptTemplate("""
You are a helpful assistant.

Use ONLY the context below to answer the question.

Context:
{context}

Question:
{query}

Answer:
""")

    # Step 2: Use the LLM to predict/generate the answer based on the template
    return await self.llm.apredict(
        template,
        context=context,
        query=query
    )

  async def answer(self, query: str):
    """
    Orchestrates the entire RAG pipeline: Retrieve, Build Context, and Generate Answer.
    """
    print(f"\n[QUERY] {query}")

    # Step 1: Retrieve context document nodes
    retrieved = await self.retrieve(query)
    print("\n[RETRIEVED]")
    for r in retrieved:
        print("#"*60)
        print(r.text)

    # Step 2: Combine nodes into a single context string
    context = await self.build_context(retrieved)
    print("\n[CONTEXT]\n", context)

    # Step 3: Call LLM to generate final response based on context
    answer = await self.generate(query, context)
    print("\n[ANSWER]\n", answer)

    return answer