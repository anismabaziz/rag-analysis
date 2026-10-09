from abc import ABC, abstractmethod

from config.params import Params
from core.chunk import REFUSAL, Chunk, build_context
from core.prompt import ANSWER_PROMPT
from llama_index.llms.groq import Groq


def get_generation_llm():
    """The one generation model every architecture answers with.

    Architectures are only comparable while the generator is held still, so the model is named
    once here, with the temperature every run is measured at, and no architecture chooses its own.
    """
    return Groq(
        model=Params.GENERATION_MODEL,
        api_key=Params.GROQ_API_KEY,
        temperature=Params.GENERATION_TEMPERATURE,
    )


class BaseRAG(ABC):
    """
    Abstract base class defining the standard structure and lifecycle of a RAG pipeline.
    Concrete subclasses must implement the retrieve method.

    Every pipeline is built the same way, with the generation model, the encoder, and the name of
    the collection it reads, because that triple is what the registry hands to every architecture
    it knows about.
    """

    def __init__(self, llm, embed_model, collection_name):
        """
        Initializes the base RAG class with an LLM, embedding model, and the collection to read.
        """
        self.llm = llm
        self.embed_model = embed_model
        self.collection_name = collection_name

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = Params.TOP_K) -> list[Chunk]:
        """
        Retrieves scored chunks for the query, each carrying its text and its provenance.

        The depth is the frozen one, taken from the configuration rather than chosen here, because
        a depth an architecture could pick for itself is a second axis varying underneath the one
        the run is measuring. To be implemented by subclasses.
        """
        pass

    async def generate(self, query: str, context: str):
        """
        Queries the LLM using the retrieved context to answer the user question.
        """

        # use the LLM to predict/generate the answer based on the shared template
        return await self.llm.apredict(ANSWER_PROMPT, context=context, query=query)

    async def answer(self, query: str):
        """
        Orchestrates the entire RAG pipeline: Retrieve, Build Context, and Generate Answer.
        """
        print(f"[QUERY] {query}")

        # retrieve scored chunks, each carrying the score and provenance that selected it
        chunks = await self.retrieve(query)
        print("[RETRIEVED]")
        for chunk in chunks:
            print("#" * 60)
            print(chunk.report_line())
            print(chunk.text)

        # nothing retrieved means the model is never asked, so it is never asked to invent
        if not chunks:
            print(f"[REFUSAL]\n{REFUSAL}")
            return REFUSAL

        # combine chunks into a single context string
        context = build_context(chunks)
        print("[CONTEXT]\n", context)

        # call LLM to generate final response based on context
        answer = await self.generate(query, context)
        print("[ANSWER]\n", answer)

        return answer
