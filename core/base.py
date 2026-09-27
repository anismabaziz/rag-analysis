from abc import ABC, abstractmethod

from config.params import Params
from core.chunk import REFUSAL, Chunk, build_context
from llama_index.core import PromptTemplate
from llama_index.llms.groq import Groq


def get_generation_llm():
	"""The one generation model every architecture answers with.

	Architectures are only comparable while the generator is held still, so the model is named
	once here and no architecture chooses its own.
	"""
	return Groq(model=Params.GENERATION_MODEL, api_key=Params.GROQ_API_KEY)


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
	async def retrieve(self, query: str) -> list[Chunk]:
		"""
		Retrieves scored chunks for the query, each carrying its text and its provenance.
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

Use ONLY the context below to answer the question. If the context does not contain the
answer, say so instead of guessing.

Context:
{context}

Question:
{query}

Answer:
""")

		# use the LLM to predict/generate the answer based on the template
		return await self.llm.apredict(
				template,
				context=context,
				query=query
		)

	async def answer(self, query: str):
		"""
		Orchestrates the entire RAG pipeline: Retrieve, Build Context, and Generate Answer.
		"""
		print(f"[QUERY] {query}")

		# retrieve scored chunks, each carrying the score and provenance that selected it
		chunks = await self.retrieve(query)
		print("[RETRIEVED]")
		for chunk in chunks:
			print("#"*60)
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
