from rag.naive_rag import NaiveRAG
from llama_index.llms.groq import Groq
from data.embed import get_embed_model
from data.loader import load_documents
from data.splitter import split_documents
from vector.store import get_vector_store
from dotenv import load_dotenv
import os 
import asyncio


load_dotenv()



def init():
  embedding_model = get_embed_model()
  vector_store = get_vector_store()

  print(f"[INIT] create embedding model and vector store")

  doc_path = "./documents"
  documents = load_documents(doc_path)

  print(f"[DOC] read documents: {len(documents)} documents")

  nodes = split_documents(documents)

  print(f"[SPLIT] split document to nodes: {len(nodes)} nodes")

  for node in nodes:
      text = node.text
      node.embedding = embedding_model.get_text_embedding(text)

  print("[EMBED] embeddings created")

  vector_store.add(nodes)

  print(f"[DB] added {len(nodes)} nodes to vector store")

async def test_naive():
  # test naive rag
  llm = Groq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY")
  )
  embedding_model = get_embed_model()
  vector_store = get_vector_store()

  naive_rag = NaiveRAG(llm, embedding_model, vector_store)

  await naive_rag.answer("What is the architecture of a transformer?")


if __name__ == "__main__":
  asyncio.run(test_naive())


