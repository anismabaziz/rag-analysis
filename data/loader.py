from llama_index.core import SimpleDirectoryReader


def load_documents(path: str):
  return SimpleDirectoryReader(path).load_data() 