from llama_index.core import SimpleDirectoryReader


def load_documents(path: str):
  return SimpleDirectoryReader(input_dir=path).load_data() 