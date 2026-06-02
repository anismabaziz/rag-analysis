from llama_index.core.node_parser import SentenceSplitter


def split_documents(documents):
  splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50
  )

  return splitter.get_nodes_from_documents(documents)