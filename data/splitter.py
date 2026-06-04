from llama_index.core.node_parser import SemanticSplitterNodeParser


def split_documents(documents, embed_model):
  splitter = SemanticSplitterNodeParser(
    embed_model=embed_model,
    buffer_size=1,
    breakpoint_percentile_threshold=95
  )

  return splitter.get_nodes_from_documents(documents)