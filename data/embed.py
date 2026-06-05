from fastembed import SparseTextEmbedding, TextEmbedding


def get_embed_model():
  return TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_sparse_embed_model():
  return SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")