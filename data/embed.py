from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def get_embed_model():
  return HuggingFaceEmbedding(model_name="BAAI/bge-m3")