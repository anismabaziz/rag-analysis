from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def get_embed_model():
  return HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")