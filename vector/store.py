from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
		VectorParams,
		Distance,
		SparseVectorParams,
		SparseIndexParams,
)




def get_qdrant_client():
	return QdrantClient(url="http://localhost:6333")


def get_vector_store(collection_name: str = "rag_naive", enable_hybrid: bool = False):
	"""
	Gets Qdrant vector store for a certain collection 
	"""

	client = get_qdrant_client()

	# create collection if it doesn't exist
	if not client.collection_exists(collection_name):
		_create_collection(client, collection_name, enable_hybrid)
		
	
	return QdrantVectorStore(
		client=get_qdrant_client(),
		collection_name="rag_benchmark"
	)

def _create_collection(client: QdrantClient, collection_name: str, enable_hybrid: bool):
	"""
	Creates collection with appropriate config
	"""
	
	# dense vector config
	dense_config = VectorParams(
		size=384, # matches all-MiniLM-L6-v2
		distance=Distance.COSINE
	)
	
	# sparse vector config 
	if enable_hybrid:
		sparse_config = SparseVectorParams(
			index=SparseIndexParams(on_disk=False)
		)

		client.create_collection(
			collection_name=collection_name,
			vectors_config=dense_config,
			sparse_vectors_config={"text_sparse": sparse_config}
		)

		print(f"[INFO] created hybrid collection: {collection_name}")
	
	else:
		client.create_collection(
			collection_name=collection_name,
			vectors_config=dense_config
		)

		print(f"[INFO] created dense-only collection: {collection_name}")
	

	

def reset_vector_store(collection_name: str):
	""" 
	Resets a specific collection by name
	"""
	client = get_qdrant_client()


	if client.collection_exists(collection_name):
		client.delete_collection(collection_name)
		print(f"[INFO] deleted collection: {collection_name}")
	else:
		print(f"[INFO] collection '{collection_name}' does not exist")

