from qdrant_client import QdrantClient, models


def get_qdrant_client():
	return QdrantClient(url="http://localhost:6333")


def create_collection(client: QdrantClient, collection_name: str, enable_hybrid: bool):
	"""
	Creates collection with appropriate config
	"""
	
	# dense vector config
	dense_config = models.VectorParams(
		size = 384, # matches all-MiniLM-L6-v2
		distance = models.Distance.COSINE
	)
	
	# sparse vector config 
	if enable_hybrid:
		sparse_config = models.SparseVectorParams(
			index=models.SparseIndexParams(on_disk=False)
		)

		client.create_collection(
			collection_name=collection_name,
			vectors_config={
				'dense': dense_config
			},
			sparse_vectors_config={
				'sparse': sparse_config
			}
		)

		print(f"[INFO] created hybrid collection: {collection_name}")
	
	else:
		client.create_collection(
			collection_name=collection_name,
			vectors_config={
				'dense': dense_config
			}
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

