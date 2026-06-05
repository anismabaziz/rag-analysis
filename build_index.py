from data.embed import get_embed_model, get_sparse_embed_model
from data.loader import PDFLoader
from data.splitter import PDFSplitter
from vector.store import reset_vector_store, get_qdrant_client, create_collection
from qdrant_client import models
from glob import glob


def build_index(collection_name: str, enable_hybrid: bool):
	"""
	Clears the existing Qdrant collection to prevent contamination, 
	loads raw documents, splits them into text nodes, embeds them, 
	and indexes them into the Qdrant vector store.
	"""

	# delete existing collection to avoid duplicated or contaminated points
	print("[INDEX] resetting vector store collection...")
	try:
			reset_vector_store(collection_name)
	except Exception as e:
			print(f"[INDEX] warning: could not reset collection: {e}")
	qdrant_client = get_qdrant_client()
	create_collection(qdrant_client, collection_name, enable_hybrid)


	# initialize vector store and embedding models
	print("[INDEX] Loading the embedding model & vector store...")
	embedding_model = get_embed_model()

	# initialize loader and splitter
	print("[INDEX] loading documents...")
	loader = PDFLoader(
		infer_table_structure=True,
		fallback_strategy='hi_res'
	)
	splitter = PDFSplitter(
		chunking_strategy='semantic',
		chunk_size=512,
		chunk_overlap=128
	)

	# find all pdfs
	pdf_files = glob('./documents/**/*.pdf', recursive=True)
	print(f"[INDEX] found {len(pdf_files)} PDF files")

	points = []

	for pdf_path in pdf_files:
		print(f"[INDEX] processing: {pdf_path}...")
		
		# load elements from pdf
		elements = loader.load(pdf_path)
		print(f"  loaded {len(elements)} elements")
			
		# split into nodes
		nodes = splitter.process(elements)
		print(f"  created {len(nodes)} nodes")

		for node in nodes:

			# generate dense embeddings
			dense_embedding = list(embedding_model.embed([node['text']]))[0]

			# create point
			point = None
			if enable_hybrid:
				# generate sparse embeddings
				sparse_embedding = create_sparse_vector(node['text'])

				point = models.PointStruct(
					id=node['node_id'],
					vector= {
						'dense': dense_embedding.tolist(),
						'sparse': sparse_embedding,
					},
					payload= {
						'text': node['text'],
						'metadata': node['metadata']
					}
				)
			else:

				point = models.PointStruct(
					id=node['node_id'],
					vector= {
						'dense': dense_embedding.tolist(),
					},
					payload= {
						'text': node['text'],
						'metadata': node['metadata']
					}
				)

			# add source file
			point.payload['metadata']['source_file'] = pdf_path

			points.append(point)

	print(f"[INDEX] total nodes: {len(points)}")


	# add to vector store
	print("[INDEX] indexing points to Qdrant...")
	qdrant_client.upsert(
		collection_name=collection_name,
		points=points
	)

	print(f"[INDEX] stored {len(points)} points successfully")
	
	return points


def create_sparse_vector(text: str):
	"""
	Creates a sparse vector from text using SPLADE
	"""

	embedding_model = get_sparse_embed_model()
	embeddings = list(embedding_model.embed([text]))[0]

	sparse_vector = models.SparseVector(
		indices=embeddings.indices.tolist(),
		values=embeddings.values.tolist()
	)

	return sparse_vector



if __name__ == "__main__":
	build_index()