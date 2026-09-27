import argparse

from core.registry import DENSE, SPARSE, UnknownArchitecture, architecture
from data.embed import get_embed_model, get_sparse_embed_model
from data.loader import PDFLoader
from data.splitter import PDFSplitter
from vector.store import reset_vector_store, get_qdrant_client, create_collection
from qdrant_client import models
from glob import glob



def build_index(collection_name: str, vectors: tuple[str, ...] = (DENSE,)):
	"""
	Clears the existing Qdrant collection to prevent contamination, 
	loads raw documents, splits them into text nodes, embeds them under the vectors the
	architecture declares, and indexes them into the Qdrant vector store.
	"""

	# delete existing collection to avoid duplicated or contaminated points
	print("[INDEX] resetting vector store collection...")
	try:
			reset_vector_store(collection_name)
	except Exception as e:
			print(f"[INDEX] warning: could not reset collection: {e}")
	qdrant_client = get_qdrant_client()
	create_collection(qdrant_client, collection_name, vectors)


	# initialize vector store and embedding models
	print("[INDEX] Loading the embedding model & vector store...")
	embedding_model = get_embed_model() if DENSE in vectors else None
	sparse_model = get_sparse_embed_model() if SPARSE in vectors else None

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

			# embed the chunk under each vector this architecture retrieves on
			point_vector = {
				name: embed_vector(name, node['text'], embedding_model, sparse_model)
				for name in vectors
			}

			point = models.PointStruct(
				id=node['node_id'],
				vector=point_vector,
				payload={
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


def embed_vector(name: str, text: str, embedding_model, sparse_model):
	"""One chunk as the named vector, so an architecture is written and queried under the same
	names rather than under a flag meaning roughly that."""
	if name == SPARSE:
		return create_sparse_vector(text, sparse_model)

	return list(embedding_model.embed([text]))[0].tolist()


def create_sparse_vector(text: str, embedding_model):
	"""
	Creates a sparse vector from text using BM42
	"""
	embeddings = list(embedding_model.embed([text]))[0]

	sparse_vector = models.SparseVector(
		indices=embeddings.indices.tolist(),
		values=embeddings.values.tolist()
	)

	return sparse_vector


def main():
	"""
	Command line entry point so ingestion can be run without the top-level CLI.
	"""
	parser = argparse.ArgumentParser(
		description="Ingest the PDFs under ./documents for one registered architecture."
	)
	parser.add_argument(
		"--architecture",
		type=str,
		required=True,
		help="Name of a registered architecture. Its collection is replaced, so a rerun never mixes corpora."
	)

	args = parser.parse_args()

	try:
		ingested_for = architecture(args.architecture)
	except UnknownArchitecture as error:
		parser.error(str(error))

	ingested_for.ingest()


if __name__ == "__main__":
	main()
