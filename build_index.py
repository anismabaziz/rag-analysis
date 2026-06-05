from data.embed import get_embed_model
from data.loader import PDFLoader
from data.splitter import PDFSplitter
from vector.store import get_vector_store, reset_vector_store
from llama_index.core.schema import TextNode
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
			reset_vector_store()
	except Exception as e:
			print(f"[INDEX] warning: could not reset collection: {e}")

	# initialize vector store and embedding models
	embedding_model = get_embed_model()
	vector_store = get_vector_store(collection_name, enable_hybrid)

	# initialize loader and splitter
	print("[INDEX] loading documents")
	loader = PDFLoader(
		infer_table_structure=True,
		fallback_strategy="hi_res"
	)
	splitter = PDFSplitter(
		chunking_strategy="semantic",
		chunk_size=512,
		chunk_overlap=128
	)

	# find all pdfs
	pdf_files = glob("./documents/**/*.pdf", recursive=True)
	print(f"[INDEX] found {len(pdf_files)} PDF files")

	all_nodes = []

	for pdf_path in pdf_files:
		print(f"[INDEX] processing: {pdf_path}")
		
		# load elements from pdf
		elements = loader.load(pdf_path)
		print(f"  loaded {len(elements)} elements")
			
		# split into nodes
		nodes = splitter.process(elements)
		print(f"  created {len(nodes)} nodes")

		# convert to LlamaIndex TextNodes
		for node in nodes:
			text_node = TextNode(
				text=node["text"],
				id_=node["node_id"],
				metadata=node["metadata"]
			)

			# add source file
			text_node.metadata["source_file"] = pdf_path

			# generate emebeddings
			text_node.embedding = embedding_model.get_text_embedding(text_node.text)

			all_nodes.append(text_node)

	print(f"[INDEX] total nodes: {len(all_nodes)}")

	# add to vector store
	print("[INDEX] indexing nodes to Qdrant")
	vector_store.add(all_nodes)

	print(f"[INDEX] stored {len(all_nodes)} nodes successfully")
	
	return all_nodes

if __name__ == "__main__":
	build_index()