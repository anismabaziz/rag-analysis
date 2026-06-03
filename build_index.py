# build_index.py
from data.embed import get_embed_model
from data.loader import load_documents
from data.splitter import split_documents
from vector.store import get_vector_store, reset_vector_store
from data.utils import get_deterministic_node_id


def build_index():
    """
    Clears the existing Qdrant collection to prevent contamination, 
    loads raw documents, splits them into text nodes, embeds them, 
    and indexes them into the Qdrant vector store using deterministic UUIDs.
    """
    # Step 1: Delete existing collection to avoid duplicated or contaminated points
    print("[INDEX] resetting vector store collection...")
    try:
        reset_vector_store()
    except Exception as e:
        print(f"[INDEX] warning: could not reset collection: {e}")

    # Step 2: Initialize vector store and embedding models
    embedding_model = get_embed_model()
    vector_store = get_vector_store()

    # Step 3: Load document files from the target directory
    print("[INDEX] loading documents...")
    documents = load_documents("./documents")
    print(f"[INDEX] loaded {len(documents)} documents")
    
    # Step 4: Split documents into smaller text chunks/nodes
    nodes = split_documents(documents)
    print(f"[INDEX] split into {len(nodes)} nodes")

    # Step 5: Generate embeddings and deterministic UUIDs for each node
    print("[INDEX] generating embeddings and IDs...")
    for node in nodes:
        file_path = node.metadata.get("file_path", "unknown_path")
        node.id_ = get_deterministic_node_id(file_path, node.text)
        node.embedding = embedding_model.get_text_embedding(node.text)

    # Step 6: Write nodes and embeddings to Qdrant vector database using our deterministic IDs
    print(f"[INDEX] writing {len(nodes)} nodes to Qdrant...")
    vector_store.add(nodes, ids=[node.id_ for node in nodes])

    print(f"[INDEX] stored {len(nodes)} nodes successfully")
    return nodes


if __name__ == "__main__":
    # Execute build_index if the script is run directly
    build_index()