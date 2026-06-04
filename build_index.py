from data.embed import get_embed_model
from data.loader import load_documents
from data.splitter import split_documents
from data.filters import filter_nodes
from vector.store import get_vector_store, reset_vector_store
from data.utils import get_deterministic_node_id


def build_index():
    """
    Clears the existing Qdrant collection to prevent contamination, 
    loads raw documents, splits them into text nodes, embeds them, 
    and indexes them into the Qdrant vector store using deterministic UUIDs.
    """
    # delete existing collection to avoid duplicated or contaminated points
    print("[INDEX] resetting vector store collection...")
    try:
        reset_vector_store()
    except Exception as e:
        print(f"[INDEX] warning: could not reset collection: {e}")

    # initialize vector store and embedding models
    embedding_model = get_embed_model()
    vector_store = get_vector_store()

    # load document files from the target directory
    print("[INDEX] loading documents...")
    documents = load_documents("./documents")
    print(f"[INDEX] loaded {len(documents)} documents")
    
    # split documents into smaller text chunks/nodes
    nodes = split_documents(documents, embed_model=embedding_model)
    print(f"[INDEX] split into {len(nodes)} nodes")

    # filter low quality chunks
    print("[INDEX] filtering low-quality chunks...")
    nodes = filter_nodes(nodes)
    print(f"[INDEX] remaining {len(nodes)} nodes after filtering")

    # generate embeddings and deterministic UUIDs for each node
    print("[INDEX] generating embeddings and IDs...")
    for node in nodes:
        file_path = node.metadata.get("file_path", "unknown_path")
        node.id_ = get_deterministic_node_id(file_path, node.text)
        node.embedding = embedding_model.get_text_embedding(node.text)

    # write nodes and embeddings to Qdrant vector database using our deterministic IDs
    print(f"[INDEX] writing {len(nodes)} nodes to Qdrant...")
    vector_store.add(nodes, ids=[node.id_ for node in nodes])

    print(f"[INDEX] stored {len(nodes)} nodes successfully")
    return nodes


if __name__ == "__main__":
    # execute build_index if the script is run directly
    build_index()