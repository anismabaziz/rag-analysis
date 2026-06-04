from typing import List, Dict
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document
from llama_index.core.schema import MetadataMode
from llama_index.core.node_parser import (
  NodeParser, 
  SemanticSplitterNodeParser, 
  HierarchicalNodeParser, 
  SentenceSplitter
)

class PDFSplitter:
  """
  Bridge between your PDFLoader output and LlamaIndex's native chunking.
  Preserves all metadata (page, section, type) while leveraging LlamaIndex's
  semantic chunking capabilities.
  """

  def __init__(
      self,
      chunking_strategy = "semantic",
      embedding_model = "sentence-transformers/all-MiniLM-L6-v2",
      semantic_threshold: int = 95,
      hierarchical_chunk_sizes: List[int] = [2048, 512, 128],
      chunk_size: int = 512,
      chunk_overlap: int = 128
      ):
    
    # setup chunk params
    self.chunking_strategy = chunking_strategy
    self.chunk_size = chunk_size
    self.chunk_overlap = chunk_overlap
    
    # setup embeddings
    self.embed_model = HuggingFaceEmbedding(model_name=embedding_model)
    
    # initialize the appropriate node parser
    self.node_parser = self._create_node_parser(
        semantic_threshold, hierarchical_chunk_sizes
    )


  def process(self, elements: List[Dict]):
    """
    Complete splitting pipeline execution
    """

    documents = self._elements_to_llama_documents(elements)

    nodes = self._chunk_documents(documents)

    return nodes



  def _create_node_parser(self, semantic_threshold: int, hierarchical_chunk_sizes: List[int]) -> NodeParser:
    """
    Creates Llama-index node parser based on a certain strategy
    """

    if self.chunking_strategy == "semantic":
      return SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=semantic_threshold,
        embed_model=self.embed_model,
        include_metadata=True
      )
    
    if self.chunking_strategy == "hierarchical":
      return HierarchicalNodeParser.from_defaults(
        chunk_sizes=hierarchical_chunk_sizes,
        chunk_overlap=self.chunk_overlap,
        include_metadata=True
      )
    

    return SentenceSplitter(
      chunk_size=self.chunk_size,
      chunk_overlap=self.chunk_overlap,
      include_metadata=True,
      paragraph_separator="\n\n"
      )
  

  def _elements_to_llama_documents(self, elements: List[Dict]) -> List[Document]:
    """
    Convert structured PDFLoader elements to LlamaIndex documents
    """

    documents = []

    # group by section
    sections = {}
    for element in elements:
      section = element.get("current_section", "Unknown")
      
      if section not in sections:
        sections[section] = []
      
      sections[section].append(element)

    # create one document per section
    for s_name, s_elements in sections.items():

      # combine text within sections
      text_parts = []
      page_numbers = set()
      element_types = set()

      for element in s_elements:
        text_parts.append(element["text"])
        if element.get("metadata", {}).get("page"):
          page_numbers.add(element["metadata"]["page"])

        element_types.add(element["type"])

      
      # create document with rich metadata
      doc = Document(
        text = "\n\n".join(text_parts),
        metadata = {
          "section": s_name,
          "pages" : list(page_numbers),
          "element_types": list(element_types),
          "num_elements": len(s_elements),
          "source": "pdf"
        },
        excluded_embed_metadata_keys=["element_types"],
        excluded_llm_metadata_keys=["num_elements"],
      )

      documents.append(doc)

    return documents
  

  def _chunk_documents(self, documents: List[Document]):
    """
    Chunk documents using the selected LlamaIndex strategy returns nodes ready for embedding
    """
    all_nodes = []


    for doc in documents:
      # get nodes from the parser
      nodes = self.node_parser.get_nodes_from_documents([doc])

      for node in nodes:
        all_nodes.append({
          "text": node.get_content(metadata_mode=MetadataMode.NONE),
          "node_id": node.node_id,
          "metadata": node.metadata,
          "embedding": None,
          "relationships": {
              "parent_id": getattr(node, 'parent_node', None),
              "child_ids": getattr(node, 'child_nodes', []),
          } if hasattr(node, 'parent_node') else {},
        })

    return all_nodes






