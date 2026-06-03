import uuid

def get_deterministic_node_id(file_path: str, text: str) -> str:
  """
  Generates a stable and valid UUID string for a document node based on file path and text.
  """

  # fixed namespace UUID for this project
  namespace = uuid.UUID("3d813cbb-47fb-32ba-91df-831e1593ac9e")
  
  # Generate a deterministic UUID based on the namespace and the node context
  return str(uuid.uuid5(namespace, f"{file_path}:{text}"))
