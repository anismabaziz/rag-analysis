from dotenv import load_dotenv
import os

load_dotenv()

class Params:
  GROQ_API_KEY = os.getenv("GROQ_API_KEY")
  QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
  NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
  NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
