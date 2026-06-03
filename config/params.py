from dotenv import load_dotenv
import os 

class Params:
  GROQ_API_KEY = os.getenv("GROQ_API_KEY")