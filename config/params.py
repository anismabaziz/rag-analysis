from dotenv import load_dotenv
import os 

load_dotenv()

class Params:
  GROQ_API_KEY = os.getenv("GROQ_API_KEY")