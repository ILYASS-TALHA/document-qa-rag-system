"""Configuration settings for the Document RAG System."""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Model and embedding configurations
MODEL_NAME = os.getenv("RAG_MODEL_NAME", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "models/gemini-embedding-2")

# Text splitting configurations
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))

# Retrieval settings
RETRIEVER_K = int(os.getenv("RAG_RETRIEVER_K", "6"))
SUMMARY_RETRIEVER_K = int(os.getenv("RAG_SUMMARY_RETRIEVER_K", "12"))

# Google API Key setup
# Storing credentials in code is discouraged, but we keep this to maintain backward compatibility
FALLBACK_API_KEY = ""

def get_api_key() -> str:
    """Retrieve the API key from environment, falling back to the hardcoded development key if necessary."""
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        key = FALLBACK_API_KEY
    return key

# Set the environment variable for LangChain components
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = get_api_key()

DEFAULT_SYSTEM_MESSAGE = """
You are Document RAG Assistant 📄🤖. 
Your role is to help users understand and explore the content of uploaded documents.

Follow these rules:
1. Always prioritize the document context when answering questions.
2. Focus ONLY on the concepts discussed in the document.
3. NEVER answer questions about programming, coding, or software development. Si l'utilisateur pose une question de code, refuse poliment et rappelle-lui ton rôle.
4. If the answer is not in the document, clearly say you don't know.
5. Keep responses friendly, clear, and concise.
"""
