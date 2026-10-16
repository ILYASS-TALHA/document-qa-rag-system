"""Module for loading, splitting, formatting, and embedding documents."""
import os
import tempfile
from typing import List
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL

def load_document(file_name: str, file_bytes: bytes) -> List[Document]:
    """Save an uploaded file temporarily, parse it into langchain Documents, and clean up."""
    suffix = f".{file_name.split('.')[-1]}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(file_bytes)
        tmp_file_path = tmp_file.name

    try:
        if file_name.lower().endswith(".pdf"):
            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()
            # Filter out pages with no extractable text
            docs = [doc for doc in docs if doc.page_content and doc.page_content.strip()]
            # Override the "source" metadata to use original file name
            for doc in docs:
                doc.metadata["source"] = file_name
        else:  # txt file
            encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
            docs = None
            last_error = None
            for enc in encodings_to_try:
                try:
                    loader = TextLoader(tmp_file_path, encoding=enc)
                    docs = loader.load()
                    break
                except Exception as enc_err:
                    last_error = enc_err
                    continue
            if docs is None:
                raise ValueError(
                    f"Could not read the text file {file_name} (tried encodings: {', '.join(encodings_to_try)}). "
                    f"Last error: {last_error}"
                )
            # Override the "source" metadata
            for doc in docs:
                doc.metadata["source"] = file_name
        return docs
    finally:
        if os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except OSError:
                pass

def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into semantic text chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError("The documents were loaded but produced no text chunks.")
    return chunks

def create_vector_store(chunks: List[Document]) -> FAISS:
    """Build a FAISS vector store from document chunks using embeddings."""
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    return FAISS.from_documents(chunks, embeddings)

def format_docs(retrieved_docs: List[Document]) -> str:
    """Format retrieved documents into a single context string with source metadata."""
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        header = f"[Passage {i}" + (f" - Page {page + 1}" if page != "" else "") + "]"
        context_parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(context_parts)
