# 📄 Document RAG Assistant

### Chat with Your Documents Using Retrieval-Augmented Generation (RAG)

An AI-powered document question-answering application that allows users to upload documents and ask questions about their content using natural language. Built with Streamlit, LangChain, FAISS, and Google Gemini.

## 🚀 Overview

**Document RAG Assistant** is a document question-answering system based on Retrieval-Augmented Generation (RAG). It processes uploaded PDF and TXT files, extracts their text, splits the content into smaller chunks, and retrieves relevant information to help generate context-aware answers.

The goal is to make document analysis easier by allowing users to interact with their documents through a conversational interface.

## ✨ Features

- 📁 **Document Upload:** Upload PDF and TXT documents.
- 🤖 **AI-Powered Answers:** Use Google Gemini models to answer questions about document content.
- 🔍 **Semantic Search:** Retrieve relevant text chunks using FAISS vector similarity search.
- 🧠 **RAG Pipeline:** Combine document retrieval with language-model generation.
- 💬 **Interactive Chat:** Ask questions in natural language.
- 📚 **Document Processing:** Extract and split text into manageable chunks.
- 📊 **Progress Tracking:** Follow document processing progress.
- 💾 **Chat Export:** Export conversation history.
- 🔄 **Embedding Fallback:** Support for alternative Hugging Face embeddings when configured.

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python 3.11+ |
| User Interface | Streamlit |
| LLM Integration | Google Gemini API |
| RAG Framework | LangChain |
| Vector Database | FAISS |
| Embeddings | Google Generative AI Embeddings |
| Document Loaders | PyPDFLoader, TextLoader |
| Environment Configuration | Environment variables |
| Containerization | Docker support, if configured |

## 🏗️ System Architecture

The application follows a typical Retrieval-Augmented Generation workflow:

1. **Document Upload:** The user uploads a PDF or TXT file.
2. **Text Extraction:** The application extracts the document's textual content.
3. **Text Chunking:** The content is divided into smaller, overlapping chunks.
4. **Embedding Generation:** Text chunks are converted into vector representations.
5. **Vector Indexing:** FAISS stores the vectors for similarity-based retrieval.
6. **Question Processing:** The user's question is used to retrieve relevant chunks.
7. **Answer Generation:** Google Gemini generates an answer using the retrieved context.
8. **Response Display:** The answer is displayed in the Streamlit interface.

## 📋 Prerequisites

Before running the application, make sure you have:

- Python 3.11 or later.
- A Google Gemini API key.
- An internet connection.
- The project dependencies installed.

Get an API key from [Google AI Studio](https://aistudio.google.com/apikey).

## ⚙️ Installation and Setup

### 1. Download the project

Download or clone this repository from GitHub, then open the project directory.

### 2. Install dependencies

If the project contains `pyproject.toml` and `uv.lock`, install its dependencies using [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Alternatively, if the project provides a compatible `requirements.txt`, install the dependencies with:

```bash
pip install -r requirements.txt
```

Use the installation method that matches the dependency files included in the repository.

### 3. Configure the API key

Create a `.env` file in the project root if the application reads environment variables from this file.

Add your own key:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

Replace the placeholder with your actual API key.

**Security:** Never commit your real `.env` file or API key to GitHub. Use `.env.example` to document the required configuration without exposing secrets.

If the application supports entering the API key through its sidebar, you can use that option instead.

### 4. Run the application

Start the Streamlit application:

```bash
streamlit run app.py
```

### 5. Open the application

Open your browser and navigate to:

http://localhost:8501

## 📖 How to Use

1. Open the application.
2. Enter your Google Gemini API key using the supported configuration method.
3. Upload a PDF or TXT document.
4. Start document processing.
5. Ask questions about the document in natural language.
6. Review the generated answers.
7. Export your conversation if the feature is available in your version.

### Example Questions

- What is the main topic of this document?
- Summarize the main findings.
- Explain the methodology used in this research.
- What are the key recommendations?
- What does the document say about machine learning?

## 📂 Supported Documents

The application is designed to work with:

- **PDF:** Text-based PDF documents.
- **TXT:** Plain-text files.

Scanned PDFs may require OCR support, which is not included in the documented functionality.

## 🔐 Security and Privacy

- Keep API keys and credentials private.
- Do not upload `.env` files containing secrets to a public repository.
- Exclude virtual environments, caches, and temporary files from version control.
- Be careful when uploading confidential or personal documents to an AI-powered application.

## ⚠️ Limitations

- Document processing time depends on file size and complexity.
- Google Gemini API usage is subject to applicable quotas and rate limits.
- Complex PDF layouts may affect text extraction.
- Image-based PDFs may not work without OCR.
- Answer quality depends on document content, retrieval quality, and the selected language model.

## 🔮 Future Improvements

Potential improvements include:

- Support for DOCX, HTML, and Markdown files.
- OCR for scanned documents.
- Multi-document question answering.
- Multilingual document processing.
- Improved source citations and answer traceability.
- More embedding model options.
- Advanced retrieval and filtering.
- Cloud storage integration.

## 🤝 Contributing

Contributions, suggestions, and bug reports are welcome.

To suggest an improvement, open an issue in the repository. If you want to contribute code, submit a pull request.

## 📄 License

Refer to the repository's `LICENSE` file for licensing terms. Do not claim a specific license unless that file is included and confirms it.

## 👨‍💻 Author

**ILYASS TALHA**

GitHub: https://github.com/ILYASS-TALHA

---

*Built with Python, Streamlit, LangChain, FAISS, and Google Gemini.*
