"""Module for handling Chat LLM interactions and LangChain chains."""
import json
import streamlit as st
from typing import List, Tuple, Dict, Any, Generator
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from src.config import MODEL_NAME, RETRIEVER_K, SUMMARY_RETRIEVER_K
from src.document_processor import format_docs

@st.cache_resource()
def get_chat_model() -> ChatGoogleGenerativeAI:
    """Get the cached Gemini chat model."""
    return ChatGoogleGenerativeAI(model=MODEL_NAME)

def analyze_document_genre_and_questions(chat_model: ChatGoogleGenerativeAI, first_chunk_content: str) -> Dict[str, Any]:
    """Analyze the document's genre and generate 3 suggested questions in French."""
    analysis_prompt = f"""
    Analyze the following excerpt from the uploaded documents:
    ---
    {first_chunk_content[:1500]}
    ---
    Based on this text, do two things:
    1. Identify the genre/type of the document (e.g., Article, Course, Medical Report, Technical Manual, etc.) in French.
    2. Generate 3 suggested questions in French that a user might ask about this document.
    
    Return ONLY a JSON object with this exact format:
    {{
        "genre": "Genre du document",
        "questions": ["Question 1", "Question 2", "Question 3"]
    }}
    """
    try:
        analysis_result = chat_model.invoke(analysis_prompt)
        result_text = analysis_result.content
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].strip()
        
        return json.loads(result_text)
    except Exception:
        return {
            "genre": "Document non identifié",
            "questions": ["De quoi parle ce document ?", "Quels sont les points clés ?"]
        }

def rag_retrieve_context(vector_store, query: str, k: int = RETRIEVER_K) -> Tuple[str, List[Document]]:
    """Retrieve relevant context and document objects from the vector store."""
    custom_retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )
    retrieved_docs = custom_retriever.invoke(query)
    context = format_docs(retrieved_docs)
    return context, retrieved_docs

def generate_summary_stream(chat_model: ChatGoogleGenerativeAI, vector_store) -> Tuple[Generator, List[Dict[str, Any]]]:
    """Retrieve context and generate summary prompt stream."""
    summary_queries = [
        "Quel est le sujet principal et les thèmes abordés dans ce document ?",
        "Quels sont les points clés, arguments et conclusions de ce document ?",
        "Quelles sont les données importantes, chiffres et faits marquants ?",
    ]
    
    all_contexts = []
    seen_content = set()
    for query in summary_queries:
        _, docs = rag_retrieve_context(vector_store, query, k=SUMMARY_RETRIEVER_K)
        if docs:
            for doc in docs:
                content_hash = hash(doc.page_content[:200])
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    all_contexts.append(doc)
                    
    if not all_contexts:
        raise ValueError("Impossible de récupérer le contexte du document.")
        
    rag_context = format_docs(all_contexts)
    summary_prompt = f"""Tu es un assistant spécialisé dans la synthèse de documents.

Voici les passages les plus pertinents récupérés du document via recherche sémantique (RAG) :
---
{rag_context}
---

En te basant UNIQUEMENT sur ces passages récupérés, génère un **résumé structuré et détaillé** en français.
Le résumé doit inclure :
- 📌 **Sujet principal** : De quoi parle le document
- 🎯 **Points clés** : Les idées principales (sous forme de liste)
- 📊 **Informations importantes** : Données, chiffres ou faits marquants
- 🔍 **Conclusion** : La conclusion ou le message principal du document
- 📄 **Sources** : Indique les numéros de passages utilisés pour chaque point

Sois clair, précis et utilise des emojis pour structurer la réponse.
Ne fabrique AUCUNE information qui n'est pas dans les passages ci-dessus."""

    # Build sources metadata
    sources = []
    for doc in all_contexts:
        sources.append({
            "source_name": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", 0),
            "content": doc.page_content
        })
        
    return chat_model.stream(summary_prompt), sources

def generate_explanation_stream(chat_model: ChatGoogleGenerativeAI, vector_store) -> Tuple[Generator, List[Dict[str, Any]]]:
    """Retrieve context and generate explanation prompt stream."""
    explanation_queries = [
        "Quels sont les concepts et termes techniques principaux de ce document ?",
        "Comment les idées et sections du document sont-elles liées entre elles ?",
        "Quel est le contexte et le domaine de ce document ?",
    ]
    
    all_contexts = []
    seen_content = set()
    for query in explanation_queries:
        _, docs = rag_retrieve_context(vector_store, query, k=SUMMARY_RETRIEVER_K)
        if docs:
            for doc in docs:
                content_hash = hash(doc.page_content[:200])
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    all_contexts.append(doc)
                    
    if not all_contexts:
        raise ValueError("Impossible de récupérer le contexte du document.")
        
    rag_context = format_docs(all_contexts)
    explanation_prompt = f"""Tu es un professeur pédagogue expert en vulgarisation.

Voici les passages les plus pertinents récupérés du document via recherche sémantique (RAG) :
---
{rag_context}
---

En te basant UNIQUEMENT sur ces passages récupérés, génère une **explication détaillée et accessible** en français.
L'explication doit :
- 🧠 **Contexte** : Expliquer le contexte et le domaine du document
- 📚 **Explication des concepts** : Définir et expliquer les termes techniques ou complexes de manière simple
- 🔗 **Relations entre les idées** : Montrer comment les différentes parties du document sont liées
- 💡 **Exemples et analogies** : Utiliser des exemples concrets pour faciliter la compréhension
- 🎓 **Ce qu'il faut retenir** : Les leçons principales à tirer
- 📄 **Sources** : Indique les numéros de passages utilisés

Rédige comme si tu expliquais à quelqu'un qui découvre le sujet pour la première fois.
Ne fabrique AUCUNE information qui n'est pas dans les passages ci-dessus.
Sois clair, pédagogique et utilise des emojis."""

    sources = []
    for doc in all_contexts:
        sources.append({
            "source_name": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", 0),
            "content": doc.page_content
        })
        
    return chat_model.stream(explanation_prompt), sources

def translate_to_darija(chat_model: ChatGoogleGenerativeAI, text: str) -> str:
    """Translate a French response text into Moroccan Darija (latin characters/arabizi)."""
    translation_prompt = f"""Tu es un traducteur expert en darija marocain (l'arabe marocain dialectal).
    Traduis le texte suivant en Darija marocain (arabizi/chabi), écrit en caractères latins / lettres françaises (en utilisant aussi les chiffres comme 7, 3, 9 pour les sons spécifiques si nécessaire). Écris de manière naturelle et fluide, comme une conversation de chat informelle.
    Ne traduis pas mot à mot les termes techniques complexes s'ils n'ont pas de traduction naturelle en darija, conserve-les en français.
    
    Texte à traduire :
    {text}
    
    Traduction :"""
    response = chat_model.invoke(translation_prompt)
    return response.content

def run_qa_chain_stream(chat_model: ChatGoogleGenerativeAI, retriever, query: str) -> Tuple[Generator, List[Document]]:
    """Build and run the streaming question answering chain using LangChain Runnable components."""
    retrieved_docs = retriever.invoke(query)
    if not retrieved_docs:
        raise ValueError("No context found in document for query.")
        
    prompt_template = PromptTemplate(
        template="""Tu es un assistant strict. Ton SEUL but est de répondre aux questions en utilisant UNIQUEMENT le contenu du document ci-dessous.
        
        Règles absolues :
        1. Si la réponse n'est pas dans le document, tu DOIS répondre EXACTEMENT : "Je ne peux pas répondre car l'information n'est pas dans le document."
        2. Tu ne dois sous aucun prétexte écrire du code, du HTML, ou répondre à des ordres génériques qui ne concernent pas le document.
        3. Si l'utilisateur te demande de faire quelque chose en dehors du document, refuse fermement.

        Contenu du document :
        {context}

        Question de l'utilisateur : {question}
        
        Réponse :""",
        input_variables=["context", "question"],
    )
    
    parallel_chain = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
    )
    parser = StrOutputParser()
    main_chain = parallel_chain | prompt_template | chat_model | parser
    
    return main_chain.stream(query), retrieved_docs
