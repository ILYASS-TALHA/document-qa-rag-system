"""Module for handling Streamlit UI rendering and page styling."""
import streamlit as st
import time
from datetime import datetime
from typing import List, Optional, Any, Callable, Dict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.config import MODEL_NAME, DEFAULT_SYSTEM_MESSAGE

def configure_page():
    """Configure Streamlit page defaults."""
    st.set_page_config(
        page_title="Document RAG Chat",
        page_icon="📄",
        layout="centered",
    )
    st.title("📄 Document RAG Assistant")
    st.markdown("### Chat with your documents using AI")

def apply_custom_styles():
    """Apply custom CSS styling for UI responsiveness and visual aesthetics."""
    st.markdown(
        """
        <style>
        .block-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            max-width: 800px;
            margin: auto;
        }

        #transform-any-you-tube-video-into-an-interactive-conversation,#you-tube-x-rag-assistant{
            text-align: center;
        }

        .stTextInput, .stButton {
            width: 100% !important;
            margin: auto;
        }

        @media (max-width: 768px) {
            .stVerticalBlock {
                align-items: center;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def render_sidebar() -> Optional[List[Any]]:
    """Render side controls (config, controls, session metrics) and return uploaded documents."""
    st.sidebar.header("⚙️ Configuration")
    st.sidebar.info(f"**Modèle :** {MODEL_NAME}")
    st.sidebar.divider()

    st.sidebar.subheader("💬 Chat Controls")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.sidebar.button("🗑️ Clear Chat", use_container_width=True, key="clear_chat_btn"):
            st.session_state.messages = [SystemMessage(content=DEFAULT_SYSTEM_MESSAGE)]
            st.rerun()

    with col2:
        if st.sidebar.button("🔄 Clear Cache", use_container_width=True, key="clear_cache_btn"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Cache cleared!")

    # New Document button logic
    if st.sidebar.button("🔄 New Document", use_container_width=True, key="new_doc_btn"):
        for key in ["retriever", "vector_store", "document_name", "document_names", "document_genre", "suggested_questions"]:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.messages = [SystemMessage(content=DEFAULT_SYSTEM_MESSAGE)]
        st.success("🔄 Ready for new document!")
        time.sleep(1)
        st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("📊 Session Info")

    message_count = len(st.session_state.messages) - 1  # Exclude system message
    document_processed = "retriever" in st.session_state and st.session_state.get("retriever") is not None

    m_col1, m_col2 = st.sidebar.columns(2)
    with m_col1:
        st.metric("Messages", message_count)
    with m_col2:
        st.metric("Document", "✅" if document_processed else "❌")

    if document_processed:
        st.sidebar.success("📄 Document ready for chat")
    else:
        st.sidebar.info("📄 No Document processed yet")

    # Export Chat history download button
    if message_count > 0:
        st.sidebar.divider()
        chat_text = ""
        for msg in st.session_state.messages[1:]:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            chat_text += f"{role}: {msg.content}\n\n"

        st.sidebar.download_button(
            "📥 Download Chat",
            chat_text,
            f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "text/plain",
            use_container_width=True,
            help="Download your conversation history",
        )

    # Main area file uploader
    uploaded_files = st.file_uploader(
        "📁 Upload your documents",
        type=["pdf", "txt"],
        help="Upload PDF or text files to chat with",
        accept_multiple_files=True,
    )
    return uploaded_files

def render_sources_expander(sources: List[Dict[str, Any]]):
    """Render the expander detailing RAG sources used for a response."""
    with st.expander("🔍 Sources utilisées"):
        for s_idx, src in enumerate(sources):
            st.markdown(f"📄 **Nom du PDF :** `{src['source_name']}`")
            page_num = src['page']
            page_str = f"{page_num + 1}" if isinstance(page_num, int) else f"{page_num}"
            st.markdown(f"📖 **Numéro de la page :** {page_str}")
            st.markdown(f"💬 **Extrait utilisé :**\n> {src['content']}")
            if s_idx < len(sources) - 1:
                st.divider()

def render_message_actions(msg_index: int, content: str, has_translation: bool, on_translate: Callable):
    """Render action buttons (Download response and Translate to Darija)."""
    col_dl, col_trans = st.columns(2)
    with col_dl:
        st.download_button(
            label="📥 Télécharger la réponse",
            data=content,
            file_name=f"reponse_{msg_index}.txt",
            mime="text/plain",
            key=f"dl_{msg_index}",
            use_container_width=True
        )
    with col_trans:
        if not has_translation:
            if st.button("🇲🇦 Traduire en Darija", key=f"trans_{msg_index}", use_container_width=True):
                on_translate()

def display_chat_history(chat_model, translate_callback: Callable):
    """Render the entire conversation history in Streamlit chat messages."""
    for idx, message in enumerate(st.session_state.messages[1:]):
        msg_index = idx + 1
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)

        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)
                
                sources = message.additional_kwargs.get("sources", []) if message.additional_kwargs else []
                translation_darija = message.additional_kwargs.get("translation_darija") if message.additional_kwargs else None
                
                if sources:
                    render_sources_expander(sources)
                
                if translation_darija:
                    st.info(f"🇲🇦 **Traduction en Darija :**\n\n{translation_darija}")
                
                # Setup wrapper function for translating this specific message
                def make_translate_handler(msg=message):
                    translate_callback(msg)

                render_message_actions(
                    msg_index=msg_index,
                    content=message.content,
                    has_translation=bool(translation_darija),
                    on_translate=make_translate_handler
                )
